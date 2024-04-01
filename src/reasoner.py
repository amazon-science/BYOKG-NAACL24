import copy
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed
from gensim.summarization.bm25 import BM25
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sbert_util

from src.llm_wrapper import LLMWrapper
from src.utils.arguments import Arguments
from src.utils.generation import get_logprob_score_from_gen, fix_posthoc, construct_query_prompt_from_args, \
    prompt_arr_2_text, get_logprob_score, stringify_schema
from src.utils.generation_reasoner import default_instructions, \
    default_instructions_schema, default_query_prefix, default_schema_prefix, \
    default_output_prefix, default_output_prefix_chat, default_output_prefix_chat_beluga, default_inv_instructions, \
    default_inv_output_prefix, default_metrics, default_instructions_R, \
    default_inv_instructions_R, default_inv_instructions_schema, default_inv_output_prefix_chat, \
    default_inv_output_prefix_chat_beluga, construct_args_from_example, default_inv_query_prefix, \
    default_inv_schema_prefix, skip_mismatch, expand_candidates, process_candidates, bm25_tokenizer, \
    mask_question_entities, get_fewshot_samples_coverage_1, get_fewshot_samples_coverage_2, mask_sexpr_entities, \
    skip_empty, default_instructions_l2m, default_instructions_pangu, default_query_prefix_pangu, \
    default_output_prefix_pangu, rerank_results
from src.utils.helpers import setup_logger, split_underscore_period, remove_special_characters
from src.utils.parser import parse_bottom_up, bottom_up_to_sexpr
from src.utils.sparql import SPARQLUtil

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Reasoner(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, is_beluga=None, load_in_8bit=False, openai_api_key=None):
        super().__init__(model, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, is_beluga=is_beluga, load_in_8bit=load_in_8bit,
                         openai_api_key=openai_api_key)
        self.default_instructions = default_instructions
        self.default_instructions_R = default_instructions_R
        self.default_instructions_schema = default_instructions_schema
        self.default_query_prefix = default_query_prefix
        self.default_schema_prefix = default_schema_prefix
        self.default_question_prefix = default_output_prefix  # Used in few-shot demonstrations
        self.default_output_prefix = default_output_prefix
        if self.is_chat:
            self.default_output_prefix = default_output_prefix_chat
            if self.is_beluga:
                self.default_output_prefix = default_output_prefix_chat_beluga
        self.default_inv_instructions = default_inv_instructions
        self.default_inv_instructions_R = default_inv_instructions_R
        self.default_inv_instructions_schema = default_inv_instructions_schema
        self.default_inv_query_prefix = default_inv_query_prefix
        self.default_inv_schema_prefix = default_inv_schema_prefix
        self.default_inv_question_prefix = default_inv_output_prefix  # Used in few-shot demonstrations
        self.default_inv_output_prefix = default_inv_output_prefix
        if self.is_chat:
            self.default_inv_output_prefix = default_inv_output_prefix_chat
            if self.is_beluga:
                self.default_inv_output_prefix = default_inv_output_prefix_chat_beluga
        self.default_metrics = default_metrics

        try:
            if args.pangu_prompt:
                self.default_instructions = default_instructions_pangu
                self.default_instructions_R = []
                self.default_query_prefix = default_query_prefix_pangu
                self.default_output_prefix = default_output_prefix_pangu
                self.default_question_prefix = default_output_prefix_pangu  # Used in few-shot demonstrations
        except NameError:
            logger.info('args not defined')
            pass

    def custom_get_metric_scores(self, metric, predictions, references):
        _preds = fix_posthoc(predictions)
        _refs = fix_posthoc(references)
        n = len(_preds)
        per_ex_hits = defaultdict(list)
        rtn = {}
        if metric['name'] == 'accuracy':  # EM
            hits = 0.
            for p, r in zip(_preds, _refs):
                if type(p) is list:
                    # set-level
                    per_ex_hits[metric['name']].append(float(set(p) == set(r)))
                else:
                    # sequence-level
                    per_ex_hits[metric['name']].append(float(p == r))
                hits += per_ex_hits[metric['name']][-1]
            rtn = {
                'accuracy': hits * 1. / n
            }
        elif metric['name'] == 'f1':  # Token-level F1
            hits = 0.
            for p, r in zip(_preds, _refs):
                f1 = 0.
                if type(p) is list:
                    # set-level
                    n_same = len(set(p).intersection(set(r)))
                    if n_same != 0:
                        precision = 1.0 * n_same / len(p)
                        recall = 1.0 * n_same / len(r)
                        f1 = (2 * precision * recall) / (precision + recall)
                else:
                    # token-level
                    p_tokens = p.split()
                    r_tokens = r.split()
                    common = Counter(p_tokens) & Counter(r_tokens)
                    n_same = sum(common.values())
                    if n_same != 0:
                        precision = 1.0 * n_same / len(p_tokens)
                        recall = 1.0 * n_same / len(r_tokens)
                        f1 = (2 * precision * recall) / (precision + recall)
                per_ex_hits[metric['name']].append(f1)
                hits += per_ex_hits[metric['name']][-1]
            rtn = {
                'f1': hits * 1. / n
            }
        elif metric['name'] == 'hits@1':  # hits@1
            hits = 0.
            for p, r in zip(_preds, _refs):
                if type(p) is list:
                    # set-level
                    per_ex_hits[metric['name']].append(float(len(set(p).intersection(set(r))) > 0))
                else:
                    # sequence-level not computable for hits@1; compute EM instead
                    per_ex_hits[metric['name']].append(float(p == r))
                hits += per_ex_hits[metric['name']][-1]
            rtn = {
                'hits@1': hits * 1. / n
            }
        return rtn, per_ex_hits

    def get_byokg_zeroshot_prompt(self, query, anon_type, schema=None, inverse=False):
        if not inverse:
            instruction = self.default_instructions
            instructions_R = self.default_instructions_R
            instructions_schema = self.default_instructions_schema
            query_prefix = self.default_query_prefix
            schema_prefix = self.default_schema_prefix
            output_prefix = self.default_output_prefix
        else:
            instruction = self.default_inv_instructions
            instructions_R = self.default_inv_instructions_R
            instructions_schema = self.default_inv_instructions_schema
            query_prefix = self.default_inv_query_prefix
            schema_prefix = self.default_inv_schema_prefix
            output_prefix = self.default_inv_output_prefix

        prompt_arr = []
        instructions = copy.copy(instruction)
        if not anon_type.endswith('-rev'):
            instructions += instructions_R
        if schema is not None:
            instructions += instructions_schema
        instructions = " ".join(instructions)
        prompt_arr.append(instructions)
        prompt_arr += construct_query_prompt_from_args(query, query_prefix,
                                                       None, None, schema,
                                                       schema_prefix,
                                                       None, None, None)
        prompt = prompt_arr_2_text(prompt_arr, '\n\n', self.is_llama2, self.is_beluga,
                                   self.is_chat, output_prefix)
        return prompt

    def get_byokg_fewshot_prompt(self, query, anon_type, sampled_demos, schema=None, inverse=False,
                                 restrict_to_step=None, per_cand_schema=True, l2m_demos=False):
        if not inverse:
            instruction = self.default_instructions if not l2m_demos else default_instructions_l2m
            instructions_R = self.default_instructions_R
            instructions_schema = self.default_instructions_schema
            query_prefix = self.default_query_prefix
            schema_prefix = self.default_schema_prefix
            output_prefix = self.default_output_prefix
            question_prefix = self.default_question_prefix
        else:
            instruction = self.default_inv_instructions
            instructions_R = self.default_inv_instructions_R
            instructions_schema = self.default_inv_instructions_schema
            query_prefix = self.default_inv_query_prefix
            schema_prefix = self.default_inv_schema_prefix
            output_prefix = self.default_inv_output_prefix
            question_prefix = self.default_inv_question_prefix

        prompt_arr = []
        instructions = copy.copy(instruction)
        if not anon_type.endswith('-rev'):
            instructions += instructions_R
        if schema is not None:
            instructions += instructions_schema
        instructions = " ".join(instructions)
        prompt_arr.append(instructions)
        # Demonstrations
        demo_fields = []
        common_schema, common_schema_str = None, None
        for d in sampled_demos:
            d_inf_args, d_ref = construct_args_from_example(d, anon_type, schema is not None)
            d_schema = d_inf_args['schema']  # stringified
            if restrict_to_step is not None:
                # Restrict d_ref to at most the current step's expansion
                d_ref = bottom_up_to_sexpr(parse_bottom_up(d_ref)[:restrict_to_step + 1])
                d_schema = None if d_schema is None else stringify_schema(schema=d['schema'],
                                                                          anon_type=anon_type,
                                                                          query=d_ref)
            if l2m_demos:
                d_ref = '\n'.join(bottom_up_to_sexpr(d_ref, return_all=True))
            if not per_cand_schema:
                if common_schema is None:
                    common_schema = d['schema']
                else:
                    common_schema = {k: v | d['schema'][k] for k, v in common_schema.items()}
            input = d_inf_args['query'] if not inverse else d_ref
            output = d_ref if not inverse else d_inf_args['query']
            demo_fields.append((input, output, d_schema))
        if not per_cand_schema:
            common_schema_str = stringify_schema(schema=common_schema, anon_type=anon_type,
                                                 query=[d[0] if inverse else d[1] for d in demo_fields])
        prompt_arr_demos = []
        for d in demo_fields:
            prompt_arr_demos.append("\n".join(construct_query_prompt_from_args(d[0], query_prefix,
                                                                               None, None, d[
                                                                                   2] if per_cand_schema else common_schema_str,
                                                                               schema_prefix,
                                                                               None, None, None, question=d[1],
                                                                               question_prefix=question_prefix)))
        # Query
        query_arr = construct_query_prompt_from_args(query, query_prefix,
                                                     None, None, schema,
                                                     schema_prefix,
                                                     None, None, None)
        query_arr = ["\n".join(query_arr) + f"\n{output_prefix}"]

        prompt_arr += prompt_arr_demos + query_arr
        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, '\n\n', self.is_llama2, self.is_beluga,
                                   self.is_chat, output_prefix="")
        return prompt

    def get_res_dir_name(self, dataset_name, split, n_hops, n_samples, inf_fn_key, n_shots, retrieval_strategy,
                         restrict_to_step, demos_label, anon_type, _strat, use_schema, per_cand_schema, beam_size,
                         sparql_timeout, run_id):
        res_dir = ["eval-byokg"]
        if dataset_name is not None:
            res_dir += [dataset_name]
        res_dir += [self.model_name + ("_chat" if self.is_chat else ""), split]
        if n_hops is not None and n_hops != 0:
            res_dir += [f"{n_hops}-hop"]
        if n_samples is not None:
            res_dir += [str(n_samples)]
        method = inf_fn_key
        if method == 'fewshot':
            method = f'{method}-{n_shots}-{retrieval_strategy}'
            if restrict_to_step:
                method = f'{method}-restrict'
            if demos_label is not None:
                method = f'{method}-{demos_label}'
        res_dir += [method, anon_type, _strat]
        if use_schema:
            res_dir += ['schema']
            if per_cand_schema:
                res_dir[-1] = f'{res_dir[-1]}-per-cand'
        res_dir += [f'beam{beam_size}', f'timeout{sparql_timeout}']
        res_dir += [run_id]
        res_dir = f"{'_'.join(res_dir)}"
        return res_dir

    def eval_byokg(self, oracle_el=True, expand_using_classes_only=False, beam_size=5, max_steps=5, sparql_timeout=None,
                   inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None, sparql_url=None, demos_label=None,
                   dataset_sample_strategy='static', use_schema=False, anon_type='machine-rev', per_cand_schema=False,
                   exit_on_sparql_error=False, output_sampling_strategy='max', inv_consistency_alpha=1.,
                   restrict_to_step=False, kg_name="freebase", demos_n_samples=-1, l2m_demos=False,
                   all_inv_consistency_alpha=False, sparql_cache=None, n_shots=None, retrieval_strategy='random',
                   demos_split='train', n_hops=None, rev_schema_fpath=None, scorer_batch_size=10, verbose=False,
                   out_dir=None, dataset_name=None, run_id=str(int(time.time())),
                   **inf_fn_kwargs):
        if metrics is None:
            metrics = copy.deepcopy(self.default_metrics)
        eval_args = locals()
        del eval_args["self"]
        logger.info(f"Eval arguments: {eval_args}")
        dataset = self.datasets[split]
        if (n_samples is not None and n_samples != -1) and n_samples < len(dataset):
            if dataset_sample_strategy == 'static':
                dataset = dataset[:n_samples]
            elif dataset_sample_strategy == 'random':
                dataset = random.sample(dataset, n_samples)
        if inf_fn_key == "fewshot":
            demos = self.datasets[demos_split]
            if demos_n_samples != -1 and demos_n_samples < len(demos):
                demos = random.sample(demos, demos_n_samples)
        references_sexpr, references_ans, _examples = [], [], []
        _sparql = SPARQLUtil(sparql_url, cache_fpath=sparql_cache,
                             retry_on_cache_none_override=args.retry_on_cache_none if args.retry_on_cache_none else None,
                             graph_name=args.kg_name)
        if sparql_timeout is not None:
            _sparql.wrapper.setTimeout(sparql_timeout)
        rev_schema = None
        legal_relation_set = None
        if rev_schema_fpath is not None:
            with open(rev_schema_fpath, 'r') as fh:
                if rev_schema_fpath.endswith('.json'):
                    # Our style of schema
                    rev_schema = json.load(fh)
                else:
                    # GrailQA style of reverse properties
                    rev_schema = {"relations": {}, "inverse_relations": {}}
                    while _line := fh.readline():
                        _rel, _inv_rel = _line.split('\t')
                        rev_schema["relations"][_rel.strip()] = {"reverse": _inv_rel.strip()}
                        rev_schema["inverse_relations"][_inv_rel.strip()] = _rel.strip()
            legal_relation_set = set(rev_schema["relations"].keys()).union(set(rev_schema["inverse_relations"].keys()))

        # Setup retriever if few-shot mode
        bm25 = None
        sbert_model = None
        if inf_fn_key == "fewshot":
            if retrieval_strategy == "bm25":
                if args.pangu_prompt:
                    demos_text = [d['question_label'] for d in demos]
                else:
                    demos_text = [d['question_machine'] for d in demos]
                demos_corpus = [bm25_tokenizer(t, skip_entities=True) for t in demos_text]
                bm25 = BM25(demos_corpus)
            elif retrieval_strategy == "dense":
                if args.fewshot_anon_strat == 'mask':
                    demos_corpus = [mask_question_entities(d['question_machine']) for d in demos]
                    q_corpus = [mask_question_entities(d['question_machine']) for d in dataset]
                else:
                    demos_corpus = [mask_question_entities(d[f'question_{args.fewshot_anon_strat}']) for d in demos]
                    q_corpus = [mask_question_entities(d[f'question_{args.fewshot_anon_strat}']) for d in dataset]
                sbert_model = SentenceTransformer('all-mpnet-base-v2')
                demos_embeddings = sbert_model.encode(demos_corpus, convert_to_tensor=True, batch_size=10,
                                                      device=args.sbert_device)
                q_embeddings = sbert_model.encode(q_corpus, convert_to_tensor=True, batch_size=10,
                                                  device=args.sbert_device)
                if len(q_embeddings) <= 3000:
                    dense_scores = sbert_util.cos_sim(q_embeddings, demos_embeddings)
                else:
                    dense_scores = []
                    # Compute per example on the fly
                    # for _b in range(0, len(q_embeddings), 1000):
                    #     dense_scores += list(sbert_util.cos_sim(q_embeddings[_b:_b+1000], demos_embeddings))
                if args.fewshot_coverage_strat1:
                    sampled_demo_set = get_fewshot_samples_coverage_1(dataset, demos, dense_scores, n_shots=n_shots)
                elif args.fewshot_coverage_strat2:
                    sampled_demo_set = get_fewshot_samples_coverage_2(dataset, demos, dense_scores, n_shots=n_shots)

        if args.prune_candidates and sbert_model is None:
            if args.fewshot_anon_strat == 'mask':
                q_corpus = [mask_question_entities(d['question_machine']) for d in dataset]
            else:
                q_corpus = [mask_question_entities(d[f'question_{args.fewshot_anon_strat}']) for d in dataset]
            sbert_model = SentenceTransformer('all-mpnet-base-v2')
            q_embeddings = sbert_model.encode(q_corpus, convert_to_tensor=True, batch_size=10, device=args.sbert_device)

        _strats = ['max', 'inverse-len-norm'] if output_sampling_strategy == 'all' else [output_sampling_strategy]
        predictions, prediction_sepxrs, examples = defaultdict(list), defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)

        # Sanity check
        sanity_correct_idxs = []
        sanity_error_idxs_step = []
        n_cands_enumerated = []

        # Load predicted starting candidates (entity linking)
        pred_start_cands = None
        if args.pred_start_cands_fpath is not None:
            with open(args.pred_start_cands_fpath, 'r') as fh:
                pred_start_cands = json.load(fh)
        elif not oracle_el:
            logger.info('--pred_start_cands_fpath not defined in non-oracle mode. Changing execution to oracle-mode.')
            oracle_el = True

        # Run BYOKG-reasoner
        start_time = time.time()

        if args.save_interval == -1:
            _datasets = [dataset]
        else:
            n_intervals = math.ceil(len(dataset) / args.save_interval)
            _datasets = [dataset[i * args.save_interval: (i + 1) * args.save_interval] for i in range(n_intervals)]
            assert sum([len(d) for d in _datasets]) == len(dataset)

        for interval_idx, _dataset in enumerate(_datasets):
            if interval_idx < args.start_interval:
                logger.info(f'Skipping interval idx {interval_idx}')
                continue
            for idx, d in enumerate(
                    tqdm(_dataset, desc=f"Evaluating dataset interval {interval_idx + 1}/{len(_datasets)}")):
                idx = interval_idx * len(_datasets[0]) + idx
                sanity_skip_to_next_example = False
                if args.sanity_check:
                    logger.info(f"# IDX {idx}")

                nl_query = d['question_label']
                if 's_expression_machine' in d:
                    references_sexpr.append(d['s_expression_machine'])
                if "answer" in d:
                    if args.kg_name != "freebase":
                        references_ans.append([d_ans["value_readable"] for d_ans in d["answer"]])
                    else:
                        references_ans.append([d_ans["value"] for d_ans in d["answer"]])

                # Get starting entities, classes, and/or literals
                if oracle_el:
                    start_cands = d['question_entities']
                else:
                    start_cands = pred_start_cands[str(d['original_qid'])]

                # Construct inf_fn arguments
                inf_args, _ = construct_args_from_example(d, anon_type=anon_type, use_schema=False)
                anon_query = inf_args['query']

                # Construct prompt
                if inf_fn_key == "zeroshot":
                    prompt = self.get_byokg_zeroshot_prompt(anon_query, anon_type)
                else:
                    # Few-shot
                    if retrieval_strategy == "random":
                        sampled_demos = random.sample(demos, n_shots)
                    elif retrieval_strategy == "bm25":
                        bm25_scores = bm25.get_scores(
                            bm25_tokenizer(d['question_label'] if args.pangu_prompt else d['question_machine'],
                                           skip_entities=True))
                        top_idxs = sorted(range(len(demos)), key=lambda i: bm25_scores[i], reverse=True)[:n_shots]
                        sampled_demos = [demos[i] for i in top_idxs]
                    elif retrieval_strategy == "dense":
                        if args.fewshot_coverage_strat1 or args.fewshot_coverage_strat2:
                            sampled_demos = sampled_demo_set[idx]
                        else:
                            if len(dense_scores) == 0:
                                _dense_scores = sbert_util.cos_sim(q_embeddings[idx:idx + 1], demos_embeddings)[0]
                            else:
                                _dense_scores = dense_scores[idx]
                            if args.fewshot_coverage_strat3:
                                top_scores, top_idxs = torch.topk(_dense_scores, k=len(_dense_scores))
                                seen_patterns = defaultdict(int)
                                sampled_demos = []
                                for i in top_idxs:
                                    cand_demo_pattern = mask_sexpr_entities(demos[i]['s_expression_machine'])
                                    seen_patterns[cand_demo_pattern] += 1
                                    if seen_patterns[cand_demo_pattern] <= args.dup_fewshot_threshold:
                                        sampled_demos.append(demos[i])
                                    if len(sampled_demos) == n_shots:
                                        break
                                assert len(sampled_demos) == n_shots
                            else:
                                top_scores, top_idxs = torch.topk(_dense_scores, k=n_shots)
                                sampled_demos = [demos[i] for i in top_idxs]
                    # Construct prompt (few-shot)
                    prompt = self.get_byokg_fewshot_prompt(anon_query, anon_type=anon_type, sampled_demos=sampled_demos)

                _example = {
                    "idx": idx + 1,
                    "id": d["id"],
                    "original_qid": d["original_qid"],
                    "original_fname": d.get("original_fname", "none"),
                    "split": d.get("level", d.get("original_fname", "nosplit")),
                    "prompt": prompt,
                    # Actual input to model might be differentbased on demonstrations / candidate schema in each step of reasoning
                    "query": nl_query,
                    "start_cands": start_cands,
                    "reference_sexpr": references_sexpr[-1] if len(references_sexpr) > 0 else None,
                    "reference_ans": references_ans[-1] if len(references_ans) > 0 else None,
                    "fewshot_examples": None,
                    "prediction_sexpr": None,
                    "prediction_ans": None
                }
                if inf_fn_key == 'zeroshot':
                    del _example['fewshot_examples']
                else:
                    _example['fewshot_examples'] = []
                    for _s in sampled_demos:
                        _example['fewshot_examples'].append(
                            (_s[f'question_{anon_type}'], _s[f's_expression_{anon_type}']))

                if verbose:
                    logger.info(f"idx={_example['idx']}:")
                    logger.info(f"Question: {nl_query}")
                    logger.info(f"Gold: {_example['reference_sexpr']}\n")

                # TODO: Think about whether search should be grounded or not.
                # If based on executing the logical form against some datasource, we might prematurely stop the parse because a sub-clause already leads to an empty set.
                # Works for question answering, but if the goal is semantic parsing, we might want to use the domain and range of the relations to determine next candidates.
                # This would be cheaper too? (expand using domain-range info, but get LLM scores with groundings)

                # Candidate set expansion by KG querying and LLM scoring
                candidates = defaultdict(list)
                top_candidates = defaultdict(list)
                per_step_top_candidates = defaultdict(dict)  # used for logging
                stop_step = defaultdict(int)
                stopped_strats = []
                for _strat in _strats:
                    stop_step[_strat] = max_steps
                    for step in range(max_steps):  # start multi-step reasoning
                        if _strat in stopped_strats:
                            continue
                        cands = candidates[_strat]

                        if step == 0:
                            cands.append(start_cands)

                        if verbose:
                            logger.info(f"[{_strat}] All candidates at step {step}: {cands[step]}")
                            logger.info(f"[{_strat}] Building candidates for step {step + 1}.")

                        # Expand candidate set based on reachable relations/classes/functions
                        cands.append([])  # initialize next step with empty list
                        expand_candidates(_sparql, cands, step,
                                          list(zip(*top_candidates[_strat]))[0] if len(
                                              top_candidates[_strat]) > 0 else [],
                                          use_classes_only=expand_using_classes_only, freebase=args.kg_name == "freebase",
                                          skip_on_error=not exit_on_sparql_error, verbose=verbose, schema=rev_schema,
                                          legal_relation_set=legal_relation_set,
                                          use_type_constraint=args.type_constraint,
                                          sanity_check=args.sanity_check)

                        # If no new candidate added, then terminate
                        if len(cands[step + 1]) == 0:
                            stopped_strats.append(_strat)
                            stop_step[_strat] = step + 1
                            if verbose:
                                logger.info(f"[{_strat}] Step {step + 1} is empty. Stopping.")
                            continue
                        n_cands_enumerated.append(len(cands[step + 1]))

                        if args.prune_candidates and len(cands[step + 1]) > args.prune_candidates_threshold:
                            if args.pruning_anon_strat not in ['machine', 'mask']:
                                cands_to_score, step_schema = process_candidates(cands[step + 1],
                                                                                 args.pruning_anon_strat, _sparql,
                                                                                 rev_schema=rev_schema,
                                                                                 per_cand_schema=per_cand_schema)
                            else:
                                cands_to_score = cands[step + 1]
                            if verbose:
                                logger.info(f'step={step + 1}, n_cands={len(cands_to_score)}')
                            _cands_to_embed = cands_to_score
                            q_prune = [d[f'question_{args.pruning_anon_strat}'] if args.pruning_anon_strat != 'mask' \
                                           else mask_question_entities(d[f'question_machine'])]
                            if args.strip_cands_before_pruning:
                                _cands_to_embed = list(map(split_underscore_period, _cands_to_embed))
                            if args.preprocess_before_pruning:
                                _cands_to_embed = [str.lower(
                                    remove_special_characters(split_underscore_period(_c))) for _c in _cands_to_embed]
                                q_prune = [str.lower(remove_special_characters(split_underscore_period(_q))) for _q in
                                           q_prune]
                            if not args.bm25_pruning:
                                cand_embeddings = sbert_model.encode(_cands_to_embed, convert_to_tensor=True,
                                                                     show_progress_bar=False, batch_size=10,
                                                                     device=args.sbert_device)
                                q_embedding = sbert_model.encode(q_prune, convert_to_tensor=True, show_progress_bar=False,
                                                                 batch_size=10, device=args.sbert_device)
                                cand_scores = sbert_util.cos_sim(q_embedding[:1], cand_embeddings)
                                top_scores, top_idxs = torch.topk(cand_scores[0], k=args.prune_candidates_threshold)
                            else:
                                cand_embeddings = [bm25_tokenizer(_cand, skip_entities=False) for _cand in
                                                   _cands_to_embed]
                                bm25 = BM25(cand_embeddings)
                                _scores = bm25.get_scores(bm25_tokenizer(q_prune[0], skip_entities=False))
                                top_idxs = sorted(range(len(cand_embeddings)), key=lambda i: _scores[i], reverse=True)[
                                           :args.prune_candidates_threshold]
                            cands[step + 1] = [cands[step + 1][_i] for _i in top_idxs]

                        if args.sanity_check:  # for oracle recall experiment
                            cands_to_score = cands[step + 1]
                            sanity_valid_step = False
                            sanity_correct_cands = []
                            for c in cands_to_score:
                                if not sanity_skip_to_next_example:
                                    if c == _example['reference_sexpr']:
                                        sanity_correct_idxs.append(idx)
                                        sanity_valid_step = True
                                        sanity_skip_to_next_example = True
                                        logger.info(f"Passed.")
                                        break
                                    if c in _example['reference_sexpr']:
                                        sanity_valid_step = True
                                        sanity_correct_cands.sort(key=len, reverse=True)
                                        add_new_cand = True
                                        for saved_cand_i, saved_cand in enumerate(sanity_correct_cands):
                                            if c in saved_cand:
                                                add_new_cand = False
                                                break
                                            if saved_cand in c and len(c) > len(saved_cand):
                                                sanity_correct_cands[saved_cand_i] = c
                                                for ti, t in enumerate(top_candidates[_strat]):
                                                    if t[0] == saved_cand:
                                                        top_candidates[_strat][ti] = (c, 1., step + 1)
                                                        break
                                                add_new_cand = False
                                                break
                                        if add_new_cand:
                                            sanity_correct_cands.append(c)
                                            top_candidates[_strat].append((c, 1., step + 1))
                            if not sanity_valid_step:
                                sanity_error_idxs_step.append((idx, step + 1))
                                logger.info(f"#{idx + 1} failed: {_example['reference_sexpr']}")
                                sanity_skip_to_next_example = True
                                if args.sanity_check_breakpoint:
                                    breakpoint()
                            elif not sanity_skip_to_next_example:
                                if args.sanity_check_keep_gold_only:
                                    cands[step + 1] = sanity_correct_cands
                                else:
                                    cands[step + 1] = sanity_correct_cands
                                    for c in cands[step + 1][:args.beam_size]:
                                        if len(cands[step + 1]) == args.beam_size:
                                            break
                                        if c not in cands[step + 1]:
                                            cands[step + 1].append(c)
                            logger.info(
                                f'n_sanity_correct_idxs = {len(sanity_correct_idxs)} / {idx + 1} ({round(len(sanity_correct_idxs) / (idx + 1) * 100, 2)}%)')
                        else:
                            # Convert candidates to anon_type before scoring
                            cands_to_score, step_schema = process_candidates(cands[step + 1], anon_type, _sparql,
                                                                             rev_schema=rev_schema,
                                                                             per_cand_schema=per_cand_schema)
                            if verbose:
                                logger.info(
                                    f"[{_strat}] Step {step + 1}: New candidates to score:\n{cands_to_score}")

                            if not use_schema:
                                step_schema = None

                            cand_scores = [0.] * len(cands_to_score)
                            if 'inverse' not in _strat or inv_consistency_alpha != 1:
                                cand_prompts = []
                                for _c_idx, c in enumerate(cands_to_score):
                                    _step_schema = step_schema[_c_idx] if type(step_schema) is list else step_schema
                                    # Score candidates added to (step+1) and retain top-k (beam-size)
                                    if inf_fn_key == "zeroshot":
                                        prompt = self.get_byokg_zeroshot_prompt(anon_query, anon_type,
                                                                                schema=_step_schema)
                                    else:
                                        # fewshot
                                        prompt = self.get_byokg_fewshot_prompt(anon_query, anon_type, sampled_demos,
                                                                               schema=_step_schema,
                                                                               per_cand_schema=per_cand_schema,
                                                                               restrict_to_step=step if restrict_to_step else None,
                                                                               l2m_demos=l2m_demos)
                                    if l2m_demos:
                                        prompt += '\n'.join(
                                            bottom_up_to_sexpr(parse_bottom_up(c.replace('""', '"'), expand=True)[:-1],
                                                               return_all=True))
                                        if not prompt.endswith('\n'):
                                            prompt += '\n'
                                    cand_prompts.append(prompt)
                                cand_scores = get_logprob_score(target=cands_to_score,
                                                                prefix=cand_prompts,
                                                                model=self.model,
                                                                tokenizer=self.tokenizer,
                                                                len_norm=True,
                                                                bsz=scorer_batch_size)
                            if 'inverse' in _strat and inv_consistency_alpha != 0:
                                inv_cand_prompts = []
                                for _c_idx, c in enumerate(cands_to_score):
                                    _step_schema = step_schema[_c_idx] if type(step_schema) is list else step_schema
                                    if inf_fn_key == "zeroshot":
                                        inv_prompt = self.get_byokg_zeroshot_prompt(c, anon_type, schema=_step_schema,
                                                                                    inverse=True)
                                    else:
                                        # fewshot
                                        inv_prompt = self.get_byokg_fewshot_prompt(c, anon_type, sampled_demos,
                                                                                   schema=_step_schema, inverse=True,
                                                                                   per_cand_schema=per_cand_schema,
                                                                                   restrict_to_step=step if restrict_to_step else None)
                                    inv_cand_prompts.append(inv_prompt)
                                inv_cand_scores = get_logprob_score(target=[anon_query] * len(inv_cand_prompts),
                                                                    prefix=inv_cand_prompts,
                                                                    model=self.model,
                                                                    tokenizer=self.tokenizer,
                                                                    len_norm=True,
                                                                    bsz=scorer_batch_size)
                                assert 0. <= inv_consistency_alpha <= 1.
                                cand_scores = list(
                                    map(lambda x: (1 - inv_consistency_alpha) * x[0] + inv_consistency_alpha * x[1],
                                        zip(cand_scores, inv_cand_scores)))

                            n_enumerated = len(cands[step + 1])
                            if verbose:
                                logger.info(
                                    f"[{_strat}] Step {step + 1}: Enumerated {n_enumerated} paths for step {step + 1}: {cands[step + 1]}")

                            if legal_relation_set is not None and 1. > args.relation_repetition_factor > 0.:
                                for ci, c in enumerate(cands[step + 1]):
                                    rel_count = defaultdict(int)
                                    for c_tkn in c.split():
                                        c_tkn = c_tkn.replace('(', '').replace(')', '')
                                        if c_tkn in legal_relation_set:
                                            rel_count[c_tkn] += 1
                                    assert max(rel_count.values()) != 0
                                    cand_scores[ci] += (max(rel_count.values()) - 1) * np.log(
                                        args.relation_repetition_factor)

                            sorted_cand_scores = sorted(
                                list(zip(cands[step + 1], cand_scores, [step + 1] * len(cand_scores))) + top_candidates[
                                    _strat],
                                key=lambda x: x[1],
                                reverse=True)
                            # Retain top candidates and prune from next step
                            top_candidates[_strat] = sorted_cand_scores[:beam_size]
                            per_step_top_candidates[_strat][step + 1] = top_candidates[_strat]

                            # Enumerate top candidates that will be expanded in the next iteration
                            next_step = []
                            for tc in top_candidates[_strat]:
                                if tc[2] == step + 1:
                                    next_step.append(tc[0])
                            n_retained = len(next_step)
                            cands[step + 1] = next_step
                            if verbose:
                                logger.info(
                                    f"[{_strat}] Step {step + 1}: Retained {n_retained} (of {n_enumerated}) in the top-{beam_size} candidates.")
                                logger.info(
                                    f"[{_strat}] Step {step + 1}: Top-{beam_size} candidates:\n{json.dumps(top_candidates[_strat], indent=2)}")
                            if n_retained == 0:
                                stopped_strats.append(_strat)
                                stop_step[_strat] = step + 1
                                if verbose:
                                    logger.info(
                                        f"[{_strat}] Step {step + 1} is empty. Stopping.")

                        if sanity_skip_to_next_example:
                            break

                    if args.sanity_check:
                        break

                    if len(top_candidates[_strat]) > 0:
                        examples[_strat].append(dict(_example, prediction_sexpr_candidates=top_candidates[_strat]))
                        pred_sexpr = top_candidates[_strat][0][0]
                        pred_step = top_candidates[_strat][0][2]
                        examples[_strat][-1].update({'prediction_sexpr': pred_sexpr})
                        _sparql.wrapper.setTimeout(30)
                        ans_set = _sparql.get_answer_set_ent_val(pred_sexpr, retry_on_cache_none=True)
                        _sparql.wrapper.setTimeout(sparql_timeout if sparql_timeout is not None else 5)
                        prediction = []
                        for a in ans_set:
                            if args.kg_name != "freebase" and a['answer_type'] == 'Entity':
                                prediction.append(a['entity_name'])
                            else:
                                prediction.append(a['answer_argument'])  # 'Value'
                        examples[_strat][-1].update({'prediction_ans': prediction})
                        examples[_strat][-1].update({'per_step_top_cands': per_step_top_candidates[_strat]})
                        predictions[_strat].append(prediction)
                        prediction_sepxrs[_strat].append(pred_sexpr)
                        examples[_strat][-1].update({
                            'pred_step': pred_step,
                            'search_stop_step': stop_step[_strat],
                        })
                    else:
                        assert len(start_cands) == 0 or (len(cands[0]) > 0 and len(cands[1]) == 0)
                        examples[_strat].append(dict(_example, prediction_sexpr_candidates=[]))
                        examples[_strat][-1].update({'prediction_sexpr': ''})
                        examples[_strat][-1].update({'prediction_ans': []})
                        examples[_strat][-1].update({'per_step_top_cands': {}})
                        predictions[_strat].append([])
                        prediction_sepxrs[_strat].append('')
                        examples[_strat][-1].update({
                            'pred_step': None,
                            'search_stop_step': None,
                        })

            if args.sanity_check:
                logger.info('### SANITY CHECK:')
                logger.info(f'n_sanity_correct_idxs = {len(sanity_correct_idxs)} / {len(dataset)}')
                logger.info(f'sanity_error_idxs_step = {sanity_error_idxs_step}')
                if len(sanity_correct_idxs) != len(dataset):
                    logger.info('Check failed. Invoking breakpoint...')
                else:
                    logger.info('Check passed. Invoking breakpoint...')
                breakpoint()

            end_time = time.time()

            for _strat in _strats:
                # Compute metrics
                if len(references_ans) > 0:
                    metric_results[_strat]['per_split'] = {}
                    for metric in metrics:
                        preds = predictions[_strat] if metric['name'] == 'f1' else predictions[
                            _strat]  # prediction_sepxrs[_strat]
                        refs = references_ans if metric['name'] == 'f1' else references_ans  # references_sexpr
                        # TODO: Currently using answer set to measure EM; get sexpr EM to work; challenge is type constraint
                        scores, per_ex_scores = self.custom_get_metric_scores(metric, preds, refs)
                        for k in metric["score_keys"]:
                            metric_results[_strat][f"{metric['name']}.{k}"] = round(scores[k], 4)
                            examples[_strat] = [dict(_ex, **{f"{metric['name']}.{k}": per_ex_scores[k][_i]}) for _i, _ex
                                                in enumerate(examples[_strat])]
                            for _i, _ex in enumerate(examples[_strat]):
                                if _ex['split'] not in metric_results[_strat]['per_split']:
                                    metric_results[_strat]['per_split'][_ex['split']] = {}
                                if f"{metric['name']}.{k}" not in metric_results[_strat]['per_split'][_ex['split']]:
                                    metric_results[_strat]['per_split'][_ex['split']][f"{metric['name']}.{k}"] = []
                                metric_results[_strat]['per_split'][_ex['split']][f"{metric['name']}.{k}"].append(
                                    per_ex_scores[k][_i])
                    for _k, _v in metric_results[_strat]['per_split'].items():
                        for __k, __v in _v.items():
                            metric_results[_strat]['per_split'][_k][__k] = round(float(np.mean(__v)), 4)
                        metric_results[_strat]['per_split'][_k]['n_total'] = len(__v)
                    if verbose:
                        logger.info(metric_results[_strat])

                # Save results
                res_dir = self.get_res_dir_name(dataset_name, split, n_hops, n_samples, inf_fn_key, n_shots,
                                                retrieval_strategy, restrict_to_step, demos_label, anon_type, _strat,
                                                use_schema, per_cand_schema, beam_size, sparql_timeout, run_id)

                results[_strat] = {
                    "experiment": res_dir,
                    "interval_idx": interval_idx,
                    "n_total_in_interval": len(_datasets[0]) * interval_idx + len(_dataset),
                    "n_total": len(dataset),
                    "eval_args": eval_args,
                    "scores": metric_results[_strat],
                    "openai_usage": None,
                    "time_taken": end_time - start_time,
                    "time_taken_per_q": (end_time - start_time) / (len(_datasets[0]) * interval_idx + len(_dataset)),
                    "avg_pred_step": float(
                        np.mean([e['pred_step'] for e in examples[_strat] if e['pred_step'] is not None])),
                    "avg_search_stop_step": float(np.mean(
                        [e['search_stop_step'] for e in examples[_strat] if e['search_stop_step'] is not None])),
                    "avg_cands_enumerated_per_step": float(np.mean(n_cands_enumerated)),
                    "max_cands_enumerated_per_step": int(np.max(n_cands_enumerated)),
                    "examples": examples[_strat]
                }
                if not llm.is_openai:
                    del results[_strat]['openai_usage']
                else:
                    results[_strat]['openai_usage'] = llm.model['session_tokens']

                if out_dir is not None:
                    res_dir_fpath = os.path.join(out_dir, res_dir)
                    os.makedirs(res_dir_fpath, exist_ok=True)
                    out_fname = "results.json"
                    out_fpath = os.path.join(res_dir_fpath, out_fname)
                    with open(out_fpath, 'w') as fh:
                        fh.write(json.dumps(results[_strat], indent=2))
                    logger.info(f"Saved results to {out_fpath}")
                    if args.save_eval_dataset:
                        with open(os.path.join(res_dir_fpath, 'eval_dataset.json'), 'w') as fh:
                            fh.write(json.dumps(dataset, indent=2))

        return results


if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups=["llm", "reasoner"])
    global args
    args = cli_args.parse_args()
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else str(args.run_id)
    setup_logger(RUN_ID)
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)
    # Create output dir
    out_dir = os.path.join(args.out_dir, "reasoning")
    if args.clean_out_dir:
        for f in os.listdir(out_dir):
            fpath = os.path.join(out_dir, f)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
    os.makedirs(out_dir, exist_ok=True)

    llm = Reasoner(model=args.model if not args.sanity_check else None, is_chat=args.eval_chat_mode,
                   load_in_8bit=args.load_in_8bit, openai_api_key=args.openai_api_key)
    if llm.is_openai:
        logger.info(f"As of 01/2024, OpenAI API does not provide access to the logits of the full sequence, which is required for the BYOKG reasoner. We're therefore currently only supporting models hosted on HuggingFace.")
        exit(1)

    if args.rerank_fpath is not None:
        rerank_results(llm, args.rerank_fpath, args.rev_schema_fpath, args.sparql_url, args.kg_name,
                       sparql_cache=args.sparql_cache, sparql_retry_on_cache_none=args.retry_on_cache_none,
                       alpha=args.rerank_alpha, rel_rep_factor=args.rerank_rel_rep_factor, anon_type=args.eval_anon_type,
                       inverse=args.rerank_inverse, pmi=args.rerank_pmi)
    else:
        filter_fns = []
        if args.skip_mismatch:
            filter_fns.append(skip_mismatch)
        if args.skip_empty:
            filter_fns.append(skip_empty)
        if args.eval_split == 'train' or args.eval_inf_fn_key == 'fewshot':
            llm.load_train_set(fpath=args.load_train_fpath, filter_fns=filter_fns)
        if args.eval_split == 'dev':
            llm.load_dev_set(fpath=args.load_dev_fpath, filter_fns=filter_fns)
        if args.eval_split == 'test':
            llm.load_test_set(fpath=args.load_test_fpath, filter_fns=filter_fns)

        results = llm.eval_byokg(
            dataset_name=args.dataset_name, inf_fn_key=args.eval_inf_fn_key, split=args.eval_split,
            n_hops=args.eval_n_hops, n_samples=args.eval_n_samples, out_dir=out_dir, anon_type=args.eval_anon_type,
            use_schema=args.eval_use_schema, verbose=args.verbose, n_shots=args.eval_n_shots,
            demos_label=args.demos_label, retrieval_strategy=args.eval_retrieval_strategy, sparql_url=args.sparql_url,
            max_steps=args.max_steps, dataset_sample_strategy=args.dataset_sample_strategy,
            demos_n_samples=args.demos_n_samples,
            sparql_cache=args.sparql_cache, per_cand_schema=args.per_cand_schema, demos_split=args.demos_split,
            restrict_to_step=args.restrict_to_step, kg_name=args.kg_name, oracle_el=args.oracle_el,
            output_sampling_strategy=args.eval_output_sampling_strategy, rev_schema_fpath=args.rev_schema_fpath,
            expand_using_classes_only=args.expand_using_classes_only, exit_on_sparql_error=args.exit_on_sparql_error,
            inv_consistency_alpha=args.eval_inv_consistency_alpha, beam_size=args.beam_size, l2m_demos=args.l2m_demos,
            all_inv_consistency_alpha=args.eval_all_inv_consistency_alpha, sparql_timeout=args.sparql_timeout,
            use_gen_norm_seq_scores=args.eval_use_gen_norm_seq_scores, scorer_batch_size=args.scorer_batch_size,
            use_alt_seq_scores=args.eval_use_alt_seq_scores, run_id=RUN_ID
        )

    if args.debug:
        breakpoint()
