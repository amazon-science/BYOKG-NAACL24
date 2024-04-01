import json
import logging
import copy
import os
import random
import time
import shutil
import math

import torch
from transformers import set_seed
from tqdm import tqdm
import numpy as np

from src.llm_wrapper import LLMWrapper
from src.utils.generation import fix_posthoc, default_metrics, \
    default_instructions, default_instructions_answer, default_instructions_schema, default_query_prefix, \
    default_answer_prefix, default_schema_prefix, default_output_prefix, default_output_prefix_chat, \
    default_output_prefix_chat_beluga, construct_args_from_example, \
    default_inv_instructions, default_inv_question_prefix, \
    default_inv_output_prefix, get_logprob_score, get_logprob_score_from_gen, construct_query_prompt_from_args, \
    prompt_arr_2_text, default_instructions_R, default_instructions_l2m
from src.utils.helpers import setup_logger
from src.utils.arguments import Arguments
from src.utils.parser import parse_bottom_up

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, is_beluga=None, load_in_8bit=False, openai_api_key=None):
        super().__init__(model, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, is_beluga=is_beluga, load_in_8bit=load_in_8bit,
                         openai_api_key=openai_api_key)

        self.default_instructions = default_instructions
        self.default_instructions_R = default_instructions_R
        self.default_instructions_answer = default_instructions_answer
        self.default_instructions_schema = default_instructions_schema
        self.default_instructions_l2m = default_instructions_l2m
        self.default_query_prefix = default_query_prefix
        self.default_answer_prefix = default_answer_prefix
        self.default_schema_prefix = default_schema_prefix
        self.default_question_prefix = default_output_prefix  # Used in few-shot demonstrations
        self.default_output_prefix = default_output_prefix
        if self.is_chat:
            self.default_output_prefix = default_output_prefix_chat
            if self.is_beluga:
                self.default_output_prefix = default_output_prefix_chat_beluga
        self.default_inv_instructions = default_inv_instructions
        self.default_inv_question_prefix = default_inv_question_prefix
        self.default_inv_output_prefix = default_inv_output_prefix
        self.default_metrics = default_metrics

    def zero_shot_l2m(self, inf_args_list, prev_predictions, step_idx, anon_type, prompt_sep="\n\n", **kwargs):
        # Construct prompt
        prompt_arr = []
        # Task instructions
        instructions = copy.copy(self.default_instructions)
        if not anon_type.endswith('-rev'):
            instructions += self.default_instructions_R
        if args.eval_use_schema:
            instructions += self.default_instructions_schema
        instructions += self.default_instructions_l2m

        instructions = " ".join(instructions)
        if instructions != "":
            prompt_arr.append(instructions)

        inf_args = inf_args_list[step_idx]

        # Previous queries
        prompt_arr_demos = []
        n_prev = 0
        if step_idx != 0:
            for _i in range(step_idx - 1, -1, -1):
                if args.limit_l2m_context != -1 and n_prev == args.limit_l2m_context:
                    break
                _inf_args, _pred = inf_args_list[_i], prev_predictions[_i]
                if args.only_keep_contributors:
                    if _inf_args['query'] not in inf_args['query']:
                        continue
                    if any(_inf_args['query'] in d for d in prompt_arr_demos):
                        continue
                prompt_arr_demos.insert(0, "\n".join(construct_query_prompt_from_args(_inf_args['query'], None, None, None,
                                                                                   None if args.add_schema_only_once else
                                                                                   _inf_args['schema'], None,
                                                                                   self.default_query_prefix, None,
                                                                                   self.default_schema_prefix,
                                                                                   question=_pred,
                                                                                   question_prefix=self.default_question_prefix)))
                n_prev += 1
        # Query
        query_arr = construct_query_prompt_from_args(inf_args['query'], None, None, None, inf_args['schema'],
                                                     None, self.default_query_prefix,
                                                     None, self.default_schema_prefix)
        if args.eval_use_schema and args.add_schema_only_once:
            # Add the schema right after the instructions
            prompt_arr_demos = [query_arr[-1]] + prompt_arr_demos
            query_arr = query_arr[:-1]
        query_arr = ["\n".join(query_arr) + f"\n{self.default_output_prefix}"]

        prompt_arr += prompt_arr_demos + query_arr
        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, prompt_sep, self.is_llama2, self.is_beluga, self.is_chat,
                                   output_prefix="")

        return self._base_generator(prompt, **kwargs)

    def generate_and_sample_l2m(self, dataset, start_idx, step_idx, per_ex_steps, per_ex_inf_args, per_ex_preds,
                                anon_type, use_gen_norm_seq_scores, use_alt_seq_scores, prev_step_examples,
                                inf_fn_kwargs, output_sampling_strategy='inverse-len-norm', verbose=False):
        examples = []
        # Get LLM generations
        for idx, d in enumerate(tqdm(dataset, desc="Generating")):
            _idx = start_idx + idx
            if step_idx >= per_ex_steps[idx]:
                examples.append(prev_step_examples[idx])
                continue
            inf_args_list, preds_list = per_ex_inf_args[idx], per_ex_preds[idx]
            # Make LLM call
            llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = self.zero_shot_l2m(inf_args_list=inf_args_list,
                                                                                         prev_predictions=preds_list,
                                                                                         step_idx=step_idx,
                                                                                         anon_type=anon_type,
                                                                                         **inf_fn_kwargs)
            if not self.is_openai:
                llm_seq_scores = llm_outputs.sequences_scores.tolist() if not use_gen_norm_seq_scores \
                    else get_logprob_score_from_gen(llm_outputs, llm_prompt, self.model, self.tokenizer,
                                                    llm_decoding_args.get('length_penalty', 1.),
                                                    use_alt=use_alt_seq_scores)
            else:
                llm_seq_scores = [0.]*len(llm_decoded)
            llm_decoded = fix_posthoc(llm_decoded)
            if verbose:
                logger.info(f"Example #{_idx + 1}:")
                logger.info(f"Prompt:\n{llm_prompt}")
                logger.info(f"Predictions: {llm_decoded}")
            ex = {
                "idx": _idx + 1,
                "id": d["id"],
                "prompt": llm_prompt,
                "query": inf_args_list[step_idx]['query'],
                "reference": None,
                "prediction": None,
                "prediction_l2m": None,
                "prediction_candidates": sorted(list(zip(llm_decoded, llm_seq_scores)), key=lambda x: x[1],
                                                reverse=True)
            }

            # Select prediction from generations
            _strat = output_sampling_strategy
            if self.is_openai:
                logger.info(f"Force setting the sampling strategy to 'max' for OpenAI API")
                _strat = 'max'
            pred_cands, pred_seq_scores = zip(*ex["prediction_candidates"])
            if _strat == 'max':
                pred_selected = pred_cands[0]  # the decoded list is sorted
            elif _strat == 'random':
                pred_selected = random.choice(pred_cands)
            elif 'inverse' in _strat:  # inverse-consistency
                cand_prompts = []
                for cand_q in pred_cands:
                    inv_question = f"""{self.default_inv_question_prefix}{cand_q}"""
                    inv_prompt = [self.default_inv_instructions, inv_question, self.default_inv_output_prefix]
                    inv_prompt = "\n\n".join(inv_prompt)
                    cand_prompts.append(inv_prompt)
                cand_logprobs = get_logprob_score(target=[ex['query']] * len(cand_prompts),
                                                  prefix=cand_prompts,
                                                  model=self.model,
                                                  tokenizer=self.tokenizer,
                                                  len_norm='len-norm' in _strat)
                pred_cands_scored = sorted(list(zip(pred_cands, cand_logprobs)), key=lambda x: x[1], reverse=True)
                pred_selected = pred_cands_scored[0][0]
                ex[f"prediction_candidates"] = pred_cands_scored
            elif _strat == 'pmi':
                pmi_prompts = [f"{self.default_output_prefix}"] * len(pred_cands)
                pmi_logprobs = get_logprob_score(target=pred_cands,
                                                 prefix=pmi_prompts,
                                                 model=self.model,
                                                 tokenizer=self.tokenizer,
                                                 len_norm=True)
                pmi_scores = [(sc_yx - sc_y) for sc_yx, sc_y in zip(pred_seq_scores, pmi_logprobs)]
                pred_cands_scored = sorted(list(zip(pred_cands, pmi_scores)), key=lambda x: x[1], reverse=True)
                pred_selected = pred_cands_scored[0][0]
                ex[f"prediction_candidates"] = pred_cands_scored
            else:
                raise ValueError()
            per_ex_preds[idx].append(pred_selected)
            ex["prediction"] = pred_selected
            examples.append(ex)
        return examples


    def eval(self, inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None,
             dataset_sample_strategy='static', anon_type='label', use_answer=False, use_schema=False,
             output_sampling_strategy='max', inv_consistency_alpha=1., all_inv_consistency_alpha=False,
             use_gen_norm_seq_scores=False, use_alt_seq_scores=False, verbose=False, out_dir=None, n_shots=None,
             retrieval_strategy=None, kg_name=None, force_type_constraint=False, kg_schema_fpath=None,
             run_id=str(int(time.time())), **inf_fn_kwargs):
        if metrics is None:
            metrics = copy.deepcopy(self.default_metrics)
        eval_args = locals()
        del eval_args["self"]
        logger.info(f"Eval arguments: {eval_args}")
        inf_fn = {
            'zeroshot': self.zero_shot,
            'fewshot': self.few_shot
        }[inf_fn_key]
        dataset = self.datasets[split]
        if (n_samples is not None and n_samples != -1) and n_samples < len(dataset):
            if dataset_sample_strategy == 'static':
                dataset = dataset[:n_samples]
            elif dataset_sample_strategy == 'random':
                dataset = random.sample(dataset, n_samples)

        if inf_fn_key == "fewshot":
            inf_fn_kwargs.update({
                "anon_type": anon_type,
                "use_schema": use_schema,
                "use_answer": use_answer
            })

        kg_schema = None
        if kg_schema_fpath is not None:
            with open(kg_schema_fpath, 'r') as fh:
                kg_schema = json.load(fh)

        references, examples = [], []
        per_ex_inf_args, per_ex_ref = [], []
        metric_results, results = {}, {}
        per_ex_steps = []
        per_ex_preds = []
        for d in dataset:
            # Construct inf_fn arguments list
            inf_args_list, ref = construct_args_from_example(d, anon_type, use_schema, bottom_up_parse=True)
            inf_args_list = inf_args_list if type(inf_args_list) is list else [inf_args_list]
            if force_type_constraint:
                assert kg_schema is not None
                for _i, inf_args in enumerate(inf_args_list):
                    type_con = None
                    q = inf_args['query']
                    q_parse = parse_bottom_up(q, expand=True)[-1]
                    if q_parse[0] in ['JOIN', 'gt', 'lt', 'ge', 'le']:
                        if type(q_parse[1]) is str:
                            _rel = q_parse[1]
                            _is_rev = False
                        elif type(q_parse[1]) is list and q_parse[1][0] == 'R':
                            _rel = q_parse[1][1]
                            _is_rev = True
                        if _rel in kg_schema['relations']:
                            type_con = kg_schema['relations'][_rel]['domain' if not _is_rev else 'range']
                        elif _rel in kg_schema['inverse_relations']:
                            type_con = kg_schema['relations'][kg_schema['inverse_relations'][_rel]]['range']
                    if type_con is not None:
                        q_with_type_con = f'(AND {type_con} {q})'
                        if _i != (len(inf_args_list) - 1) and inf_args_list[_i + 1]['query'] != q_with_type_con:
                            inf_args['query'] = q_with_type_con
            per_ex_inf_args.append(inf_args_list)
            if ref is not None:
                references.append(ref)
            per_ex_steps.append(len(per_ex_inf_args[-1]))
            per_ex_preds.append([])

        # Setup results dir
        res_dir = ["eval-l2m"]
        if kg_name is not None:
            res_dir += [kg_name]
        res_dir += [self.model_name + ("_chat" if self.is_chat else ""), split]
        if n_samples is not None:
            res_dir += [str(n_samples)]
        res_dir += [inf_fn_key, anon_type, output_sampling_strategy]
        if use_schema:
            res_dir += ['schema']
        if use_answer:
            res_dir += ['answer']
        res_dir += [run_id]
        res_dir = f"{'_'.join(res_dir)}"

        if args.save_interval == -1:
            _datasets = [dataset]
        else:
            n_intervals = math.ceil(len(dataset) / args.save_interval)
            _datasets = [dataset[i * args.save_interval: (i + 1) * args.save_interval] for i in range(n_intervals)]
            assert sum([len(d) for d in _datasets]) == len(dataset)

        # Results object
        results = {
            "experiment": res_dir,
            "n_total": len(dataset),
            "n_intervals": n_intervals,
            "n_processed": 0,
            "interval_processed": -1,
            "eval_args": eval_args,
            "scores": None,
            "time_taken": 0,
            "examples": []
        }

        start_time = time.time()

        for interval_idx, _dataset in enumerate(_datasets):
            if interval_idx < args.start_interval:
                logger.info(f'Skipping interval idx {interval_idx}')
                continue
            start_idx = interval_idx * len(_datasets[0])
            end_idx = start_idx + len(_dataset)
            _per_ex_steps = per_ex_steps[start_idx:end_idx]
            _per_ex_steps = per_ex_steps[start_idx:end_idx]
            _per_ex_inf_args = per_ex_inf_args[start_idx:end_idx]
            _per_ex_preds = per_ex_preds[start_idx:end_idx]

            for step_idx in range(max(_per_ex_steps)):
                logger.info(f"INTERVAL={interval_idx}, L2M STEP={step_idx + 1}")
                examples = self.generate_and_sample_l2m(_dataset, start_idx, step_idx, _per_ex_steps, _per_ex_inf_args,
                                                        _per_ex_preds, anon_type, use_gen_norm_seq_scores,
                                                        use_alt_seq_scores, examples, inf_fn_kwargs,
                                                        output_sampling_strategy=output_sampling_strategy,
                                                        verbose=verbose)
            end_time = time.time()
            predictions = [p[-1] for p in _per_ex_preds]
            # Compute metrics
            if len(references) > 0:
                _references = []
                for idx, ex in enumerate(examples):
                    idx = start_idx + idx
                    ex['reference'] = references[idx]
                    _references.append(references[idx])
                    ex['prediction_l2m'] = [[tup[0]['query'], tup[1]] for tup in
                                            zip(per_ex_inf_args[idx], per_ex_preds[idx])]
                for metric in metrics:
                    scores = self.get_metric_scores(metric, predictions, _references)
                    for k in metric["score_keys"]:
                        metric_results[f"{metric['name']}.{k}"] = round(np.mean(scores[k]), 4)
                        if results["scores"] is not None:
                            metric_results[f"{metric['name']}.{k}"] = round((metric_results[
                                                                                 f"{metric['name']}.{k}"] * len(
                                _dataset) + results["scores"][f"{metric['name']}.{k}"] * results["n_processed"]) / (
                                                                                    len(_dataset) + results[
                                                                                "n_processed"]), 4)
                if verbose:
                    logger.info(metric_results)
            else:
                for idx, ex in enumerate(examples):
                    idx = start_idx + idx
                    ex['prediction_l2m'] = [[tup[0]['query'], tup[1]] for tup in
                                            zip(per_ex_inf_args[idx], per_ex_preds[idx])]

            results.update({
                "n_processed": results["n_processed"] + len(_dataset),
                "interval_processed": interval_idx,
                "scores": metric_results,
                "time_taken": end_time - start_time,
                "examples": results["examples"] + examples
            })

            if out_dir is not None:
                res_dir_fpath = os.path.join(out_dir, res_dir)
                os.makedirs(res_dir_fpath, exist_ok=True)
                out_fname = "results.json"
                out_fpath = os.path.join(res_dir_fpath, out_fname)
                with open(out_fpath, 'w') as fh:
                    fh.write(json.dumps(results, indent=2))
                logger.info(f"Saved results to {out_fpath}")

        return results


if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups=["llm", "question_generator", "question_generator_l2m"])
    global args
    args = cli_args.parse_args()
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else str(args.run_id)
    setup_logger(RUN_ID)
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)
    # Create output dir
    out_dir = os.path.join(args.out_dir, "question_generation")
    if args.clean_out_dir:
        for f in os.listdir(out_dir):
            fpath = os.path.join(out_dir, f)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
    os.makedirs(out_dir, exist_ok=True)

    llm = QuestionGenerator(model=args.model, is_chat=args.eval_chat_mode, load_in_8bit=args.load_in_8bit,
                            openai_api_key=args.openai_api_key)
    if args.eval_split == 'train' or args.eval_inf_fn_key == 'fewshot':
        llm.load_train_set(fpath=args.load_train_fpath)
    if args.eval_split == 'dev':
        llm.load_dev_set(fpath=args.load_dev_fpath)
    if args.eval_split == 'test':
        llm.load_test_set(fpath=args.load_test_fpath)
    results = llm.eval(inf_fn_key=args.eval_inf_fn_key, split=args.eval_split, n_samples=args.eval_n_samples,
                       out_dir=out_dir, anon_type=args.eval_anon_type, use_answer=args.eval_use_answer,
                       use_schema=args.eval_use_schema, verbose=args.verbose, n_shots=args.eval_n_shots,
                       retrieval_strategy=args.eval_retrieval_strategy, kg_name=args.kg_name,
                       output_sampling_strategy=args.eval_output_sampling_strategy, run_id=RUN_ID,
                       inv_consistency_alpha=args.eval_inv_consistency_alpha, kg_schema_fpath=args.kg_schema_fpath,
                       dataset_sample_strategy=args.dataset_sample_strategy, force_type_constraint=args.force_type_constraint,
                       all_inv_consistency_alpha=args.eval_all_inv_consistency_alpha,
                       use_gen_norm_seq_scores=args.eval_use_gen_norm_seq_scores,
                       use_alt_seq_scores=args.eval_use_alt_seq_scores)

    if args.debug:
        breakpoint()
