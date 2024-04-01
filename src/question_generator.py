import json
import logging
import copy
import os
import random
import time
import shutil
from collections import defaultdict

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
    default_inv_output_prefix, get_logprob_score, get_logprob_score_from_gen
from src.utils.helpers import setup_logger
from src.utils.arguments import Arguments

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator(LLMWrapper):
    def __init__(self, model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, is_beluga=None, load_in_8bit=False):
        super().__init__(model, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map,
                         is_chat=is_chat, is_llama2=is_llama2, is_beluga=is_beluga, load_in_8bit=load_in_8bit)

        self.default_instructions = default_instructions
        self.default_instructions_answer = default_instructions_answer
        self.default_instructions_schema = default_instructions_schema
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

    def eval(self, inf_fn_key="zeroshot", split="dev", metrics=None, n_samples=None,
             dataset_sample_strategy='static', anon_type='label', use_answer=False, use_schema=False,
             output_sampling_strategy='max', inv_consistency_alpha=1., all_inv_consistency_alpha=False,
             use_gen_norm_seq_scores=False, use_alt_seq_scores=False, verbose=False, out_dir=None, n_shots=None,
             retrieval_strategy=None, kg_name=None, run_id=str(int(time.time())), **inf_fn_kwargs):
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

        references, _examples = [], []
        # Get LLM generations
        start_time = time.time()
        for idx, d in enumerate(tqdm(dataset, desc="Generating")):
            # Construct inf_fn arguments
            try:
                inf_args, ref = construct_args_from_example(d, anon_type, use_schema, use_answer)
            except ValueError:
                logger.info('`sample_answer` not found in example while using use_answer=True mode')
                logger.info('Exiting...')
                exit()
            # Make LLM call
            llm_decoded, llm_outputs, llm_prompt, llm_decoding_args = inf_fn(**inf_args, **inf_fn_kwargs)
            llm_seq_scores = llm_outputs.sequences_scores.tolist() if not use_gen_norm_seq_scores \
                else get_logprob_score_from_gen(llm_outputs, llm_prompt, self.model, self.tokenizer,
                                                llm_decoding_args.get('length_penalty', 1.),
                                                use_alt=use_alt_seq_scores)
            llm_decoded = fix_posthoc(llm_decoded)
            if verbose:
                logger.info(f"Example #{idx + 1}:")
                logger.info(f"Prompt:\n{llm_prompt}")
                logger.info(f"Gold: {ref}")
                logger.info(f"Predictions: {llm_decoded}")
            if ref is not None:
                references.append(ref.lower())
            _examples.append({
                "idx": idx + 1,
                "id": d["id"],
                "prompt": llm_prompt,
                "query": inf_args['query'],
                "reference": references[-1] if len(references) > 0 else None,
                "prediction": None,
                "prediction_candidates_max": sorted(list(zip(llm_decoded, llm_seq_scores)), key=lambda x: x[1],
                                                    reverse=True)
            })

        end_time = time.time()
        generation_time = end_time - start_time

        _strats = ['max', 'random', 'inverse-len-norm'] if output_sampling_strategy == 'all' else [
            output_sampling_strategy]
        predictions, examples = defaultdict(list), defaultdict(list)
        metric_results, results = defaultdict(dict), defaultdict(dict)
        # Optionally evaluate all intermediate ([0,1] in steps of 0.1) values of inverse-consistency alpha
        _inv_alpha = list(np.arange(0, 11) / 10) if all_inv_consistency_alpha else []
        _inv_alpha_preds = defaultdict(list)
        # Select prediction from generations
        for _strat in _strats:
            logger.info(f"Sampling generations (strategy={_strat}):")
            start_time = time.time()
            for _ex in tqdm(_examples, desc=f"Sampling ({_strat})"):
                ex = copy.deepcopy(_ex)
                pred_cands, pred_seq_scores = zip(*ex["prediction_candidates_max"])
                if _strat == 'max':
                    pred_selected = pred_cands[0]  # the decoded list is sorted
                elif _strat == 'random':
                    pred_selected = random.choice(pred_cands)
                elif 'inverse' in _strat:  # inverse-consistency
                    cand_prompts = []
                    for cand_q in pred_cands:
                        inv_question = f"""{self.default_inv_question_prefix}{cand_q}"""
                        inv_prompt = [self.default_inv_instructions, inv_question, self.default_inv_output_prefix]
                        # TODO: Add schema information if use_schema == True
                        # TODO: Use chat-based prompts if using chat models
                        inv_prompt = "\n\n".join(inv_prompt)
                        cand_prompts.append(inv_prompt)
                    cand_logprobs = get_logprob_score(target=[ex['query']] * len(cand_prompts),
                                                      prefix=cand_prompts,
                                                      model=self.model,
                                                      tokenizer=self.tokenizer,
                                                      len_norm='len-norm' in _strat)
                    assert 0. <= inv_consistency_alpha <= 1.
                    cand_scores = list(map(lambda x: (1 - inv_consistency_alpha) * x[0] + inv_consistency_alpha * x[1],
                                           zip(pred_seq_scores, cand_logprobs)))
                    pred_cands_scored = sorted(list(zip(pred_cands, cand_scores)), key=lambda x: x[1], reverse=True)
                    pred_selected = pred_cands_scored[0][0]
                    ex[f"prediction_candidates_{_strat}"] = pred_cands_scored
                    for _alpha in _inv_alpha:
                        _alpha_cand_scores = list(
                            map(lambda x: (1 - _alpha) * x[0] + _alpha * x[1], zip(pred_seq_scores, cand_logprobs)))
                        _alpha_pred_cands_scored = sorted(list(zip(pred_cands, _alpha_cand_scores)), key=lambda x: x[1],
                                                          reverse=True)
                        ex[f"prediction_candidates_{_strat}_{_alpha}"] = _alpha_pred_cands_scored
                        _inv_alpha_preds[_alpha].append(_alpha_pred_cands_scored[0][0])
                else:
                    raise ValueError()
                ex["prediction"] = pred_selected
                predictions[_strat].append(pred_selected)
                examples[_strat].append(ex)
            end_time = time.time()

            # Compute metrics
            if len(references) > 0:
                for metric in metrics:
                    scores = self.get_metric_scores(metric, predictions[_strat], references)
                    for k in metric["score_keys"]:
                        metric_results[_strat][f"{metric['name']}.{k}"] = round(np.mean(scores[k]), 4)
                if 'inverse' in _strat:
                    # Add scores for all intermediate alpha values, if requested
                    for _alpha in _inv_alpha_preds:
                        if 'all_alpha' not in metric_results[_strat]:
                            metric_results[_strat]['all_alpha'] = defaultdict(dict)
                        for metric in metrics:
                            _alpha_scores = self.get_metric_scores(metric, _inv_alpha_preds[_alpha], references)
                            for k in metric["score_keys"]:
                                metric_results[_strat]['all_alpha'][_alpha][f"{metric['name']}.{k}"] = round(
                                    np.mean(_alpha_scores[k]), 4)
                if verbose:
                    logger.info(metric_results[_strat])

            # Save results
            res_dir = ["eval"]
            if kg_name is not None:
                res_dir += [kg_name]
            res_dir += [self.model_name + ("_chat" if self.is_chat else ""), split]
            if n_samples is not None:
                res_dir += [str(n_samples)]
            res_dir += [inf_fn_key, anon_type, _strat]
            if use_schema:
                res_dir += ['schema']
            if use_answer:
                res_dir += ['answer']
            res_dir += [run_id]
            res_dir = f"{'_'.join(res_dir)}"

            results[_strat] = {
                "experiment": res_dir,
                "n_total": len(dataset),
                "eval_args": eval_args,
                "scores": metric_results[_strat],
                "time_taken": {
                    "generation": generation_time,
                    "sampling": end_time - start_time,
                    "total": generation_time + (end_time - start_time)
                },
                "examples": examples[_strat]
            }

            if out_dir is not None:
                res_dir_fpath = os.path.join(out_dir, res_dir)
                os.makedirs(res_dir_fpath, exist_ok=True)
                out_fname = "results.json"
                out_fpath = os.path.join(res_dir_fpath, out_fname)
                with open(out_fpath, 'w') as fh:
                    fh.write(json.dumps(results[_strat], indent=2))
                logger.info(f"Saved results to {out_fpath}")

        return results


if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups=["llm", "question_generator"])
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

    llm = QuestionGenerator(model=args.model, is_chat=args.eval_chat_mode, load_in_8bit=args.load_in_8bit)
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
                       inv_consistency_alpha=args.eval_inv_consistency_alpha,
                       dataset_sample_strategy=args.dataset_sample_strategy,
                       all_inv_consistency_alpha=args.eval_all_inv_consistency_alpha,
                       use_gen_norm_seq_scores=args.eval_use_gen_norm_seq_scores,
                       use_alt_seq_scores=args.eval_use_alt_seq_scores)

    # TODO: Implement prompting strategies -- CoT, L2M, ToT, Self-Ask?

    if args.debug:
        breakpoint()
