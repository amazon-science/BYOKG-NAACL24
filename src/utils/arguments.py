import argparse


# TODO: Add descriptions for all flags
class Arguments(argparse.ArgumentParser):
    def __init__(self, groups=None):
        super().__init__(conflict_handler='resolve')
        # Common flags
        self.add_argument(
            "--out_dir", type=str, default="outputs"
        )
        self.add_argument(
            "--clean_out_dir", action="store_true",
        )
        self.add_argument(
            "--debug", action="store_true",
        )
        self.add_argument(
            "--verbose", action="store_true",
        )
        self.add_argument(
            "--seed", type=int, default=17
        )
        self.add_argument(
            "--run_id", type=int, default=None
        )

        if type(groups) is not list:
            groups = [groups]

        for group in groups:
            if group == "explorer":
                self.add_argument(
                    "--kg_name", type=str, required=True
                )
                self.add_argument(
                    "--kg_prefix", type=str, default=None
                )
                self.add_argument(
                    "--kg_path", type=str, required=True
                )
                self.add_argument(
                    "--triples_fname", type=str, default="kb.txt"
                )
                self.add_argument(
                    "--kg_lowercase", action="store_true"
                )
                self.add_argument(
                    "--kg_sep", type=str, default="|"
                )
                self.add_argument(
                    "--kg_write_ntriples", action="store_true"
                )
                self.add_argument(
                    "--kg_explore", action="store_true"
                )
                self.add_argument(
                    "--kg_n_walks", type=int, default=None
                )
                self.add_argument(
                    "--kg_n_walk_edges", type=int, default=4
                )
                self.add_argument(
                    "--kg_walk_functions", nargs='+', default=None,
                    help="If None, then use all functions, else pass one or more of the following strings as a \
                    space-separated list: 'none' 'count' 'argmax' 'argmin' '>' '>=' '<' '<='"
                )
                self.add_argument(
                    "--kg_walk_max_skip", type=int, default=None
                )
                self.add_argument(
                    "--max_retries", type=int, default=None
                )
                self.add_argument(
                    "--sparql_url", type=str, default="http://localhost:3001/sparql",
                    help="URL of the Virtuoso SPARQL endpoint"
                )
                self.add_argument(
                    "--sparql_cache", type=str, default=None,
                    help="Cache file path to use for sparql queries"
                )
                self.add_argument(
                    "--filter_empty_walks", action=argparse.BooleanOptionalAction, default=True,
                    help="Remove explored queries that result in an empty answer set on execution"
                )
                self.add_argument(
                    "--prune_redundant", action=argparse.BooleanOptionalAction, default=True,
                    help="Prune parts of explored graph queries that do not change the final answers"
                )
                self.add_argument(
                    "--additional_save_dir", type=str, default=None,
                    help="Additional directory where the walks file should be saved as `walks.json`"
                )  # e.g. "data/graphs/metaqa"
                self.add_argument(
                    "--always_ground_literals", action=argparse.BooleanOptionalAction, default=True
                )
                self.add_argument(
                    "--always_ground_classes", action=argparse.BooleanOptionalAction, default=False
                )
                self.add_argument(
                    "--n_per_pattern", type=int, default=5
                )
                self.add_argument(
                    "--prev_exploration_stats_fpath", nargs='+', default=None,
                    help="Continue exploring based on a previous run's results file."
                )
                self.add_argument(
                    "--save_interval", type=int, default=-1,
                    help="-1 means do not save in intervals - save once full exploration terminates."
                )

            elif group == "llm":
                # Common group for scripts that will use the LLMWrapper class
                self.add_argument(
                    "--load_train_fpath", type=str
                )  # 'data/datasets/grailqa/q_gen_100_train.json'
                self.add_argument(
                    "--load_dev_fpath", type=str
                )  # 'data/datasets/grailqa/q_gen_100_dev.json'
                self.add_argument(
                    "--load_test_fpath", type=str
                )
                self.add_argument(
                    "--model", type=str, default="meta-llama/Llama-2-7b-hf"
                )
                self.add_argument(
                    "--eval_chat_mode", action="store_true", default=None
                )
                self.add_argument(
                    "--load_in_8bit", action="store_true"
                )
                self.add_argument(
                    "--eval_inf_fn_key", default="zeroshot", choices=['zeroshot', 'fewshot']
                )
                self.add_argument(
                    "--eval_n_shots", type=int, default=3
                )
                self.add_argument(
                    "--eval_retrieval_strategy", default="random", choices=['random', 'function', 'bm25', 'dense']
                )
                self.add_argument(
                    "--eval_split", default="dev", choices=['train', 'dev', 'test']
                )
                self.add_argument(
                    "--eval_output_sampling_strategy", default="all",
                    choices=['all', 'max', 'random', 'inverse', 'inverse-len-norm', 'pmi'],
                    help="Strategy to pick a sequence from the set of generations returned by beam search.\
                                    `all` will evaluate `max`, `random`, and `inverse-len-norm`."
                )
                self.add_argument(
                    "--eval_inv_consistency_alpha", type=float, default=1.,
                    help="""Value in [0,1] that controls the weight of inverse-consistency in the final score 
                                    compared to the forward sequence score, i.e. score = (1-alpha)*seq_score + alpha*inv_score"""
                )
                self.add_argument(
                    "--eval_all_inv_consistency_alpha", action="store_true",
                    help="Evaluate all intermediate inv-consistency alpha values, i.e. [0.1,0.9] in steps of 0.1"
                )
                self.add_argument(
                    "--eval_use_gen_norm_seq_scores", action="store_true",
                    help="Whether to use sequence scores normalized by the length of only the generated tokens \
                                    or by the full (prompt+generation) sequence (i.e. the default HF implementation)."
                )
                self.add_argument(
                    "--eval_use_alt_seq_scores", action="store_true",
                    help="[Experimental] Use log avg. transition probability to score sequence"
                )
                self.add_argument(
                    "--openai_api_key", type=str, default=None,
                )

            elif group == "question_generator":
                self.add_argument(
                    "--kg_name", type=str
                )
                self.add_argument(
                    "--eval_n_samples", type=int, default=-1, help="-1 means use all examples"
                )
                self.add_argument(
                    "--dataset_sample_strategy", default="static", choices=['static', 'random']
                )
                self.add_argument(
                    "--eval_anon_type", default="label-rev", choices=['label', 'label-rev', 'anon', 'machine']
                )
                self.add_argument(
                    "--eval_use_answer", action="store_true",
                )
                self.add_argument(
                    "--eval_use_schema", action=argparse.BooleanOptionalAction, default=True
                )

            elif group == "question_generator_l2m":
                self.add_argument(
                    "--add_schema_only_once", action="store_true",
                    help="Injects the schema only once after the instructions, instead of per demonstration."
                )
                self.add_argument(
                    "--only_keep_contributors", action="store_true",
                    help="Retain only those few-shot query breakdowns that directly contribute to the current step"
                )
                self.add_argument(
                    "--kg_schema_fpath", type=str, default=None
                )
                self.add_argument(
                    "--force_type_constraint", action=argparse.BooleanOptionalAction, default=False
                )
                self.add_argument(
                    "--save_interval", type=int, default=-1,
                    help="-1 means do not save in intervals - save only when all samples have been generated."
                )
                self.add_argument(
                    "--start_interval", type=int, default=0,
                    help="defines the start index of the interval to process. Useful when resuming a crashed run."
                )
                self.add_argument(
                    "--limit_l2m_context", type=int, default=-1,
                    help="Use at most this many previous l2m steps as context for the current step."
                )

            elif group == "dataset_metaqa":
                self.add_argument(
                    "--data_dir", type=str, default="data/datasets/metaqa"
                )
                self.add_argument(
                    "--data_out_dir", type=str, default=None
                )
                self.add_argument(
                    "--sparql_url", type=str, default="http://localhost:3001/sparql",
                    help="URL of the Virtuoso SPARQL endpoint"
                )

            elif group == "qa_metaqa":
                self.add_argument(
                    "--data_dir", type=str, default="data/datasets/metaqa"
                )
                self.add_argument(
                    "--rev_schema_fpath", type=str, default=None
                )

            elif group == "reasoner":
                self.add_argument(
                    "--eval_inf_fn_key", default="fewshot", choices=['zeroshot', 'fewshot']
                )
                self.add_argument(
                    "--dataset_name", type=str
                )
                self.add_argument(
                    "--eval_n_hops", type=int, default=None
                )
                self.add_argument(
                    "--demos_label", type=str, default=None
                )
                self.add_argument(
                    "--oracle_el", action=argparse.BooleanOptionalAction, default=False
                )
                self.add_argument(
                    "--pred_start_cands_fpath", type=str, default=None
                )
                self.add_argument(
                    "--eval_n_samples", type=int, default=-1,
                    help="-1 means use all examples"
                )
                self.add_argument(
                    "--demos_split", default="train", choices=['train', 'dev', 'test']
                )
                self.add_argument(
                    "--demos_n_samples", type=int, default=-1,
                    help="-1 means use all examples"
                )
                self.add_argument(
                    "--dataset_sample_strategy", default="static", choices=['static', 'random']
                )
                self.add_argument(
                    "--eval_anon_type", default="label",
                    choices=['label', 'label-rev', 'anon', 'anon-rev', 'machine', 'machine-rev']
                )
                self.add_argument(
                    "--eval_use_schema", action=argparse.BooleanOptionalAction, default=False
                )
                self.add_argument(
                    "--eval_n_shots", type=int, default=5
                )
                self.add_argument(
                    "--skip_mismatch", action="store_true",
                )
                self.add_argument(
                    "--skip_empty", action="store_true",
                )
                self.add_argument(
                    "--sparql_url", type=str, default="http://localhost:3001/sparql",
                    help="URL of the Virtuoso SPARQL endpoint"
                )
                self.add_argument(
                    "--sparql_cache", type=str, default=None,
                    help="Cache file path to use for sparql queries"
                )
                self.add_argument(
                    "--kg_name", type=str, required=True,
                    help="Name of the graph being queried"
                )
                self.add_argument(
                    "--expand_using_classes_only", action="store_true",
                )
                self.add_argument(
                    "--exit_on_sparql_error", action="store_true",
                )
                self.add_argument(
                    "--rev_schema_fpath", type=str
                )
                self.add_argument(
                    "--beam_size", type=int, default=4
                )
                self.add_argument(
                    "--max_steps", type=int, default=10
                )
                self.add_argument(
                    "--sparql_timeout", type=int, default=5,
                    help="Timeout for sparql requests (in s)"
                )
                self.add_argument(
                    "--scorer_batch_size", type=int, default=10,
                    help="Batch size for computing log probability scores"
                )
                self.add_argument(
                    "--per_cand_schema", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to use schema descriptions in the prompt from each candidate separately or \
                         a union of all schemas common to all candidates."
                )
                self.add_argument(
                    "--eval_retrieval_strategy", default="dense", choices=['random', 'function', 'bm25', 'dense']
                )
                self.add_argument(
                    "--fewshot_coverage_strat1", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--fewshot_coverage_strat2", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--fewshot_coverage_strat3", action=argparse.BooleanOptionalAction, default=True,
                )
                self.add_argument(
                    "--fewshot_anon_strat", default="anon", choices=['mask', 'anon', 'machine', 'label']
                )
                self.add_argument(
                    "--dup_fewshot_threshold", type=int, default=1,
                )
                self.add_argument(
                    "--restrict_to_step", action=argparse.BooleanOptionalAction, default=False,
                    help="Whether to restrict the expansion of few-shot sexpr demonstrations to the current decoding step or not."
                )
                self.add_argument(
                    "--save_eval_dataset", action="store_true"
                )
                self.add_argument(
                    "--eval_output_sampling_strategy", default="max",
                    choices=['all', 'max', 'inverse-len-norm'],
                    help="Strategy to pick a sequence from the set of generations returned by beam search. \
                    `all` will evaluate both `max` and `inverse-len-norm`."
                )
                self.add_argument(
                    "--sanity_check", action="store_true",
                )
                self.add_argument(
                    "--sanity_check_breakpoint", action="store_true",
                )
                self.add_argument(
                    "--sanity_check_keep_gold_only", action="store_true",
                )
                self.add_argument(
                    "--retry_on_cache_none", action="store_true",
                )
                self.add_argument(
                    "--save_interval", type=int, default=-1,
                    help="-1 means do not save in intervals - save only when the full dataset has been evaluated."
                )
                self.add_argument(
                    "--start_interval", type=int, default=0,
                    help="defines the start index of the interval to process. Useful when resuming a crashed run."
                )
                self.add_argument(
                    "--prune_candidates", action=argparse.BooleanOptionalAction, default=True,
                )
                self.add_argument(
                    "--prune_candidates_threshold", type=int, default=10,
                )
                self.add_argument(
                    "--pruning_anon_strat", default="anon", choices=['mask', 'anon', 'machine', 'label']
                )
                self.add_argument(
                    "--strip_cands_before_pruning", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--preprocess_before_pruning", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--bm25_pruning", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--sbert_device", type=str, default="cuda"
                )
                self.add_argument(
                    "--l2m_demos", action=argparse.BooleanOptionalAction, default=False,
                )
                self.add_argument(
                    "--pangu_prompt", action="store_true",
                )
                self.add_argument(
                    "--type_constraint", action=argparse.BooleanOptionalAction, default=True,
                )
                self.add_argument(
                    "--relation_repetition_factor", type=float, default=1.,
                    help="1. means no penalty; allowed values \in [0, 1]."
                )
                self.add_argument(
                    "--rerank_fpath", type=str, default=None,
                )
                self.add_argument(
                    "--rerank_alpha", type=float, default=0.5,
                )
                self.add_argument(
                    "--rerank_rel_rep_factor", type=float, default=0.7,
                )
                self.add_argument(
                    "--rerank_inverse", action=argparse.BooleanOptionalAction, default=True,
                )
                self.add_argument(
                    "--rerank_pmi", action=argparse.BooleanOptionalAction, default=False,
                )

            elif group == "prep_data_for_qgen":
                self.add_argument(
                    "--rev_schema_fpath", type=str, default=None
                )
                self.add_argument(
                    "--generate_rev_schema", action="store_true",
                )
                self.add_argument(
                    "--walks_fpath", type=str, default=None
                )
                self.add_argument(
                    "--walks_out_fname", type=str, default="qgen_walks"
                )
                self.add_argument(
                    "--train_fpath", type=str, default=None
                )
                self.add_argument(
                    "--train_out_fname", type=str, default="qgen_train"
                )
                self.add_argument(
                    "--dev_fpath", type=str, default=None
                )
                self.add_argument(
                    "--dev_out_fname", type=str, default="qgen_dev"
                )
                self.add_argument(
                    "--test_fpath", type=str, default=None
                )
                self.add_argument(
                    "--test_out_fname", type=str, default="qgen_test"
                )
                self.add_argument(
                    "--sexpr_machine_key", type=str, default="s_expression_machine",
                    help="Set to 's_expression' when using curated GrailQA data"
                )
                self.add_argument(
                    "--metaqa_dir", type=str, default=None
                )

            elif group == "prep_data_for_qa":
                self.add_argument(
                    "--rev_schema_fpath", type=str, default=None
                )
                self.add_argument(
                    "--generate_rev_schema", action="store_true",
                )
                self.add_argument(
                    "--walks_qgen_in_fpath", type=str, default=None
                )
                self.add_argument(
                    "--walks_qgen_out_fpath", type=str, default=None
                )
                self.add_argument(
                    "--walks_out_fname", type=str, default="qa_walks"
                )
                self.add_argument(
                    "--train_fpath", type=str, default=None
                )
                self.add_argument(
                    "--train_out_fname", type=str, default="qa_train"
                )
                self.add_argument(
                    "--dev_fpath", type=str, default=None
                )
                self.add_argument(
                    "--dev_out_fname", type=str, default="qa_dev"
                )
                self.add_argument(
                    "--test_fpath", type=str, default=None
                )
                self.add_argument(
                    "--test_out_fname", type=str, default="qa_test"
                )
                self.add_argument(
                    "--sexpr_machine_key", type=str, default="s_expression"
                )
                self.add_argument(
                    "--metaqa_dir", type=str, default=None
                )
                self.add_argument(
                    "--prediction_sampling_strategy", choices=["inverse-len-norm", "max", ""],
                    default=""
                )
                self.add_argument(
                    "--n_samples_per_q", type=int, default=1
                )
                self.add_argument(
                    "--type_constraint", action=argparse.BooleanOptionalAction, default=True,
                )

            elif group == "merge_qgen_outputs":
                self.add_argument(
                    "--qgen_out_1", type=str
                )
                self.add_argument(
                    "--qgen_out_2", type=str
                )
                self.add_argument(
                    "--qgen_in_1", type=str
                )
                self.add_argument(
                    "--qgen_in_2", type=str
                )
