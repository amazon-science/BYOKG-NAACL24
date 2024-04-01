import copy
import json
import logging
import os
import random

from src.utils.arguments import Arguments
from transformers import set_seed

from src.utils.kg import create_splits_for_q_gen

"""
Script that takes either a qgen input+output file or a dataset file and outputs a file that can be consumed by the reasoner.
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_anon_question(question, schema, anon_type):
    rtn = question
    type_map = {v: k for k, v in schema[f'nodes_{anon_type}'].items()}
    for k, v in schema['nodes_machine'].items():
        if k.startswith('m.') or k.startswith('g.'):
            rtn = rtn.replace(v, f'"{type_map[v]}"')
    return rtn


if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups="prep_data_for_qa")
    global args
    args = cli_args.parse_args()
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)

    rev_schema = None
    if args.rev_schema_fpath is not None:
        with open(args.rev_schema_fpath, 'r') as fh:
            if args.rev_schema_fpath.endswith('.json'):
                # Our style of schema
                rev_schema = json.load(fh)
            else:
                # GrailQA style of reverse properties
                rev_schema = {"relations": {}}
                while _line := fh.readline():
                    _rel, _inv_rel = _line.split('\t')
                    rev_schema["relations"][_rel.strip()] = {"reverse": _inv_rel.strip()}
    elif args.generate_rev_schema:
        # TODO: LLM reverse relation generation
        pass

    if args.walks_qgen_in_fpath is not None and args.walks_qgen_out_fpath is not None:
        prep_data = []
        anon_types = []
        with open(args.walks_qgen_in_fpath, 'r') as fh:
            qgen_in = json.load(fh)
        with open(args.walks_qgen_out_fpath, 'r') as fh:
            qgen_out = json.load(fh)
            qgen_out = qgen_out['examples']
        max_len = max(len(qgen_in), len(qgen_out))
        qgen_in = qgen_in[:max_len]
        qgen_out = qgen_out[:max_len]
        for qi, qo in zip(qgen_in, qgen_out):
            for i in range(args.n_samples_per_q):
                obj = copy.deepcopy(qi)
                pred_key = "prediction_candidates"
                if args.prediction_sampling_strategy != "":
                    pred_key += '_' + args.prediction_sampling_strategy
                obj['answer'] = [obj.pop('sample_answer')]
                obj['original_fname'] = args.walks_qgen_out_fpath.split('/')[-2]  # dir name
                question = qo[pred_key][i][0]
                question = question[:-1] if question.endswith('?') else question
                entities = []
                for k in obj['schema']['nodes_machine']:
                    if k.startswith('m.') or k.startswith('g.'):
                        entities.append(k)
                obj['question_entities'] = entities
                if len(anon_types) == 0:
                    for k in obj:
                        if k.startswith('s_expression_'):
                            anon_type = k[len('s_expression_'):]
                            anon_types.append(anon_type)
                for anon_type in anon_types:
                    if 'label' in anon_type:
                        obj[f'question_{anon_type}'] = question
                    elif any(t in anon_type for t in ['anon', 'machine']):
                        obj[f'question_{anon_type}'] = get_anon_question(question, obj['schema'], anon_type)
                obj['id'] = len(prep_data)
                prep_data.append(obj)
        out_dir = os.path.dirname(args.walks_qgen_in_fpath)
        out_fpath = os.path.join(out_dir, f"{args.walks_out_fname}.json")
        with open(out_fpath, 'w') as fh:
            fh.write(json.dumps(prep_data, indent=2))
        logger.info(f"Wrote {args.walks_out_fname} to {out_fpath}")

    if args.metaqa_dir is not None:
        fname_map = {'train': args.train_out_fname, 'dev': args.dev_out_fname, 'test': args.test_out_fname}
        for hop in [1, 2, 3]:
            for split in fname_map:
                logger.info(f"Hop: {hop}")
                logger.info(f"Split: {split}")
                fpath = os.path.join(args.metaqa_dir, f'{split}_{hop}-hop.json')
                create_splits_for_q_gen(fpath=fpath,
                                        n_samples_per_split=0,  # 0 means all
                                        out_fnames=[f"{fname_map[split]}_{hop}-hop"],
                                        keep_all_answers=True,
                                        additional_keys=["question_entities", "ans_original", "ans_mismatch"],
                                        sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema,
                                        type_constraint=args.type_constraint)
        logger.info(f"Creating single train/dev/test splits with all hops")
        for split in fname_map:
            split_data = []
            for hop in [1, 2, 3]:
                in_fpath = os.path.join(args.metaqa_dir, f"{fname_map[split]}_{hop}-hop.json")
                with open(in_fpath, 'r') as fh:
                    split_hop_data = json.load(fh)
                split_data += split_hop_data
            random.shuffle(split_data)
            split_data = [dict(s, id=i) for i, s in enumerate(split_data)]
            out_fpath = os.path.join(args.metaqa_dir, f"{fname_map[split]}.json")
            with open(out_fpath, 'w') as fh:
                fh.write(json.dumps(split_data, indent=2))
            logger.info(f"Wrote merged {split} set to {out_fpath}")
    else:
        if args.train_fpath is not None:
            create_splits_for_q_gen(fpath=args.train_fpath,
                                    n_samples_per_split=0,
                                    out_fnames=[args.train_out_fname],  # set to "qa_train"
                                    keep_all_answers=True,
                                    additional_keys=["question_entities", "ans_original", "ans_mismatch", "level"],
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema,
                                    type_constraint=args.type_constraint)

        if args.dev_fpath is not None:
            create_splits_for_q_gen(fpath=args.dev_fpath,
                                    n_samples_per_split=0,
                                    out_fnames=[args.dev_out_fname],  # set to "qa_dev"
                                    keep_all_answers=True,
                                    additional_keys=["question_entities", "ans_original", "ans_mismatch", "level"],
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema,
                                    type_constraint=args.type_constraint)

        if args.test_fpath is not None:
            create_splits_for_q_gen(fpath=args.test_fpath,
                                    n_samples_per_split=0,
                                    out_fnames=[args.test_out_fname],  # set to "qa_test"
                                    keep_all_answers=True,
                                    additional_keys=["question_entities", "ans_original", "ans_mismatch", "level"],
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema,
                                    type_constraint=args.type_constraint)
