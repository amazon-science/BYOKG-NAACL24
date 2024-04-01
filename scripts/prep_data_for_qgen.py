import json
import logging
import os
import random

from src.utils.arguments import Arguments
from transformers import set_seed

from src.utils.kg import create_splits_for_q_gen

"""
Script that takes either the output from the Explorer or a dataset file and outputs a file that can be consumed by the question generator.
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups="prep_data_for_qgen")
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

    if args.walks_fpath is not None:
        create_splits_for_q_gen(fpath=args.walks_fpath,
                                n_samples_per_split=0,  # 0 means use all samples
                                out_fnames=[args.walks_out_fname],  # set to "qgen_walks"
                                sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema)

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
                                        sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema)
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
                                    out_fnames=[args.train_out_fname],  # set to "qgen_train"
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema)

        if args.dev_fpath is not None:
            create_splits_for_q_gen(fpath=args.dev_fpath,
                                    n_samples_per_split=0,
                                    out_fnames=[args.dev_out_fname],  # set to "qgen_dev"
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema)

        if args.test_fpath is not None:
            create_splits_for_q_gen(fpath=args.test_fpath,
                                    n_samples_per_split=0,
                                    out_fnames=[args.test_out_fname],  # set to "qgen_test"
                                    sexpr_machine_key=args.sexpr_machine_key, rev_schema=rev_schema)
