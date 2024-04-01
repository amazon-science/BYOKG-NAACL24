import json
import logging
import os
import time

from src.utils.arguments import Arguments
from transformers import set_seed

from src.utils.helpers import setup_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups="merge_qgen_outputs")
    global args
    args = cli_args.parse_args()
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else str(args.run_id)
    setup_logger(RUN_ID)
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)

    with open(args.qgen_out_1, 'r') as fh:
        qgen_out_1 = json.load(fh)
        qgen_out_1 = qgen_out_1['examples']
    with open(args.qgen_out_2, 'r') as fh:
        qgen_out_2 = json.load(fh)
        qgen_out_2 = qgen_out_2['examples']
    qgen_out = qgen_out_1 + qgen_out_2
    for i, n in enumerate(qgen_out):
        n['idx'] = i + 1
        n['id'] = i

    out_dir = os.path.join(args.out_dir, "question_generation", f'merged_{len(qgen_out)}_{RUN_ID}')
    os.makedirs(out_dir, exist_ok=True)

    qgen_out_fpath = os.path.join(out_dir, f'qgen_out.json')
    with open(qgen_out_fpath, 'w') as fh:
        fh.write(json.dumps({"examples": qgen_out}, indent=2))
    logger.info(f'Wrote merged output file to {qgen_out_fpath}')

    with open(args.qgen_in_1, 'r') as fh:
        qgen_in_1 = json.load(fh)
    with open(args.qgen_in_2, 'r') as fh:
        qgen_in_2 = json.load(fh)
    qgen_in = qgen_in_1 + qgen_in_2
    for i, n in enumerate(qgen_in):
        n['id'] = i

    qgen_in_fpath = os.path.join(out_dir, f'qgen_in.json')
    with open(qgen_in_fpath, 'w') as fh:
        fh.write(json.dumps(qgen_in, indent=2))
    logger.info(f'Wrote merged input file to {qgen_in_fpath}')
