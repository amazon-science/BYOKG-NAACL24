import json
import logging
import os
import random

from src.utils.arguments import Arguments
from tqdm import tqdm
from transformers import set_seed

from src.utils.helpers import split_underscore_period
from src.utils.kg import get_readable_relation
from src.utils.parser import sexpr_to_sparql, graph_query_to_sexpr
from src.utils.sparql import SPARQLUtil

"""
Script to process the raw MetaQA dataset files into a format that is friendly for our codebase
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups="dataset_metaqa")
    global args
    args = cli_args.parse_args()
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)

    _sparql = SPARQLUtil(args.sparql_url, graph_name='metaqa')

    rel_map = {
        'actor': 'movie.starred_actors',
        'genre': 'movie.has_genre',
        'language': 'movie.in_language',
        'director': 'movie.directed_by',
        'year': 'movie.release_year',
        'writer': 'movie.written_by',
        'tag': 'movie.has_tags',
        'imdbvote': 'movie.has_imdb_votes',
        'imdbrating': 'movie.has_imdb_rating'
    }
    cls_map = {
        'movie': 'movie.movie',
        'actor': 'movie.person',
        'genre': 'movie.genre',
        'language': 'movie.language',
        'director': 'movie.person',
        'year': 'type.datetime',
        'writer': 'movie.person',
        'tag': 'movie.tag',
        'imdbvote': 'movie.vote',
        'imdbrating': 'movie.rating'
    }

    with open('data/graphs/metaqa/schema.json', 'r') as fh:
        schema = json.load(fh)

    for n_hop in [1, 2, 3]:
        logger.info(f'Hop: {n_hop}')
        for split in ['train', 'dev', 'test']:
            logger.info(f'Split: {split}')
            dataset = []
            mismatches = []
            qa_fpath = os.path.join(args.data_dir, f'{n_hop}-hop', f'qa_{split}.txt')
            qa_type_fpath = os.path.join(args.data_dir, f'{n_hop}-hop', f'qa_{split}_qtype.txt')
            idx = 0
            with open(qa_fpath, 'r') as fh:
                qa_lines = fh.readlines()
            with open(qa_type_fpath, 'r') as fh:
                qa_type_lines = fh.readlines()

            for _i, line in enumerate(tqdm(qa_lines)):
                q, _ans = line.split('\t')
                _ans = _ans.strip()
                ans = _ans.split('|')
                _ent = q[q.index('[') + 1: q.index(']')]
                qa_type_line = qa_type_lines[_i].strip()
                qtype = list(map(lambda x: x[:-1] if x.endswith('s') else x, qa_type_line.split('_to_')))
                q_entids = _sparql.get_entid_by_cls_label(cls=cls_map[qtype[0]], label=_ent)
                match_q_entid, mismatch_q_entid = [], []
                for q_entid in q_entids:
                    q_ent = {
                        'class': cls_map[qtype[0]],  # e.g. 'movie.movie'
                        'value': q_entid,
                        'value_readable': _ent
                    }
                    sexpr = q_ent['value']
                    for i, t in enumerate(qtype):  # e.g. movie_to_director_to_movie_to_genre
                        if i == 0:
                            continue
                        if t != 'movie':
                            sexpr = f"(JOIN (R {rel_map[t]}) {sexpr})"
                        else:
                            prev_t = qtype[i - 1]
                            sexpr = f"(JOIN {rel_map[prev_t]} {sexpr})"
                    sexpr = f"(AND {cls_map[qtype[-1]]} {sexpr})"
                    sparql_query = sexpr_to_sparql(sexpr, ' ')
                    sparql_ans = _sparql.get_answer_set_ent_val(sexpr)
                    ans_mismatch = set([a['answer_argument' if a['answer_type'] == 'Value' else 'entity_name'] for a in
                                        sparql_ans]) != set(ans)

                    gq = {'nodes': [], 'edges': []}

                    for ti, t in enumerate(qtype[::-1]):
                        node = {
                            "nid": ti,
                            "node_type": "class",
                            "id": cls_map[t],
                            "class": cls_map[t],
                            "readable_name": split_underscore_period(cls_map[t]) if cls_map[t].startswith('type.') else
                            schema['classes'][cls_map[t]]['description'],
                            "question_node": 1 if ti == 0 else 0,
                            "function": "none"
                        }
                        if ti > 0:
                            # Check if the class needs to be grounded
                            if ti == len(qtype) - 1:
                                assert node["class"] == q_ent["class"]
                                node["node_type"] = "entity"
                                node["id"] = q_ent["value"]
                                node["readable_name"] = q_ent["value_readable"]

                            # Add edge
                            if node["class"] != "movie.movie":
                                assert gq["nodes"][ti - 1]["class"] == "movie.movie"
                            if node["class"] == "movie.movie":
                                assert gq["nodes"][ti - 1]["class"] != "movie.movie"

                            rel = rel_map[t] if t != "movie" else rel_map[qtype[::-1][ti - 1]]
                            edge = {
                                "start": ti if node["class"] == "movie.movie" else ti - 1,
                                "end": ti if node["class"] != "movie.movie" else ti - 1,
                                "relation": rel,
                                "readable_name": schema['relations'][rel]['description'],
                                "reverse_relation": schema['relations'][rel]['reverse'],
                                "reverse_readable_name": get_readable_relation(schema['relations'][rel]['reverse'])
                            }
                            gq["edges"].append(edge)
                        gq["nodes"].append(node)

                    assert graph_query_to_sexpr(gq) == sexpr

                    instance = {
                        'qid': idx,
                        'question': q.replace('[', '').replace(']', ''),
                        'question_linked': q,
                        'question_entities': [q_entid],
                        'answer': sparql_ans,
                        'function': 'none',
                        'num_node': n_hop + 1,
                        'num_edge': n_hop,
                        'graph_query': gq,
                        'sparql_query': sparql_query,
                        'question_type': '_to_'.join(qtype),
                        's_expression': sexpr,
                        'ans_original': _ans,
                        'ans_mismatch': ans_mismatch  # i.e. SPARQL answer doesn't match published dataset's answer
                    }
                    if not ans_mismatch:
                        match_q_entid.append(instance)
                    else:
                        mismatch_q_entid.append(instance)
                if len(match_q_entid) > 0:
                    dataset.append(random.choice(match_q_entid))
                else:
                    dataset.append(random.choice(mismatch_q_entid))
                    mismatches.append(dict(dataset[-1], orig_answer=ans))
                idx += 1

            assert len(qa_lines) == len(dataset)

            # Write data files
            out_dir = args.data_out_dir if args.data_out_dir is not None else args.data_dir
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'{split}_{n_hop}-hop.json'), 'w') as fh:
                fh.write(json.dumps(dataset, indent=2))
                logger.info(f"Created {os.path.join(out_dir, f'{split}_{n_hop}-hop.json')} (n={len(dataset)})")
            if len(mismatches) > 0:
                os.makedirs(os.path.join(out_dir, 'mismatches'), exist_ok=True)
                with open(os.path.join(out_dir, 'mismatches', f'{split}_{n_hop}-hop.json'), 'w') as fh:
                    fh.write(json.dumps(mismatches, indent=2))
                    logger.info(
                        f"Created {os.path.join(out_dir, 'mismatches', f'{split}_{n_hop}-hop.json')} (n={len(mismatches)})")
