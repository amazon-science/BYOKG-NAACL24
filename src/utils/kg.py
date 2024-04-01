import copy
import json
import logging
import os.path
import random

import networkx as nx

from .helpers import rename_key_in_dict, split_underscore_period
from .maps import fn_description
from .parser import graph_query_to_sexpr, comp_map_2_alpha, sexpr_to_sparql, graph_query_to_graph, sexpr_to_struct

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_readable_class(name, schema=None):
    if schema is not None:
        if "classes" in schema:
            if name in schema["classes"]:
                if "description" in schema["classes"][name]:
                    return schema["classes"][name]["description"]
    return split_underscore_period(name, 2)


def get_readable_relation(name, schema=None):
    if name.endswith("#R"):
        name = name[:-2]
    if schema is not None:
        if "relations" in schema:
            if name in schema["relations"]:
                if "description" in schema["relations"][name]:
                    return schema["relations"][name]["description"]
            elif "inverse_relations" in schema:
                if name in schema["inverse_relations"]:
                    inv_rel = schema["inverse_relations"][name]
                    if "reverse_description" in schema["relations"][inv_rel]:
                        return schema["relations"][inv_rel]["reverse_description"]
    return split_underscore_period(name, 2)


def get_reverse_relation(name, schema=None):
    if name.endswith("#R"):
        name = name[:-2]
    if schema is not None:
        if "relations" in schema:
            if name in schema["relations"]:
                if "reverse" in schema["relations"][name]:
                    return schema["relations"][name]["reverse"]
            elif "inverse_relations" in schema:
                if name in schema["inverse_relations"]:
                    return schema["inverse_relations"][name]
    return None


def get_reverse_readable_relation(name, schema=None):
    if name.endswith("#R"):
        name = name[:-2]
    if schema is not None:
        if "relations" in schema:
            if name in schema["relations"]:
                if "reverse_description" in schema["relations"][name]:
                    return schema["relations"][name]["reverse_description"]
                if "reverse" in schema["relations"][name]:
                    return split_underscore_period(schema["relations"][name]["reverse"], 1)
            elif "inverse_relations" in schema:
                if name in schema["inverse_relations"]:
                    inv_rel = schema["inverse_relations"][name]
                    if inv_rel in schema["relations"]:
                        if "description" in schema["relations"][inv_rel]:
                            return schema["relations"][inv_rel]["description"]
                        return split_underscore_period(inv_rel, 2)
    return None


def _anonymize_q(question, node_map, add_quotes=False) -> str:
    for node in node_map:
        split_q = question.split(node)
        if len(split_q) == 1:
            question = split_q[0]
        if len(split_q) >= 2:
            mapped_node = f'"{node_map[node]}"' if add_quotes else node_map[node]
            question = mapped_node.join(list(map(lambda x: _anonymize_q(x, node_map, add_quotes=add_quotes), split_q)))
            break
    return question


def add_reverse_info_from_schema(gq, schema):
    for e in gq["edges"]:
        if "reverse_relation" not in e or e["reverse_relation"] is None:
            if e["relation"] in schema["relations"]:
                e["reverse_relation"] = schema["relations"][e["relation"]].get("reverse", None)
                if "reverse_readable_name" not in e or e["reverse_readable_name"] is None:
                    e["reverse_readable_name"] = schema["relations"][e["relation"]].get("reverse_description", None)
                    if e["reverse_readable_name"] is None and e["reverse_relation"] is not None:
                        e["reverse_readable_name"] = split_underscore_period(e["reverse_relation"], 1)


def create_splits_for_q_gen(fpath, n_samples_per_split, out_fnames: list, type_constraint=True,
                            sexpr_machine_key="s_expression", rev_schema=None, keep_all_answers=False,
                            additional_keys=None):
    with open(fpath, 'r') as fh:
        data = json.load(fh)

    n_splits = len(out_fnames)
    if n_samples_per_split == 0:
        sampled = data
        n_samples_per_split = len(data)
        assert n_splits == 1
    else:
        sampled = random.sample(data, (n_samples_per_split) * n_splits)

    # Add friendly_name/readable_name to edges where it may be missing (WebQSP needed this)
    for s in sampled:
        for e in s['graph_query']['edges']:
            if e.get('readable_name', e.get('friendly_name', None)) is None:
                e['readable_name'] = get_readable_relation(e['relation'], schema=rev_schema)

    sample_idx = 0
    for split_idx in range(n_splits):
        dataset = []
        while len(dataset) < n_samples_per_split:
            sample = sampled[sample_idx]
            rename_key_in_dict(sample, "friendly_name", "readable_name")
            sample_answers = None
            if "answer" in sample:
                sample_answers = []
                for sample_answer in sample["answer"]:
                    sample_answers.append({
                        "type": sample_answer["answer_type"],
                        "value": sample_answer["answer_argument"],
                        "value_readable": sample_answer["entity_name"] if "entity_name" in sample_answer else
                        sample_answer["answer_argument"]
                    })
            gq = sample["graph_query"]
            if rev_schema is not None:
                add_reverse_info_from_schema(gq, rev_schema)
            sexpr_machine = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                                 readable=False)
            if type_constraint:
                assert sexpr_machine == sample[sexpr_machine_key]
            sexpr_machine_rev = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                                     readable=False, use_reverse_relations=True)
            sexpr_anon = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                              readable=True, readable_type='anon')
            sexpr_anon_rev = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                                  readable=True, readable_type='anon',
                                                  use_reverse_relations=True)
            sexpr_label = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                               readable=True, readable_type='label')
            sexpr_label_rev = graph_query_to_sexpr(gq, type_constraint=type_constraint,
                                                   readable=True, readable_type='label',
                                                   use_reverse_relations=True)
            node_2_machine = {}
            machine_2_anon = {}
            schema = {"nodes_machine": {}, "nodes_machine-rev": {}, "nodes_anon": {}, "nodes_anon-rev": {},
                      "nodes_label": {}, "nodes_label-rev": {}, "relations": {}, "functions": {}}
            for n in gq["nodes"]:
                if n["id"] in sexpr_machine:
                    if n["node_type"] != "class":
                        node_2_machine[n["readable_name"].lower()] = n['id']
                        machine_2_anon[n['id']] = f"{n['class'].split('.')[-1]}_{n['nid']}"
                    if n["node_type"] in ["class", "entity"]:
                        schema["nodes_machine"][n["id"]] = n["readable_name"].lower()
                        anon_key = n["id"] if n["node_type"] == "class" else machine_2_anon[n["id"]]
                        schema["nodes_anon"][anon_key] = n["readable_name"].lower()
                        if n["node_type"] == "class":
                            schema["nodes_label"][n["id"]] = n["readable_name"].lower()
            schema["nodes_machine-rev"] = schema["nodes_machine"]
            schema["nodes_anon-rev"] = schema["nodes_anon"]
            schema["nodes_label-rev"] = schema["nodes_label"]
            for e in gq["edges"]:
                if e["relation"] in sexpr_machine:
                    schema["relations"][e["relation"]] = e["readable_name"].lower()
                if e.get("reverse_relation", None) is not None and e["reverse_relation"] in sexpr_label_rev:
                    schema["relations"][e["reverse_relation"]] = e["reverse_readable_name"].lower()
            if sample["function"] != "none":
                if sample["function"] in comp_map_2_alpha:
                    fn_key = comp_map_2_alpha[sample["function"]]
                else:
                    fn_key = sample["function"].upper()
                schema["functions"][fn_key] = fn_description[fn_key]

            instance = {
                "id": len(dataset),
                "original_fname": '.'.join(fpath.split('/')[-1].split('.')[:-1]),
                "original_qid": sample["qid"],
                "s_expression_anon": sexpr_anon,
                "s_expression_anon-rev": sexpr_anon_rev,
                "s_expression_label": sexpr_label,
                "s_expression_label-rev": sexpr_label_rev,
                "s_expression_machine": sexpr_machine,
                "s_expression_machine-rev": sexpr_machine_rev,
                "schema": schema
            }
            question_label = sample["question"] if "question" in sample else None
            if question_label is not None:
                question_machine = _anonymize_q(question_label.lower(), node_2_machine, add_quotes=True)
                question_anon = _anonymize_q(question_machine.lower(), machine_2_anon,
                                             add_quotes=False)  # Quotes already added
                instance.update({
                    "question_anon": question_anon,
                    "question_anon-rev": question_anon,
                    "question_label": question_label,
                    "question_label-rev": question_label,
                    "question_machine": question_machine,
                    "question_machine-rev": question_machine
                })
            if sample_answers is not None:
                if not keep_all_answers:
                    if len(sample_answers) == 0:
                        sample_answer = {
                            "type": "Value",
                            "value": "Not available",
                            "value_readable": "Not available"
                        }
                    else:
                        sample_answer = random.choice(sample_answers)
                    instance['sample_answer'] = sample_answer
                else:
                    instance['answer'] = sample_answers

            if additional_keys is not None:
                for k in additional_keys:
                    if k in sample:
                        instance[k] = sample[k]
                    else:
                        if k == "question_entities":
                            _entities = []
                            for n in gq["nodes"]:
                                if n["node_type"] == "entity" and n["id"] in sexpr_machine:
                                    _entities.append(n["id"])
                            instance[k] = _entities

            dataset.append(instance)
            sample_idx += 1

        out_dir = os.path.dirname(fpath)
        out_fpath = os.path.join(out_dir, f"{out_fnames[split_idx]}.json")
        with open(out_fpath, 'w') as fh:
            fh.write(json.dumps(dataset, indent=2))
        logger.info(f'Wrote {out_fnames[split_idx]} set (n={len(dataset)}) to {out_fpath}')


def get_non_literals(nodes, except_nid=None):
    res = []
    for n in nodes:
        if n["node_type"] != "literal":
            if except_nid is None or n["nid"] not in except_nid:
                res.append(n)
    return res


def get_nodes_by_class(nodes, cls, except_nid=None):
    res = []
    if not cls.startswith("type."):
        for n in nodes:
            if n["node_type"] == "class" and n["class"] == cls:
                if except_nid is None or n["nid"] not in except_nid:
                    res.append(n)
    return res


def _renumber_nodes(_gq):
    renumber_edges = False
    node_map_old_to_new = {}
    for i, n in enumerate(_gq['nodes']):
        node_map_old_to_new[n['nid']] = i
        renumber_edges = renumber_edges or node_map_old_to_new[n['nid']] != i
    if renumber_edges:
        new_edges = []
        for i in range(_gq['edges']):
            _start = _gq['edges'][i]['start']
            _end = _gq['edges'][i]['end']
            if any(_old not in node_map_old_to_new for _old in [_start, _end]):
                continue
            edge = _gq['edges'][i]
            edge['start'] = node_map_old_to_new[_start]
            edge['end'] = node_map_old_to_new[_end]
            new_edges.append(edge)
        _gq['edges'] = new_edges


def prune_graph_query(_sparql, gq, orig_ans=None, final_check=False, verbose=False):
    discard_edges = set()
    _gq = copy.deepcopy(gq)
    n_nodes = len(_gq['nodes'])
    n_edges = len(_gq['edges'])

    orig_sexpr = graph_query_to_sexpr(_gq, type_constraint=True, readable=False)
    if verbose:
        logger.info(f'Orig sexpr: {orig_sexpr}')
    orig_ans = set(_sparql.execute_query(sexpr_to_sparql(orig_sexpr))) if orig_ans is None else set(orig_ans)
    if verbose:
        logger.info(f'Orig ans: {orig_ans}')

    # Find edges to discard
    for i in range(n_edges - 1, -1, -1):
        if len(_gq['edges']) == 1:
            continue
        popped = _gq['edges'].pop()
        if verbose:
            logger.info(f'Popped: {popped}')
        try:
            new_sexpr = graph_query_to_sexpr(_gq, type_constraint=True, readable=False)
            if verbose:
                logger.info(f'New sexpr: {new_sexpr}')
            new_struct = sexpr_to_struct(new_sexpr)
            if type(new_struct) is str or len(new_struct) == 1:
                _gq['edges'].insert(0, popped)  # Put back in
                continue
            new_sparql = sexpr_to_sparql(new_sexpr)
            if verbose:
                logger.info(f'New sparql: {new_sparql}')
        except:
            # Taking this to mean that the removed edge is non-computable
            # TODO: Add better logic?
            _gq['edges'].insert(0, popped)  # Put back in
            continue
        new_ans = set(_sparql.execute_query(new_sparql, catch_timeout=True))
        if verbose:
            logger.info(f'New ans: {new_ans}')
        if new_ans != orig_ans:
            if verbose:
                logger.info(f'Different result. Retain node idx={i}.')
            _gq['edges'].insert(0, popped)  # Put back in
        else:
            if verbose:
                logger.info(f'Same result. Discard node idx={i}.')
            discard_edges.add(i)
    # Add retained edges back in the same order
    _gq['edges'] = []
    for i in range(n_edges):
        if i not in discard_edges:
            _gq['edges'].append(gq['edges'][i])

    # Remove any disconnected nodes
    discard_nodes = set()
    G, aggregation, arg_node, qid = graph_query_to_graph(_gq)
    for i, node in enumerate(_gq['nodes']):
        if node['nid'] == qid:
            continue
        paths_to_qid = list(nx.all_simple_paths(G, qid, node['nid']))
        if len(paths_to_qid) == 0:
            discard_nodes.add(i)
        else:
            # TODO: Add code to remove ungrounded class nodes that are unnatural constructions
            pass
    _gq['nodes'] = []
    for i in range(n_nodes):
        if i not in discard_nodes:
            _gq['nodes'].append(gq['nodes'][i])

    _renumber_nodes(_gq)  # since pruning nodes could've made some nid's non-contiguous

    if final_check:
        new_sexpr = graph_query_to_sexpr(_gq, type_constraint=True, readable=False)
        try:
            assert set(_sparql.execute_query(sexpr_to_sparql(new_sexpr))) == orig_ans
        except AssertionError:
            logger.info('Final check failed during query pruning. Retaining unpruned program.')
            return gq, orig_ans
    return _gq, orig_ans


def legal_relation(r):
    if r.startswith('common.') or r.startswith('kg.') or r.startswith('user.'):
        return False
    return True

def legal_class(c):
    if c.startswith('common.') or c.startswith('kg.') or c.startswith('user.') or c.startswith('base.type_ontology'):
        return False
    return True
