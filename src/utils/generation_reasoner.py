# TODO: Change file structure to src.utils.generation.reasoner
import copy
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

from src.utils.generation import stringify_schema, prompt_arr_2_text, get_logprob_score
from src.utils.helpers import deep_get
from src.utils.kg import get_readable_class, get_readable_relation
from src.utils.maps import fn_description, literal_map, literal_map_inv
from src.utils.parser import parse_bottom_up, bottom_up_to_sexpr

from src.utils.generation import default_instructions as gen_default_instructions
from src.utils.generation import default_instructions_R as gen_default_instructions_R
from src.utils.generation import default_query_prefix as gen_default_query_prefix
from src.utils.generation import default_output_prefix as gen_default_output_prefix
from src.utils.sparql import SPARQLUtil

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

default_instructions = [
    """### Instructions:\nWrite a logical form expression using only elements mentioned in the provided natural language question."""]
default_instructions_l2m = [
    """### Instructions:\nWrite a logical form expression in a step-by-step manner using only elements mentioned in the provided natural language question."""]
default_instructions_pangu = [
    """### Please translate the follow questions to Lisp-like query language."""]
default_instructions_R = [
    """An "R" before a relation in the logical expression may be used to indicate a reverse or inverse relation."""
]
default_instructions_schema = [
    """You may use elements from the provided schema, which describe entities, relations, and functions for the logical form.""",
    """Only use those elements from the schema that are meaningful for the provided question."""]
default_query_prefix = """### Question:\n"""
default_query_prefix_pangu = """# """
default_schema_prefix = """### Schema:\n"""
default_output_prefix = f"""### Logical Form:\n"""
default_output_prefix_pangu = f""""""
default_output_prefix_chat = f"""Assistant: Sure, I'm happy to help you with that. Here is a possible logical form expression for the provided question:\n\n"""
default_output_prefix_chat_beluga = f"""### Assistant:\nSure, I'm happy to help you with that. Here is a possible logical form expression for the provided question:\n\n"""

# Inverse-consistency prompts
default_inv_instructions = [
    """### Instructions:\nWrite a plausible question in English that can be formed from the provided logical query as a starting point.""",
    """The question must contain at least all of the information present in the logical query."""]
default_inv_instructions_R = []
default_inv_instructions_schema = [
    """You may refer to the provided schema, which describes the relevant entities, relations, and functions in the provided logical query."""]
default_inv_query_prefix = """### Logical Query:\n"""
default_inv_schema_prefix = """### Schema:\n"""
default_inv_output_prefix = """### Plausible Question:\n"""
default_inv_output_prefix_chat = f"""Assistant: Sure, I'm happy to help you with that. Here is a plausible question for the provided logical query:\n\n"""
default_inv_output_prefix_chat_beluga = f"""### Assistant:\nSure, I'm happy to help you with that. Here is a plausible question for the provided logical query:\n\n"""

llama_chat_B_INST, llama_chat_E_INST = "[INST]", "[/INST]"
llama_chat_B_SYS, llama_chat_E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
llama_chat_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

beluga_DEFAULT_SYSTEM_PROMPT = "### System:\nYou are a helpful AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

default_metrics = [
    {"name": "f1", 'score_keys': ['f1'], 'args': {}},
    {"name": "accuracy", 'score_keys': ['accuracy'], 'args': {}},
    {"name": "hits@1", 'score_keys': ['hits@1'], 'args': {}},
]


def construct_args_from_example(example, anon_type, use_schema, use_answer=False):
    inf_args = {
        'query': example[f'question_{anon_type}'],
        'schema': None,
        'answer': None,
    }
    ref = example[f's_expression_{anon_type}'] if f's_expression_{anon_type}' in example else None
    if use_schema:
        inf_args['schema'] = stringify_schema(schema=example['schema'], anon_type=anon_type, query=ref)
    return inf_args, ref


def skip_mismatch(obj):
    return not obj.get('ans_mismatch', False)


def skip_empty(obj):
    answer = obj.get('answer', [])
    return type(answer) is list and len(answer) != 0


def expand_candidates(_sparql, candidates, step, top_candidates, use_classes_only=False, skip_on_error=True, max_join=5,
                      freebase=False, legal_relation_set=None, schema=None, verbose=False, use_type_constraint=True,
                      sanity_check=False):
    for c in tqdm(candidates[step], disable=True):
        bottom_up = parse_bottom_up(c)
        last_op = bottom_up[-1][0]
        last_op_arg = bottom_up[-1][1]
        prev_was_type_constraint = False
        classes = []
        init_type = None

        if step == 0:
            if c.startswith('m.') or c.startswith('g.'):
                init_type = 'entity'
            elif '^^' in c:
                init_type = 'literal'
                init_literal_type = literal_map_inv[c.split('^^')[1]]
            else:
                init_type = 'class'
                if c in literal_map:
                    continue
                # # AND with initial class
                # cand = f'(AND {c} (*))'
                # candidates[step + 1].append(cand)
        else:
            if last_op in ['COUNT', 'ARGMIN', 'ARGMAX']:
                continue
            if (last_op == 'AND' and not last_op_arg.startswith('#')) or not use_type_constraint:
                if (last_op == 'AND' and not last_op_arg.startswith('#')):
                    prev_was_type_constraint = True
                # COUNT
                cand = f'(COUNT {c})'
                candidates[step + 1].append(cand)

        if prev_was_type_constraint:
            out_relations = []
            in_relations = []
            if last_op_arg not in literal_map:
                out_relations = filter_rel(
                    lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                    _sparql.get_relations(c, direction='out', use_classes_only=use_classes_only,
                                          catch_timeout=skip_on_error, filter_entities=False))
                in_relations = filter_rel(
                    lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                    _sparql.get_relations(c, direction='in', use_classes_only=use_classes_only,
                                          catch_timeout=skip_on_error, filter_entities=False))
        else:
            if init_type == 'literal':
                out_relations = []
                in_relations = filter_rel(
                    lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                    get_literal_relations(init_literal_type, schema))
            else:
                out_relations = filter_rel(
                    lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                    _sparql.get_relations(c, direction='out', use_classes_only=use_classes_only,
                                          catch_timeout=skip_on_error, filter_entities=False))
                in_relations = filter_rel(
                    lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                    _sparql.get_relations(c, direction='in', use_classes_only=use_classes_only,
                                          catch_timeout=skip_on_error, filter_entities=False))
            if step > 0:
                classes = filter_cls(legal_class if freebase else lambda x: True,
                                     _sparql.get_cls_by_query(c, use_classes_only=use_classes_only,
                                                              catch_timeout=skip_on_error,
                                                              filter_entities=False))

        for rel in out_relations:
            r = rel['relation']
            if (not prev_was_type_constraint or not use_type_constraint) and init_type != 'class':
                # Reverse JOIN
                cand = f'(JOIN (R {r}) {c})'
                if cand.count('JOIN') <= max_join:
                    candidates[step + 1].append(cand)

            if (prev_was_type_constraint or not use_type_constraint) or init_type == 'class':
                # ARGMAX / ARGMIN
                if rel['range'] in literal_map:
                    literal_cls = rel['range'][len('type.'):]
                    if literal_cls in ['int', 'integer', 'float', 'datetime']:
                        for op in ['ARGMAX', 'ARGMIN']:
                            cand = f'({op} {c} {r})'
                            candidates[step + 1].append(cand)
                else:
                    # Check if 2nd hop relation is numerical
                    # Get relations that have rel['range'] as the domain and check if their range is numerical
                    c2 = f'(JOIN (R {r}) {c})'
                    if c2.count('JOIN') <= max_join:
                        out_relations_2 = filter_rel(
                            lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                            _sparql.get_relations(c2, direction='out', use_classes_only=use_classes_only,
                                                  catch_timeout=skip_on_error, filter_entities=False))
                        for rel2 in out_relations_2:
                            r2 = rel2['relation']
                            if rel2['range'] in literal_map:
                                literal_cls = rel2['range'][len('type.'):]
                                if literal_cls in ['int', 'integer', 'float', 'datetime']:
                                    for op in ['ARGMAX', 'ARGMIN']:
                                        cand = f'({op} {c} (JOIN {r} {r2}))'
                                        candidates[step + 1].append(cand)

        # Forward JOIN
        for rel in in_relations:
            r = rel['relation']
            if (not prev_was_type_constraint or not use_type_constraint) and init_type != 'class':
                cand = f'(JOIN {r} {c})'
                if cand.count('JOIN') <= max_join:
                    candidates[step + 1].append(cand)
            # Check if 2nd hop relation is numerical
            # Get relations that have rel['domain'] as the domain and check if their range is numerical
            if (prev_was_type_constraint or not use_type_constraint) or init_type == 'class':
                c2 = f'(JOIN {r} {c})'
                if c2.count('JOIN') <= max_join:
                    out_relations_2 = filter_rel(
                        lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                        _sparql.get_relations(c2, direction='out', use_classes_only=use_classes_only,
                                              catch_timeout=skip_on_error, filter_entities=False))
                    for rel2 in out_relations_2:
                        r2 = rel2['relation']
                        if rel2['range'] in literal_map:
                            literal_cls = rel2['range'][len('type.'):]
                            if literal_cls in ['int', 'integer', 'float', 'datetime']:
                                for op in ['ARGMAX', 'ARGMIN']:
                                    cand = f'({op} {c} (JOIN (R {r}) {r2}))'
                                    candidates[step + 1].append(cand)

        if step > 0:
            for cls in classes:
                # if use_type_constraint:
                # AND
                cand = f'(AND {cls} {c})'
                candidates[step + 1].append(cand)

                # lt / le / gt / ge
                if cls in literal_map:
                    literal_cls = cls[len('type.'):]
                    if literal_cls in ['int', 'integer', 'float', 'datetime']:
                        literal_relations = filter_rel(
                            lambda x: legal_relation(x, legal_relation_set) if freebase else lambda x: True,
                            _sparql.get_relations(cls, direction="in",
                                                  catch_timeout=skip_on_error,
                                                  filter_entities=False))
                        for rel in literal_relations:
                            r = rel['relation']
                            for op in ['lt', 'le', 'gt', 'ge']:
                                cand = f'({op} {r} {c})'
                                candidates[step + 1].append(cand)
        elif init_type == 'literal' and init_literal_type[len('type.'):] in ['int', 'integer', 'float', 'datetime']:
            for rel in in_relations:
                r = rel['relation']
                for op in ['lt', 'le', 'gt', 'ge']:
                    cand = f'({op} {r} {c})'
                    candidates[step + 1].append(cand)

    if step > 0:
        # AND c1 c2
        cand_len = len(top_candidates)
        for _i in range(cand_len - 1):
            for _j in range(_i + 1, cand_len):
                c1 = top_candidates[_i]
                c2 = top_candidates[_j]
                bottom_up1 = parse_bottom_up(c1)
                last_op1 = bottom_up1[-1][0]
                last_op_arg1 = bottom_up1[-1][1]
                prev_was_type_constraint1 = False
                bottom_up2 = parse_bottom_up(c2)
                last_op2 = bottom_up2[-1][0]
                last_op_arg2 = bottom_up2[-1][1]
                prev_was_type_constraint2 = False

                if any(lop in ['AND', 'COUNT', 'ARGMIN', 'ARGMAX'] for lop in [last_op1, last_op2]):
                    continue

                classes1 = filter_cls(legal_class if freebase else lambda x: True,
                                      _sparql.get_cls_by_query(c1, use_classes_only=use_classes_only,
                                                               catch_timeout=skip_on_error,
                                                               filter_entities=False))
                classes2 = filter_cls(legal_class if freebase else lambda x: True,
                                      _sparql.get_cls_by_query(c2, use_classes_only=use_classes_only,
                                                               catch_timeout=skip_on_error,
                                                               filter_entities=False))
                if any(len(cls) == 0 for cls in [classes1, classes2]):
                    continue

                and_c1_c2 = filter_cls(legal_class if freebase else lambda x: True,
                                       _sparql.get_cls_by_query(f'(AND {c1} {c2})', use_classes_only=use_classes_only,
                                                                catch_timeout=skip_on_error, filter_entities=False))
                if len(and_c1_c2) > 0:
                    cand = f'(AND {c1} {c2})'
                    candidates[step + 1].append(cand)
                    if sanity_check:
                        cand = f'(AND {c2} {c1})'
                        candidates[step + 1].append(cand)

    candidates[step + 1] = list(set(candidates[step + 1]))


def swap_mid_with_name(_sparql, sexprs, ent):
    rtn = []
    label = _sparql.get_label_by_entid(ent)
    if type(sexprs) is str:
        sexprs = [sexprs]
    for sexpr in sexprs:
        bottom_up = parse_bottom_up(sexpr)
        for level in range(len(bottom_up)):
            try:
                idx = bottom_up[level].index(ent)
                bottom_up[level][idx] = label
            except:
                continue
        rtn.append(bottom_up_to_sexpr(bottom_up))
    return rtn


def process_candidates(sexprs, anon_type, _sparql, rev_schema=None, per_cand_schema=False):
    nodes_key = f'nodes_{anon_type}'
    step_schema_template = {nodes_key: {}, 'relations': {}, 'functions': {}}
    cand_schemas = []  # used to return per candidate schemas
    if anon_type.endswith('-rev'):
        assert rev_schema is not None
    rtn = []  # processed candidates
    if type(sexprs) is str:
        sexprs = [sexprs]
    if not per_cand_schema:
        step_schema = copy.deepcopy(step_schema_template)
    for sexpr in sexprs:
        if per_cand_schema:
            step_schema = copy.deepcopy(step_schema_template)
        replaced_idx = 1
        bottom_up = parse_bottom_up(sexpr)
        for level in range(len(bottom_up)):
            for idx in range(len(bottom_up[level])):
                tkn = bottom_up[level][idx]
                if type(tkn) is str:
                    if tkn in fn_description and tkn not in step_schema['functions']:
                        step_schema['functions'][tkn] = fn_description[tkn]
                        continue
                    # Check for entity
                    elif tkn.startswith('m.') or tkn.startswith('g.'):
                        ent_label = _sparql.get_label_by_entid(tkn, use_id_on_empty=True)
                        if 'anon' in anon_type:
                            cls = _sparql.get_cls_by_query(tkn)[0]  # TODO: handle multiple class scenario
                            anon_id = f'{cls.split(".")[-1]}_{replaced_idx}'
                            bottom_up[level][idx] = f'"{anon_id}"'
                            replaced_idx += 1
                            step_schema[nodes_key][anon_id] = ent_label
                        elif 'label' in anon_type:
                            bottom_up[level][idx] = f'"{ent_label}"'
                        elif 'machine' in anon_type:
                            step_schema[nodes_key][tkn] = ent_label
                    elif idx - 1 == 0:
                        if bottom_up[level][idx - 1] == 'AND':
                            # current index is a class
                            cls = bottom_up[level][idx]
                            cls_desc = get_readable_class(cls, rev_schema)
                            step_schema[nodes_key][cls] = cls_desc
                        elif bottom_up[level][idx - 1] in ['JOIN', 'lt', 'le', 'gt', 'ge']:
                            # current index is a relation
                            rel = bottom_up[level][idx]
                            rel_desc = get_readable_relation(rel, rev_schema)
                            step_schema['relations'][rel] = rel_desc
                    elif idx - 2 == 0:
                        if bottom_up[level][idx - 2] in ['ARGMAX', 'ARGMIN']:
                            rel = bottom_up[level][idx]
                            rel_desc = get_readable_relation(rel, rev_schema)
                            step_schema['relations'][rel] = rel_desc
                elif type(tkn) is list:
                    # Check for reverse
                    if tkn[0] == 'R':
                        assert len(tkn) == 2
                        rel = tkn[1]
                        rel_desc = get_readable_relation(rel, rev_schema)
                        if anon_type.endswith('-rev'):
                            if rel in rev_schema['relations'] and 'reverse' in rev_schema['relations'][rel]:
                                bottom_up[level][idx] = rev_schema['relations'][rel]['reverse']
                                rel = bottom_up[level][idx]
                                rel_desc = get_readable_relation(rel, rev_schema)
                            elif rel in rev_schema['inverse_relations']:
                                bottom_up[level][idx] = rev_schema['inverse_relations'][rel]
                                rel = bottom_up[level][idx]
                                rel_desc = get_readable_relation(rel, rev_schema)
                        step_schema['relations'][rel] = rel_desc
        sexpr_processed = bottom_up_to_sexpr(bottom_up)
        rtn.append(sexpr_processed)
        if per_cand_schema:
            cand_schemas.append(stringify_schema(schema=step_schema, anon_type=anon_type, query=sexpr_processed))
    if not per_cand_schema:
        step_schema = stringify_schema(schema=step_schema, anon_type=anon_type)
    return rtn, cand_schemas if per_cand_schema else step_schema


def bm25_tokenizer(text: str, skip_entities=False):
    # Note: skip_entities will only work if the string being passed uses machine ID entities (i.e. starting with "m.")
    return [s for s in text.split() if not skip_entities or not s.startswith('m.') or not s.startswith('g.')]


def remove_id_from_anon_sexpr(sexpr):
    rtn = []
    for s in sexpr.split():
        if s.startswith('"'):
            val = s[1:s[1:].index('"') + 1]
            for i in range(10):
                if val.endswith(f'_{i}'):
                    val = val[:-len(f'_{i}')]
                    s = f'"{val}"'
                    break
        rtn.append(s)
    return " ".join(rtn)


def mask_question_entities(question):
    tkns = question.split()
    for _i, tkn in enumerate(tkns):
        if tkn.startswith('"') and tkn.endswith('"'):
            if tkn[1:].startswith('m.') or tkn[1:].startswith('g.'):
                tkns[_i] = "[MASK]"
    return " ".join(tkns)


def mask_sexpr_entities(sexpr):
    parsed = parse_bottom_up(sexpr)
    for i in range(len(parsed)):
        for j in range(len(parsed[i])):
            if type(parsed[i][j]) is str and (parsed[i][j].startswith('m.') or parsed[i][j].startswith('g.')):
                parsed[i][j] = '[MASK]'
    rtn = bottom_up_to_sexpr(parsed)
    return rtn


def get_exprs(sexpr):
    parsed = parse_bottom_up(sexpr)
    exprs = set()
    for p in parsed:
        new_p = []
        for _p in p:
            if type(_p) is list:
                assert _p[0] == 'R'
                _p = f"({' '.join(_p)})"
            elif _p.startswith('m.') or _p.startswith('g.') or _p == '[MASK]':
                _p = '#ent'
            elif _p.startswith('#'):
                _p = '#var'
            elif "^^" in _p:
                _p = '#lit'
            new_p.append(_p)
        new_p = f"({' '.join(new_p)})"
        exprs.add(new_p)
    return exprs


def get_fewshot_samples_coverage_1(dataset, demos, dense_scores, delta_thresh=7, search_topk=20, n_shots=5):
    # coverage = []
    sampled_fewshot = []
    for idx, q in enumerate(tqdm(dataset, desc="Re-ranking fewshots")):
        top_scores, top_idxs = torch.topk(dense_scores[idx], k=search_topk)
        sampled_demos = [demos[i] for i in top_idxs]
        # gold_exprs = get_exprs(q['s_expression_machine'])
        fewshot_exprs = [get_exprs(s['s_expression_machine']) for s in sampled_demos]
        retained_exprs = set()
        retained_samples = []
        _delta_thresh = delta_thresh + 1
        while len(retained_samples) < n_shots:
            _delta_thresh -= 1
            assert _delta_thresh >= 0
            for _i, fexpr in enumerate(fewshot_exprs):
                if _i not in retained_samples:
                    union_exprs = retained_exprs.union(fexpr)
                    if len(union_exprs) - len(retained_exprs) >= _delta_thresh:
                        retained_samples.append(_i)
                        retained_exprs = retained_exprs.union(fexpr)
                        if len(retained_samples) == n_shots:
                            break
        assert len(retained_samples) == n_shots
        sampled_fewshot.append([sampled_demos[_i] for _i in retained_samples])
        # coverage.append(len(gold_exprs.intersection(retained_exprs)) / len(gold_exprs))
    return sampled_fewshot


def get_fewshot_samples_coverage_2(dataset, demos, dense_scores, search_topk=20, n_shots=5):
    # coverage = []
    sampled_fewshot = []
    for idx, q in enumerate(tqdm(dataset)):
        top_scores, top_idxs = torch.topk(dense_scores[idx], k=search_topk)
        sampled_demos = [demos[i] for i in top_idxs]
        # gold_exprs = get_exprs(q['s_expression_machine'])
        fewshot_exprs = [get_exprs(s['s_expression_machine']) for s in sampled_demos]
        retained_samples = list(range(n_shots))  # Initialize
        for cand_i in range(n_shots, len(sampled_demos)):
            retained_exprs = set().union(*[fewshot_exprs[_i] for _i in retained_samples])
            cur_coverage = len(retained_exprs)
            swap = -1
            max_coverage = cur_coverage
            for swap_i in range(n_shots - 1, -1, -1):
                cand_retained_samples = list(retained_samples)
                cand_retained_samples[swap_i] = cand_i
                cand_retained_exprs = set().union(*[fewshot_exprs[_i] for _i in cand_retained_samples])
                cand_coverage = len(cand_retained_exprs)
                if cand_coverage > max_coverage:
                    max_coverage = cand_coverage
                    swap = swap_i
            if swap > -1:
                retained_samples[swap] = cand_i
                retained_samples.sort(key=lambda x: -top_scores[x])
        assert len(retained_samples) == n_shots
        # retained_exprs = set().union(*[fewshot_exprs[_i] for _i in retained_samples])
        sampled_fewshot.append([sampled_demos[_i] for _i in retained_samples])
        # coverage.append(len(gold_exprs.intersection(retained_exprs)) / len(gold_exprs))
    return sampled_fewshot


def legal_relation(r, legal_relation_set=None):
    if r.startswith('common.') or r.startswith('type.') or r.startswith('kg.') or r.startswith('user.'):
        return False
    if legal_relation_set is not None and r not in legal_relation_set:
        return False
    return True


def legal_class(c):
    if any(c.startswith(k) for k in ["http://www.w3.org/2000/01/rdf-schema",
                                     "http://www.w3.org/2002/07/"]):
        return False
    if c.startswith('base.type_ontology'):
        return False
    # if c.startswith('common.') or c.startswith('kg.') or c.startswith('user.'):
    #     return False
    return True


def filter_rel(fn, seq):
    rtn = []
    for s in seq:
        if fn(s['relation']):
            rtn.append(s)
    return rtn


def filter_cls(fn, seq):
    return list(filter(fn, seq))


def get_literal_relations(literal_type, schema):
    rtn = []
    if literal_type == 'type.integer':
        literal_type = 'type.int'
    for r, robj in schema['relations'].items():
        if robj['range'] == literal_type:
            rtn.append({
                "relation": r,
                "domain": robj['domain'],
                "range": robj['range']
            })
    return rtn


def rerank_results(llm, res_fpath, schema_fpath, sparql_url, kg_name, sparql_cache=None, sparql_retry_on_cache_none=None,
                   alpha=0.5, rel_rep_factor=0.7, anon_type='label', inverse=True, bsz=5, pmi=False):
    with open(res_fpath, 'r') as fh:
        res = json.load(fh)
    with open(schema_fpath, 'r') as fh:
        schema = json.load(fh)

    _sparql = SPARQLUtil(sparql_url, cache_fpath=sparql_cache, retry_on_cache_none_override=sparql_retry_on_cache_none,
                         graph_name=kg_name)

    res['eval_args'].update({
        'rerank_inverse': inverse,
        'rerank_pmi': pmi,
        'rerank_alpha': alpha,
        'rerank_rel_rep_factor': rel_rep_factor,
        'rerank_anon_type': anon_type
    })

    references = []
    reranked_answers = []
    for i, r in enumerate(tqdm(res['examples'], desc='Re-ranking')):
        question = r['query']
        if r['reference_ans'] is not None:
            references.append(r['reference_ans'])

        top_cands = r['prediction_sexpr_candidates']
        if len(top_cands) == 0:
            r['orig_prediction_sexpr'] = ""
            r['orig_prediction_sexpr_candidates'] = []
            r['orig_prediction_ans'] = []
            reranked_answers.append([])
            continue

        top_cand_sexprs = [t[0] for t in top_cands]
        top_cand_scores = [t[1] for t in top_cands]
        top_cand_sexprs_anon, _ = process_candidates(top_cand_sexprs, anon_type, _sparql, rev_schema=schema)

        # Get zero-shot scores for the top candidates
        if inverse:
            prefixes = []
            prompt_arr = [" ".join(gen_default_instructions + gen_default_instructions_R)]
            for cand in top_cand_sexprs_anon:
                prompt = prompt_arr_2_text(prompt_arr + [f"{gen_default_query_prefix}{cand}"],
                                           '\n\n', llm.is_llama2, llm.is_beluga, llm.is_chat, gen_default_output_prefix)
                prefixes.append(prompt)
            targets = [question] * len(prefixes)
        else:
            targets = top_cand_sexprs_anon
            prompt_arr = [" ".join(default_instructions + default_instructions_R), f"{default_query_prefix}{question}"]
            prompt = prompt_arr_2_text(prompt_arr, '\n\n', llm.is_llama2, llm.is_beluga, llm.is_chat,
                                       default_output_prefix)
            prefixes = [prompt] * len(targets)
            pmi_prefixes = None
            if pmi:
                pmi_prefixes = [default_output_prefix] * len(targets)

        zshot_scores = get_logprob_score(target=targets, prefix=prefixes, model=llm.model, tokenizer=llm.tokenizer,
                                         len_norm=True, bsz=bsz)
        if not inverse and pmi_prefixes is not None:
            zshot_pmi_scores = get_logprob_score(target=targets, prefix=pmi_prefixes, model=llm.model,
                                                 tokenizer=llm.tokenizer, len_norm=True, bsz=bsz)
            zshot_scores = np.array(zshot_scores) - np.array(zshot_pmi_scores)

        # Compute relation repetition penalty
        max_rel_reps = []
        for cand in top_cand_sexprs_anon:
            tkns = cand.replace('(', '').replace(')', '').split()
            _rel_counts = defaultdict(int)
            for t in tkns:
                if t in schema['relations'] or t in schema['inverse_relations']:
                    _rel_counts[t] += 1
            max_rel_count = max(_rel_counts.values())
            assert max_rel_count != 0
            max_rel_reps.append(max_rel_count)

        # Compute new scores
        rerank_scores = (alpha * np.array(top_cand_scores) + (1 - alpha) * np.array(zshot_scores)) + \
                        np.log(rel_rep_factor) * (np.array(max_rel_reps) - 1)
        reranked_preds = sorted(list(zip(top_cand_sexprs, rerank_scores)), key=lambda x: -x[1])

        reranked_sexpr = reranked_preds[0][0]
        _sparql.wrapper.setTimeout(30)
        reranked_ans_set = _sparql.get_answer_set_ent_val(reranked_sexpr, retry_on_cache_none=True)
        _sparql.wrapper.setTimeout(5)
        reranked_ans = []
        for a in reranked_ans_set:
            if kg_name != 'freebase' and a['answer_type'] == 'Entity':
                reranked_ans.append(a['entity_name'])
            else:
                reranked_ans.append(a['answer_argument'])  # 'Value'
        reranked_answers.append(reranked_ans)

        # Retain old predictions
        r['orig_prediction_sexpr'] = r['prediction_sexpr']
        r['orig_prediction_sexpr_candidates'] = r['prediction_sexpr_candidates']
        r['orig_prediction_ans'] = r['prediction_ans']

        # Add new predictions
        r['prediction_sexpr'] = reranked_sexpr
        r['prediction_sexpr_candidates'] = reranked_preds
        r['prediction_ans'] = reranked_ans

    if llm.is_openai:
        res['eval_args']['rerank_openai_usage'] = llm.model['session_tokens']

    # Compute new scores
    if len(references) > 0:
        scores = {}
        metrics = llm.default_metrics
        scores['per_split'] = {}
        for metric in metrics:
            _scores, per_ex_scores = llm.custom_get_metric_scores(metric, reranked_answers, references)
            for k in metric["score_keys"]:
                scores[f"{metric['name']}.{k}"] = round(_scores[k], 4)
                res['examples'] = [dict(_ex, **{f"{metric['name']}.{k}": per_ex_scores[k][_i]}) \
                                   for _i, _ex in enumerate(res['examples'])]
                if 'split' in res['examples'][0]:
                    for _i, _ex in enumerate(res['examples']):
                        if _ex['split'] not in scores['per_split']:
                            scores['per_split'][_ex['split']] = {}
                        if f"{metric['name']}.{k}" not in scores['per_split'][_ex['split']]:
                            scores['per_split'][_ex['split']][f"{metric['name']}.{k}"] = []
                        scores['per_split'][_ex['split']][f"{metric['name']}.{k}"].append(per_ex_scores[k][_i])
        for _k, _v in scores['per_split'].items():
            for __k, __v in _v.items():
                scores['per_split'][_k][__k] = round(float(np.mean(__v)), 4)
            scores['per_split'][_k]['n_total'] = len(__v)
        res['scores'] = scores

    # Save results
    new_fpath = os.path.join(Path(res_fpath).parent, f"reranked_{Path(res_fpath).name}")
    with open(new_fpath, 'w') as fh:
        fh.write(json.dumps(res, indent=2))
    logger.info(f'Wrote re-ranked results to {new_fpath}')
