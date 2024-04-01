import json
import logging
import os
import random
import shutil
import time
import pickle
from collections import defaultdict

from src.utils.helpers import setup_logger
from src.utils.parser import graph_query_to_sexpr, is_inv_rel, get_inv_rel, graph_query_to_sparql
from src.utils.kg import get_readable_relation, get_readable_class, get_non_literals, get_nodes_by_class, \
    get_reverse_relation, get_reverse_readable_relation, prune_graph_query, legal_class, legal_relation
from src.utils.arguments import Arguments
from src.utils.sparql import SPARQLUtil, get_freebase_label, get_freebase_literals_by_cls_rel, \
    get_freebase_entid_lbl_by_cls
from src.utils.maps import literal_map

from transformers import set_seed
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Explorer:
    def __init__(self, kg_name):
        self.kg_name = kg_name
        self.ntriples = None
        self.classes = None
        self.cls_ent_2_entid = None
        self.entid_2_cls_ent = None
        self.cls_2_entid = None
        self.out_relations = None
        self.in_relations = None
        self.out_relations_cls = None
        self.in_relations_cls = None
        self.triples = None
        self.literals_by_cls_rel = None
        self.schema = None
        self.schema_dr = None  # domain-range for each relation

    def read_triples(self, kg_path, triples_fname='kb.txt', schema_fname='schema.json', sep='|',
                     entity_prefix='m', lowercase=False, write_ntriples=True, triple_pred_prefix=None):
        processed_fpath = os.path.join(kg_path, 'processed.pkl')
        if os.path.exists(processed_fpath):
            logger.info(f'Loading stored processed KG objects from {processed_fpath}')
            with open(processed_fpath, 'rb') as fh:
                processed = pickle.load(fh)
            self.schema_dr = processed['schema_dr']
            self.classes = processed['classes']
            self.schema = processed['schema']
            self.out_relations_cls = processed['out_relations_cls']
            self.in_relations_cls = processed['in_relations_cls']
            self.cls_2_entid = processed['cls_2_entid']
            self.entid_2_cls_ent = processed['entid_2_cls_ent']
            self.literals_by_cls_rel = processed['literals_by_cls_rel']
        else:
            logger.info(f'Processing KG objects')
            ntriples = []
            classes = set()
            schema_dr = {}
            cls_ent_2_entid = defaultdict(list)  # dict: k=(entity_class, entity_name), v=list(entity_ids)
            entid_2_cls_ent = {}  # dict: k=entity_id, v={'class': entity_class, 'name': entity_name}
            cls_2_entid = defaultdict(set)  # for each entity class
            out_relations = defaultdict(set)  # for each entity
            in_relations = defaultdict(set)  # for each entity
            out_relations_cls = defaultdict(set)  # for each entity class
            in_relations_cls = defaultdict(set)  # for each entity class
            triples = defaultdict(set)  # for each (entity,pred) pair; stores the set of objects
            literals_by_cls_rel = defaultdict(
                set)  # for each (entity_class, pred) pair; stores the set of possible literals

            def _add_new_entity(name, cls, cls_ent_2_entid, ent_prefix, entid_2_cls_ent, cls_2_entid, ntriples,
                                prev_sub=None):
                # MetaQA KG uses textual labels that may overalp despite being different entities;
                # Identify different entities based on their grouping in the file
                if (self.kg_name == 'metaqa' and prev_sub is not None and prev_sub != name) or \
                        (cls, name) not in cls_ent_2_entid:
                    entity_id = (f"{ent_prefix}." if ent_prefix is not None else "") + \
                                str(sum(map(len, cls_ent_2_entid.values())))
                    cls_ent_2_entid[(cls, name)].append(entity_id)
                    entid_2_cls_ent[entity_id] = {"class": cls, "name": name}
                    cls_2_entid[cls].add(entity_id)
                    ntriples.append(f'<{entity_id}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{cls}>')
                    ntriples.append(f'<{entity_id}> <http://www.w3.org/2000/01/rdf-schema#label> "{name}"')
                return cls_ent_2_entid[(cls, name)][-1]

            schema_fpath = os.path.join(kg_path, schema_fname)
            with open(schema_fpath, 'r') as fh:
                schema = json.load(fh)
            for rel in list(schema["relations"]):
                rel_obj = schema["relations"][rel]
                ntriples.append(
                    f'<{rel}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>')
                rel_name = get_readable_relation(rel.split('.')[-1])
                ntriples.append(f'<{rel}> <http://www.w3.org/2000/01/rdf-schema#label> "{rel_name}"')
                domain_cls = rel_obj['domain']
                ntriples.append(
                    f'<{rel}> <http://www.w3.org/2000/01/rdf-schema#domain> <{literal_map[domain_cls] if domain_cls.startswith("type.") else domain_cls}>')
                range_cls = rel_obj['range']
                ntriples.append(
                    f'<{rel}> <http://www.w3.org/2000/01/rdf-schema#range> <{literal_map[range_cls] if range_cls.startswith("type.") else range_cls}>')
                if "description" in rel_obj:
                    ntriples.append(
                        f'<{rel}> <http://www.w3.org/2000/01/rdf-schema#comment> "{rel_obj["description"]}"')
                if "alt_labels" in rel_obj:
                    ntriples.append(f'<{rel}> <meta.alt_labels> "{rel_obj["alt_labels"]}"')
                if "wikidata_propertyLabel" in rel_obj:
                    ntriples.append(f'<{rel}> <meta.wikidata_propertyLabel> "{rel_obj["wikidata_propertyLabel"]}"')
                if "wikidata_property" in rel_obj:
                    ntriples.append(f'<{rel}> <meta.wikidata_property> "{rel_obj["wikidata_property"]}"')
                classes.add(domain_cls)
                classes.add(range_cls)
                schema_dr[rel] = (domain_cls, range_cls)

            triples_fpath = os.path.join(kg_path, triples_fname)
            with open(triples_fpath, 'r') as fh:
                lines = fh.readlines()
                prev_sub = ''
                for line in tqdm(lines, desc="Reading KG"):
                    if lowercase:
                        line = str.lower(line)
                    line = line.strip()
                    sub, pred, obj = line.split(sep)
                    pred = (f"{triple_pred_prefix}." if triple_pred_prefix is not None else "") + pred

                    # Add new entities
                    sub_cls = schema['relations'][pred]['domain']
                    if not schema['relations'][pred]['domain'].startswith("type."):
                        # Subject is an entity
                        sub_ent = _add_new_entity(sub, sub_cls, cls_ent_2_entid, entity_prefix, entid_2_cls_ent,
                                                  cls_2_entid, ntriples, prev_sub=prev_sub)
                        sub_nt = f'<{sub_ent}>'  # nt : n-triples format
                        prev_sub = sub
                    else:
                        # Subject is a literal
                        sub_ent = None
                        sub_nt = f'"{sub}"^^<{literal_map[sub_cls]}>'
                    obj_cls = schema['relations'][pred]['range']
                    if not schema['relations'][pred]['range'].startswith("type."):  # check for literal
                        obj_ent = _add_new_entity(obj, obj_cls, cls_ent_2_entid, entity_prefix, entid_2_cls_ent,
                                                  cls_2_entid, ntriples)
                        obj_nt = f'<{obj_ent}>'
                    else:
                        obj_cls = schema['relations'][pred]['range']
                        obj_ent = None
                        obj_nt = f'"{obj}"^^<{literal_map[obj_cls]}>'

                    # Add new triples
                    ntriples.append(f'{sub_nt} <{pred}> {obj_nt}')

                    if sub_ent is not None:  # not a literal
                        out_relations[sub_ent].add(pred)
                        triples[(sub_ent, pred)].add(obj_ent if obj_ent is not None else obj)
                    else:
                        literals_by_cls_rel[(obj_cls, get_inv_rel(pred))].add(sub)
                    out_relations_cls[sub_cls].add(pred)
                    if obj_ent is not None:  # not a literal
                        in_relations[obj_ent].add(pred)
                        triples[(obj_ent, get_inv_rel(pred))].add(sub_ent if sub_ent is not None else sub)
                    else:
                        literals_by_cls_rel[(sub_cls, pred)].add(obj)
                    in_relations_cls[obj_cls].add(pred)

            if write_ntriples:
                out_ntriples_fpath = os.path.join(kg_path, 'graph.nt')
                with open(out_ntriples_fpath, 'w') as fh:
                    for nt in tqdm(ntriples, desc="Writing n-triples"):
                        fh.write(nt)
                        fh.write(" .\n")
                logger.info(f"Saved KG n-triples to {out_ntriples_fpath}")

            self.ntriples = ntriples
            self.classes = classes
            self.cls_ent_2_entid = cls_ent_2_entid
            self.entid_2_cls_ent = entid_2_cls_ent
            self.cls_2_entid = cls_2_entid
            self.out_relations = out_relations
            self.in_relations = in_relations
            self.out_relations_cls = out_relations_cls
            self.in_relations_cls = in_relations_cls
            self.triples = triples
            self.literals_by_cls_rel = literals_by_cls_rel
            self.schema = schema
            self.schema_dr = schema_dr

            with open(processed_fpath, 'wb') as fh:
                pickle.dump({
                    'schema_dr': self.schema_dr,
                    'classes': self.classes,
                    'schema': self.schema,
                    'out_relations_cls': self.out_relations_cls,
                    'in_relations_cls': self.in_relations_cls,
                    'cls_2_entid': self.cls_2_entid,
                    'entid_2_cls_ent': self.entid_2_cls_ent,
                    'literals_by_cls_rel': self.literals_by_cls_rel
                }, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Wrote processed KG objects to {processed_fpath}')

    def read_freebase(self, sparql_url, kg_path='data/graphs/freebase', sparql_cache=None):
        processed_fpath = os.path.join(kg_path, 'processed.pkl')
        if os.path.exists(processed_fpath):
            logger.info(f'Loading stored processed Freebase objects from {processed_fpath}')
            with open(processed_fpath, 'rb') as fh:
                processed = pickle.load(fh)
            self.schema_dr = processed['schema_dr']
            self.classes = processed['classes']
            self.schema = processed['schema']
            self.out_relations_cls = processed['out_relations_cls']
            self.in_relations_cls = processed['in_relations_cls']
            self.cls_2_entid = processed['cls_2_entid']
            self.entid_2_cls_ent = processed['entid_2_cls_ent']
            self.literals_by_cls_rel = processed['literals_by_cls_rel']
        else:
            logger.info(f'Processing Freebase objects')
            _sparql = SPARQLUtil(sparql_url, cache_fpath=sparql_cache, timeout=30, graph_name="freebase")
            self.schema_dr = {}
            self.classes = {}
            self.schema = None
            self.out_relations_cls = defaultdict(set)
            self.in_relations_cls = defaultdict(set)
            self.cls_2_entid = defaultdict(set)
            self.entid_2_cls_ent = {}
            self.literals_by_cls_rel = defaultdict(set)

            errors = []
            with open(os.path.join(kg_path, 'fb_roles'), 'r') as fh:
                # head predicate tail
                lines = fh.readlines()
                for line in tqdm(lines, desc='Reading relations'):
                    try:
                        head, pred, tail = line.split()
                    except:
                        errors.append(line)
                        logger.info(f'Error processing: {line}')
                        continue
                    self.schema_dr[pred] = (head, tail)
                    if head not in self.classes:
                        if not head.startswith('type.'):
                            self.classes[head] = {"description": get_freebase_label(_sparql, head).lower()}
                        else:
                            self.classes[head] = {}
                    if tail not in self.classes:
                        if not tail.startswith('type.'):
                            self.classes[tail] = {"description": get_freebase_label(_sparql, tail).lower()}
                        else:
                            self.classes[tail] = {}
                logger.info(f'{len(errors)} errors while processing fb_roles.')
            if os.path.exists(os.path.join(kg_path, 'schema.json')):
                with open(os.path.join(kg_path, 'schema.json'), 'r') as fh:
                    self.schema = json.load(fh)
            else:
                relations = {}
                inv_relations = set()
                errors = []
                with open(os.path.join(kg_path, 'fb_schema_properties'), 'r') as fh:
                    # predicate _ predicate_name _ _ inverse_predicate predicate_desc
                    while line := fh.readline():
                        try:
                            pred, _, pred_name, _, _, inv_pred, pred_desc = list(
                                map(lambda x: x.strip(), line.split('\t')))
                        except:
                            errors.append(line)
                            logger.info(f'Error processing: {line}')
                            continue
                        if pred in inv_relations:
                            relations[inv_pred]['reverse_description'] = pred_desc if pred_desc != 'null' else pred_name
                        else:
                            if pred not in self.schema_dr:
                                continue
                            relations[pred] = {
                                'description': pred_desc if pred_desc != 'null' else pred_name,
                                'domain': self.schema_dr[pred][0],
                                'range': self.schema_dr[pred][1],
                            }
                            if inv_pred != 'null':
                                relations[pred]['reverse'] = inv_pred
                                inv_relations.add(inv_pred)
                    logger.info(f'{len(errors)} errors while processing fb_schema_properties.')
                self.schema = {"classes": self.classes, "relations": relations, "inverse_relations": {}}
                for r in self.schema["relations"]:
                    if (_rev := self.schema["relations"][r].get("reverse", None)) is not None:
                        self.schema["inverse_relations"][_rev] = r
                with open(os.path.join(kg_path, 'schema.json'), 'w') as fh:
                    fh.write(json.dumps(self.schema, indent=2))

            for rel, rel_obj in self.schema['relations'].items():
                self.out_relations_cls[rel_obj['domain']].add(rel)
                self.in_relations_cls[rel_obj['range']].add(rel)

            for cls in tqdm(self.classes, desc="Fetching entities per class"):
                if not cls.startswith('type.'):
                    # ent_ids = get_freebase_entid_lbl_by_cls(_sparql, cls, only_entid=True)
                    # for ent_id in ent_ids:
                    #     ent_lbl = get_freebase_label(_sparql, ent_id)
                    #     if ent_lbl is None or len(ent_lbl) == 0:
                    #         ent_lbl = get_readable_class(cls, schema=self.schema)
                    #     self.cls_2_entid[cls].add(ent_id)
                    #     if ent_id not in self.entid_2_cls_ent:
                    #         self.entid_2_cls_ent[ent_id] = {'class': set(), 'name': ent_lbl}
                    #     self.entid_2_cls_ent[ent_id]['class'].add(cls)
                    ent_lbl_set = get_freebase_entid_lbl_by_cls(_sparql, cls)
                    for ent_lbl in ent_lbl_set:
                        entid, entlbl = ent_lbl
                        self.cls_2_entid[cls].add(entid)
                        if entid not in self.entid_2_cls_ent:
                            self.entid_2_cls_ent[entid] = {'class': set(), 'name': entlbl}
                        self.entid_2_cls_ent[entid]['class'].add(cls)

            for rel in tqdm(self.schema_dr, desc="Fetching literals per class+relation"):
                domain, range = self.schema_dr[rel]
                if range in literal_map:
                    literals = get_freebase_literals_by_cls_rel(_sparql, domain, rel)
                    for l in literals:
                        self.literals_by_cls_rel[(domain, rel)].add(l['answer_argument'])

            with open(processed_fpath, 'wb') as fh:
                pickle.dump({
                    'schema_dr': self.schema_dr,
                    'classes': self.classes,
                    'schema': self.schema,
                    'out_relations_cls': self.out_relations_cls,
                    'in_relations_cls': self.in_relations_cls,
                    'cls_2_entid': self.cls_2_entid,
                    'entid_2_cls_ent': self.entid_2_cls_ent,
                    'literals_by_cls_rel': self.literals_by_cls_rel
                }, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Wrote processed Freebase objects to {processed_fpath}')

    def explore(self, n_walks, edge_lengths: list, max_retries_per_iter: int = 5, always_ground_classes=False,
                always_ground_literals: bool = False, sexpr_type_constraint: bool = True, n_per_pattern=1,
                use_functions: [list, None] = None, max_skip: [int, None] = None, verbose: bool = False,
                max_retries=None, sparql_url=None, filter_empty=False, prune_redundant=False, save_answers=True,
                sparql_cache=None, out_dir=None, additional_save_dir=None, run_id: str = str(int(time.time()))):
        edge_dup, walk_dup, empty = 0, 0, 0
        iter = 0
        walked_sexpr = defaultdict(int)
        walked_sexpr_all = set()
        walks = []
        fn_counts = defaultdict(int)
        edge_counts = defaultdict(int)
        node_counts = defaultdict(int)
        retained_iter = []
        _sparql = SPARQLUtil(sparql_url, cache_fpath=sparql_cache, timeout=20,
                             graph_name=self.kg_name) if sparql_url is not None else None

        if args.prev_exploration_stats_fpath is not None:
            for prev_exploration_stats_fpath in args.prev_exploration_stats_fpath:
                with open(os.path.join(prev_exploration_stats_fpath, 'stats.json'), 'r') as fh:
                    prev_stats = json.load(fh)
                for k, v in prev_stats['n_programs_per_pattern'].items():
                    walked_sexpr[k] += v
                with open(os.path.join(prev_exploration_stats_fpath, 'results.json'), 'r') as fh:
                    prev_res = json.load(fh)
                for r in prev_res:
                    walked_sexpr_all.add(r['s_expression_machine'])
            logger.info(
                f'Loaded previous exploration results. Found {len([k for k in walked_sexpr if walked_sexpr[k] > 0])} ' +
                f'patterns and {len(walked_sexpr_all)} programs.')

        # Create output directory
        res_dir = ["walks", self.kg_name, '-'.join(map(str, edge_lengths)), ]  # str(len(walks))
        if use_functions is not None:
            res_dir += ['-'.join(use_functions)]
        res_dir += [run_id]
        res_dir = f"{'_'.join(res_dir)}"
        if out_dir is not None:
            res_dir_fpath = os.path.join(out_dir, res_dir)
            os.makedirs(res_dir_fpath, exist_ok=True)

        pbar = tqdm(total=n_walks, desc="Exploring")
        start_time = time.time()
        retries = 0
        retries_n_walks = len(walks)
        retries_per_program = []
        reached_retry_limit = False
        while n_walks is None or (
                n_walks is not None and len(walks) < n_walks and (max_skip is None or iter < n_walks + max_skip)):
            # Implement a retry loop with a count-out
            if len(walks) == retries_n_walks:
                retries += 1
                if max_retries is not None and retries > max_retries:
                    reached_retry_limit = True
                    break
            else:
                retries_per_program.append(retries)
                retries = 0
                retries_n_walks = len(walks)

            if verbose:
                logger.info(f"iter {iter}:")
            iter += 1
            sampled_n_edges = random.choice(edge_lengths)
            res = self.generate_graph_query(n_edges=sampled_n_edges, max_retries=max_retries_per_iter,
                                            always_ground_literals=always_ground_literals,
                                            always_ground_classes=always_ground_classes,
                                            use_functions=use_functions, verbose=verbose)
            if res is None:
                edge_dup += 1
                if verbose:
                    logger.info(f"Skipping (count={edge_dup})")
                continue
            gq, gq_fn, gq_n_groundings = res
            sexpr_anon_noid = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                                   readable=True, readable_type='anon_noid')
            sexpr_machine = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint)
            # Note: using sexpr_anon_noid for duplicate check since we don't want the pattern to repeat
            # (i.e. we don't want the same pattern with different entities to be included again)
            if walked_sexpr[sexpr_anon_noid] >= n_per_pattern or sexpr_machine in walked_sexpr_all:
                walk_dup += 1
                if verbose:
                    logger.info(f"Already walked. Skipping (count={walk_dup})")
                continue

            ans_list = None
            if filter_empty:
                ans_list, is_empty = _sparql.get_answer_set(gq, zero_is_empty=True, verbose=verbose)
                if is_empty:
                    empty += 1
                    if verbose:
                        logger.info(f"Empty answer set. Skipping (count={empty})")
                    continue
            # walked_sexpr[sexpr_anon_noid] += 1  # Add non-pruned sexpr to seen list

            if prune_redundant:
                gq, ans_list = prune_graph_query(_sparql, gq, orig_ans=ans_list, final_check=True, verbose=verbose)
                # TODO: Change final_check to false once confident with code

            ans_objs = []
            if save_answers:
                ans_objs = _sparql.get_answer_set_ent_val(gq, verbose=verbose)

            sexpr_anon_noid = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                                   readable=True, readable_type='anon_noid')
            sexpr_machine = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint)

            if walked_sexpr[sexpr_anon_noid] >= n_per_pattern or sexpr_machine in walked_sexpr_all:
                walk_dup += 1
                if verbose:
                    logger.info(f"Already walked. Skipping (count={walk_dup})")
                continue

            walked_sexpr[sexpr_anon_noid] += 1  # Add pruned sexpr to seen list
            walked_sexpr_all.add(sexpr_machine)

            sexpr_anon = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                              readable=True, readable_type='anon')
            sexpr_anon_rev = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                                  readable=True, readable_type='anon', use_reverse_relations=True)
            sexpr_machine_rev = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint, readable=False,
                                                     use_reverse_relations=True)
            sexpr_label = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                               readable=True, readable_type='label')
            sexpr_label_rev = graph_query_to_sexpr(gq, type_constraint=sexpr_type_constraint,
                                                   readable=True, readable_type='label', use_reverse_relations=True)
            sparql_query = _sparql.add_headers_to_query(graph_query_to_sparql(gq))

            walks.append({
                "qid": len(walks),
                "function": gq_fn,
                "num_node": len(gq["nodes"]),
                "num_edge": len(gq["edges"]),
                "graph_query": gq,
                "s_expression_anon": sexpr_anon,
                "s_expression_anon-rev": sexpr_anon_rev,
                "s_expression_label": sexpr_label,
                "s_expression_label-rev": sexpr_label_rev,
                "s_expression_machine": sexpr_machine,
                "s_expression_machine-rev": sexpr_machine_rev,
                "sparql_query": sparql_query
            })
            retained_iter.append(iter)
            if save_answers:
                walks[-1].update({"answer": ans_objs})

            fn_counts[gq_fn] += 1
            edge_counts[len(gq["edges"])] += 1
            node_counts[len(gq["nodes"])] += 1
            if verbose:
                logger.info(walks[-1])

            if args.save_interval > 0 and len(walks) % args.save_interval == 0:
                if out_dir is not None:
                    out_fname = "results.json"
                    out_fpath = os.path.join(res_dir_fpath, out_fname)
                    with open(out_fpath, 'w') as fh:
                        fh.write(json.dumps(walks, indent=2))
                    logger.info(f"Saved walks to {out_fpath}")
            pbar.update(1)
        pbar.close()

        if max_retries is not None and reached_retry_limit:
            logger.info(f"Stopping exploration: reached retry limit ({max_retries})")
        elif n_walks is not None and max_skip is not None and len(walks) < n_walks:
            logger.info(f"Stopping exploration: reached maximum attempts ({n_walks + max_skip})")

        end_time = time.time()

        seen_patterns = {k: v for k, v in dict(walked_sexpr).items() if v > 0}

        stats = {
            'n_iters': iter,
            'n_programs': len(walks),
            'time_taken': end_time - start_time,
            'retries': {
                'avg': sum(retries_per_program) * 1. / len(retries_per_program),
                'max': max(retries_per_program)
            },
            'patterns': {
                'total': len(seen_patterns),
                'min_programs': min(seen_patterns.values()),
                'max_programs': max(seen_patterns.values()),
                'avg_programs': sum(seen_patterns.values()) * 1. / len(seen_patterns)
            },
            'n_skipped_seen_node-rel_pair': edge_dup,
            'n_skipped_seen_pattern': walk_dup,
            'n_skipped_empty_ans': empty,
            'n_programs_per_fn': dict(fn_counts),
            'n_programs_per_node_count': dict(node_counts),
            'n_programs_per_edge_count': dict(edge_counts),
            'n_programs_per_pattern': seen_patterns,
            'retain_iter_count': retained_iter
        }

        if out_dir is not None:
            out_fname = "results.json"
            out_fpath = os.path.join(res_dir_fpath, out_fname)
            with open(out_fpath, 'w') as fh:
                fh.write(json.dumps(walks, indent=2))
            logger.info(f"Saved walks to {out_fpath}")
            stats_fname = "stats.json"
            stats_fpath = os.path.join(res_dir_fpath, stats_fname)
            with open(stats_fpath, 'w') as fh:
                fh.write(json.dumps(stats, indent=2))
            logger.info(f"Saved exploration statistics to {stats_fpath}")
        if additional_save_dir is not None:
            res_dir_fpath = os.path.join(additional_save_dir)
            os.makedirs(res_dir_fpath, exist_ok=True)
            out_fname = "walks.json"
            out_fpath = os.path.join(res_dir_fpath, out_fname)
            with open(out_fpath, 'w') as fh:
                fh.write(json.dumps(walks, indent=2))
            logger.info(f"Saved walks to {out_fpath}")

        return walks

    def generate_graph_query(self, n_edges: int, max_retries: int = 3, always_ground_literals=True,
                             always_ground_classes=False, verbose=False, use_functions=None, ground_attempts_max=5):
        graph = {"nodes": [], "edges": []}
        class_2_nid = defaultdict(set)
        sampled_node_rel = set()
        nodes_2_ground = set()
        ungrounded_terminal_node = set()
        n_nodes_grounded = 0
        # Sample an initial question node from all classes

        q_class = random.choice(list(self.classes))
        while not legal_class(q_class):
            q_class = random.choice(list(self.classes))

        q_node = {
            "nid": 0,
            "node_type": "class",
            "id": q_class,
            "class": q_class,
            "readable_name": get_readable_class(q_class, schema=self.schema),
            "question_node": 1,
            "function": "none"
        }
        graph["nodes"].append(q_node)
        class_2_nid[q_node["class"]].add(q_node["nid"])

        n_attempts = 0
        n_legal_relation_attempts = 0
        while len(graph["edges"]) < n_edges and n_attempts < max_retries and n_legal_relation_attempts < max_retries:
            next_edge = {}
            # Sample a node from the set of ungrounded and non-literal nodes
            node_2_expand = random.choice(get_non_literals(graph["nodes"], except_nid=nodes_2_ground))
            # Sample adjacent node relation
            next_rel = random.choice(list(self.out_relations_cls[node_2_expand["class"]]) +
                                     list(map(lambda x: f"{x}#R", self.in_relations_cls[node_2_expand["class"]])))
            if not legal_relation(next_rel):
                n_legal_relation_attempts += 1
                if verbose:
                    logger.info(f"Illegal relation sampled. Retrying ({n_legal_relation_attempts})")
                continue
            n_legal_relation_attempts = 0

            # Re-try if the sampled (node, relation) pair has been sampled before
            if (node_2_expand["nid"], next_rel) in sampled_node_rel:
                n_attempts += 1
                if verbose:
                    logger.info(f"Already seen. Retrying ({n_attempts})")
                continue
            sampled_node_rel.add((node_2_expand["nid"], next_rel))
            sampled_node_rel.add((node_2_expand["nid"], get_inv_rel(next_rel)))
            n_attempts = 0

            ungrounded_terminal_node.discard(node_2_expand["nid"])

            if is_inv_rel(next_rel):
                rel_domain, rel_range = self.schema['relations'][get_inv_rel(next_rel)]['range'], \
                                        self.schema['relations'][get_inv_rel(next_rel)]['domain']
            else:
                rel_domain, rel_range = self.schema['relations'][next_rel]['domain'], \
                                        self.schema['relations'][next_rel]['range']

            # Select next node from the sampled relation
            # We randomly either add a new node with the range class or sample an existing compatible node
            cand_nodes_from_existing = get_nodes_by_class(graph["nodes"], cls=rel_range,
                                                          except_nid=[node_2_expand["nid"]])
            new_node = {
                "nid": len(graph["nodes"]),
                "node_type": "class",
                "readable_name": get_readable_class(rel_range, schema=self.schema),
                "question_node": 0,
                "function": "none",
                "id": rel_range,
                "class": rel_range
            }
            if len(cand_nodes_from_existing) == 0:
                add_new_node = True
                next_node = new_node
            else:
                if add_new_node := random.choice([True, False]):
                    next_node = new_node
                else:
                    next_node = random.choice(cand_nodes_from_existing)

            if is_inv_rel(next_rel):
                next_edge.update({
                    "start": next_node["nid"],
                    "end": node_2_expand["nid"],
                    "relation": get_inv_rel(next_rel),  # this removes "#R" from the name if called on an inv rel
                    "readable_name": get_readable_relation(get_inv_rel(next_rel), schema=self.schema),
                    "reverse_relation": get_reverse_relation(get_inv_rel(next_rel), schema=self.schema),
                    "reverse_readable_name": get_reverse_readable_relation(get_inv_rel(next_rel), schema=self.schema)
                })
            else:
                next_edge.update({
                    "start": node_2_expand["nid"],
                    "end": next_node["nid"],
                    "relation": next_rel,
                    "readable_name": get_readable_relation(next_rel, schema=self.schema),
                    "reverse_relation": get_reverse_relation(next_rel, schema=self.schema),
                    "reverse_readable_name": get_reverse_readable_relation(next_rel, schema=self.schema)
                })

            if add_new_node:
                if next_node["class"] in literal_map:  # literal class
                    if always_ground_literals:
                        next_node["node_type"] = "literal"
                        # Ground node immediately
                        grounding_cands = self.literals_by_cls_rel[(node_2_expand["class"], next_edge["relation"])]
                        if len(grounding_cands) == 0:
                            if verbose:
                                logger.info(
                                    f"No grounding values found for class `{node_2_expand['class']}` + relation `{next_edge['relation']}`. Skipping.")
                            return None
                        next_node["readable_name"] = random.choice(list(grounding_cands))
                        next_node["id"] = f'"{next_node["readable_name"]}"^^{literal_map[next_node["class"]]}'
                        n_nodes_grounded += 1
                    else:
                        next_node['grounding_helper'] = {
                            'cls': node_2_expand["class"],
                            'rel': next_edge["relation"]
                        }
                        if random.choice([True, False]):
                            nodes_2_ground.add(next_node["nid"])
                else:  # non-literal class
                    if always_ground_classes or random.choice([True, False]):
                        if len(self.cls_2_entid[next_node["class"]]) > 0:
                            nodes_2_ground.add(next_node["nid"])

                graph["nodes"].append(next_node)
                class_2_nid[next_node["class"]].add(next_node["nid"])
                if next_node["node_type"] == "class" and next_node["nid"] not in nodes_2_ground:
                    ungrounded_terminal_node.add(next_node["nid"])

            graph["edges"].append(next_edge)
            sampled_node_rel.add((next_node["nid"], next_rel))
            sampled_node_rel.add((next_node["nid"], get_inv_rel(next_rel)))

        if len(graph["edges"]) < n_edges:
            if verbose:
                logger.info('Reached maximum attempts in trying to sample new node-edge pairs')
            return None

        # Mark ungrounded terminal nodes for grounding
        nodes_2_ground.update(ungrounded_terminal_node)

        # Add a function to the query
        function_cands = ['none']
        if not q_class.startswith('type.'):  # Only allow COUNT if the question node is not a literal class
            function_cands.append('count')
        # if q_node['class'] in ['type.integer', 'type.float', 'type.dateTime']:
        #     function_cands.extend(['MAX', 'MIN'])  # TODO: Think about adding these
        numerical_nodes_not_q = \
            class_2_nid['type.integer'].union(class_2_nid['type.float']).union(class_2_nid['type.datetime']) - \
            {q_node['nid']}
        if len(numerical_nodes_not_q) > 0:
            function_cands.extend(['argmax', 'argmin', '>', '<', '>=', '<='])
        else:
            function_cands += ['none'] * 2  # reduce likelihood of 'count' being sampled
        # Sample a function (including none) and assign to a compatible node
        sampled_fn = random.choice(function_cands)

        if use_functions is not None:
            if sampled_fn not in use_functions:
                return None

        if sampled_fn == 'count':
            assert graph['nodes'][0]['nid'] == q_node['nid']
            graph['nodes'][0]['function'] = 'count'
        elif sampled_fn in ['argmin', 'argmax']:
            sampled_node = random.choice(list(numerical_nodes_not_q))
            assert graph['nodes'][sampled_node]['nid'] == sampled_node
            graph['nodes'][sampled_node].update({
                "node_type": "literal",
                "id": '"0"^^http://www.w3.org/2001/XMLSchema#int',
                "readable_name": "0",
                "function": sampled_fn
            })
            nodes_2_ground -= {sampled_node}
            n_nodes_grounded += 1
        elif sampled_fn in ['>', '<', '>=', '<=']:
            sampled_node_id = random.choice(list(numerical_nodes_not_q))
            assert graph['nodes'][sampled_node_id]['nid'] == sampled_node_id
            graph['nodes'][sampled_node_id]['function'] = sampled_fn
            if graph['nodes'][sampled_node_id]['node_type'] == "class":
                nodes_2_ground.add(sampled_node_id)

        # Try to ground something if nothing is grounded so far
        if n_nodes_grounded == 0:
            ground_attempts = 0
            while len(nodes_2_ground) == 0 and ground_attempts < ground_attempts_max:
                sampled_node = random.choice(graph["nodes"][1:])
                if sampled_node["node_type"] == "class":
                    nodes_2_ground.add(sampled_node["nid"])
                ground_attempts += 1

        # Ground nodes
        grounded_ents = set()
        for nid in nodes_2_ground:
            node = graph["nodes"][nid]
            node_cls = node["class"]
            if node_cls in literal_map:  # literal class
                grounding_cands = list(
                    self.literals_by_cls_rel[(node["grounding_helper"]["cls"], node["grounding_helper"]["rel"])])
                if len(grounding_cands) == 0:
                    raise ValueError(
                        f"No grounding values found for `{node['grounding_helper']['cls']}->{node['grounding_helper']['rel']}`")
                grounded_lit = random.choice(grounding_cands)
                node.update({
                    "node_type": "literal",
                    "id": f'"{grounded_lit}"^^{literal_map[node_cls]}',
                    "readable_name": grounded_lit
                })
            else:  # entity class
                if len(self.cls_2_entid[node_cls]) == 0:
                    if verbose:
                        logger.info(f'No entities with labels found for class {node_cls}. Skipping.')
                    return None
                grounding_cands = list(
                    self.cls_2_entid[node_cls] - grounded_ents)  # ensuring that same entity is not grounded twice
                if len(grounding_cands) == 0:
                    if verbose:
                        logger.info(f"No new grounding values found for class `{node_cls}`. Skipping")
                    return None
                grounded_ent = random.choice(grounding_cands)
                grounded_ents.add(grounded_ent)
                node.update({
                    "node_type": "entity",
                    "id": grounded_ent,
                    "readable_name": self.entid_2_cls_ent[grounded_ent]['name']
                })
            n_nodes_grounded += 1

        return graph, sampled_fn, n_nodes_grounded


if __name__ == '__main__':
    # Setup
    cli_args = Arguments(groups="explorer")
    global args
    args = cli_args.parse_args()
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else str(args.run_id)
    setup_logger(RUN_ID)
    logger.info("Script arguments:")
    logger.info(args.__dict__)
    set_seed(args.seed)
    # Create output dir
    out_dir = os.path.join(args.out_dir, "exploration")
    if args.clean_out_dir:
        for f in os.listdir(out_dir):
            fpath = os.path.join(out_dir, f)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
    os.makedirs(out_dir, exist_ok=True)

    # Init explorer
    explorer = Explorer(kg_name=args.kg_name)

    if args.kg_name == 'freebase':
        explorer.read_freebase(kg_path=args.kg_path, sparql_url=args.sparql_url, sparql_cache=args.sparql_cache)
    else:
        # Load KG from a triples file
        explorer.read_triples(kg_path=args.kg_path, triples_fname=args.triples_fname,
                              write_ntriples=args.kg_write_ntriples, lowercase=args.kg_lowercase,
                              triple_pred_prefix=args.kg_prefix, sep=args.kg_sep)

    # Run exploration
    if args.kg_explore:
        walks = explorer.explore(n_walks=args.kg_n_walks, edge_lengths=list(range(1, args.kg_n_walk_edges + 1)),
                                 use_functions=args.kg_walk_functions, max_skip=args.kg_walk_max_skip,
                                 sparql_url=args.sparql_url, filter_empty=args.filter_empty_walks,
                                 prune_redundant=args.prune_redundant, out_dir=out_dir, run_id=RUN_ID,
                                 verbose=args.verbose, additional_save_dir=args.additional_save_dir,
                                 always_ground_literals=args.always_ground_literals, max_retries=args.max_retries,
                                 always_ground_classes=args.always_ground_classes, sparql_cache=args.sparql_cache,
                                 n_per_pattern=args.n_per_pattern)

    if args.debug:
        breakpoint()
