import copy
import json
import logging
import socket
import atexit

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
import urllib

from src.utils.helpers import split_underscore_period, deep_get
from src.utils.maps import literal_map_inv, literal_map, literal_map_variants
from src.utils.parser import graph_query_to_sparql, sexpr_to_sparql, sexpr_to_struct, graph_query_to_sexpr, \
    parse_bottom_up, bottom_up_to_sexpr, get_where_clause_content

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_freebase_label(_sparql, entity_cls: str) -> str:
    query = f"""
        SELECT (?x0 AS ?label) WHERE {{
        SELECT DISTINCT ?x0  WHERE {{
        :{entity_cls} rdfs:label ?x0 .
        FILTER (lang(?x0) = "" OR langMatches(lang(?x0), "en") )
        }}
        }}"""
    rtn = _sparql.execute_query(query)
    assert type(rtn) is list
    return rtn[0] if len(rtn) >= 1 else rtn


def get_freebase_literals_by_cls_rel(_sparql, cls, rel):
    query = f"""SELECT DISTINCT ?x0 WHERE {{?s :{rel} ?x0 . ?s a :{cls} }}"""
    return _sparql.get_answer_set_ent_val(query, is_sparql=True)


def get_freebase_entid_lbl_by_cls(_sparql, cls, only_entid=False):
    query = f"""SELECT DISTINCT ?x0 {'?label' if not only_entid else ' '} WHERE 
    {{?x0 a :{cls} . {'?x0 rdfs:label ?label .' if not only_entid else ' '}
    {'FILTER(lang(?label) = "" OR langMatches(lang(?label), "en") )' if not only_entid else ' '}
    }}"""
    ans_set = _sparql.get_answer_set(query, is_sparql=True, return_dict=False)[0]
    for i, ans in enumerate(ans_set):
        if not only_entid:
            ans[0] = ans[0].split('/')[-1]
            assert ans[0].startswith('m.') or ans[0].startswith('g.')
        else:
            ans = ans.split('/')[-1]
            assert ans.startswith('m.') or ans.startswith('g.')
            ans_set[i] = ans
    return ans_set


class SPARQLUtil:
    def __init__(self, url, graph_name='freebase', timeout=60,
                 cache_fpath=None, retry_on_cache_none_override=None,
                 **kwargs):
        self.wrapper = SPARQLWrapper(url, **kwargs)  # "http://localhost:3001/sparql"
        self.wrapper.setReturnFormat(JSON)
        self.wrapper.setTimeout(timeout)
        self.cache = {}  # keys: sparql queries, values: responses
        self.cache_fpath = cache_fpath
        self.load_cache()
        self.retry_on_cache_none_override = retry_on_cache_none_override
        self.graph_name = graph_name
        self.prefix = 'http://rdf.freebase.com/ns/' if self.graph_name == 'freebase' else f'http://{self.graph_name}.com/'
        atexit.register(self.save_cache)

    def load_cache(self):
        if self.cache_fpath is not None:
            try:
                with open(self.cache_fpath, 'r') as fh:
                    self.cache = json.load(fh)
                logger.info(f'Loaded SPARQL cache from {self.cache_fpath}')
            except:
                logger.info(f'SPARQL cache not found at {self.cache_fpath}')

    def save_cache(self):
        if self.cache_fpath is not None and len(self.cache) != 0:
            with open(self.cache_fpath, 'w') as fh:
                json.dump(self.cache, fh)
            logger.info(f'Saved SPARQL cache to {self.cache_fpath}')

    def reset_cache(self, q=None):
        if q is not None and q in self.cache:
            del self.cache[q]
        else:
            self.cache = {}

    def process_results(self, rtn):
        if rtn is None or len(rtn) == 0:
            return rtn
        if type(rtn) is str:
            return rtn.replace(self.prefix, '')
        res = copy.deepcopy(rtn)
        for row_i in range(len(res)):
            if type(res[row_i]) is str:
                res[row_i] = res[row_i].replace(self.prefix, '')
            else:
                assert type(res[row_i]) is list
                for col_i in range(len(res[row_i])):
                    res[row_i][col_i] = res[row_i][col_i].replace(self.prefix, '')
        return res

    def add_headers_to_query(self, query):
        _q = query
        graph_uri = f'define input:default-graph-uri <http://{self.graph_name}.com>'
        graph_prefix = f'PREFIX : <{self.prefix}>'
        standard_prefixes = 'PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ' + \
                            'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>'
        headers_to_add = [graph_prefix, standard_prefixes, graph_uri]
        for header in headers_to_add:
            if header not in _q:
                _q = f'{header} {_q}'
        return _q

    def execute_query(self, query, catch_timeout=False, return_dict=False, retry_on_cache_none=False,
                      return_none_on_timeout=False):
        if self.retry_on_cache_none_override is not None:
            retry_on_cache_none = self.retry_on_cache_none_override
        rtn = []
        results = None
        _q = ' '.join(query.split())  # remove new-lines

        # Set graph-specific headers
        _q = self.add_headers_to_query(_q)

        if self.cache is not None and _q in self.cache:
            results = self.cache[_q]
            if results is None and not retry_on_cache_none:
                if return_none_on_timeout:  # None saved in cache means it was a timeout in the past
                    return results
                results = []

        if results is None:
            self.wrapper.setQuery(_q)
            if not catch_timeout:
                results = self.wrapper.query().convert()
            else:
                try:
                    results = self.wrapper.query().convert()
                except (
                        socket.error, socket.timeout, urllib.error.HTTPError,
                        SPARQLExceptions.EndPointInternalError) as e:
                    logger.info(f"Caught: {e}\nwhile running {query}")
                    if any([reset_error in str(e) for reset_error in ['[Errno 104]', '[Errno 111]']]):
                        logger.info(f"SERVER CRASHED! RESTART SERVER and press 'c' TO CONTINUE EXECUTION.")
                        breakpoint()
                    if self.cache is not None:
                        self.cache[_q] = None
                    return None if return_none_on_timeout else rtn
            if self.cache is not None:
                self.cache[_q] = results

        if return_dict:
            return deep_get(results, 'results', 'bindings', default={})
        for result in deep_get(results, 'results', 'bindings', default=[]):
            if len(result) == 1:  # single column result
                k = next(iter(result))
                rtn.append(result[k]['value'])
            else:
                row = []
                for k in result:
                    row.append(result[k]['value'])
                if len(row) > 0:
                    rtn.append(row)
        return rtn

    def get_answer_set(self, query: [dict, str], zero_is_empty=False, catch_timeout=True,
                       return_dict=False, is_sparql=False, retry_on_cache_none=False, verbose=False):
        # query: graph-query or s-expression
        _q = query if is_sparql else (
            graph_query_to_sparql(query) if type(query) is dict else sexpr_to_sparql(
                query))
        res = self.execute_query(_q, catch_timeout=catch_timeout, return_dict=return_dict,
                                 retry_on_cache_none=retry_on_cache_none)
        if verbose:
            logger.info(res)
        is_empty = len(res) == 0
        if not is_empty and zero_is_empty:
            if len(res) == 1:
                if not return_dict:
                    is_empty = res[0] == '0'
                elif len(res[0]) == 1:  # check if dict has only 1 key
                    k = next(iter(res[0]))
                    if res[0][k]['type'] == 'typed-literal' and '#int' in res[0][k]['datatype']:
                        is_empty = res[0][k]['value'] == '0'
        return res, is_empty

    def get_answer_set_ent_val(self, query, is_sparql=False, retry_on_cache_none=False, verbose=False):
        # query: graph-query or s-expression
        rtn = []
        ans_list, _ = self.get_answer_set(query, return_dict=True, is_sparql=is_sparql,
                                          retry_on_cache_none=retry_on_cache_none, verbose=verbose)
        for a in ans_list:
            assert len(a.keys()) == 1
            k = next(iter(a))
            v = a[k]
            v['value'] = v['value'].replace(self.prefix, '')
            if v['type'] == 'uri' and (v['value'].startswith('m.') or v['value'].startswith('g.')):
                label = self.get_label_by_entid(v['value'], use_id_on_empty=True)
                a_obj = {
                    "answer_type": "Entity" if v['value'].startswith('m.') or v['value'].startswith('g.') else "Class",
                    "answer_argument": v['value'],
                    "entity_name": label
                }
            elif v['type'] == 'literal':
                # string
                a_obj = {
                    "answer_type": "Value",
                    "answer_argument": v['value']
                }
            elif v['type'] == 'typed-literal':
                a_obj = {
                    "answer_type": "Value",
                    "answer_argument": v['value'][:4] if '#date' in v['datatype'] else v['value'],
                    "datatype": f"type.{v['datatype'].split('#')[-1].lower()}",
                    "raw_datatype": v['datatype']
                }
                if any(typ in a_obj["raw_datatype"].lower() for typ in ['date', 'year']):
                    a_obj["datatype"] = "type.datetime"
                    typ = a_obj["raw_datatype"].split('#')[-1].lower()
                    if typ == 'datetime':
                        a_obj["answer_argument"] = v['value'][:4]  # only keep year; used by MetaQA
                    else:  # date, gYear, gYearMonth
                        a_obj["answer_argument"] = v['value'].replace(f'^^<{a_obj["raw_datatype"]}>', '').replace('"',
                                                                                                                  '').replace(
                            '-08:00', '')
            else:
                # treat as string
                a_obj = {
                    "answer_type": "Value",
                    "answer_argument": v['value']
                }
            rtn.append(a_obj)
        return rtn

    def get_entid_by_cls(self, cls):
        _q = f"""SELECT DISTINCT ?x WHERE {{
            ?x a :{cls} .
        }}"""
        entid = self.execute_query(_q)  # returns a list; could be more than one entry
        entid = self.process_results(entid)
        return entid

    def get_entid_by_cls_label(self, cls, label):
        _q = f"""SELECT DISTINCT ?x WHERE {{
            ?x rdfs:label ?label .
            ?x a :{cls} .
            FILTER (?label="{label}" or ?label="{label}"@en)
        }}"""
        entid = self.execute_query(_q)  # returns a list; could be more than one entry
        entid = self.process_results(entid)
        return entid

    def get_label_by_entid(self, entid, use_id_on_empty=False):
        _q = f"""SELECT DISTINCT ?x  WHERE {{
            :{entid} rdfs:label ?x .
            FILTER (lang(?x) = '' OR langMatches(lang(?x), 'en'))
        }}"""
        label = self.execute_query(_q)
        assert type(label) is list
        if len(label) == 0:
            # check if there's a label in some other language
            _q = f"""SELECT DISTINCT ?x  WHERE {{
                :{entid} rdfs:label ?x
            }}"""
            label = self.execute_query(_q)
            assert type(label) is list
            if len(label) != 0:
                return label[0]
            if use_id_on_empty:
                return split_underscore_period(entid)
            else:
                raise ValueError(f"SPARQLUtil: No label found for {entid}")
        return label[0]

    def get_cls_by_entid(self, entid):
        _q = f"""SELECT DISTINCT ?x WHERE {{
            :{entid} a ?x .
        }}"""
        cls = self.execute_query(_q)
        cls = self.process_results(cls)
        return cls

    def get_cls_by_query(self, query: [str, dict], use_classes_only=False, catch_timeout=True,
                         filter_entities=True):
        # query: graph-query or s-expression
        entity_or_class = None
        if type(query) is str:
            struct = sexpr_to_struct(query)
            if type(struct) is str:
                entity_or_class = struct
            elif type(struct) is list and len(struct) == 1:
                entity_or_class = struct[0]
        if entity_or_class is not None:
            if not entity_or_class.startswith('m.') and not entity_or_class.startswith('g.'):
                if self.get_answer_set(f'(COUNT (AND {entity_or_class} (*)))', zero_is_empty=True)[1]:
                    return []  # class doesn't exist
                return [entity_or_class]
            _cls = self.get_cls_by_entid(entity_or_class)
            return _cls if type(_cls) is list else [_cls]

        # query is an s-expression or graph-query
        sexpr = query if type(query) is str else graph_query_to_sexpr(query)
        if use_classes_only:
            bottom_up_parse = parse_bottom_up(sexpr)
            for subexp in bottom_up_parse:
                for ti, token in enumerate(subexp):
                    if ti == 0 or type(token) is list:
                        continue
                    if token.startswith('m.') or token.startswith('g.'):
                        subexp[ti] = self.get_cls_by_entid(token)  # TODO: handle multiple classes
            sexpr = bottom_up_to_sexpr(bottom_up_parse)
        _q_sparql = sexpr_to_sparql(sexpr, filter_entities=filter_entities)
        _q = f"""SELECT DISTINCT ?class WHERE {{
            ?x a ?class .
            {get_where_clause_content(_q_sparql)}
        }}"""
        rtn = self.execute_query(_q, catch_timeout=catch_timeout, return_none_on_timeout=True)  # returns a list of rows
        rtn = self.process_results(rtn)
        if rtn is not None and len(rtn) == 0:
            # check for literals
            _q = f"""SELECT DISTINCT datatype(?x) WHERE {{
                {get_where_clause_content(_q_sparql)}
            }}"""
            rtn = self.execute_query(_q, catch_timeout=catch_timeout)
            rtn = self.process_results(rtn)
            try:
                rtn = list(map(lambda x: literal_map_inv[x], rtn))
            except KeyError:
                logger.info(f'KeyError in literal_map_inv with:\nrtn={rtn}\nfor query={query}')
                return []
        return [] if rtn is None else rtn

    def get_cls_by_label(self, label: str):
        return self.get_cls_by_query(f'(JOIN http://www.w3.org/2000/01/rdf-schema#label "{label}")')

    def get_relations(self, query: [str, dict], direction, use_classes_only=False, catch_timeout=True,
                      filter_entities=True):
        # query : str -> entity id, class name, sexpr; dict -> graph query
        # TODO: Optimize code when use_classes_only==True by using bottom up parsing and replacing subexprs with classes
        assert direction in ['in', 'out']
        entity_or_class = None
        if type(query) is str:
            struct = sexpr_to_struct(query)
            if type(struct) is str:
                entity_or_class = struct
            elif type(struct) is list and len(struct) == 1:
                entity_or_class = struct[0]
        if entity_or_class is not None:
            is_entity = entity_or_class.startswith('m.') or entity_or_class.startswith('g.')
            if is_entity and use_classes_only:
                entity_or_class = self.get_cls_by_entid(entity_or_class)  # TODO: handle multiple classes
                is_entity = False
            if is_entity:
                _q = f"""SELECT DISTINCT ?rel ?domain ?range WHERE {{
                    {'?x' if direction == 'in' else (':' + entity_or_class)} ?rel {(':' + entity_or_class) if direction == 'in' else '?x'} . 
                    {{{{?rel a rdfs:Property}} UNION {{?rel a rdf:Property}} 
                    UNION {{?rel :type.object.type :type.property}} 
                    UNION {{?rel a <http://www.w3.org/2002/07/owl#FunctionalProperty>}}}} .
                    ?rel rdfs:domain ?domain .
                    ?rel rdfs:range ?range
                }}"""
            else:
                if entity_or_class in literal_map:
                    if direction == 'out':
                        return []
                    # literal class
                    if entity_or_class in literal_map_variants:
                        conditions = [f'datatype(?y) = <{lmv}>' for lmv in literal_map_variants[entity_or_class]]
                        filter = f"FILTER ({' || '.join(conditions)})"
                    else:
                        filter = f"FILTER (datatype(?y) = <{literal_map[entity_or_class]}>)"
                    _q = f"""SELECT DISTINCT ?rel ?domain :{entity_or_class} WHERE {{
                        ?x ?rel ?y .
                        {filter} .
                        {{{{?rel a rdfs:Property}} UNION {{?rel a rdf:Property}} 
                        UNION {{?rel :type.object.type :type.property}} 
                        UNION {{?rel a <http://www.w3.org/2002/07/owl#FunctionalProperty>}}}} .
                        ?rel rdfs:domain ?domain .
                        ?rel rdfs:range ?range
                    }}"""
                else:
                    _q = f"""SELECT DISTINCT ?rel ?domain ?range WHERE {{
                        ?x ?rel ?y .
                        {'?y' if direction == 'in' else '?x'} a :{entity_or_class} .
                        {{{{?rel a rdfs:Property}} UNION {{?rel a rdf:Property}} 
                        UNION {{?rel :type.object.type :type.property}} 
                        UNION {{?rel a <http://www.w3.org/2002/07/owl#FunctionalProperty>}}}} .
                        ?rel rdfs:range ?range .
                        ?rel rdfs:domain ?domain .
                    }}"""
        else:
            # query is an s-expression or graph-query
            sexpr = query if type(query) is str else graph_query_to_sexpr(query)
            if use_classes_only:
                bottom_up_parse = parse_bottom_up(sexpr)
                for subexp in bottom_up_parse:
                    for ti, token in enumerate(subexp):
                        if ti == 0 or type(token) is list:
                            continue
                        if token.startswith('m.') or token.startswith('g.'):
                            subexp[ti] = self.get_cls_by_entid(token)  # TODO: handle multiple classes
                sexpr = bottom_up_to_sexpr(bottom_up_parse)
            _q = sexpr_to_sparql(sexpr, filter_entities=filter_entities)
            _q = f"""SELECT DISTINCT ?rel ?domain ?range WHERE {{
                {'?x' if direction == 'out' else '?_y'} ?rel {'?_y' if direction == 'out' else '?x'} .
                {{{{?rel a rdfs:Property}} UNION {{?rel a rdf:Property}} 
                UNION {{?rel :type.object.type :type.property}} 
                UNION {{?rel a <http://www.w3.org/2002/07/owl#FunctionalProperty>}}}} .
                ?rel rdfs:domain ?domain .
                ?rel rdfs:range ?range .
                {get_where_clause_content(_q)}
            }}"""
        rtn = self.execute_query(_q, catch_timeout=catch_timeout)  # returns a list of rows
        rtn = self.process_results(rtn)
        # Return unique relations
        return list({"_".join(r): {
            "relation": r[0],
            "domain": literal_map_inv.get(r[1], r[1]),
            "range": literal_map_inv.get(r[2], r[2])
        } for r in rtn}.values())
