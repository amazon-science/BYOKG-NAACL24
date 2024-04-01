from collections import defaultdict
from typing import List, Union
import logging
import networkx as nx

from src.utils.helpers import merge_quotes
from src.utils.maps import comp_map_2_alpha, comp_map_2_sym, literal_map, literal_map_variants

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def sexpr_to_struct(sexpr: str) -> List:
    """
    Takes an s-expression and returns its nested list representation.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    Text in quotations are retained as a single unit.
    """
    stack: List = []
    current_expression: List = []
    tokens = sexpr.split()
    try:
        tokens = merge_quotes(tokens)
    except:
        logger.info(f'Error merging quotes in {sexpr}')
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def struct_to_sexpr(expression: list) -> str:
    rtn = '('
    for i, e in enumerate(expression):
        if isinstance(e, list):
            rtn += struct_to_sexpr(e)
        else:
            rtn += e
        if i != len(expression) - 1:
            rtn += ' '
    rtn += ')'
    return rtn


def get_nesting_level(struct: list) -> int:
    max_sub = 0
    for item in struct:
        if isinstance(item, list):
            level = get_nesting_level(item)
            if level > max_sub:
                max_sub = level
    return 1 + max_sub


def get_inv_rel(rel: str):
    return f"{rel[:-2]}" if rel.endswith("#R") else f"{rel}#R"


def is_inv_rel(rel: str):
    return rel.endswith("#R")


def get_2hop_relations(entity: str, in_relations, out_relations, triples):
    paths = set()
    # entity in in
    for rel in in_relations[entity]:
        rel1 = rel
        ent2_set = triples[(entity, get_inv_rel(rel1))]
        for ent2 in ent2_set:
            for rel2 in in_relations[ent2]:
                paths.add((rel1, rel2))

    # entity in out
    for rel in in_relations[entity]:
        rel1 = rel
        ent2_set = triples[(entity, get_inv_rel(rel1))]
        for ent2 in ent2_set:
            for rel2 in out_relations[ent2]:
                paths.add((rel1, get_inv_rel(rel2)))

    # entity out in
    for rel in out_relations[entity]:
        rel1 = rel
        ent2_set = triples[(entity, get_inv_rel(rel1))]
        for ent2 in ent2_set:
            for rel2 in in_relations[ent2]:
                paths.add((get_inv_rel(rel1), rel2))

    # entity out out
    for rel in out_relations[entity]:
        rel1 = rel
        ent2_set = triples[(entity, get_inv_rel(rel1))]
        for ent2 in ent2_set:
            for rel2 in out_relations[ent2]:
                paths.add((get_inv_rel(rel1), get_inv_rel(rel2)))
    return paths


def generate_all_logical_forms_2(entity_id: int, hop2_paths: dict[str, list], schema_desc: dict,
                                 entity_dict=None, add_quotes=True):
    lfs = []

    if entity_id not in hop2_paths:
        return lfs

    paths = hop2_paths[entity_id]
    for path in paths:
        # TODO: Add check for legal relation?
        # TODO: Handle inverse relations differently the way GraphQ does?
        relation_1 = path[0]
        relation_2 = path[1]
        cls = schema_desc[relation_2]['domain']
        entity = str(entity_id) if entity_dict is None else entity_dict[entity_id][0]
        if add_quotes:
            entity = f'"{entity}"'
        lf = '(AND ' + cls + ' (JOIN ' + relation_2 + ' (JOIN ' + relation_1 + ' ' + entity + ')))'
        lfs.append(lf)
        lf = '(COUNT (AND ' + cls + ' (JOIN ' + relation_2 + ' (JOIN ' + relation_1 + ' ' + entity + '))))'
        lfs.append(lf)
    return lfs


def get_end_num(G, s):
    end_num = defaultdict(lambda: 0)
    for edge in list(G.edges(s)):  # for directed graph G.edges is the same as G.out_edges, not including G.in_edges
        end_num[list(edge)[1]] += 1
    return end_num


def set_visited(G, s, e, relation):
    end_num = get_end_num(G, s)
    for i in range(0, end_num[e]):
        if G.edges[s, e, i]['relation'] == relation:
            G.edges[s, e, i]['visited'] = True


def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print("error: binary function should have 2 parameters!")
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    else:
        if len(elements) == 2:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' \
                   + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'


def count_function(G, start, type_constraint=True, readable=False, readable_type='label', use_reverse_relations=False,
                   entry_point=False):
    return '(COUNT ' + none_function(G, start, type_constraint=type_constraint, readable=readable,
                                     readable_type=readable_type, use_reverse_relations=use_reverse_relations,
                                     entry_point=entry_point) + ')'


def none_function(G, start, arg_node=None, type_constraint=True, readable=False, readable_type='label',
                  use_reverse_relations=False, entry_point=False):
    # NOTE: we assume that there is only one function per query
    if arg_node is not None:
        # ARGMAX or ARGMIN function
        arg = G.nodes[arg_node]['function']
        path = list(nx.all_simple_paths(G, start, arg_node))
        # try:
        #     assert len(path) == 1
        # except:
        #     breakpoint()
        path = path[0]
        arg_clause = []
        for i in range(0, len(path) - 1):
            edge = G.edges[path[i], path[i + 1], 0]
            if edge['reverse']:
                if use_reverse_relations and edge['reverse_relation'] is not None:
                    relation = edge['reverse_relation']
                else:
                    relation = '(R ' + edge['relation'] + ')'
            else:
                relation = edge['relation']
            arg_clause.append(relation)
        # Deleting edges until the first node with out-degree > 2 is meet
        # (conceptually it should be 1, but remember that add edges is both directions)
        while i >= 0:
            flag = False
            if G.out_degree[path[i]] > 2:
                flag = True
            G.remove_edge(path[i], path[i + 1], 0)
            i -= 1
            if flag:
                break
        if len(arg_clause) > 1:
            arg_clause = binary_nesting(function='JOIN', elements=arg_clause)
        else:
            arg_clause = arg_clause[0]
        return '(' + arg.upper() + ' ' + none_function(G, start,
                                                       type_constraint=type_constraint,
                                                       readable=readable,
                                                       readable_type=readable_type,
                                                       use_reverse_relations=use_reverse_relations,
                                                       entry_point=True) + ' ' + arg_clause + ')'

    if G.nodes[start]['type'] != 'class':
        if readable:
            entity_or_literal = G.nodes[start]["readable_name"]  # readable_type == "label" (default)
            if readable_type == "anon":
                cls_text = G.nodes[start]["cla"].split(".")[-1]
                entity_or_literal = f"{cls_text}_{G.nodes[start]['nid']}"  # e.g. movie_1, person_3
            elif readable_type == "anon_noid":
                cls_text = G.nodes[start]["cla"].split(".")[-1]
                entity_or_literal = f"{cls_text}"  # e.g. movie, person
            entity_or_literal = f'"{entity_or_literal}"'
        else:
            entity_or_literal = G.nodes[start]['id']
        return entity_or_literal

    clauses = []
    if G.nodes[start]['question'] and type_constraint and entry_point:
        clauses.append(G.nodes[start]['id'])  # Adds an AND clause with the class of the question node

    end_num = get_end_num(G,
                          start)  # gives a dict of node_id to number of incoming edges (including inverse) from `start` to those nodes
    # Iterate over all nodes that have an incoming edge from `start`
    for key in end_num.keys():
        # Iterate over all edges between the start and that node
        for i in range(0, end_num[key]):
            if not G.edges[start, key, i]['visited']:
                relation = G.edges[start, key, i]['relation']
                G.edges[start, key, i]['visited'] = True
                set_visited(G, key, start, relation)
                if G.edges[start, key, i]['reverse']:
                    if use_reverse_relations and G.edges[start, key, i]['reverse_relation'] is not None:
                        relation = G.edges[start, key, i]['reverse_relation']
                    else:
                        relation = '(R ' + relation + ')'
                if G.nodes[key]['function'].__contains__('<') or G.nodes[key]['function'].__contains__('>'):
                    # Comparative function
                    fn = comp_map_2_alpha[G.nodes[key]['function']]  # functions use inequality ops e.g. >, <=
                    clauses.append(
                        f'({fn} {relation} ' +
                        f'{none_function(G, key, type_constraint=type_constraint, readable=readable, readable_type=readable_type, use_reverse_relations=use_reverse_relations)})')  # e.g. 'gt capacity 1000'
                else:
                    clauses.append('(JOIN ' + relation + ' ' + none_function(G, key, type_constraint=type_constraint,
                                                                             readable=readable,
                                                                             readable_type=readable_type,
                                                                             use_reverse_relations=use_reverse_relations) + ')')

    if len(clauses) == 0:
        return G.nodes[start]['id']

    if len(clauses) == 1:
        return clauses[0]
    else:
        return binary_nesting(function='AND', elements=clauses)


def graph_to_logical_form(G, start, count: bool = False):
    if count:
        return '(COUNT ' + none_function(G, start) + ')'
    else:
        return none_function(G, start)


def graph_query_to_graph(graph_query: dict):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    qid = None
    for node in graph_query['nodes']:
        G.add_node(node['nid'], nid=node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                   function=node['function'], cla=node['class'], readable_name=node['readable_name'])
        if node['question_node'] == 1:
            qid = node['nid']
        if node['function'] != 'none':
            aggregation = node['function']
            if node['function'].__contains__('arg'):
                arg_node = node['nid']
    for edge in graph_query['edges']:
        G.add_edge(edge['start'], edge['end'], relation=edge['relation'], readable_name=edge['readable_name'],
                   reverse=False, visited=False, reverse_relation=edge.get('reverse_relation', None),
                   reverse_readable_name=edge.get('reverse_readable_name', None))
        G.add_edge(edge['end'], edge['start'], relation=edge['relation'], readable_name=edge['readable_name'],
                   reverse=True, visited=False, reverse_relation=edge.get('reverse_relation', None),
                   reverse_readable_name=edge.get('reverse_readable_name', None))
    return G, aggregation, arg_node, qid


def graph_query_to_sexpr(graph_query: dict, type_constraint=True, readable=False, readable_type='label',
                         use_reverse_relations=False):
    G, aggregation, arg_node, qid = graph_query_to_graph(graph_query)

    if 'count' == aggregation:
        return count_function(G, qid, type_constraint=type_constraint, readable=readable, readable_type=readable_type,
                              use_reverse_relations=use_reverse_relations, entry_point=True)
    else:
        return none_function(G, qid, arg_node=arg_node, type_constraint=type_constraint, readable=readable,
                             readable_type=readable_type, use_reverse_relations=use_reverse_relations, entry_point=True)


def _expand(tkn, sub_formulas):
    if type(tkn) is str:
        if tkn.startswith('#'):
            idx = int(tkn[1:])
            return _expand(sub_formulas[idx], sub_formulas)
        return tkn
    assert type(tkn) is list
    return [_expand(t, sub_formulas) for t in tkn]


def parse_bottom_up(sexpr_or_struct: [str, list], sub_formula_id=None, expand=False):
    # Parses sexpr string or nested list in a bottom-up manner
    # Returns the list of sub-expression steps, tagging the result of each by the index of their appearance
    expression = sexpr_or_struct if type(sexpr_or_struct) is list else sexpr_to_struct(sexpr_or_struct)
    if sub_formula_id is None:
        sub_formula_id = [0]
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(parse_bottom_up(e, sub_formula_id))
            sub_exp_idx = sub_formula_id[0] - 1
            expression[i] = f'#{sub_exp_idx}'

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    if expand:
        return _expand(tkn=sub_formulas, sub_formulas=sub_formulas)
    return sub_formulas


def bottom_up_to_sexpr(parse, return_all=False):
    if type(parse) is str:
        # input is sexpr instead of the parse (list)
        parse = parse_bottom_up(parse)

    def swap_placeholder(cur, stored):
        if type(cur) is list:
            for _i, _c in enumerate(cur):
                if _c[0] == '#':
                    cur[_i] = stored[int(_c[1:])]
                    swap_placeholder(cur[_i], stored)

    sub_formulas = []
    for p in parse:
        swap_placeholder(p, sub_formulas)
        sub_formulas.append(struct_to_sexpr(p))
    if return_all:
        return sub_formulas
    return sub_formulas[-1]


def get_symbol_type(symbol: str, types, relations) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations:
        return 4
    elif symbol:
        return 1


def _get_graph(
        expression: List, types, relations, relation_dr,
        reverse_properties,
        upper_types) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
    if isinstance(expression, str):
        G = nx.MultiDiGraph()
        if get_symbol_type(expression, types, relations) == 1:
            G.add_node(1, id=expression, type='entity')
        elif get_symbol_type(expression, types, relations) == 2:
            G.add_node(1, id=expression, type='literal')
        elif get_symbol_type(expression, types, relations) == 3:
            G.add_node(1, id=expression, type='class')
            # G.add_node(1, id="common.topic", type='class')
        elif get_symbol_type(expression, types, relations) == 4:  # relation or attribute
            domain, rang = relation_dr[expression]
            G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
            G.add_node(2, id=domain, type='class')
            G.add_edge(2, 1, relation=expression)

            if reverse_properties is not None:
                if expression in reverse_properties:
                    G.add_edge(1, 2, relation=reverse_properties[expression])

        return G

    if expression[0] == 'R':
        G = _get_graph(expression[1], types, relations, relation_dr, reverse_properties, upper_types)
        size = len(G.nodes())
        mapping = {}
        for n in G.nodes():
            mapping[n] = size - n + 1
        G = nx.relabel_nodes(G, mapping)
        return G

    elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
        G1 = _get_graph(expression=expression[1], types=types, relations=relations, relation_dr=relation_dr,
                        reverse_properties=reverse_properties, upper_types=upper_types)
        G2 = _get_graph(expression=expression[2], types=types, relations=relations, relation_dr=relation_dr,
                        reverse_properties=reverse_properties, upper_types=upper_types)

        size = len(G2.nodes())
        qn_id = size
        if upper_types is not None:
            if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
                if G2.nodes[qn_id]['id'] in upper_types[G1.nodes[1]['id']]:
                    G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G = nx.compose(G1, G2)

        if expression[0] != 'JOIN':
            G.nodes[1]['function'] = comp_map_2_sym[expression[0]]

        return G

    elif expression[0] == 'AND':
        G1 = _get_graph(expression[1], types, relations, relation_dr, reverse_properties, upper_types)
        G2 = _get_graph(expression[2], types, relations, relation_dr, reverse_properties, upper_types)

        size1 = len(G1.nodes())
        size2 = len(G2.nodes())
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']
            # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
            # So here for the AND function we force it to choose the type explicitly provided in the logical form
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'COUNT':
        G = _get_graph(expression[1], types, relations, relation_dr, reverse_properties, upper_types)
        size = len(G.nodes())
        G.nodes[size]['function'] = 'count'

        return G

    elif expression[0].__contains__('ARG'):
        G1 = _get_graph(expression[1], types, relations, relation_dr, reverse_properties, upper_types)
        size1 = len(G1.nodes())
        G2 = _get_graph(expression[2], types, relations, relation_dr, reverse_properties, upper_types)
        size2 = len(G2.nodes())
        # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
        G2.nodes[1]['id'] = 0
        G2.nodes[1]['type'] = 'literal'
        G2.nodes[1]['function'] = expression[0].lower()
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']

        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'TC':
        G = _get_graph(expression[1], types, relations, relation_dr, reverse_properties, upper_types)
        size = len(G.nodes())
        G.nodes[size]['tc'] = (expression[2], expression[3])

        return G


def logical_form_to_graph(expression: List, types, relations, relation_dr, reverse_properties=None,
                          upper_types=None) -> nx.MultiGraph:
    # expression : nested s-expression
    # types : set of all classes
    # relations : set of all relations
    # reverse_properties : maps a relation to its inverse (if exists, i.e. not using #R)
    # upper_types : maps a class to its set of parent classes

    G = _get_graph(expression, types, relations, relation_dr, reverse_properties, upper_types)
    G.nodes[len(G.nodes())]['question_node'] = 1
    return G


def same_logical_form(form1: str, form2: str, types, relations) -> bool:
    if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
        return False
    try:
        G1 = logical_form_to_graph(sexpr_to_struct(form1), types, relations)
    except Exception:
        return False
    try:
        G2 = logical_form_to_graph(sexpr_to_struct(form2), types, relations)
    except Exception:
        return False

    def node_match(n1, n2):
        if n1['id'] == n2['id'] and n1['type'] == n2['type']:
            func1 = n1.pop('function', 'none')
            func2 = n2.pop('function', 'none')
            tc1 = n1.pop('tc', 'none')
            tc2 = n2.pop('tc', 'none')

            if func1 == func2 and tc1 == tc2:
                return True
            else:
                return False
        else:
            return False

    def multi_edge_match(e1, e2):
        if len(e1) != len(e2):
            return False
        values1 = []
        values2 = []
        for v in e1.values():
            values1.append(v['relation'])
        for v in e2.values():
            values2.append(v['relation'])
        return sorted(values1) == sorted(values2)

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)


def sexpr_to_sparql(sexpr: str, sep='\n', filter_entities=True):
    # Freebase-specific
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = sexpr_to_struct(sexpr)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = parse_bottom_up(expression)
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    ue = 0  # ungrounded entity
    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append(":" + subp[2] + " :" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " :" + subp[1][1] + " ?x" + i + " .")
                elif '"' not in subp[2] and '^^' not in subp[2]:  # ungrounded entity
                    clauses.append(f"?ue{ue} :{subp[1][1]} ?x{i} .")
                    clauses.append(f"?ue{ue} a :{subp[2]} .")
                    ue += 1
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        lit_val = subp[2].split("^^")[0].replace('"', '')
                        if data_type in ['int', 'integer', 'float', 'double', 'dateTime', 'boolean']:
                            subp[2] = f'"{lit_val}"^^<{subp[2].split("^^")[1]}>'
                        elif any(typ in data_type.lower() for typ in ['date', 'year']):
                            subp[2] = f'"{lit_val + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            raise ValueError(f'Unknown datatype found: {data_type}')
                    clauses.append(subp[2] + " :" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " :" + subp[1] + " :" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " :" + subp[1] + " ?x" + subp[2][1:] + " .")
                elif '"' not in subp[2] and '^^' not in subp[2]:  # ungrounded entity
                    clauses.append(f"?x{i} :{subp[1]} ?ue{ue} .")
                    clauses.append(f"?ue{ue} a :{subp[2]} .")
                    ue += 1
                else:  # literal
                    if '^^' in subp[2]:
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        lit_val = subp[2].split("^^")[0].replace('"', '')
                        if data_type in ['int', 'integer', 'float', 'double', 'dateTime', 'boolean']:
                            subp[2] = f'"{lit_val}"^^<{subp[2].split("^^")[1]}>'
                        elif any(typ in data_type.lower() for typ in ['date', 'year']):
                            subp[2] = f'"{lit_val + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            raise ValueError(f'Unknown datatype found: {data_type}')
                    clauses.append("?x" + i + " :" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                if subp[1] in literal_map:
                    if subp[1] in literal_map_variants:
                        conditions = [f'datatype( ?x{i} ) = <{lmv}>' for lmv in literal_map_variants[subp[1]]]
                        clauses.append(f"FILTER ({' || '.join(conditions)})")
                    else:
                        clauses.append(f"FILTER (datatype( ?x{i} ) = <{literal_map[subp[1]]}>)")
                else:
                    clauses.append(f"?x{i} a :{subp[1]} .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " :" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2][0] == '#':  # variable
                subp[2] = f'?x{subp[2][1:]}'
            elif '^^' in subp[2]:
                data_type = subp[2].split("^^")[1].split("#")[1]
                lit_val = subp[2].split("^^")[0].replace('"', '')
                if data_type in ['int', 'integer', 'float', 'double', 'dateTime', 'boolean']:
                    subp[2] = f'"{lit_val}"^^<{subp[2].split("^^")[1]}>'
                elif any(typ in data_type.lower() for typ in ['date', 'year']):
                    subp[2] = f'"{lit_val + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    raise ValueError(f'Unknown datatype found: {data_type}')
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} :{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} :{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} :{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} :{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} :{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} :{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                if subp[1] in literal_map:
                    if subp[1] in literal_map_variants:
                        conditions = [f'datatype( ?x{i} ) = <{lmv}>' for lmv in literal_map_variants[subp[1]]]
                        clauses.append(f"FILTER ({' || '.join(conditions)})")
                    else:
                        clauses.append(f"FILTER (datatype( ?x{i} ) = <{literal_map[subp[1]]}>)")
                else:
                    clauses.append(f'?x{i} a :{subp[1]} .')

            if len(subp) == 3:
                clauses.append(f'?x{i} :{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} :{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} :{relation} ?{var1} .')

                clauses.append(f'?c{j} :{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?sk0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    if filter_entities:
        for entity in entities:
            clauses.append(f'FILTER (?x != :{entity})')

    clauses.insert(0, f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    return sep.join(clauses)


def get_where_clause_content(sparql):
    where_clause = sparql.split('WHERE')[-1].strip()
    assert where_clause[0] == '{'
    balance = 0
    for i, t in enumerate(where_clause):
        if t == '{':
            balance += 1
        if t == '}':
            balance -= 1
            if balance == 0:
                break
    where_content = where_clause[1:i].strip()
    return where_content


def graph_query_to_sparql(gq: dict, type_constraint=True, readable=False):
    sexpr = graph_query_to_sexpr(gq, type_constraint=type_constraint, readable=readable)
    return sexpr_to_sparql(sexpr)


if __name__ == '__main__':
    # Tests passed:
    # sexpr = "(AND type.datetime (JOIN (R movie.release_year) m.38536))"
    # sexpr = "(AND movie.movie (JOIN movie.has_genre (JOIN (R movie.has_genre) m.8095)))"
    # sexpr = "(ARGMAX (AND movie.tag (JOIN (R movie.has_tags) (JOIN movie.directed_by movie.person))) (JOIN (R movie.has_tags) movie.release_year))"
    # sexpr = "(AND movie.language (JOIN (R movie.in_language) m.23162))"  # Empty set: correct
    # sexpr = "(COUNT (AND movie.language (JOIN (R movie.in_language) (AND (JOIN movie.has_tags (JOIN (R movie.has_tags) m.5359)) (JOIN movie.in_language m.37555)))))" # Returns 0
    # sexpr = "(COUNT (AND movie.movie (AND (JOIN movie.directed_by movie.person) (JOIN movie.starred_actors m.269))))"
    # sexpr = "(ARGMAX movie.language (JOIN (R movie.in_language) movie.release_year))"
    # sexpr = """(AND movie.tag (JOIN (R movie.has_tags) (AND (lt movie.release_year "1946"^^http://www.w3.org/2001/XMLSchema#dateTime) (JOIN movie.has_tags (JOIN (R movie.has_tags) movie.movie)))))"""
    # sparql = sexpr_to_sparql(sexpr)
    # print(sparql)
    # test = {'nodes': [{'nid': 0,
    #                    'node_type': 'class',
    #                    'id': 'movie.movie',
    #                    'class': 'movie.movie',
    #                    'readable_name': 'film or movie',
    #                    'question_node': 1,
    #                    'function': 'none'},
    #                   {'nid': 1,
    #                    'node_type': 'class',
    #                    'readable_name': 'person involved in a movie',
    #                    'question_node': 0,
    #                    'function': 'none',
    #                    'id': 'movie.person',
    #                    'class': 'movie.person'},
    #                   {'nid': 2,
    #                    'node_type': 'entity',
    #                    'readable_name': 'sam rockwell',
    #                    'question_node': 0,
    #                    'function': 'none',
    #                    'id': 'm.9679',
    #                    'class': 'movie.tag'}],
    #         'edges': [{'start': 0,
    #                    'end': 1,
    #                    'relation': 'movie.written_by',
    #                    'readable_name': 'person(s) who wrote the script for the movie',
    #                    'reverse_relation': 'person.wrote_movie',
    #                    'reverse_readable_name': 'wrote movie'},
    #                   {'start': 0,
    #                    'end': 1,
    #                    'relation': 'movie.directed_by',
    #                    'readable_name': 'person(s) who directed the movie',
    #                    'reverse_relation': 'person.directed_movie',
    #                    'reverse_readable_name': 'directed movie'},
    #                   {'start': 0,
    #                    'end': 2,
    #                    'relation': 'movie.has_tags',
    #                    'readable_name': 'additional features or information about the movie',
    #                    'reverse_relation': 'tag.of_movie',
    #                    'reverse_readable_name': 'of movie'}]}
    # print(graph_query_to_sexpr(test, type_constraint=False))
    # print(graph_query_to_sexpr(test, type_constraint=True))
    # print(graph_query_to_sparql(test, type_constraint=True))
    pass
