literal_map = {
    "type.integer": "http://www.w3.org/2001/XMLSchema#integer",
    "type.int": "http://www.w3.org/2001/XMLSchema#integer",
    "type.float": "http://www.w3.org/2001/XMLSchema#float",
    "type.datetime": "http://www.w3.org/2001/XMLSchema#dateTime",
    # TODO: Support list values -> ['http://www.w3.org/2001/XMLSchema#dateTime', 'http://www.w3.org/2001/XMLSchema#gYear', 'http://www.w3.org/2001/XMLSchema#date']
    "type.string": "http://www.w3.org/2001/XMLSchema#string",
    "type.boolean": "http://www.w3.org/2001/XMLSchema#boolean",
    # TODO: what about type.lang, type.media_type, type.type
}

literal_map_variants = {
    "type.datetime": [
        'http://www.w3.org/2001/XMLSchema#dateTime',
        'http://www.w3.org/2001/XMLSchema#gYear',
        'http://www.w3.org/2001/XMLSchema#gYearMonth',
        'http://www.w3.org/2001/XMLSchema#date'
    ],
    "type.float": [
        'http://www.w3.org/2001/XMLSchema#float',
        'http://www.w3.org/2001/XMLSchema#double'
    ]
}

literal_map_inv = {v: k for k, v in literal_map.items()}

for k, v in literal_map_variants.items():
    for vi in v:
        literal_map_inv[vi] = k

rel_map = {
    # RDF relations
    "rdf.label": "http://www.w3.org/2000/01/rdf-schema#label",
    "rdf.type": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "rdf.domain": "http://www.w3.org/2000/01/rdf-schema#domain",
    "rdf.range": "http://www.w3.org/2000/01/rdf-schema#range",
    "rdf.comment": "http://www.w3.org/2000/01/rdf-schema#comment",
    # Wiki relations
    "wiki.label": "meta.wikidata_propertyLabel",
    "wiki.id": "meta.wikidata_property",
    # Custom
    "_.alias": "meta.alt_labels"
}

fn_description = {
    'COUNT': "function to get the number of answer entities returned by the query",
    'ARGMAX': "function to get the answer entity that maximizes a numerical quantity",
    'ARGMIN': "function to get the answer entity that minimizes a numerical quantity",
    'gt': "function to get answer entities that are greater than the specified numerical quantity",
    'lt': "function to get answer entities that are lesser than the specified numerical quantity",
    'ge': "function to get answer entities that are greater than or equal to the specified numerical quantity",
    'le': "function to get answer entities that are lesser than or equal to the specified numerical quantity"
}

# Comparative function maps
comp_map_2_sym = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}
comp_map_2_alpha = {'<=': 'le', '>=': 'ge', '<': 'lt', '>': 'gt'}
