import datetime
import re
from typing import Dict, Tuple, Any, List

URL_RE = r'''(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
URL_RE_COMPILED = re.compile(URL_RE)

NUMBER_RE = r'''^[+-]?(?:\d+\.?\d*|\d*\.\d+)$'''
NUMBER_RE_COMPILED = re.compile(NUMBER_RE)

_DATE_FORMATS = {
    '%y0000': (True, False, False),
    '%y%m00': (True, True, False),
    '%y%m%d': (True, True, True),
    '%Y0000': (True, False, False),
    '%Y%m00': (True, True, False),
    '%d %B %Y': (True, True, True),
    '%d %B': (True, True, False),
    '%d %Y': (True, False, True),
    '%Y%m%d': (True, True, True),
    '%Y-%m-%d': (True, True, True),
    '%m/%d': (False, True, True),
    '%m/%d/%Y': (True, True, True),
    '%m - %d - %Y': (True, True, True),
    '%B %Y': (True, True, False),
    '%B , %Y': (True, True, False),
    '%B %d %Y': (True, True, True),
    '%B %d , %Y': (True, True, True),
    '%B %d': (False, True, True),
    '%B %dst': (False, True, True),
    '%B %dnd': (False, True, True),
    '%B %drd': (False, True, True),
    '%B %dth': (False, True, True),
    '%B': (False, True, False),
    '%Y': (True, False, False),
    '%y': (True, False, False),
}


def parse_date(expression: str):
    results = []
    for format_ in _DATE_FORMATS:
        try:
            result = datetime.datetime.strptime(expression, format_)
            if format_[0] and result.year < 2018:
                results.append((result, _DATE_FORMATS[format_]))
        except:
            pass
    if len(results) > 1:
        return results[0]
    elif len(results) == 1:
        return results[0]
    else:
        return None, (False, False, False)


def is_url(s: str):
    global URL_RE_COMPILED
    return URL_RE_COMPILED.match(s) is not None


def is_number(s: str):
    global NUMBER_RE_COMPILED
    return NUMBER_RE_COMPILED.match(s) is not None


def unquote(s: str):
    return s[1:-1] if len(s) > 2 and s[0] == '"' and s[-1] == '"' else s


def is_attribute(s: str):
    return s in ('-',  # polarity
                 '+',  # polite
                 'BC', 'AD', 'Heisei',  # era
                 'expressive', 'imperative'  # mode
                 ) \
           or s.isdigit() \
           or is_url(s) \
           or is_number(s)


def is_const_relation(s: str):
    return s.startswith('op') \
           or s in ('month',
                    'decade',
                    'polarity',
                    'day',
                    'quarter',
                    'year',
                    'era',
                    'century',
                    'timezone',
                    'polite',
                    'mode',
                    'value',
                    'quant',
                    'unit',
                    'range',
                    'scale',
                    'li',
                    'year2')


def extract_mrp_dict(existing_edges: Dict[int, List[Tuple[str, int]]],
                     sent_len: int,
                     id_cnt: int,
                     node_labels: Dict[int, str],
                     node_types: Dict[int, str],
                     metadata: Dict[str, Any]):
    assert (sent_len > 0)
    assert (id_cnt > sent_len)
    for n in existing_edges:
        assert (0 <= n and n < id_cnt)
        for e in existing_edges[n]:
            assert (len(e) == 2)
            assert (0 <= e[1] and e[1] < id_cnt)
    assert (node_types[0] == 'ROOT')
    for i in range(1, sent_len + 1):
        assert (node_types[i] == 'TokenNode')
    for i in range(sent_len + 1, id_cnt):
        assert (node_types[i] in ('EntityNode', 'ConceptNode', 'AttributeNode'))
    assert ('tokens' in metadata)
    assert (len(metadata['tokens']) == sent_len)

    _NO_NORMAL = {
        'prep-on-behalf-of',
        'prep-out-of'
    }
    _EDGE_LABEL_NORMAL = {
        "accompanier-of": "accompanier",
        "age-of": "age",
        "ARG0-of": "ARG0",
        "ARG1-of": "ARG1",
        "ARG2-of": "ARG2",
        "ARG3-of": "ARG3",
        "ARG4-of": "ARG4",
        "ARG5-of": "ARG5",
        "ARG6-of": "ARG6",
        "ARG7-of": "ARG7",
        "beneficiary-of": "beneficiary",
        "concession-of": "concession",
        "condition-of": "condition",
        "consist-of": "consist",
        "degree-of": "degree",
        "destination-of": "destination",
        "direction-of": "direction",
        "duration-of": "duration",
        "example-of": "example",
        "extent-of": "extent",
        "frequency-of": "frequency",
        "instrument-of": "instrument",
        "location-of": "location",
        "manner-of": "manner",
        "medium-of": "medium",
        "mod": "domain",
        "name-of": "name",
        "op1-of": "op1",
        "ord-of": "ord",
        "part-of": "part",
        "path-of": "path",
        "polarity-of": "polarity",
        "poss-of": "poss",
        "purpose-of": "purpose",
        "quant-of": "quant",
        "source-of": "source",
        "subevent-of": "subevent",
        "subset-of": "subset",
        "time-of": "time",
        "topic-of": "topic",
        "value-of": "value",
    }

    amr_dict = {'id': metadata['id'],
                'flavor': 2,
                'framework': 'amr',
                'version': metadata['version'] if 'version' in metadata else 1.0,
                'time': metadata['time'] if 'time' in metadata else datetime.datetime.now().strftime(
                    '%Y-%m-%d (%H:%M)'),
                'input': metadata['input'] if 'input' in metadata else ' '.join(metadata['tokens']),
                'tops': [0],
                'nodes': [{'id': 0,
                           'label': 'null'}],
                'edges': []}

    if any([tp == 'ConceptNode' for tp in node_types]):
        property_node_set = set()
        amr_dict['tops'] = []
        amr_dict['nodes'] = []
        amr_dict['edges'] = []
        for node_idx in range(sent_len + 1, id_cnt):
            if node_types[node_idx] != 'ConceptNode':
                continue
            cur_node = {'id': node_idx,
                        'label': unquote(node_labels[node_idx]),
                        'properties': [],
                        'values': []}
            if node_idx in existing_edges:
                for cur_edge in existing_edges[node_idx]:
                    if cur_edge[1] == 0:
                        continue
                    if cur_edge[1] not in existing_edges:
                        if cur_edge[1] < sent_len:
                            cur_node['properties'].append(unquote(cur_edge[0]))
                            cur_node['values'].append(unquote(node_labels[cur_edge[1]]))
                            property_node_set.add(cur_edge[1])
                        elif (is_const_relation(cur_edge[0]) and is_attribute(node_labels[cur_edge[1]])) \
                                or (node_types[cur_edge[1]] == 'AttributeNode'):
                            cur_node['properties'].append(unquote(cur_edge[0]))
                            cur_node['values'].append(unquote(node_labels[cur_edge[1]]))
                            property_node_set.add(cur_edge[1])
                        else:
                            amr_dict['edges'].append({'source': node_idx,
                                                      'target': cur_edge[1],
                                                      'label': cur_edge[0]})
                    else:
                        if is_const_relation(cur_edge[0]) and is_attribute(node_labels[cur_edge[1]]):
                            cur_node['properties'].append(unquote(cur_edge[0]))
                            cur_node['values'].append(unquote(node_labels[cur_edge[1]]))
                            property_node_set.add(cur_edge[1])
                        else:
                            amr_dict['edges'].append({'source': node_idx,
                                                      'target': cur_edge[1],
                                                      'label': cur_edge[0]})
            if len(cur_node['properties']) == 0:
                del cur_node['properties']
                del cur_node['values']
            amr_dict['nodes'].append(cur_node)

        for e in amr_dict['edges']:
            if e['label'] in _EDGE_LABEL_NORMAL:
                e['normal'] = _EDGE_LABEL_NORMAL[e['label']]
            elif re.match(r'(ARG|op)\d+-of', e['label']):
                e['normal'] = e['label'][:-3]

        amr_dict['nodes'] = [x for x in amr_dict['nodes'] if x['id'] not in property_node_set]
        amr_dict['edges'] = [x for x in amr_dict['edges'] if
                             x['source'] not in property_node_set and x['target'] not in property_node_set]

        if 0 in existing_edges and len(existing_edges[0]) > 0:
            amr_dict['tops'] = [e[1] for e in existing_edges[0] if e[1] not in property_node_set]

    return amr_dict
