import json
import re
from argparse import ArgumentParser
from typing import Dict


def mrp_dict_to_amr_str(mrp_dict: Dict, amr_str_only=True, all_nodes=False):
    amr_str = ''
    if not amr_str_only:
        if 'id' in mrp_dict:
            amr_str += f"# ::id {mrp_dict['id']}\n"
        if 'input' in mrp_dict:
            amr_str += f"# ::snt {mrp_dict['input']}\n"
    if 'nodes' not in mrp_dict or 'edges' not in mrp_dict:
        amr_str += f"(n / null)"
        return amr_str
    nodes = mrp_dict['nodes']
    edges = mrp_dict['edges']
    node_list = {info['id']: info.copy() for info in nodes}
    pattern = re.compile(r'''[\s()":/,\\']+''')
    for k in node_list:
        info = node_list[k]
        if 'values' in info:
            info['values'] = [f"\"{s}\"" if pattern.search(s) else s for s in info['values']]
        if 'label' in info:
            s = info['label']
            if not (s[0] == '"' and s[-1] == '"'):
                info['label'] = f"\"{s}\"" if pattern.search(s) else s
    edge_list = {info['id']: [] for info in nodes}
    for info in edges:
        edge_list[info['source']].append(info)
    expanded = set()

    def rec(id: int, prefix_indent: str, prefix_label: str):
        nonlocal amr_str
        amr_str += prefix_indent
        amr_str += prefix_label
        if id in expanded:
            amr_str += f"MRPNode-{id}"
        else:
            info = node_list[id]
            expanded.add(id)
            amr_str += f"(MRPNode-{id} / {info['label']}"
            if 'properties' in info:
                for p, v in zip(info['properties'], info['values']):
                    amr_str += f" :{p} {v}"
            for edge in edge_list[info['id']]:
                amr_str += '\n'
                rec(id=edge['target'],
                    prefix_indent=prefix_indent + ' ' * 6,
                    prefix_label=f":{edge['label']} ")
            amr_str += ')'

    if all_nodes:
        in_deg = {info['id']: 0 for info in nodes}
        for node in edge_list:
            for edge in edge_list[node]:
                in_deg[edge['target']] += 1
        top_list = mrp_dict['tops']
        for k in in_deg:
            if in_deg[k] == 0 and k not in top_list:
                top_list.append(k)
        if len(top_list) == 0 and len(node_list) > 0:
            top_list.append(list(node_list.keys())[0])
        for t in top_list:
            if t not in expanded:
                rec(id=t,
                    prefix_indent='',
                    prefix_label='')
                amr_str += '\n'
    else:
        rec(id=mrp_dict['tops'][0] if 'tops' in mrp_dict and len(mrp_dict['tops']) > 0 else mrp_dict['nodes'][0]['id'],
            prefix_indent='',
            prefix_label='')
    amr_str = amr_str.strip()
    return amr_str


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', '-i', required=True)
    argparser.add_argument('--output', '-o', required=True)
    argparser.add_argument('--not_amr_str_only', action='store_true')
    argparser.add_argument('--all_nodes', action='store_true')
    argparser.add_argument('--read_mode', default='r')
    argparser.add_argument('--write_mode', default='w')
    argparser.add_argument('--encoding', default='utf8')
    args = argparser.parse_args()

    mrp = []
    with open(args.input, args.read_mode, encoding=args.encoding) as f:
        mrp += [mrp_dict_to_amr_str(json.loads(l), not args.not_amr_str_only, args.all_nodes) for l in f]
    with open(args.output, args.write_mode, encoding=args.encoding) as f:
        for s in mrp:
            f.write(''.join(filter(lambda c: c == '\n' or c.isprintable(), s)))
            f.write('\n\n')
