import json
from argparse import ArgumentParser
from typing import List, Tuple


def amr_add_extra(amr: str, output: str, extra: List[Tuple[str, str]] = []):
    with open(amr, 'r', encoding='utf8') as f:
        amr = f.read()
    amr = amr.strip().split('\n\n')
    ex = []
    if extra:
        for k in extra:
            pt = k[1]
            with open(pt, 'r', encoding='utf8') as f:
                pt = f.read()
            pt = {json.loads(s)['id']: s.strip() for s in pt.strip().split('\n')}
            ex.append((k[0], pt))
    print(len(ex))
    for i in range(len(amr)):
        t = amr[i].split()
        for j in range(len(t)):
            if t[j] == '::id':
                eid = t[j + 1]
                break
        for exe in ex:
            print(i, eid)
            amr[i] += f"\n# ::{exe[0]} {exe[1][eid]}"
    with open(output, 'w', encoding='utf8') as f:
        f.write('\n\n'.join(amr))


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', '-i', required=True)
    argparser.add_argument('--output', '-o', required=True)
    argparser.add_argument('--extra', '-e', nargs=2, action='append')
    args = argparser.parse_args()

    amr_add_extra(args.input, args.output, args.extra)
