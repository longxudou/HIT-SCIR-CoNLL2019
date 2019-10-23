import json
import collections
import argparse
import copy


def ud2companion(ud):
    lines = []
    for node in ud["nodes"]:
        if len(node["anchors"]) > 1:
            print("Multi-anchor in ud: {} \n (node: {})".format(ud, node["id"]))
        # id, token, lemma, upos, xpos, '_', head, label, '_', range
        line = [str(node["id"] + 1), node["label"], node["values"][0], node["values"][1],
                node["values"][2], '_', '_', '_', '_', 'TokenRange=' + str(node["anchors"][0]["from"])
                + ':' + str(node["anchors"][0]["to"])]
        lines.append(line)
    # print (lines)
    if "edges" in ud:
        for edge in ud["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            # print (src, tgt)
            if lines[tgt][6] is not '_':
                print("Multi-head in ud : {} \n (node: {})".format(ud, tgt))
                exit()
            lines[tgt][6] = str(src + 1)
            lines[tgt][7] = edge["label"]
    # for line in lines:
    # if line[6] is '_' or line[7] is '_':
    #	print ("No-head in ud: {} \n (node: {})".format(ud, line[0]))
    #	exit()
    ud["companion"] = lines
    del ud["nodes"]
    if "edges" in ud:
        del ud["edges"]
    # print (ud)
    return ud


parser = argparse.ArgumentParser(description="Preprocess evaluation files");
parser.add_argument("udpipe", type=str, help="udpipe input file");
parser.add_argument("input", type=str, help="mrp input file (contains predicting targets)");
parser.add_argument("--outdir", type=str, help="output dir");
args = parser.parse_args();

outputs = {"dm": [], "psd": [], "eds": [], "ucca": [], "amr": []}
flavor = {"dm": 0, "psd": 0, "eds": 1, "ucca": 1, "amr": 2}

with open(args.udpipe, 'r', encoding='utf8') as fi_ud, open(args.input, 'r', encoding='utf8') as fi_input:
    line_ud = fi_ud.readline().strip()
    line_input = fi_input.readline().strip()
    while line_ud and line_input:
        input = json.loads(line_input, object_pairs_hook=collections.OrderedDict)
        ud = json.loads(line_ud, object_pairs_hook=collections.OrderedDict)
        assert input["id"] == ud["id"]
        mrp = ud2companion(ud)
        targets = input["targets"]
        for target in targets:
            data = copy.deepcopy(mrp)
            data["framework"] = target
            data["flavor"] = flavor[target]
            outputs[target].append(data)
        line_ud = fi_ud.readline().strip()
        line_input = fi_input.readline().strip()
print("Number of each framework:\n dm:{}, psd:{}, eds:{}, ucca:{}, amr:{}".format(
    len(outputs["dm"]), len(outputs["psd"]), len(outputs["eds"]), len(outputs["ucca"]), len(outputs["amr"])))

if args.outdir is not None:
    for type in outputs.keys():
        file = args.outdir + '/' + type + '.mrp'
        with open(file, 'wb') as fo:
            for mrp in outputs[type]:
                fo.write((json.dumps(mrp) + '\n').encode('utf-8'))
