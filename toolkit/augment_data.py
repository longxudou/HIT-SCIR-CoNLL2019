import json
import collections
import argparse


parser = argparse.ArgumentParser(description='Augment Data')
parser.add_argument("conll", type=str, help="Augment CoNLL file")
parser.add_argument("mrp", type=str, help="Input MRP file")
parser.add_argument("output", type=str, help="Output Augmented file")
args = parser.parse_args()

conll_file = args.conll
mrp_file = args.mrp
out_file = args.output

augs = {}
with open(conll_file, 'r', encoding='utf8') as f_c:
  conlls = f_c.read().split('\n\n')
  for conll in conlls:
    id = conll.split('\n')[0][1:]
    augs[id] = [line.split('\t') for line in conll.strip().split('\n')[1:]]
  #print augs.keys()
with open(mrp_file, 'r', encoding='utf8') as f_m, open(out_file, 'w', encoding='utf8') as fo:
  line = f_m.readline()
  while line:
    mrp = json.loads(line, object_pairs_hook=collections.OrderedDict)
    id = mrp['id']
    if id not in augs:
      print("id:{} not in companion".format(id))
    else:
      mrp['companion'] = augs[id]
      fo.write((json.dumps(mrp)+'\n'))
    line = f_m.readline()
