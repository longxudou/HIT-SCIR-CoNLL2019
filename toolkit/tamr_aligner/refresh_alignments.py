#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from amr.aligned import AlignmentReader, Alignment


def main():
    cmd = argparse.ArgumentParser('Get the block that contains certain amr graph.')
    cmd.add_argument('-lexicon', help='the path to the alignment file.')
    cmd.add_argument('-data', help='the path to the alignment file.')
    cmd.add_argument('-keep_alignment_in_node', default=False, action='store_true', help='')
    opt = cmd.parse_args()

    lexicon = {}
    for data in open(opt.lexicon, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        assert len(lines) == 2
        lexicon[lines[0].strip()] = lines[1].strip()

    handler = AlignmentReader(opt.data)
    for block in handler:
        graph = Alignment(block)
        new_alignment = lexicon[graph.n]

        graph.alignments = Alignment._parse_alignment([new_alignment])
        graph.refill_alignment()

        for line in block:
            if line.startswith('#'):
                if line.startswith('# ::alignments'):
                    print(new_alignment)
                else:
                    if not opt.keep_alignment_in_node and line.startswith('# ::node'):
                        tokens = line.split()
                        level = tokens[2]
                        alignment = graph.get_node_by_level(level).alignment
                        print('# ::node\t{0}\t{1}\t{2}'.format(
                            tokens[2], tokens[3], '{0}-{1}'.format(alignment[0], alignment[1]) if alignment else ''))
                    else:
                        print(line)

        print(graph.amr_graph, end='\n\n')


if __name__ == "__main__":
    main()
