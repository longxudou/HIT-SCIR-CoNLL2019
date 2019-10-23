#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import sys
import traceback
import argparse
import time
from amr.aligned import Alignment, AlignmentReader
from system.eager.oracle import Oracle
from smatch.api import smatch


def main():
    cmd = argparse.ArgumentParser('Test the program.')
    cmd.add_argument('-mod', default='evaluate', choices=('parse', 'evaluate', 'dump'),
                     help='the running mode. -parse: evaluate the best AMR graph achieved by the alignment '
                          '(specified in ::alignment field) and use the resulted graph to replace the original'
                          'AMR graph; -evaluate: same as parser without replacement; -dump: dump action file.')
    cmd.add_argument('-aligned', help='the path to the filename.')
    cmd.add_argument('-verbose', default=False, action='store_true', help='verbose the actions.')
    opt = cmd.parse_args()

    align_handler = AlignmentReader(opt.aligned)
    parser = Oracle(verbose=opt.verbose)

    for align_block in align_handler:
        graph = Alignment(align_block)
        try:
            actions, state = parser.parse(graph)

            if opt.mod in ('parse', 'evaluate'):
                predicted_amr_graph = str(state.arcs_)
                f_score = smatch(predicted_amr_graph, graph.amr_graph)
                for line in align_block:
                    if line.startswith('# ::alignments'):
                        line = line + ' ::parser eager_oracle.py' \
                                      ' ::smatch {0} ::n_actions {1}'.format(f_score, len(actions))
                    # if line.startswith('('):
                    #     break

                    # do not ignore gold amr string
                    print(line)
                if opt.mod == 'parse':
                    print(str(state.arcs_))
                else:
                    print(graph.amr_graph)
            else:
                print('# ::id {0}'.format(graph.n))
                for line in align_block:
                    if line.startswith('# ::tok') or line.startswith('# ::pos') or line.startswith('('):
                        print(line)
                print('\n'.join(['# ::action {0}'.format(action) for action in actions]))
            print()

            if opt.verbose:
                print(graph.n, file=sys.stderr)
                print('\n'.join(actions), file=sys.stderr, end='\n\n')
        except Exception:
            print(graph.n, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
