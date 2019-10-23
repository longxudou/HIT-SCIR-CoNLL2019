#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import codecs
from system.node import TokenNode, EntityNode, ConceptNode
from system.eager.state import State
from amr.aligned import AlignmentReader, Alignment
from smatch.api import SmatchScorer


class Generator(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def parse(self, align, actions):
        state = State(align)
        
        for action in actions:
            if action[0] == 'SHIFT':
                state.shift()
            elif action[0] == 'DROP':
                state.drop()
            elif action[0] == 'REDUCE':
                state.reduce()
            elif action[0] == 'CACHE':
                state.cache()
            elif action[0] == 'MERGE':
                state.merge()
            elif action[0] == 'CONFIRM':
                if action[2] == '_UNK_':
                    action[2] = state.buffer_[0].name
                state.confirm(action[2])
            elif action[0] == 'ENTITY':
                state.entity(action[1], None)
            elif action[0] == 'LEFT':
                state.left(action[1])
            elif action[0] == 'RIGHT':
                state.right(action[1])
            elif action[0] == 'NEWNODE':
                state.add_newnode(ConceptNode(action[1], None, None))
                state.newnode()
            else:
                assert False

        return state


def main():
    cmd = argparse.ArgumentParser(usage='the evaluate script.')
    cmd.add_argument('-gold', help='the path to the gold amr graph.')
    cmd.add_argument('-pred_actions', help='the path to the predicted actions.')
    opt = cmd.parse_args()

    reader = AlignmentReader(opt.gold)
    generator = Generator()
    scorer = SmatchScorer()

    predict_dataset = codecs.open(opt.pred_actions, 'r', encoding='utf-8').read().strip().split('\n\n')
    for block, predict_data in zip(reader, predict_dataset):
        graph = Alignment(block)
        actions = [line.replace('# ::action\t', '').split('\t')
                   for line in predict_data.splitlines() if line.startswith('# ::action')]
        try:
            state = generator.parse(graph, actions)
            predict_amr_graph = str(state.arcs_).encode('utf-8')
        except:
            # print('{0}'.format(graph.n))
            # print('Failed to parse actions:')
            # for action in actions:
            #     print(' - {0}'.format('\t'.join(action).encode('utf-8')))

            # make the predicted graph empty to avoid crash
            predict_amr_graph = '(a / amr-empty)'
        scorer.update(graph.amr_graph, predict_amr_graph)
    print(scorer.f_score())


if __name__ == "__main__":
    main()
