#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import sys
import itertools
import datetime
import codecs
from amr.aligned import Alignment, AlignmentReader
from smatch.api import smatch
from system.eager.oracle import Oracle
from rule_based_aligner.matcher import *
from rule_based_aligner.updater import *
from rule_based_aligner.stemmer import Stemmer
from rule_based_aligner.aligned_results import AlignedResults


def number_of_enumerate_alignment(align_results):
    n_test = 1
    for level in align_results.levels_to_spans:
        n_test *= len(align_results.levels_to_spans[level])
    return n_test


def enumerate_alignment(align_results):
    items = []
    for level in align_results.levels_to_spans:
        item = [(level, dependent, start, end) for start, end, dependent in align_results.levels_to_spans[level]]
        items.append(item)
    for alignment in itertools.product(*items):
        payload = {}
        for level, dependent, start, end in alignment:
            payload[level] = start, end
        legal = True
        for level, dependent, start, end in alignment:
            if dependent is None:
                continue
            if dependent not in payload or payload[dependent] != (start, end):
                legal = False
                break
        if legal:
            yield alignment


def dump_unaligned_records(unaligned_records):
    unaligned_names = {}
    print('Unaligned records:', file=sys.stderr)
    for unaligned_record in unaligned_records:
        print(unaligned_record[0], unaligned_record[1], file=sys.stderr)
        for level, name in unaligned_record[1]:
            if name not in unaligned_names:
                unaligned_names[name] = 0
            unaligned_names[name] += 1
    print(file=sys.stderr)
    print('Unaligned names:', file=sys.stderr)
    for name, freq in sorted(unaligned_names.items(), key=lambda n: n[1], reverse=True):
        print('{0}: {1}'.format(name, freq), file=sys.stderr)


def fill_alignment(graph, alignment):
    graph.remove_alignment()
    for level, dependent, start, end in alignment:
        node = graph.get_node_by_level(level)
        node.alignment = start, end
    graph.remove_intersection()


def alignment_string(graph):
    alignment = {}
    for node in graph.true_nodes():
        if node.alignment is None:
            continue
        start, end = node.alignment
        if (start, end) not in alignment:
            alignment[start, end] = []
        alignment[start, end].append(node.level)
    return ' '.join(['{start}-{end}|{levels}'.format(start=span[0], end=span[1], levels='+'.join(levels))
                     for span, levels in sorted(alignment.items())])


def align(opt):
    reader = AlignmentReader(opt.data)
    stemmer = Stemmer()
    matchers = [WordMatcher(),
                FuzzyWordMatcher(),
                FuzzySpanMatcher(),
                NamedEntityMatcher(),
                FuzzyNamedEntityMatcher(),
                DateEntityMatcher(),
                URLEntityMatcher(),
                OrdinalEntityMatcher(),
                MinusPolarityMatcher(),
                BelocatedAtMatcher(),
                TemporalQuantityMatcher()]

    if opt.morpho_match:
        matchers.append(MorphosemanticLinkMatcher())

    if opt.semantic_match:
        matchers.append(SemanticWordMatcher(lower=not opt.cased))
        matchers.append(SemanticNamedEntityMatcher(lower=not opt.cased))

    updaters = [EntityTypeUpdater(),
                PersonOfUpdater(),
                QuantityUpdater(),
                PersonUpdater(),
                MinusPolarityPrefixUpdater(),
                RelativePositionUpdater(),
                DegreeUpdater(),
                HaveOrgRoleUpdater(),
                GovernmentOrganizationUpdater(),
                CauseUpdater(),
                ImperativeUpdater(),
                PossibleUpdater()]

    unaligned_records = []
    oracle = Oracle(verbose=False)
    fpo = codecs.open(opt.output, 'w', encoding='utf-8') if opt.output else sys.stdout
    for block in reader:
        graph = Alignment(block)
        if opt.verbose:
            print('Aligning {0}'.format(graph.n), file=sys.stderr)

        best_alignment = [(n.level, None, n.alignment[0], n.alignment[1]) for n in graph.true_nodes() if n.alignment]
        actions, states = oracle.parse(graph)
        pred_amr_graph = str(states.arcs_)
        baseline_f_score, baseline_n_actions = best_f_score, best_n_actions = \
            smatch(graph.amr_graph, pred_amr_graph), len(actions)

        words = graph.tok
        postags = graph.pos if hasattr(graph, 'pos') else [None for _ in range(len(words))]
        stemmed_words = [stemmer.stem(word, postag) for word, postag in zip(words, postags)]

        results = AlignedResults()
        for matcher in matchers:
            matcher.match(words, stemmed_words, postags, graph, results)
        added = True
        while added:
            added = False
            for updater in updaters:
                added = added or updater.update(words, graph, results)

        unaligned = [(n.level, n.name) for n in graph.true_nodes() if n.level not in results.levels_to_spans]
        if len(unaligned) > 0:
            unaligned_records.append((graph.n, unaligned))

        if opt.report_only:
            continue

        n_test = number_of_enumerate_alignment(results)
        if opt.verbose:
            print(' - Going to enumerate {0}'.format(n_test), file=sys.stderr)
        if not opt.improve_perfect and baseline_f_score == 1.:
            print(' - Best already achieved.', file=sys.stderr)
        elif n_test > opt.trials:
            print(' - Too many test!', file=sys.stderr)
        else:
            for alignment in enumerate_alignment(results):
                fill_alignment(graph, alignment)

                actions, states = oracle.parse(graph)
                pred_amr_graph = str(states.arcs_)
                pred_f_score, pred_n_actions = smatch(graph.amr_graph, pred_amr_graph), len(actions)
                if pred_f_score > best_f_score or \
                        (pred_f_score == best_f_score and pred_n_actions < best_n_actions):
                    best_f_score = pred_f_score
                    best_n_actions = pred_n_actions
                    best_alignment = alignment[:]

        if opt.verbose:
            if best_f_score > baseline_f_score or \
                    (best_f_score == baseline_f_score and best_n_actions < baseline_n_actions):
                print(' - Better achieved!'.format(graph.n), file=sys.stderr)
            else:
                print(' - Stay the same.'.format(graph.n), file=sys.stderr)

        fill_alignment(graph, best_alignment)
        output = alignment_string(graph)
        now = datetime.datetime.now()
        output = '# ::alignments {0} ::annotator aligner3.py ::date {1} ::parser {2} ::smatch {3} ' \
                 '::n_actions {4}'.format(output, now, oracle.name, best_f_score, best_n_actions)
        if not opt.show_all:
            print(graph.n, file=fpo)
            print(output, end='\n\n', file=fpo)
        else:
            block = graph.block
            for line in block:
                if line.startswith("#"):
                    if line.startswith('# ::alignments'):
                        print(output, file=fpo)
                    elif line.startswith('# ::node'):
                        tokens = line.split()
                        level = tokens[2]
                        alignment = graph.get_node_by_level(level).alignment
                        print('# ::node\t{0}\t{1}\t{2}'.format(
                            tokens[2], tokens[3], '{0}-{1}'.format(alignment[0], alignment[1]) if alignment else ''),
                            file=fpo)
                    else:
                        print(line, file=fpo)
                else:
                    print(graph.amr_graph, file=fpo, end='\n\n')
                    break

    dump_unaligned_records(unaligned_records)


def exclusively_align(opt):
    reader = AlignmentReader(opt.data)
    stemmer = Stemmer()
    matchers = [WordMatcher(),
                FuzzyWordMatcher(),
                FuzzySpanMatcher(),
                NamedEntityMatcher(),
                FuzzyNamedEntityMatcher(),
                DateEntityMatcher(),
                URLEntityMatcher(),
                OrdinalEntityMatcher(),
                MinusPolarityMatcher(),
                BelocatedAtMatcher(),
                TemporalQuantityMatcher()]

    if opt.morpho_match:
        matchers.append(MorphosemanticLinkMatcher())

    if opt.semantic_match:
        matchers.append(SemanticWordMatcher(lower=not opt.cased))
        matchers.append(SemanticNamedEntityMatcher(lower=not opt.cased))

    updaters = [EntityTypeUpdater(),
                PersonOfUpdater(),
                QuantityUpdater(),
                PersonUpdater(),
                MinusPolarityPrefixUpdater(),
                RelativePositionUpdater(),
                DegreeUpdater(),
                HaveOrgRoleUpdater(),
                GovernmentOrganizationUpdater(),
                CauseUpdater(),
                ImperativeUpdater(),
                PossibleUpdater()]

    unaligned_records = []
    fpo = codecs.open(opt.output, 'w', encoding='utf-8') if opt.output else sys.stdout
    for block in reader:
        graph = Alignment(block)
        if opt.verbose:
            print('Aligning {0}'.format(graph.n), file=sys.stderr)

        words = graph.tok
        postags = graph.pos if hasattr(graph, 'pos') else [None for _ in range(len(words))]
        stemmed_words = [stemmer.stem(word, postag) for word, postag in zip(words, postags)]

        results = AlignedResults(multiple=False)
        for matcher in matchers:
            matcher.match(words, stemmed_words, postags, graph, results)
        for updater in updaters:
            updater.update(words, graph, results)

        unaligned = [(n.level, n.name) for n in graph.true_nodes() if n.level not in results.levels_to_spans]
        if len(unaligned) > 0:
            unaligned_records.append((graph.n, unaligned))

        if opt.report_only:
            continue

        n_test = number_of_enumerate_alignment(results)
        assert n_test == 1

        for alignment in enumerate_alignment(results):
            fill_alignment(graph, alignment)
            break

        output = alignment_string(graph)
        now = datetime.datetime.now()
        output = '# ::alignments {0} ::annotator aligner_v0.py ::date {1}'.format(output, now)
        if not opt.show_all:
            print(graph.n, file=fpo)
            print(output, end='\n\n', file=fpo)
        else:
            block = graph.block
            for line in block:
                if line.startswith("#"):
                    if line.startswith('# ::alignments'):
                        print(output, file=fpo)
                    elif line.startswith('# ::node'):
                        tokens = line.split()
                        level = tokens[2]
                        alignment = graph.get_node_by_level(level).alignment
                        print('# ::node\t{0}\t{1}\t{2}'.format(
                            tokens[2], tokens[3], '{0}-{1}'.format(alignment[0], alignment[1]) if alignment else ''),
                            file=fpo)
                    else:
                        print(line, file=fpo)
                else:
                    print(graph.amr_graph, file=fpo, end='\n\n')
                    break


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('-version', default='v1', help='the version of aligner')
    cmd.add_argument('-data', help='the path to the training data.')
    cmd.add_argument('-output', help='the path to the output file.')
    cmd.add_argument('-wordvec', help='the path to the word vector')
    cmd.add_argument('-cased', type=bool, default=False, help='use to specify if the word embedding is cased.')
    cmd.add_argument('-binary', default=False, action='store_true', help='use binary wordvec')
    cmd.add_argument('-verbose', default=False, action='store_true', help='verbose output')
    cmd.add_argument('-report_only', default=False, action='store_true', help='only report')
    cmd.add_argument('-trials', default=10000, type=int, help='the number of trials.')
    cmd.add_argument('-improve_perfect', default=False, action='store_true',
                     help='try new alignment even with the baseline achieving an smatch of 1.0. This option '
                          'is recommended.')
    cmd.add_argument('-morpho_match', default=False, action='store_true',
                     help='use to specify word match with morphosemantic')
    cmd.add_argument('-semantic_match', default=False, action='store_true',
                     help='use to specify word match with embeddings')
    cmd.add_argument('-show_all', default=False, action='store_true', help='show all the comments and amr graph.')
    opt = cmd.parse_args()

    init_word2vec(opt.wordvec, opt.binary)
    print('WordVec is loaded.', file=sys.stderr)
    if opt.version == 'v1':
        align(opt)
    else:
        exclusively_align(opt)


if __name__ == "__main__":
    main()
