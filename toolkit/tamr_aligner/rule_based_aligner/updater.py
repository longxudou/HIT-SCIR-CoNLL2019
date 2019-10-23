#!/usr/bin/env python
from __future__ import unicode_literals
from __future__ import absolute_import
from .stemmer import Stemmer


class Updater(object):
    def __init__(self):
        pass

    def update(self, words, graph, align_results):
        """

        :param words: list[str]
        :param graph: Alignment
        :param align_results: AlignedResults
        :return:
        """
        raise NotImplemented


class EntityTypeUpdater(Updater):
    def __init__(self):
        super(EntityTypeUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if graph.is_entity(node, consider_alignment=False):
                # get the :name node
                edges = list(filter(lambda e: e.relation == 'name', graph.edges_by_parents[node.level]))
                if len(edges) > 0:
                    for start, end in align_results.get_spans_by_level(edges[0].tgt_level):
                        updated = updated or align_results.add(start, end, node.level, edges[0].tgt_level)
        return updated


class QuantityUpdater(Updater):
    def __init__(self):
        super(QuantityUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if not node.name.endswith('-quantity') or node.level not in graph.edges_by_parents:
                continue
            edges = list(filter(lambda e: e.relation == 'unit', graph.edges_by_parents[node.level]))
            if len(edges) > 0:
                for start, end in align_results.get_spans_by_level(edges[0].tgt_level):
                    updated = updated or align_results.add(start, end, node.level, edges[0].tgt_level)
        return updated


class PersonOfUpdater(Updater):
    def __init__(self):
        super(PersonOfUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if node.name not in ('person', 'thing') or node.level not in graph.edges_by_parents:
                continue
            edges = list(filter(lambda e: e.relation.endswith('-of'), graph.edges_by_parents[node.level]))
            if len(edges) > 0:
                for start, end in align_results.get_spans_by_level(edges[0].tgt_level):
                    updated = updated or align_results.add(start, end, node.level, edges[0].tgt_level)
        return updated


class PersonUpdater(Updater):
    def __init__(self):
        super(PersonUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if node.name != 'person' or node.level not in graph.edges_by_parents:
                continue
            edges = graph.edges_by_parents[node.level]
            if len(edges) == 1:
                for start, end in align_results.get_spans_by_level(edges[0].tgt_level):
                    updated = updated or align_results.add(start, end, node.level, edges[0].tgt_level)
        return updated


class GovernmentOrganizationUpdater(Updater):
    def __init__(self):
        super(GovernmentOrganizationUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if not edge.relation.endswith('-of') or \
                    not edge.relation.startswith('ARG') or \
                    edge.src_name != 'government-organization':
                continue
            for start, end in align_results.get_spans_by_level(edge.tgt_level):
                updated = updated or align_results.add(start, end, edge.src_level, edge.tgt_level)
        return updated


class RelativePositionUpdater(Updater):
    def __init__(self):
        super(RelativePositionUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if edge.src_name != 'relative-position':
                continue
            for start, end in align_results.get_spans_by_level(edge.tgt_level):
                updated = updated or align_results.add(start, end, edge.src_level, edge.tgt_level)
        return updated


class MinusPolarityPrefixUpdater(Updater):
    def __init__(self):
        super(MinusPolarityPrefixUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if node.name != '-':
                continue
            edges = graph.edges_by_children[node.level]
            if len(edges) == 1 and edges[0].relation == 'polarity':
                for start, end in align_results.get_spans_by_level(edges[0].src_level):
                    if start + 1 == end and (words[start][:2] in Stemmer.kMinusPrefix2 or
                                             words[start][:3] in Stemmer.kMinusPrefix3 or
                                             words[start].endswith('less') or
                                             words[start].endswith('nt') or
                                             words[start].endswith('n\'t')):
                        updated = updated or align_results.add(start, end, node.level, edges[0].src_level)
        return updated


class DegreeUpdater(Updater):
    def __init__(self):
        super(DegreeUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if edge.relation != 'degree':
                continue
            for start, end in align_results.get_spans_by_level(edge.src_level):
                if start + 1 == end and (words[start].endswith('est') or words[start].endswith('er')):
                    updated = updated or align_results.add(start, end, edge.tgt_level, edge.src_level)
        return updated


class HaveOrgRoleUpdater(Updater):
    def __init__(self):
        super(HaveOrgRoleUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for node in graph.true_nodes():
            if node.name not in ('have-org-role-91', 'have-rel-role-91') or node.level not in graph.edges_by_parents:
                continue
            edges = [edge for edge in graph.edges_by_parents[node.level] if edge.relation in ('ARG1', 'ARG2')]
            if len(edges) == 1:
                edge = edges[0]
            elif len(edges) == 2:
                edge = edges[0] if edges[0].relation == 'ARG2' else edges[1]
            else:
                continue
            for start, end in align_results.get_spans_by_level(edge.tgt_level):
                updated = updated or align_results.add(start, end, edge.src_level, edge.tgt_level)
        return updated


class CauseUpdater(Updater):
    def __init__(self):
        super(CauseUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if edge.tgt_name != 'cause-01' or not edge.relation.startswith('ARG') or not edge.relation.endswith('-of'):
                continue
            for start, end in align_results.get_spans_by_level(edge.src_level):
                if start + 1 == end:
                    updated = updated or align_results.add(start, end, edge.tgt_level, edge.src_level)
        return updated


class ImperativeUpdater(Updater):
    def __init__(self):
        super(ImperativeUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if edge.tgt_name != 'imperative' or edge.relation != 'mode':
                continue
            you_level = [e.tgt_level for e in graph.edges_by_parents[edge.src_level] if e.tgt_name == 'you']
            for start, end in align_results.get_spans_by_level(edge.src_level):
                if start + 1 == end:
                    updated = updated or align_results.add(start, end, edge.tgt_level, edge.src_level)
                if len(you_level) == 1:
                    updated = updated or align_results.add(start, end, you_level[0], edge.src_level)
        return updated


class PossibleUpdater(Updater):
    def __init__(self):
        super(PossibleUpdater, self).__init__()

    def update(self, words, graph, align_results):
        updated = False
        for edge in graph.edges:
            if edge.src_name == 'possible' and edge.relation == 'domain':
                # operable => (p / possible :domain (o / operate))
                for start, end in align_results.get_spans_by_level(edge.tgt_level):
                    if start + 1 == end and words[start].endswith('ble'):
                        updated = updated or align_results.add(start, end, edge.src_level, edge.tgt_level)
            elif edge.tgt_name == 'possible' and edge.relation == 'mod':
                for start, end in align_results.get_spans_by_level(edge.src_level):
                    if start + 1 == end and words[start].endswith('ble'):
                        updated = updated or align_results.add(start, end, edge.tgt_level, edge.src_level)
        return updated


__all__ = [
    'EntityTypeUpdater',
    'QuantityUpdater',
    'PersonOfUpdater',
    'PersonUpdater',
    'RelativePositionUpdater',
    'MinusPolarityPrefixUpdater',
    'DegreeUpdater',
    'HaveOrgRoleUpdater',
    'GovernmentOrganizationUpdater',
    'CauseUpdater',
    'ImperativeUpdater',
    'PossibleUpdater'
]