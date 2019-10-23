#!/usr/bin/env python
# transition-eager system.
#
# ** SHIFT
#
#  [stack, s0] [deque] [b0, buffer] A => [stack, s0, deque, b0 ] [] [buffer] A
#
# ** CONFIRM(B)
#
#  [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [B, buffer] A
#
# ** MERGE
#
#  [stack, s0] [deque] [b0, b1, buffer] A => [stack, s0] [deque] [b0_b1, buffer] A
#
# ** ENTITY(E)
#
#  [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [E, buffer] A U {E -name-> name, name -op-> b0}
#
# ** NEWNODE(D)
#
#  [stack, s0] [deque] [B, buffer] A => [stack, s0] [deque] [D, B, buffer] A
#
# ** REDUCE
#
#  [stack, s0] [deque] [b0, buffer] A => [stack] [deque] [b0, buffer] A
#
# ** DROP
#
#  [stack, s0] [deque] [b0, buffer] A => [stack] [deque] [buffer] A
#
# ** CACHE
#
#  [stack, s0] [deque] [b0, buffer] A => [stack] [s0, deque] [b0, buffer] A
#
# ** LEFT(R)
#
#  [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [b0, buffer] A U {b0 -R-> s0}
#
# ** RIGHT(R)
#
#  [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [b0, buffer] A U {s0 -R-> b0}
#
from __future__ import unicode_literals
from system.node import Node, TokenNode, ConceptNode, EntityNode, AttributeNode, coverage_match_alignment
from system.edge import Edge, EdgeSet
from system.misc import parse_date


class State(object):
    def __init__(self, align):
        self.align = align
        self.stack_ = []
        self.deque_ = []
        self.root_ = ConceptNode('_ROOT_', '_ROOT_', '_ROOT_')
        self.buffer_ = [TokenNode(tok, [i]) for i, tok in enumerate(align.tok)] + [self.root_]
        self.arcs_ = EdgeSet(self.root_)
        self.newnode_ = []

    def __str__(self):
        return "[{0}] [{1}] [{2} ...]".format(", ".join([str(sigma) for sigma in self.stack_]),
                                              ", ".join([str(delta) for delta in self.deque_]),
                                              ", ".join([str(beta) for beta in self.buffer_[:1]]))

    def shift(self):
        """

        :return:
        """
        assert self.can_shift()
        for node in self.deque_:
            self.stack_.append(node)
        self.stack_.append(self.buffer_[0])
        self.buffer_ = self.buffer_[1:]
        self.deque_ = []

    def can_shift(self):
        """

        :return:
        """
        return len(self.buffer_) > 0 and isinstance(self.buffer_[0], ConceptNode)

    def confirm(self, concept, level = None):
        """
        confirm is performed on buffer.

        :return:
        """
        assert self.can_confirm()
        coverage = self.buffer_[0].get_coverage()
        self.buffer_[0] = ConceptNode(self.gao(concept), coverage, level)

    def can_confirm(self):
        """

        :return:
        """
        return (len(self.buffer_) > 0
                and (isinstance(self.buffer_[0], TokenNode) or isinstance(self.buffer_[0], EntityNode)))

    def merge(self):
        """

        :return:
        """
        assert self.can_merge()
        b0, b1 = self.buffer_[:2]
        if isinstance(b0, TokenNode):
            self.buffer_[0] = EntityNode(b0, b1)
        else:
            self.buffer_[0].add(b1)
        # pop the top element on the stack.
        self.buffer_ = [self.buffer_[0]] + self.buffer_[2:]

    def can_merge(self):
        """

        :return:
        """
        if len(self.buffer_) < 2:
            return False
        b0, b1 = self.buffer_[:2]
        return isinstance(b1, TokenNode) and (isinstance(b0, TokenNode) or isinstance(b0, EntityNode))

    def entity(self, concept, level, name_level=None):
        """
        pre-condition: stack is not empty
        :param concept:
        :param level:
        :param name_level:
        :return:
        """
        assert self.can_entity()
        if concept == 'date-entity':
            coverage = self.buffer_[0].get_coverage()
            new_node = ConceptNode(concept, coverage, level)
            if isinstance(self.buffer_[0], TokenNode):
                expression = self.buffer_[0].get_name()
            else:
                expression = ' '.join([node.get_name() for node in self.buffer_[0].nodes])
            entry, flags = parse_date(expression)
            for relation, flag in zip(['year', 'month', 'day'], flags):
                if flag:
                    value = getattr(entry, relation)
                    self.add_edge(new_node, relation, AttributeNode(str(value)))
            self.buffer_[0] = new_node
        else:
            coverage = self.buffer_[0].get_coverage()
            if concept == 'name':
                name_concept = ConceptNode('name', coverage, level)
            else:
                name_concept = ConceptNode('name', coverage, name_level)
            if isinstance(self.buffer_[0], TokenNode):
                self.add_edge(name_concept, 'op1', self.buffer_[0])
            elif isinstance(self.buffer_[0], EntityNode):
                for i, node in enumerate(self.buffer_[0].nodes):
                    self.add_edge(name_concept, 'op{0}'.format(i + 1), node)
            elif len(nodes) == 1:
                pass
            else:
                assert False

            if concept == 'name':
                self.buffer_[0] = name_concept
            else:
                new_node = ConceptNode(concept, coverage, level)
                self.add_edge(new_node, 'name', name_concept)
                self.buffer_[0] = new_node

    def can_entity(self):
        """

        :return:
        """
        if len(self.buffer_) == 0:
            return False
        return isinstance(self.buffer_[0], TokenNode) or isinstance(self.buffer_[0], EntityNode)

    def reduce(self):
        """

        :return:
        """
        assert self.can_reduce()
        self.stack_ = self.stack_[:-1]

    def can_reduce(self):
        """

        :return:
        """
        return len(self.stack_) > 0 and isinstance(self.stack_[-1], ConceptNode)

    def drop(self):
        """

        :return:
        """
        assert self.can_drop()
        self.buffer_ = self.buffer_[1:]

    def can_drop(self):
        """

        :return:
        """
        return len(self.buffer_) > 0 and isinstance(self.buffer_[0], TokenNode)

    def cache(self):
        """

        :return:
        """
        assert self.can_cache()
        self.deque_ = [self.stack_[-1]] + self.deque_
        self.stack_ = self.stack_[:-1]

    def can_cache(self):
        """

        :return:
        """
        return len(self.stack_) > 0 and len(self.buffer_) > 0  # only when buffer is not empty, cache is meaningful.

    def left(self, label):
        """

        :param label:
        :return:
        """
        assert self.can_left()
        parent = self.buffer_[0]
        child = self.stack_[-1]
        self.add_edge(parent, label, child)

    def can_left(self):
        """

        :return:
        """
        return (len(self.stack_) > 0
                and len(self.buffer_) > 0
                and isinstance(self.stack_[-1], ConceptNode)
                and isinstance(self.buffer_[0], ConceptNode))

    def right(self, label):
        """

        :param label:
        :return:
        """
        assert self.can_right()
        child = self.buffer_[0]
        parent = self.stack_[-1]
        self.add_edge(parent, label, child)

    def can_right(self):
        """

        :return:
        """
        return (len(self.stack_) > 0
                and len(self.buffer_) > 0
                and isinstance(self.stack_[-1], ConceptNode)
                and isinstance(self.buffer_[0], ConceptNode))

    def newnode(self):
        """

        :return:
        """
        assert self.can_newnode()
        assert isinstance(self.newnode_[-1], ConceptNode)
        self.buffer_ = [self.newnode_[0]] + self.buffer_
        self.newnode_ = self.newnode_[1:]

    def can_newnode(self):
        """

        :return:
        """
        return len(self.buffer_) > 0 and \
                isinstance(self.buffer_[0], ConceptNode) and len(self.newnode_) > 0

    def add_newnode(self, node):
        """

        :param node:
        :return:
        """
        assert isinstance(node, ConceptNode)
        self.newnode_.append(node)

    def add_edge(self, source, relation, target):
        """

        :param source:
        :param relation:
        :param target:
        :return:
        """
        assert isinstance(source, Node)
        assert isinstance(target, Node)
        assert isinstance(relation, str) or isinstance(relation, unicode)
        self.arcs_.add(Edge(source, relation, target))

    def is_terminated(self):
        return len(self.stack_) == 0 and len(self.buffer_) == 0

    def has_edge_by_level(self, parent_level, child_level, relation=None):
        """

        :param parent_level:
        :param child_level:
        :param relation:
        :return:
        """
        for edge in self.arcs_:
            if not (isinstance(edge.source_node, ConceptNode) and isinstance(edge.target_node, ConceptNode)):
                continue
            source_level = edge.source_node.get_level()
            target_level = edge.target_node.get_level()
            if source_level == parent_level and target_level == child_level:
                return True
        return False

    def has_edge_by_name(self, parent_name, child_name):
        """

        :param parent_name:
        :param child_name:
        :return:
        """
        for edge in self.arcs_:
            source_name = edge.source_node.get_name()
            target_name = edge.target_node.get_name()
            if source_name == parent_name and target_name == child_name:
                return True
        return False

    def has_edge_by_alignment(self, parent, child, relation=None):
        """

        :param parent:
        :param child:
        :param relation:
        :return:
        """
        for edge in self.arcs_:
            source_coverage = edge.source_node.get_coverage()
            target_coverage = edge.target_node.get_coverage()
            if coverage_match_alignment(source_coverage, parent) \
                    and (target_coverage is None or  # attribute
                         coverage_match_alignment(target_coverage, child)):
                return True
        return False

    def has_edge_by_name_and_alignment(self, parent_name, parent_align, child_name, child_align, relation=None):
        """

        :param parent_name:
        :param parent_align:
        :param child_name:
        :param child_align:
        :param relation:
        :return:
        """
        for edge in self.arcs_:
            source_name = edge.source_node.get_name()
            source_coverage = edge.source_node.get_coverage()
            target_name = edge.target_node.get_name()
            target_coverage = edge.target_node.get_coverage()

            if parent_name == source_name \
                    and coverage_match_alignment(source_coverage, parent_align) \
                    and child_name == target_name \
                    and (target_coverage is None  # attribute
                         or coverage_match_alignment(target_coverage, child_align)):
                return True
        return False
    
    def gao(self, concept):
        return concept.replace('\"', '-').replace(':', '-').replace('(', '-').replace(')', '-').replace('/', '-')
