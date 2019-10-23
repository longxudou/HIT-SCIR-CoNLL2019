#!/usr/bin/env python
from __future__ import unicode_literals


class Node(object):
    def __init__(self, name, type_name, coverage):
        self.name = name
        self.type_name = type_name
        self.coverage = coverage

    def get_coverage(self):
        return self.coverage

    def get_name(self):
        return self.name

    def get_type(self):
        return self.type_name


class TokenNode(Node):
    def __init__(self, name, coverage):
        super(TokenNode, self).__init__(name, 'token', coverage)

    def __str__(self):
        return '"{0}"'.format(self.name)


class EntityNode(Node):
    def __init__(self, node1, node2):
        self.nodes = [node1, node2]
        coverage = node1.get_coverage() + node2.get_coverage()
        name = '{0}'.format('_'.join([node1.name, node2.name]))
        super(EntityNode, self).__init__(name, 'entity', coverage)

    def add(self, node):
        self.nodes.append(node)
        self.coverage = self.coverage + node.get_coverage()
        self.name = self.name + '_{0}'.format(node.name)

    def __str__(self):
        return '"{0}"'.format(self.name)


class ConceptNode(Node):
    def __init__(self, name, coverage, level=None):
        super(ConceptNode, self).__init__(name, 'concept', coverage)
        self.level = level

    def get_level(self):
        return self.level
        
    def __str__(self):
        return self.name


class AttributeNode(Node):
    def __init__(self, value):
        super(AttributeNode, self).__init__(value, 'attribute', None)

    def __str__(self):
        return '={0}'.format(self.name)


def coverage_match_alignment(coverage, align):
    assert isinstance(coverage, list)
    if len(coverage) == 1:
        return align[0] == coverage[0] and align[1] == align[0] + 1
    else:
        return align[0] == coverage[0] and align[1] == coverage[-1] + 1
