#!/usr/bin/env python
from __future__ import unicode_literals
from system.node import TokenNode, AttributeNode


class Edge(object):
    def __init__(self, source_node, relation, target_node):
        self.source_node = source_node
        self.relation = relation
        self.target_node = target_node


class EdgeSet(set):
    def __init__(self, top):
        super(EdgeSet, self).__init__()
        self.top = top

    def _traverse_print(self, root, variables, shown, in_const_edge):
        children = []
        for edge in self.__iter__():
            if edge.source_node == root:
                children.append((edge.relation, edge.target_node))
        children.sort(key=lambda x: (x[0], x[1].name))

        if len(children) == 0:
            if not shown[root]:
                shown[root] = True
                if isinstance(root, TokenNode):
                    ret = '"{0}"'.format(self._normalize_entity_token(root.name))
                elif isinstance(root, AttributeNode) or (in_const_edge and self._is_attribute(root.name)):
                    ret = root.name
                else:
                    ret = '({0} / {1})'.format(variables[root], root.name)
            else:
                ret = variables[root]
        else:
            if shown[root]:
                ret = '{0}'.format(variables[root])
            else:
                shown[root] = True
                unnamed_concept = in_const_edge and self._is_attribute(root.name)
                if unnamed_concept:
                    ret = root.name
                else:
                    ret = '({0} / {1}'.format(variables[root], root.name)
                for relation, child in children:
                    ret += ' :{0} {1}'.format(relation, self._traverse_print(child, variables, shown,
                                                                             self._is_const_relation(relation)))
                if not unnamed_concept:
                    ret += ')'
        return ret

    def _get_size(self, root, visited, covered=set()):
        if root in visited or root in covered:
            return 1
        visited.add(root)
        tree_size = 0
        children = []
        for edge in self.__iter__():
            if edge.source_node == root:
                children.append(edge.target_node)
        for child in children:
            tree_size += self._get_size(child, visited, covered) + 1
        return tree_size + 1

    def _print(self):
        roots = self._get_roots()
        variables = self._get_variables
        shown = {node: False for node in variables}
        if len(roots) == 1:
            return self._traverse_print(roots[0], variables, shown, False)
        elif len(roots) > 1:
            # return self._traverse_print(self.top, variables, shown)
            new_root = roots[0]
            for i, root in enumerate(roots[1:]):
                self.add(Edge(new_root, 'TOP{0}'.format(i), root))
            return self._traverse_print(new_root, variables, shown, False)
        else:
            return '(a / amr-empty)'

    def __str__(self):
        return self._print()

    def _get_roots(self):
        covered = set()
        for node in self._get_variables:
            if node.name == '_ROOT_':
                covered.add(node)

        ret = []
        for edge in self.__iter__():
            if edge.source_node == self.top:
                ret.append(edge.target_node)
                self._get_size(edge.target_node, covered)

        while True:
            max_sz = 0
            max_node = None
            for node in self._get_variables:
                if node not in covered:
                    visited = set()
                    sz = self._get_size(node, visited, covered)
                    if sz > max_sz:
                        max_node = node
                        max_sz = sz
            if max_node is None:
                break
            ret.append(max_node)
            visited = set()
            sz = self._get_size(max_node, covered)

        assert len(covered) == len(self._get_variables)
        return ret

    @property
    def _get_variables(self):
        nodes = set()
        for edge in self.__iter__():
            nodes.add(edge.source_node)
            nodes.add(edge.target_node)
        nodes = list(nodes)
        nodes.sort(key=lambda x: x.name)

        variables = {}
        variable_name_counts = {}
        for node in nodes:
            shortname = self._shortname(node.name)
            if shortname not in variable_name_counts:
                variable_name_counts[shortname] = 0
            variable_name_counts[shortname] += 1
            count = variable_name_counts[shortname]
            variables[node] = shortname if count == 1 else (shortname + str(count))
        return variables

    @staticmethod
    def _normalize_entity_token(token):
        if token == '"':
            return '_QUOTE_'
        return token

    @staticmethod
    def _shortname(token):
        if token[0] == '"':
            return token[1] if len(token) > 1 else 'q'
        return token[0]

    @staticmethod
    def _is_attribute(name):
        if name in ('-', 'imperative'):  # polarity
            return True
        return name.isdigit()

    @staticmethod
    def _is_const_relation(relation):
        if relation.startswith('op') or \
            relation in ('month', 'decade', 'polarity', 'day', 'quarter', 'year', 'era', 'century',
                         'timezone', 'polite', 'mode', 'value', 'quant', 'unit', 'range', 'scale'):
            return True
        return False
