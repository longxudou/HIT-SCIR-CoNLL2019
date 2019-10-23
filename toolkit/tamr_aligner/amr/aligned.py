#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import sys
import codecs
import penman


class AlignmentReader(object):
    def __init__(self, filename):
        """

        :param filename: str, the path to the filename
        """
        self.handler = codecs.open(filename, 'r', encoding='utf-8')

    def __iter__(self):
        block = []
        for line in self.handler:
            line = line.strip()
            if len(line) == 0:
                if len(block) > 1 or not block[0].startswith('# AMR release;'):
                    yield block
                block = []
            else:
                block.append(line)
        if len(block) > 0:
            yield block


class _Node(object):
    def __init__(self, level, name, alignment):
        assert alignment is None or (isinstance(alignment, tuple) and len(alignment) == 2)
        self.level = level
        self.name = name
        self.alignment = alignment

    def same_alignment(self, other, nil_as=True):
        assert isinstance(other, _Node)
        if self.alignment is None:
            return nil_as
        return self.alignment[0] == other.alignment[0] and self.alignment[1] == other.alignment[1]

    def __repr__(self):
        return '|{0}, {1}, {2}|'.format(self.level, self.name, self.alignment)

    def __str__(self):
        return '|{0}, {1}, {2}|'.format(self.level, self.name, self.alignment)


class _Edge(object):
    def __init__(self, src_name, src_level, relation, tgt_name, tgt_level):
        self.src_name = src_name
        self.src_level = src_level
        self.relation = relation
        self.tgt_name = tgt_name
        self.tgt_level = tgt_level

    def __str__(self):
        return '{0}({1}) -{2}-> {3}({4})'.format(self.src_name, self.src_level, self.relation,
                                                 self.tgt_name, self.tgt_level)


class _Alignment(object):
    def __init__(self, line):
        fields = line.split()
        key = None
        raw_mapping = {}
        for field in fields:
            if field.startswith('#'):
                continue
            if field.startswith('::'):
                key = field[2:]
                raw_mapping[key] = []
            else:
                assert key is not None
                raw_mapping[key].append(field)
        self.raw_mapping = raw_mapping
        self.aligner = raw_mapping.get('aligner', None)
        self.date = raw_mapping.get('date', None)
        self.parser = raw_mapping.get('parser', None)
        self.smatch = float(raw_mapping.get('smatch', ['0'])[0])
        self.gold = 'gold' in raw_mapping
        self.alignments = {}
        for entry in self.raw_mapping['alignments']:
            range_, levels = entry.strip('*').split('|')
            start, end = map(int, range_.split('-'))
            for level in levels.split('+'):
                self.alignments[level] = start, end


class Alignment(object):
    kDateEntityRelations = ('day', 'month', 'year', 'decade', 'time-of',
                            'weekday', 'dayperiod', 'season', 'time',
                            'timezone', 'calendar', 'century', 'quarter',
                            'era')
    kRoot = '_ROOT_'
    kRootAlignment = (-1, -1)

    def __init__(self, block):
        """

        :param block: list of str
        """
        self.block = block
        self.n = self._parse_id(block)
        self.snt = self._parse_key(block, 'snt')
        self.tok = self._parse_key(block, 'tok')
        self.alignments = self._parse_alignment(block)
        self.nodes, self.nodes_by_levels = self._get_nodes(block)
        self.remove_intersection()
        self.edges, self.edges_by_parents, self.edges_by_children = self._get_edges(block)
        self.root_level, self.root_name = self._get_root()

        self.graph, self.attributes = self._get_graph(block)
        self.amr_graph = self._get_amr(block)

        self.nodes.append(_Node(level=self.kRoot, name=self.kRoot, alignment=self.kRootAlignment))
        self.nodes_by_levels[self.nodes[-1].level] = self.nodes[-1]
        if self.root_level is not None:
            root_edge = _Edge(src_name=self.kRoot, src_level=self.kRoot, relation=self.kRoot,
                              tgt_name=self.root_name, tgt_level=self.root_level)
            self.edges.append(root_edge)
            self.edges_by_parents[self.kRoot] = [root_edge]
            self.edges_by_children[self.root_level] = [root_edge]

    @staticmethod
    def _parse_id(block):
        ret = None
        for line in block:
            if line.startswith('# ::id'):
                ret = line.split()[2]
        assert ret is not None
        return ret

    @staticmethod
    def _parse_key(block, key):
        ret = None
        for line in block:
            if line.startswith('# ::{0}'.format(key)):
                assert ret is None
                ret = line.split('::{0}'.format(key), 1)[1].split()
        assert ret is not None
        return ret

    @staticmethod
    def _parse_alignment(block):
        ret = None
        for line in block:
            if line.startswith('# ::alignments'):
                ret = _Alignment(line)
        assert ret is not None
        return ret

    @staticmethod
    def _get_amr(block):
        ret = ''
        for line in block:
            if line[0] == '(' or ret != '':
                ret += line + ' '
        return ret.strip()

    @staticmethod
    def _parse_action(block):
        ret = []
        for line in block:
            if line.startswith('# ::action'):
                ret.append(line.split('::action', 1)[1].split())
        return ret

    @staticmethod
    def _get_nodes(block):
        """
        e.g.:  # ::node 0.0.0.0 "Estonia" 0-1
        :param block:
        :return:
        """
        raw = []
        levels = {}
        for line in block:
            if not line.startswith('# ::node'):
                continue
            fields = line.split('::node', 1)[1].strip().split('\t')
            if len(fields) == 3:
                level, fullname, alignment = fields
                alignment = tuple(map(int, alignment.split('-', 1)))
            else:
                level, fullname = fields
                alignment = None
            node = _Node(level=level, name=fullname, alignment=alignment)
            raw.append(node)
            assert level not in levels
            levels[level] = node

        return raw, levels

    @staticmethod
    def _get_edges(block):
        """
        e.g.: # ::edge name op1 "Estonia" 0.0.0 0.0.0.0
        :param block:
        :return:
        """
        raw = []
        children = {}
        parents = {}
        for line in block:
            if not line.startswith('# ::edge'):
                continue
            fields = line.split('::edge', 1)[1].strip().split('\t')
            assert len(fields) == 5
            parent_name, relation, child_name, parent_level, child_level = fields

            edge = _Edge(src_name=parent_name, src_level=parent_level, relation=relation,
                         tgt_name=child_name, tgt_level=child_level)
            raw.append(edge)
            if child_level not in children:
                children[child_level] = []
            children[child_level].append(edge)
            if parent_level not in parents:
                parents[parent_level] = []
            parents[parent_level].append(edge)
        return raw, parents, children

    def _get_root(self):
        """

        :return:
        """
        for node in self.nodes:
            if node.alignment is not None and node.level == '0':
                return node.level, node.name
        all_nodes = [node for node in self.nodes if node.alignment is not None]
        ret_level, ret_name = None, None
        for node in all_nodes:
            if node.name == Alignment.kRoot:
                continue
            if ret_level is None:
                ret_level, ret_name = node.level, node.name
            else:
                l1 = len(ret_level.split('.'))
                l2 = len(node.level.split('.'))
                if l1 > l2 or (l1 == l2 and node.level < ret_level):
                    ret_level, ret_name = node.level, node.name
        return ret_level, ret_name

    @staticmethod
    def _get_graph(block):
        amr_str = ' '.join([line for line in block if not line.startswith('#')])
        graph = penman.decode(amr_str)

        named_concepts = set()
        attributes = set()
        for t in graph.triples():
            if t.relation == 'instance':
                named_concepts.add(t.source)
        for t in graph.triples():
            if t.relation != 'instance' and t.target not in named_concepts:
                attributes.add(t.target)
        return graph, attributes

    # --- get a group of nodes ---
    def get_name_nodes(self):
        return [node for node in self.nodes if node.name == 'name']

    def get_date_nodes(self):
        return [node for node in self.nodes if node.name == 'date-entity']

    # --- get a group of nodes ---
    def get_colored(self):
        """

        :return: the tokens that are reachable from the root.
        """
        color = {i: False for i in range(len(self.tok))}
        visited = set()

        def travel(root):
            if root in visited:
                return
            visited.add(root)
            node = self.nodes_by_levels[root]
            if node.alignment is not None:
                for i in range(node.alignment[0], node.alignment[1]):
                    color[i] = True
            if root in self.edges_by_parents and len(self.edges_by_parents[root]) > 0:
                for edge in self.edges_by_parents[root]:
                    travel(edge.tgt_level)
        travel('0')
        return color

    def get_entity_colored(self):
        """

        :return: the tokens that are within a entity.
        """
        color = {i: 0 for i in range(len(self.tok))}
        visited = set()

        def travel(root, is_entity, global_id):
            # global id is hacky because the python reference.
            if root in visited:
                return
            visited.add(root)
            node = self.nodes_by_levels[root]
            if is_entity and node.alignment is not None:
                for i in range(node.alignment[0], node.alignment[1]):
                    color[i] = global_id[0]
            if root in self.edges_by_parents and len(self.edges_by_parents[root]) > 0:
                for edge in self.edges_by_parents[root]:
                    new_is_entity = edge.tgt_name == 'name' or edge.tgt_name == 'date-entity'
                    if new_is_entity:
                        global_id[0] += 1
                    travel(edge.tgt_level, is_entity or new_is_entity, global_id)

        root_name = self.nodes_by_levels['0'].name
        root_is_entity = root_name == 'name' or root_name == 'date-entity'

        travel('0', root_is_entity, [int(root_is_entity)])
        return color

    def true_nodes(self):
        return filter(lambda n: n.name != self.kRoot, self.nodes)

    # --- is methods ---
    def is_attribute(self, node):
        """
        (d / date-entity :year **2008** :month 05 :day 14)

        :param node: _Node
        :return:
        """
        return node.name in self.attributes

    def is_entity_token(self, node, consider_alignment=True, verbose=False):
        """
        (c / country :name (n / name :op1 **"Estonia"**))

        :param node:
        :param consider_alignment:
        :param verbose:
        :return:
        """
        if node.level not in self.edges_by_children or len(self.edges_by_children[node.level]) != 1:
            return False
        edges = self.edges_by_children[node.level]
        edge = edges[0]
        if edge.relation.startswith('op') and edge.src_name == 'name':
            if consider_alignment:
                parent_node = self.get_node_by_level(edge.src_level)
                return node.same_alignment(parent_node, nil_as=True)
            else:
                return True
        else:
            if verbose and edge.src_name == 'name':
                print('{0}: unexpected name relation: {1}'.format(self.n, edge), file=sys.stderr)
            return False

    def is_entity_name(self, node, consider_alignment=True, verbose=False):
        """
        (c / country :name (**n / name** :op1 "Estonia"))

        :param node: _Node
        :param consider_alignment: Boolean
        :param verbose:
        :return:
        """
        if node.name != 'name':
            return False
        if node.level not in self.edges_by_children or len(self.edges_by_children[node.level]) != 1:
            if verbose:
                if node.level not in self.edges_by_children:
                    print('{0}: name({1}) has no parents'.format(self.n, node.level))
                else:
                    print('{0}: name({1}) has multiple parents'.format(self.n, node.level))
            return False
        edges = self.edges_by_children[node.level]
        edge = edges[0]
        if edge.relation == 'name':
            if consider_alignment:
                parent_node = self.get_node_by_level(edge.src_level)
                return node.same_alignment(parent_node, nil_as=True)
            else:
                return True
        else:
            if verbose:
                print('{0}: unexpected entity_token relation: {1}'.format(self.n, edge), file=sys.stderr)
            return False

    def is_entity(self, node, consider_alignment=True):
        """
        (**c / country** :name (n / name :op1 "Estonia"))

        :param node:
        :param consider_alignment:
        :return:
        """
        if node.level not in self.edges_by_parents:
            return False
        edges = self.edges_by_parents[node.level]
        has_name_node = False
        for edge in edges:
            if edge.relation == 'name' and edge.tgt_name == 'name':
                child_node = self.get_node_by_level(edge.tgt_level)
                if self.is_entity_name(child_node, consider_alignment, verbose=False):
                    has_name_node = True
        return has_name_node

    def is_date_entity(self, node, consider_alignment=True, verbose=False):
        """
        TODO: cases I cannot handle
        - eight years ago => (b / before :op1 (n / now) :quant (d / date-entity :quant 8 :unit (y / year)))

        :param node:
        :param consider_alignment:
        :param verbose:
        :return:
        """
        if node.name != 'date-entity':
            return False
        for edge in self.edges_by_parents[node.level]:
            if edge.relation in self.kDateEntityRelations:
                # mod is for utterances like: 'earlier in 2008'
                if consider_alignment:
                    child_node = self.get_node_by_level(edge.tgt_level)
                    if not node.same_alignment(child_node, nil_as=True):
                        return False
            elif edge.relation != 'mod':
                if verbose:
                    print('{0}: unexpected date entity relation: {1}'.format(self.n, edge), file=sys.stderr)
                return False
        return True

    def is_date_entity_attributes(self, node):
        """

        :param node:
        :return:
        """
        if node.level not in self.edges_by_children:
            return False
        edges = self.edges_by_children[node.level]
        if len(edges) != 1:
            return False
        edge = edges[0]
        if edge.src_name != 'date-entity' or edge.relation not in self.kDateEntityRelations:
            return False
        return True

    def is_url_entity(self, node):
        return node.name == 'url-entity'

    def is_url_entity_attributes(self, node):
        if node.level not in self.edges_by_children:
            return False
        edges = self.edges_by_children[node.level]
        if len(edges) != 1:
            return False
        edge = edges[0]
        if edge.src_name != 'url-entity':
            return False
        return True

    def is_ordinal_entity(self, node):
        return node.name == 'ordinal-entity'

    def is_ordinal_entity_attributes(self, node):
        if node.level not in self.edges_by_children:
            return False
        edges = self.edges_by_children[node.level]
        if len(edges) != 1:
            return False
        edge = edges[0]
        if edge.src_name != 'ordinal-entity':
            return False
        return True

    def has_empty_align(self):
        for node in self.nodes:
            if node.alignment is None:
                return True
        return False

    # --- get a group of nodes ---
    def get_node_by_level(self, target_level):
        for node in self.nodes:
            if node.level == target_level:
                return node
        return None

    def get_nodes_by_alignment(self, left, right=None):
        if right is None:
            right = left + 1
        ret = []
        for node in self.nodes:
            if node.alignment is None:
                continue
            if left == node.alignment[0] and right == node.alignment[1]:
                ret.append(node)
        return ret

    def get_shallowest_node_by_alignment(self, left, right=None):
        nodes = self.get_nodes_by_alignment(left, right)
        nodes.sort(key=lambda x: len(x.level))
        if len(nodes) > 0:
            return nodes[0]
        return None

    def get_nodes_by_name(self, target_fullname):
        return [node for node in self.nodes if node.name == target_fullname]

    def get_nodes_by_name_and_align(self, target_fullname, left, right=None):
        if right is None:
            right = left + 1
        ret = []
        for node in self.nodes:
            if node.alignment is None:
                continue
            if target_fullname == node.name and left == node.alignment[0] and right == node.alignment[1]:
                ret.append(node)
        return ret

    def locate_state_node(self, state_node):
        """

        :param state_node: Node
        :return: _Node
        """
        name = state_node.get_name()
        coverage = state_node.get_coverage()
        if len(coverage) == 1:
            nodes = self.get_nodes_by_name_and_align(name, coverage[0])
        else:
            nodes = self.get_nodes_by_name_and_align(name, coverage[0], coverage[-1] + 1)
        assert len(nodes) == 1, str(nodes)
        return nodes[0]

    def has_edge(self, parent, child):
        """

        :param parent: _Node
        :param child: _Node
        :return:
        """
        if parent.level not in self.edges_by_parents:
            return False
        for edge in self.edges_by_parents[parent.level]:
            if edge.tgt_level == child.level:
                return True
        return False

    def get_edge(self, parent, child):
        """

        :param parent:
        :param child:
        :return:
        """
        if parent.level not in self.edges_by_parents:
            return None
        for edge in self.edges_by_parents[parent.level]:
            if edge.tgt_level == child.level:
                return edge
        return None

    def __len__(self):
        return len(self.tok)

    # -- alignment operation --
    def remove_intersection(self):
        """
        e.g.:  # ::node 0.0 "X" 0-10
               # ::node 0.1  A  1-2
               # ::node 0.2  B  3-4

            ->
               # ::node 0.0 "X"
               # ::node 0.1  A  1-2
               # ::node 0.2  B  3-4
        :param block:
        :return:
        """
        alignments = []
        for node in self.nodes:
            if node.alignment is not None and node.alignment not in alignments and node.alignment != (-1, -1):
                alignments.append(node.alignment)

        length = len(self.tok)
        n_alignments = len(alignments)
        alignments.sort(key=lambda x: x[1])

        dp = {i: 0 for i in range(length + 1)}

        bel = {i: None for i in range(length + 1)}

        j = 0
        # print (alignments)
        for i in range(length + 1):
            if i > 0:
                dp[i] = dp[i - 1]
            while j < n_alignments and alignments[j][1] - 1 == i:
                tmp = 1 if alignments[j][0] == 0 else dp[alignments[j][0] - 1] + 1
                if tmp > dp[i] or (tmp == dp[i] and (bel[i] is not None and bel[i][0] < alignments[j][0])):
                    dp[i] = tmp
                    bel[i] = alignments[j]
                j += 1

        s = set()

        i = length
        while i >= 0:
            # print ('~', i)
            if bel[i] is None:
                i -= 1
            else:
                # print (bel[i])
                s.add(bel[i])
                i = bel[i][0] - 1

        for alignment in alignments:
            if alignment in s:
                continue
            for i in range(len(self.nodes)):
                if self.nodes[i].alignment is not None and \
                        self.nodes[i].alignment[0] == alignment[0] and \
                        self.nodes[i].alignment[1] == alignment[1]:
                    self.nodes[i].alignment = None

    # clean alignment
    def remove_alignment(self):
        for node in filter(lambda n: n.name != Alignment.kRoot, self.nodes):
            node.alignment = None

    # refill the alignment according to the original data.
    def refill_alignment(self):
        alignments = self.alignments.alignments
        for level in alignments:
            node = self.get_node_by_level(level)
            node.alignment = alignments[level]
