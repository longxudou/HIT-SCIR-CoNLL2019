#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
from .state import State
from ..node import TokenNode, EntityNode, ConceptNode


class Oracle(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.name = 'eager_oracle'

    def parse(self, align):
        """

        :param align: Alignment
        :return:
        """
        state = State(align)
        color = align.get_colored()
        entity_color = align.get_entity_colored()

        n_sp_confirm = 0
        need_merge = {i: 0 for i in range(len(align.tok))}
        for node in align.nodes:
            if node.alignment is not None and\
                    node.alignment[0] + 1 < node.alignment[1] and entity_color[node.alignment[0]] == 0:
                n_sp_confirm += 1
                for i in range(node.alignment[0], node.alignment[1]):
                    need_merge[i] = n_sp_confirm
                
        actions = []
        n_steps = 0
        while not state.is_terminated():
            n_steps += 1
            if n_steps > 2000:
                raise ValueError('Failed to parse.')

            if state.can_newnode():
                node = state.newnode_[0]
                action_name = 'NEWNODE\t{0}'.format(node.get_name())
                actions.append(action_name)
                state.newnode()
                self._report(n_steps, action_name, state)
                continue

            # rule #0: **DROP**
            # [stack, s0] [deque] [b0, buffer] A => [stack] [deque] [buffer] A
            if state.can_drop():
                top = state.buffer_[0]
                coverage = int(top.get_coverage()[0])
                if not color[coverage]:
                    action_name = 'DROP'
                    actions.append(action_name)
                    state.drop()
                    self._report(n_steps, action_name, state)
                    continue

            # rule #x: **REDUCE**
            # [stack, s0] [deque] [b0, buffer] A => [stack] [deque] [b0, buffer] A
            if state.can_reduce():
                top = state.stack_[-1]

                if not self._has_yet_generated_and_reachable_arcs(top, align, state):
                    action_name = 'REDUCE'
                    actions.append(action_name)
                    state.reduce()
                    self._report(n_steps, action_name, state)
                    continue

            # rule #x: **ENTITY**
            # [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [E, buffer] A U {E -name-> name, name -op-> b0}
            if state.can_entity():
                top = state.buffer_[0]
                coverage = top.get_coverage()
                if len(coverage) == 1:
                    node = align.get_shallowest_node_by_alignment(coverage[0])
                    nodes = align.get_nodes_by_alignment(coverage[0])
                else:
                    node = align.get_shallowest_node_by_alignment(coverage[0], coverage[-1] + 1)
                    nodes = align.get_nodes_by_alignment(coverage[0], coverage[-1] + 1)
                if node is not None:
                    is_entity = len([1 for n in nodes if n.name == 'name']) > 0 or node.name == 'date-entity'

                    # induct a token (or tokens) into a concept node
                    # if this node is entity node, do entity, otherwise do confirm
                    # need a newnode stack, priority bigger than shift.
                    assert len(state.newnode_) == 0

                    if is_entity:
                        entity_pos = 0
                        if node.name != 'date-entity' and nodes[0].name != 'name':
                            found_entity = False
                            for i in range(len(nodes) - 1):
                                if nodes[i + 1].name == 'name':
                                    found_entity = True
                                    entity_pos = i
                                    break
                                state.newnode_ = [ConceptNode(nodes[i].name, nodes[i].alignment, nodes[i].level)] + state.newnode_
                            assert found_entity

                        action_name = 'ENTITY\t{0}'.format(nodes[entity_pos].name)
                        
                        if entity_pos + 1 < len(nodes):
                            state.entity(nodes[entity_pos].name, nodes[entity_pos].level, nodes[entity_pos + 1].level)
                        else: 
                            state.entity(nodes[entity_pos].name, nodes[entity_pos].level)

                        # PROXY_LTW_ENG_20070831_0072.17 edge from other concept node to name node
                        for i in range(entity_pos + 1, len(nodes)):
                            nodes[i].alignment = None
                        
                    elif len(nodes) >= 1: # and entity_color[coverage[0]] == 0:
                        action_name = 'CONFIRM\t{0}\t{1}'.format(top.name, nodes[-1].name)
                        state.confirm(nodes[-1].name, nodes[-1].level)    
                        assert len(state.newnode_) == 0
                        for i in range(1, len(nodes)):
                            state.add_newnode(ConceptNode(nodes[-i-1].name, nodes[-i-1].alignment, nodes[-i-1].level))    

                    assert isinstance(state.buffer_[0], ConceptNode)
                    actions.append(action_name)
                    self._report(n_steps, action_name, state)
                    continue

            # rule #x: **MERGE**
            # [stack, s0] [deque] [b0, b1, buffer] A => [stack, s0] [deque] [b0_b1, buffer] A
            # NOTE: b0 is the major node
            if state.can_merge():
                top0, top1 = state.buffer_[:2]
                coverage0 = top0.get_coverage()
                coverage1 = top1.get_coverage()
                if (need_merge[coverage0[-1]] > 0 and need_merge[coverage0[-1]] == need_merge[coverage1[0]]) or \
                        (entity_color[coverage0[-1]] > 0 and entity_color[coverage0[-1]] == entity_color[coverage1[0]]):
                    action_name = 'MERGE'
                    actions.append(action_name)
                    state.merge()
                    self._report(n_steps, action_name, state)
                    continue

            # rule: **LEFT**
            # [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [b0, buffer] A U {b0 -R-> s0}
            if state.can_left():
                parent = state.buffer_[0]
                child = state.stack_[-1]
                if align.has_edge(parent, child) and \
                        not state.has_edge_by_level(parent.level, child.level):
                    edge = align.get_edge(parent, child)
                    action_name = 'LEFT\t{0}'.format(edge.relation)
                    actions.append(action_name)
                    state.left(edge.relation)
                    self._report(n_steps, action_name, state)
                    continue
                
            # rule: **RIGHT**
            # [stack, s0] [deque] [b0, buffer] A => [stack, s0] [deque] [b0, buffer] A U {s0 -R-> b0}
            if state.can_right():
                child = state.buffer_[0]
                parent = state.stack_[-1]
                if align.has_edge(parent, child) and \
                        not state.has_edge_by_level(parent.level, child.level):
                    edge = align.get_edge(parent, child)
                    action_name = 'RIGHT\t{0}'.format(edge.relation)
                    actions.append(action_name)
                    state.right(edge.relation)
                    self._report(n_steps, action_name, state)
                    continue

            # rule #x: **CACHE**
            if state.can_cache():
                beta = state.buffer_[0]
                should_cache = False
                for sigma in state.stack_:
                    parent = beta
                    child = sigma
                    if align.has_edge(parent, child) and\
                        not state.has_edge_by_level(parent.level, child.level):
                        should_cache = True
                        break
                    parent, child = child, parent
                    if align.has_edge(parent, child) and \
                            not state.has_edge_by_level(parent.level, child.level):
                        should_cache = True
                        break
                if should_cache:
                    action_name = 'CACHE'
                    actions.append(action_name)
                    state.cache()
                    self._report(n_steps, action_name, state)
                    continue

            # rule: **SHIFT**
            # [stack, s0] [deque] [b0, buffer] A => [stack, s0, deque, b0 ] [] [buffer] A
            if state.can_shift():
                action_name = 'SHIFT'
                actions.append(action_name)
                state.shift()
                self._report(n_steps, action_name, state)
                continue

            action_name = 'EMPTY'
            self._report(n_steps, action_name, state)

        return actions, state

    def _report(self, n_step, action_name, state):
        if self.verbose:
            print('{0:02d}: ADD {1:20s} - {2}'.format(n_step, action_name, state), file=sys.stderr)

    @classmethod
    def _has_yet_generated_and_reachable_arcs(cls, node, align, state):
        """

        :param node: _Node
        :param align:
        :param state:
        :return:
        """
        
        if node.level in align.edges_by_children:
            for edge in align.edges_by_children[node.level]:
                parent_node = align.get_node_by_level(edge.src_level)
                assert parent_node is not None
                if parent_node.alignment is None:
                    continue
                if not state.has_edge_by_level(edge.src_level, node.level) and \
                        edge.src_name != 'name' and edge.src_name != 'date-entity' and \
                        node.level != edge.src_level:
                        return True

        if node.level in align.edges_by_parents and node.name != 'date-entity' and node.name != 'name':
            for edge in align.edges_by_parents[node.level]:
                child_node = align.get_node_by_level(edge.tgt_level)
                assert child_node is not None
                if child_node.alignment is None:    # if the child_node's alignment is empty.
                    continue
                if not state.has_edge_by_level(node.level, edge.tgt_level) and \
                        node.level != edge.tgt_level:
                    return True
        return False
