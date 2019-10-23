import logging
from typing import Dict, Optional, Any, List

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from torch.nn.modules import Dropout

from metrics.mrp_score import MCESScore
from modules import StackRnn
from toolkit.tamr_aligner.smatch.api import SmatchScorer
from utils.extract_mrp_dict import extract_mrp_dict, unquote, parse_date
from utils.mrp_to_amr import mrp_dict_to_amr_str

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transition_parser_amr")
class TransitionParserAmr(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 entity_dim: int,
                 rel_dim: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 dropout: float = 0.0,
                 pos_tag_embedding: Embedding = None,
                 action_text_field_embedder: TextFieldEmbedder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 eval_on_training: bool = True,
                 sep_act_type_para: bool = False
                 ) -> None:

        super(TransitionParserAmr, self).__init__(vocab, regularizer)

        self._smatch_scorer = SmatchScorer()
        self._mces_scorer = MCESScore(output_type='gscprf',
                                      cores=0,
                                      trace=0)
        self.eval_on_training = eval_on_training
        self.sep_act_type_para = sep_act_type_para
        self._NEWNODE_TYPE_MAX = 20
        self._NODE_LEN_RATIO_MAX = 20.0
        self._ACTION_TYPE = [
            'SHIFT',
            'CONFIRM',
            'REDUCE',
            'MERGE',
            'ENTITY',
            'NEWNODE',
            'DROP',
            'CACHE',
            'LEFT',
            'RIGHT'
        ]
        self._ACTION_TYPE_IDX = {a: i for (i, a) in enumerate(self._ACTION_TYPE)}

        self.text_field_embedder = text_field_embedder
        self.pos_tag_embedding = pos_tag_embedding
        self.action_text_field_embedder = action_text_field_embedder

        node_dim = word_dim
        if pos_tag_embedding:
            node_dim += pos_tag_embedding.output_dim

        self.num_actions = vocab.get_vocab_size('actions')
        self.num_newnodes = vocab.get_vocab_size('newnodes')
        self.num_relations = vocab.get_vocab_size('relations')
        self.num_entities = vocab.get_vocab_size('entities')

        self.action_embedding = Embedding(num_embeddings=self.num_actions,
                                          embedding_dim=action_dim)
        self.newnode_embedding = Embedding(num_embeddings=self.num_newnodes,
                                           embedding_dim=node_dim)
        self.rel_embedding = Embedding(num_embeddings=self.num_relations,
                                       embedding_dim=rel_dim)
        self.entity_embedding = Embedding(num_embeddings=self.num_entities,
                                          embedding_dim=entity_dim)

        # merge (stack, buffer, action_stack, deque)
        self.merge = FeedForward(input_dim=hidden_dim * 4,
                                 num_layers=1,
                                 hidden_dims=hidden_dim,
                                 activations=Activation.by_name('relu')(),
                                 dropout=dropout)
        # merge (parent, rel, child) -> parent
        self.merge_parent = FeedForward(input_dim=hidden_dim * 2 + rel_dim,
                                        num_layers=1,
                                        hidden_dims=node_dim,
                                        activations=Activation.by_name('relu')(),
                                        dropout=dropout)
        # merge (parent, rel, child) -> child
        self.merge_child = FeedForward(input_dim=hidden_dim * 2 + rel_dim,
                                       num_layers=1,
                                       hidden_dims=node_dim,
                                       activations=Activation.by_name('relu')(),
                                       dropout=dropout)
        # merge (A, B) -> AB
        self.merge_token = FeedForward(input_dim=hidden_dim * 2,
                                       num_layers=1,
                                       hidden_dims=node_dim,
                                       activations=Activation.by_name('relu')(),
                                       dropout=dropout)
        # merge (AB, entity_label) -> X
        self.merge_entity = FeedForward(input_dim=hidden_dim + entity_dim,
                                        num_layers=1,
                                        hidden_dims=node_dim,
                                        activations=Activation.by_name('relu')(),
                                        dropout=dropout)
        # Q / A value scorer
        if sep_act_type_para:
            self.action_type_scorer = torch.nn.Linear(hidden_dim, len(self._ACTION_TYPE))
            action_cnt = {
                action_: len([self.vocab.get_token_index(a, namespace='actions') for a in
                              self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)])
                for action_ in self._ACTION_TYPE
            }
            self.scorers = []
            for a in self._ACTION_TYPE:
                m = torch.nn.Linear(hidden_dim, action_cnt[a])
                self.__setattr__(f'scorer_{a}', m)
                self.scorers.append(m)
        else:
            self.scorer = torch.nn.Linear(hidden_dim, self.num_actions)
        # X -> confirm (X)
        self.confirm_layer = FeedForward(input_dim=hidden_dim,
                                         num_layers=1,
                                         hidden_dims=node_dim,
                                         activations=Activation.by_name('relu')(),
                                         dropout=dropout)

        self.pempty_buffer_emb = torch.nn.Parameter(torch.randn(word_dim))
        self.proot_stack_emb = torch.nn.Parameter(torch.randn(word_dim))
        self.proot_action_emb = torch.nn.Parameter(torch.randn(action_dim))
        self.proot_deque_emb = torch.nn.Parameter(torch.randn(word_dim))

        self._input_dropout = Dropout(input_dropout)

        self.buffer = StackRnn(input_size=word_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               recurrent_dropout_probability=recurrent_dropout_probability,
                               layer_dropout_probability=layer_dropout_probability,
                               same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.stack = StackRnn(input_size=word_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.deque = StackRnn(input_size=word_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.action_stack = StackRnn(input_size=action_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=num_layers,
                                     recurrent_dropout_probability=recurrent_dropout_probability,
                                     layer_dropout_probability=layer_dropout_probability,
                                     same_dropout_mask_per_instance=same_dropout_mask_per_instance)
        initializer(self)

    def _greedy_decode(self,
                       batch_size: int,
                       sent_len: List[int],
                       embedded_text_input: torch.Tensor,
                       metadata: List[Dict[str, Any]],
                       oracle_actions: Optional[List[List[int]]] = None
                       ) -> Dict[str, Any]:

        self.buffer.reset_stack(batch_size)
        self.stack.reset_stack(batch_size)
        self.deque.reset_stack(batch_size)
        self.action_stack.reset_stack(batch_size)

        # We will keep track of all the losses we accumulate during parsing.
        # If some decision is unambiguous because it's the only thing valid given
        # the parser state, we will not model it. We only model what is ambiguous.
        losses = [[] for _ in range(batch_size)]
        node_labels = [[] for _ in range(batch_size)]
        node_types = [[] for _ in range(batch_size)]
        existing_edges = [{} for _ in range(batch_size)]
        id_cnt = [sent_len[i] + 1 for i in range(batch_size)]
        action_strs = [[] for _ in range(batch_size)]
        origin_tokens = [metadata[i]['tokens'] for i in range(batch_size)]

        for sent_idx in range(batch_size):
            node_labels[sent_idx].append('@@ROOT@@')
            node_types[sent_idx].append('ROOT')
            for i in range(sent_len[sent_idx]):
                node_labels[sent_idx].append(origin_tokens[sent_idx][i])
                node_types[sent_idx].append('TokenNode')
        # push the tokens onto the buffer (tokens is in reverse order)
        for sent_idx in range(batch_size):
            self.buffer.push(sent_idx,
                             input=self.pempty_buffer_emb,
                             extra={'token': 0,
                                    'type': -1})
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    self.buffer.push(sent_idx,
                                     input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                                     extra={'token': sent_len[sent_idx] - token_idx,
                                            'type': 0})

        # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.stack.push(sent_idx,
                            input=self.proot_stack_emb,
                            extra={'token': 0,
                                   'type': -1})

        # init deque using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.deque.push(sent_idx,
                            input=self.proot_deque_emb,
                            extra={'token': 0,
                                   'type': -1})

        action_id = {
            action_: [self.vocab.get_token_index(a, namespace='actions') for a in
                      self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)]
            for action_ in self._ACTION_TYPE
        }
        action_idx_to_param_idx = {}
        for a in self._ACTION_TYPE:
            for (i, x) in enumerate(action_id[a]):
                action_idx_to_param_idx[x] = i
        origin_token_to_confirm_action = {}
        for a in action_id['CONFIRM']:
            t = self.vocab.get_token_from_index(a, namespace='actions').split('@@:@@')[1]
            if t not in origin_token_to_confirm_action:
                origin_token_to_confirm_action[t] = []
            origin_token_to_confirm_action[t].append(a)
        # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.action_stack.push(sent_idx,
                                   input=self.proot_action_emb,
                                   extra={'token': 0,
                                          'type': -1})

        # compute probability of each of the actions and choose an action
        # either from the oracle or if there is no oracle, based on the model
        trans_not_fin = True
        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):
                try:
                    valid_action_types = set()
                    valid_actions = []

                    # given the buffer and stack, conclude the valid action list
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] > 1:
                        valid_actions += action_id['SHIFT']
                        valid_action_types.add(self._ACTION_TYPE_IDX['SHIFT'])
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] < 2:
                        tk = node_labels[sent_idx][self.buffer.get_stack(sent_idx)[-1]['token']]
                        if self.buffer.get_stack(sent_idx)[-1]['type'] == 1:
                            tk = tk.replace('@@_@@', '_')
                        if tk in origin_token_to_confirm_action:
                            valid_actions += origin_token_to_confirm_action[tk]
                            valid_action_types.add(self._ACTION_TYPE_IDX['CONFIRM'])
                    if self.buffer.get_len(sent_idx) > 2 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] < 2 \
                            and self.buffer.get_stack(sent_idx)[-2]['type'] == 0:
                        valid_actions += action_id['MERGE']
                        valid_action_types.add(self._ACTION_TYPE_IDX['MERGE'])
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] < 2:
                        valid_actions += action_id['ENTITY']
                        valid_action_types.add(self._ACTION_TYPE_IDX['ENTITY'])
                    if self.stack.get_len(sent_idx) > 1 \
                            and self.stack.get_stack(sent_idx)[-1]['type'] > 1:
                        valid_actions += action_id['REDUCE']
                        valid_action_types.add(self._ACTION_TYPE_IDX['REDUCE'])
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] == 0:
                        valid_actions += action_id['DROP']
                        valid_action_types.add(self._ACTION_TYPE_IDX['DROP'])
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['CACHE']
                        valid_action_types.add(self._ACTION_TYPE_IDX['CACHE'])
                    if self.buffer.get_len(sent_idx) > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] > 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] < self._NEWNODE_TYPE_MAX \
                            and id_cnt[sent_idx] / sent_len[sent_idx] < self._NODE_LEN_RATIO_MAX:
                        valid_actions += action_id['NEWNODE']
                        valid_action_types.add(self._ACTION_TYPE_IDX['NEWNODE'])
                    if self.stack.get_len(sent_idx) > 1 \
                            and self.stack.get_stack(sent_idx)[-1]['type'] > 1 \
                            and self.buffer.get_len(sent_idx) > 0 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] > 1:
                        u = self.stack.get_stack(sent_idx)[-1]['token']
                        v = self.buffer.get_stack(sent_idx)[-1]['token']
                        for aid in ['LEFT', 'RIGHT']:
                            u, v = v, u
                            for a in action_id[aid]:
                                rel = self.vocab.get_token_from_index(a, namespace='actions').split('@@:@@')[1]
                                if (u not in existing_edges[sent_idx]
                                    or (rel, v) not in existing_edges[sent_idx][u]) \
                                        and \
                                        (True):  # allowing different arcs with same tag coming from one node?
                                    if u in existing_edges[sent_idx]:
                                        v_cnt = 0
                                        for eg in existing_edges[sent_idx][u]:
                                            if eg[1] == v:
                                                v_cnt += 1
                                        if v_cnt >= 3:
                                            continue
                                    valid_actions.append(a)
                                    valid_action_types.add(self._ACTION_TYPE_IDX[aid])
                    if self.stack.get_len(sent_idx) > 1 \
                            and self.stack.get_stack(sent_idx)[-1]['type'] > 1 \
                            and self.buffer.get_len(sent_idx) == 1 \
                            and self.buffer.get_stack(sent_idx)[-1]['type'] == -1:
                        v = self.stack.get_stack(sent_idx)[-1]['token']
                        u = self.buffer.get_stack(sent_idx)[-1]['token']
                        rel = '_ROOT_'
                        if (u not in existing_edges[sent_idx]
                                or (rel, v) not in existing_edges[sent_idx][u]):
                            valid_actions.append(self.vocab.get_token_index('LEFT@@:@@_ROOT_',
                                                                            namespace='actions'))
                            valid_action_types.add(self._ACTION_TYPE_IDX['LEFT'])
                    valid_action_types = list(valid_action_types)
                    assert ((len(valid_actions) == 0) == (
                                self.stack.get_len(sent_idx) == 1 and self.buffer.get_len(sent_idx) == 1))
                    if len(valid_actions) == 0:
                        continue
                    trans_not_fin = True

                    if oracle_actions is not None:
                        valid_action_types = list(range(len(self._ACTION_TYPE)))

                    if self.sep_act_type_para:
                        action_type = valid_action_types[0]
                        action_type_log_probs = None
                        h = None
                        if len(valid_action_types) > 1:
                            stack_emb = self.stack.get_output(sent_idx)
                            buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                                else self.buffer.get_output(sent_idx)
                            action_emb = self.action_stack.get_output(sent_idx)
                            deque_emb = self.deque.get_output(sent_idx)

                            p_t = torch.cat([buffer_emb, stack_emb, action_emb, deque_emb])
                            h = self.merge(p_t)

                            logits = self.action_type_scorer(h)[
                                torch.tensor(valid_action_types, dtype=torch.long, device=h.device)]
                            valid_action_type_tbl = {a: i for i, a in enumerate(valid_action_types)}
                            action_type_log_probs = torch.log_softmax(logits, dim=0)

                            action_type_idx = torch.max(action_type_log_probs, 0)[1].item()
                            action_type = valid_action_types[action_type_idx]

                        if oracle_actions is not None:
                            action_type = -1
                            for a in range(len(self._ACTION_TYPE)):
                                if oracle_actions[sent_idx][0] in action_id[self._ACTION_TYPE[a]]:
                                    action_type = a
                            assert (action_type >= 0 and action_type < len(self._ACTION_TYPE))

                        if action_type_log_probs is not None:
                            losses[sent_idx].append(action_type_log_probs[valid_action_type_tbl[action_type]])
                        else:
                            losses[sent_idx].append(torch.tensor(0.0, dtype=self.action_type_scorer.weight.dtype,
                                                                 device=self.action_type_scorer.weight.device))

                        valid_actions = [x for x in valid_actions if x in action_id[self._ACTION_TYPE[action_type]]]

                        log_probs = None
                        action = valid_actions[0]
                        if len(valid_actions) > 1:
                            if h is None:
                                stack_emb = self.stack.get_output(sent_idx)
                                buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                                    else self.buffer.get_output(sent_idx)
                                action_emb = self.action_stack.get_output(sent_idx)
                                deque_emb = self.deque.get_output(sent_idx)

                                p_t = torch.cat([buffer_emb, stack_emb, action_emb, deque_emb])
                                h = self.merge(p_t)

                            valid_actions_param = [action_idx_to_param_idx[x] for x in valid_actions]
                            logits = self.scorers[action_type](h)[
                                torch.tensor(valid_actions_param, dtype=torch.long, device=h.device)]
                            valid_action_tbl = {a: i for i, a in enumerate(valid_actions_param)}
                            log_probs = torch.log_softmax(logits, dim=0)

                            action_idx = torch.max(log_probs, 0)[1].item()
                            action = valid_actions[action_idx]

                        if oracle_actions is not None:
                            action = oracle_actions[sent_idx].pop(0)

                        if log_probs is not None:
                            # append the action-specific loss
                            losses[sent_idx].append(log_probs[valid_action_tbl[action_idx_to_param_idx[action]]])
                        else:
                            losses[sent_idx].append(torch.tensor(0.0, dtype=self.action_type_scorer.weight.dtype,
                                                                 device=self.action_type_scorer.weight.device))
                    else:
                        log_probs = None
                        action = valid_actions[0]
                        if len(valid_actions) > 1:
                            stack_emb = self.stack.get_output(sent_idx)
                            buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                                else self.buffer.get_output(sent_idx)
                            action_emb = self.action_stack.get_output(sent_idx)
                            deque_emb = self.deque.get_output(sent_idx)

                            p_t = torch.cat([buffer_emb, stack_emb, action_emb, deque_emb])
                            h = self.merge(p_t)

                            logits = self.scorer(h)[torch.tensor(valid_actions, dtype=torch.long, device=h.device)]
                            valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                            log_probs = torch.log_softmax(logits, dim=0)

                            action_idx = torch.max(log_probs, 0)[1].item()
                            action = valid_actions[action_idx]

                        if oracle_actions is not None:
                            action = oracle_actions[sent_idx].pop(0)

                        if log_probs is not None:
                            # append the action-specific loss
                            losses[sent_idx].append(log_probs[valid_action_tbl[action]])
                        else:
                            losses[sent_idx].append(
                                torch.tensor(0.0, dtype=self.scorer.weight.dtype, device=self.scorer.weight.device))

                    # push action into action_stack
                    self.action_stack.push(sent_idx,
                                           input=self.action_embedding(
                                               torch.tensor(action, device=embedded_text_input.device)),
                                           extra={'token': self.vocab.get_token_from_index(action,
                                                                                           namespace='actions')})

                    action_str = self.vocab.get_token_from_index(action,
                                                                 namespace='actions')

                    action_strs[sent_idx].append(action_str)

                    if action in action_id['SHIFT']:
                        while self.deque.get_len(sent_idx) > 1:
                            e = self.deque.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=e['stack_rnn_input'],
                                            extra={k: e[k] for k in e.keys() if not k.startswith('stack_rnn_')})
                        e = self.buffer.pop(sent_idx)
                        self.stack.push(sent_idx,
                                        input=e['stack_rnn_input'],
                                        extra={k: e[k] for k in e.keys() if not k.startswith('stack_rnn_')})
                    elif action in action_id['CONFIRM']:
                        e = self.buffer.pop(sent_idx)
                        concept = self.confirm_layer(e['stack_rnn_output'])
                        self.buffer.push(sent_idx,
                                         input=concept,
                                         extra={'token': id_cnt[sent_idx],
                                                'type': 2})
                        node_labels[sent_idx].append(action_str.split('@@:@@')[-1])
                        node_types[sent_idx].append('ConceptNode')
                        id_cnt[sent_idx] += 1
                    elif action in action_id['REDUCE']:
                        self.stack.pop(sent_idx)
                    elif action in action_id['MERGE']:
                        token_a = self.buffer.pop(sent_idx)
                        token_b = self.buffer.pop(sent_idx)
                        token_ab = self.merge_token(
                            torch.cat([token_a['stack_rnn_output'], token_b['stack_rnn_output']]))
                        token_id = token_a['token']
                        if token_a['type'] == 0:
                            node_labels[sent_idx].append(node_labels[sent_idx][token_a['token']])
                            node_types[sent_idx].append('EntityNode')
                            token_id = id_cnt[sent_idx]
                            id_cnt[sent_idx] += 1
                        node_labels[sent_idx][token_id] += '@@_@@' + node_labels[sent_idx][token_b['token']]
                        self.buffer.push(sent_idx,
                                         input=token_ab,
                                         extra={'token': token_id,
                                                'type': 1})
                    elif action in action_id['ENTITY']:
                        entity_name = action_str.split('@@:@@')[-1]
                        buffer_top_id = self.buffer.get_stack(sent_idx)[-1]['token']
                        entity = self.entity_embedding(
                            torch.tensor(self.vocab.get_token_index(action_str.split('@@:@@')[1],
                                                                    namespace='entities'),
                                         device=embedded_text_input.device))
                        e = self.buffer.pop(sent_idx)
                        entity = self.merge_entity(torch.cat([e['stack_rnn_output'], entity]))
                        self.buffer.push(sent_idx,
                                         input=entity,
                                         extra={'token': id_cnt[sent_idx],
                                                'type': 2})
                        node_labels[sent_idx].append(action_str.split('@@:@@')[-1])
                        node_types[sent_idx].append('ConceptNode')
                        id_cnt[sent_idx] += 1
                        if entity_name == 'date-entity':
                            datestr = ' '.join(map(unquote, node_labels[sent_idx][buffer_top_id].split('@@_@@')))
                            entry, flags = parse_date(datestr)
                            date_node_id = id_cnt[sent_idx] - 1
                            for relation, flag in zip(['year', 'month', 'day'], flags):
                                if flag:
                                    value = getattr(entry, relation)
                                    node_labels[sent_idx].append(str(value))
                                    node_types[sent_idx].append('AttributeNode')
                                    id_cnt[sent_idx] += 1
                                    if date_node_id not in existing_edges[sent_idx]:
                                        existing_edges[sent_idx][date_node_id] = []
                                    existing_edges[sent_idx][date_node_id].append((relation, id_cnt[sent_idx] - 1))
                        if entity_name not in ['date-entity', 'capitalism', '2', 'contrast-01', '1', 'compare-01']:
                            if entity_name != 'name':
                                node_labels[sent_idx].append('name')
                                node_types[sent_idx].append('ConceptNode')
                                id_cnt[sent_idx] += 1
                                if id_cnt[sent_idx] - 2 not in existing_edges[sent_idx]:
                                    existing_edges[sent_idx][id_cnt[sent_idx] - 2] = []
                                existing_edges[sent_idx][id_cnt[sent_idx] - 2].append(('name', id_cnt[sent_idx] - 1))
                            name_node_id = id_cnt[sent_idx] - 1
                            for (i, opi) in enumerate(
                                    map(unquote, node_labels[sent_idx][buffer_top_id].split('@@_@@'))):
                                node_labels[sent_idx].append(opi)
                                node_types[sent_idx].append('AttributeNode')
                                id_cnt[sent_idx] += 1
                                if name_node_id not in existing_edges[sent_idx]:
                                    existing_edges[sent_idx][name_node_id] = []
                                existing_edges[sent_idx][name_node_id].append((f"op{i + 1}", id_cnt[sent_idx] - 1))
                    elif action in action_id['NEWNODE']:
                        node = self.newnode_embedding(
                            torch.tensor(self.vocab.get_token_index(action_str.split('@@:@@')[1],
                                                                    namespace='newnodes'),
                                         device=embedded_text_input.device))
                        self.buffer.get_stack(sent_idx)[-1]['type'] += self._NEWNODE_TYPE_MAX
                        self.buffer.push(sent_idx,
                                         input=node,
                                         extra={'token': id_cnt[sent_idx],
                                                'type': self.buffer.get_stack(sent_idx)[-1][
                                                            'type'] - self._NEWNODE_TYPE_MAX + 1})
                        node_labels[sent_idx].append(action_str.split('@@:@@')[-1])
                        node_types[sent_idx].append('ConceptNode')
                        id_cnt[sent_idx] += 1
                    elif action in action_id['DROP']:
                        self.buffer.pop(sent_idx)
                    elif action in action_id['CACHE']:
                        e = self.stack.pop(sent_idx)
                        self.deque.push(sent_idx,
                                        input=e['stack_rnn_input'],
                                        extra={k: e[k] for k in e.keys() if not k.startswith('stack_rnn_')})
                    elif action in action_id['LEFT']:
                        parent = self.buffer.pop(sent_idx)
                        child = self.stack.pop(sent_idx)
                        rel = self.rel_embedding(torch.tensor(self.vocab.get_token_index(action_str.split('@@:@@')[1],
                                                                                         namespace='relations'),
                                                              device=embedded_text_input.device))
                        parent_rep = self.merge_parent(
                            torch.cat([parent['stack_rnn_output'], rel, child['stack_rnn_output']]))
                        child_rep = self.merge_child(
                            torch.cat([parent['stack_rnn_output'], rel, child['stack_rnn_output']]))
                        self.buffer.push(sent_idx,
                                         input=parent_rep,
                                         extra={k: parent[k] for k in parent.keys() if not k.startswith('stack_rnn_')})
                        self.stack.push(sent_idx,
                                        input=child_rep,
                                        extra={k: child[k] for k in child.keys() if not k.startswith('stack_rnn_')})
                        if parent['token'] not in existing_edges[sent_idx]:
                            existing_edges[sent_idx][parent['token']] = []
                        existing_edges[sent_idx][parent['token']].append((action_str.split('@@:@@')[1], child['token']))
                    elif action in action_id['RIGHT']:
                        child = self.buffer.pop(sent_idx)
                        parent = self.stack.pop(sent_idx)
                        rel = self.rel_embedding(torch.tensor(self.vocab.get_token_index(action_str.split('@@:@@')[1],
                                                                                         namespace='relations'),
                                                              device=embedded_text_input.device))
                        parent_rep = self.merge_parent(
                            torch.cat([parent['stack_rnn_output'], rel, child['stack_rnn_output']]))
                        child_rep = self.merge_child(
                            torch.cat([parent['stack_rnn_output'], rel, child['stack_rnn_output']]))
                        self.buffer.push(sent_idx,
                                         input=child_rep,
                                         extra={k: child[k] for k in child.keys() if not k.startswith('stack_rnn_')})
                        self.stack.push(sent_idx,
                                        input=parent_rep,
                                        extra={k: parent[k] for k in parent.keys() if not k.startswith('stack_rnn_')})
                        if parent['token'] not in existing_edges[sent_idx]:
                            existing_edges[sent_idx][parent['token']] = []
                        existing_edges[sent_idx][parent['token']].append((action_str.split('@@:@@')[1], child['token']))
                    else:
                        raise ValueError(f'Illegal action: \"{action}\"')
                except BaseException as e:
                    print(e)
                    print(metadata[sent_idx]['id'])
                    raise e

        _loss = -torch.sum(
            torch.stack([torch.sum(torch.stack(cur_loss)) for cur_loss in losses if len(cur_loss) > 0])) / \
                sum([len(cur_loss) for cur_loss in losses])
        ret = {
            'loss': _loss,
            'losses': losses,
        }
        if oracle_actions is None:
            ret['existing_edges'] = existing_edges
            ret['node_labels'] = node_labels
            ret['node_types'] = node_types
            ret['id_cnt'] = id_cnt
            ret['action_strs'] = action_strs
        return ret

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                gold_newnodes: Dict[str, torch.LongTensor] = None,
                gold_entities: Dict[str, torch.LongTensor] = None,
                gold_relations: Dict[str, torch.LongTensor] = None,
                lemmas: Dict[str, torch.LongTensor] = None,
                pos_tags: torch.LongTensor = None,
                arc_tags: torch.LongTensor = None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)
        sent_len = [len(d['tokens']) for d in metadata]

        oracle_actions = None
        if gold_actions is not None:
            oracle_actions = [d['gold_actions'] for d in metadata]
            oracle_actions = [[self.vocab.get_token_index(s, namespace='actions') for s in l] for l in oracle_actions]

        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self.pos_tag_embedding is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat([embedded_text_input, embedded_pos_tags], -1)
        embedded_text_input = self._input_dropout(embedded_text_input)

        if self.training:
            ret_train = self._greedy_decode(batch_size=batch_size,
                                            sent_len=sent_len,
                                            embedded_text_input=embedded_text_input,
                                            metadata=metadata,
                                            oracle_actions=oracle_actions)
            if not self.eval_on_training:
                return {'loss': ret_train['loss']}
        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_eval = self._greedy_decode(batch_size=batch_size,
                                           sent_len=sent_len,
                                           metadata=metadata,
                                           embedded_text_input=embedded_text_input)
        self.train(training_mode)

        action_strs = ret_eval['action_strs']

        existing_edges = ret_eval['existing_edges']
        id_cnt = ret_eval['id_cnt']
        node_labels = ret_eval['node_labels']
        node_types = ret_eval['node_types']
        _loss = ret_train['loss'] if self.training else ret_eval['loss']

        amr_dicts = []
        for sent_idx in range(batch_size):
            amr_dict = extract_mrp_dict(existing_edges=existing_edges[sent_idx],
                                        sent_len=sent_len[sent_idx],
                                        id_cnt=id_cnt[sent_idx],
                                        node_labels=node_labels[sent_idx],
                                        node_types=node_types[sent_idx],
                                        metadata=metadata[sent_idx])
            amr_dicts.append(amr_dict)

        golds_mrp = [d['mrp'] for d in metadata] if 'mrp' in metadata[0] else []
        golds_amr = [d['amr'] for d in metadata] if 'amr' in metadata[0] else []
        if golds_mrp:
            self._mces_scorer(predictions=amr_dicts,
                              golds=golds_mrp)
        elif golds_amr:
            amr_strs = [mrp_dict_to_amr_str(amr_dict, all_nodes=True) for amr_dict in amr_dicts]
            for amr_str, gold_amr in zip(amr_strs, golds_amr):
                self._smatch_scorer.update(amr_str, gold_amr)

        output_dict = {
            'tokens': [d['tokens'] for d in metadata],
            'loss': _loss,
            'sent_len': sent_len,
            'metadata': metadata
        }
        if 'gold_actions' in metadata[0]:
            output_dict['oracle_actions'] = [d['gold_actions'] for d in metadata]
        if not self.training:
            output_dict['existing_edges'] = existing_edges
            output_dict['node_labels'] = node_labels
            output_dict['node_types'] = node_types
            output_dict['id_cnt'] = id_cnt

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self._smatch_scorer.total_gold_num > 0:
            p, r, f1 = self._smatch_scorer.get_prf()
            if reset:
                self._smatch_scorer.reset()
            return {
                'P': p,
                'R': r,
                'F1': f1,
            }
        else:
            return self._mces_scorer.get_metric(reset=reset)
