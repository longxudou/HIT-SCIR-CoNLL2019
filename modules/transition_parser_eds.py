import logging
from typing import Dict, Optional, Any, List

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Metric
from torch.nn.modules import Dropout

from modules import StackRnn
from utils import eds_trans_outputs_into_mrp

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transition_parser_eds")
class TransitionParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 concept_label_dim: int,
                 num_layers: int,
                 mces_metric: Metric = None,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 lemma_text_field_embedder: TextFieldEmbedder = None,
                 pos_tag_embedding: Embedding = None,
                 action_embedding: Embedding = None,
                 concept_label_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:

        super(TransitionParser, self).__init__(vocab, regularizer)

        self._unlabeled_correct = 0
        self._labeled_correct = 0
        self._total_edges_predicted = 0
        self._total_edges_actual = 0
        self._exact_unlabeled_correct = 0
        self._exact_labeled_correct = 0
        self._total_sentences = 0

        self.num_actions = vocab.get_vocab_size('actions')
        self.num_concept_label = vocab.get_vocab_size('concept_label')
        self.text_field_embedder = text_field_embedder
        self.lemma_text_field_embedder = lemma_text_field_embedder
        self._pos_tag_embedding = pos_tag_embedding
        self._mces_metric = mces_metric

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.concept_label_dim = concept_label_dim
        self.action_embedding = action_embedding
        self.concept_label_embedding = concept_label_embedding

        if concept_label_embedding is None:
            self.concept_label_embedding = Embedding(num_embeddings=self.num_concept_label,
                                                     embedding_dim=self.concept_label_dim,
                                                     trainable=False)
        if action_embedding is None:
            self.action_embedding = Embedding(num_embeddings=self.num_actions,
                                              embedding_dim=self.action_dim,
                                              trainable=False)

        # syntactic composition
        self.p_comp = torch.nn.Linear(self.hidden_dim * 6, self.word_dim)
        # parser state to hidden
        self.p_s2h = torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        # hidden to action
        self.p_act = torch.nn.Linear(self.hidden_dim, self.num_actions)

        self.start_concept_node = torch.nn.Linear(self.hidden_dim + self.concept_label_dim, self.word_dim)
        self.end_concept_node = torch.nn.Linear(self.hidden_dim * 2 + self.concept_label_dim, self.word_dim)

        self.pempty_buffer_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.proot_stack_emb = torch.nn.Parameter(torch.randn(self.word_dim))
        self.pempty_action_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.pempty_stack_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.pempty_deque_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))

        self._input_dropout = Dropout(input_dropout)

        self.buffer = StackRnn(input_size=self.word_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=num_layers,
                               recurrent_dropout_probability=recurrent_dropout_probability,
                               layer_dropout_probability=layer_dropout_probability,
                               same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.stack = StackRnn(input_size=self.word_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.deque = StackRnn(input_size=self.word_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=num_layers,
                              recurrent_dropout_probability=recurrent_dropout_probability,
                              layer_dropout_probability=layer_dropout_probability,
                              same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.action_stack = StackRnn(input_size=self.action_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=num_layers,
                                     recurrent_dropout_probability=recurrent_dropout_probability,
                                     layer_dropout_probability=layer_dropout_probability,
                                     same_dropout_mask_per_instance=same_dropout_mask_per_instance)
        initializer(self)

    def _greedy_decode(self,
                       batch_size: int,
                       sent_len: List[int],
                       embedded_text_input: torch.Tensor,
                       oracle_actions: Optional[List[List[int]]] = None,
                       ) -> Dict[str, Any]:

        self.buffer.reset_stack(batch_size)
        self.stack.reset_stack(batch_size)
        self.deque.reset_stack(batch_size)
        self.action_stack.reset_stack(batch_size)

        # We will keep track of all the losses we accumulate during parsing.
        # If some decision is unambiguous because it's the only thing valid given
        # the parser state, we will not model it. We only model what is ambiguous.
        losses = [[] for _ in range(batch_size)]
        ratio_factor_losses = [[] for _ in range(batch_size)]
        edge_list = [[] for _ in range(batch_size)]
        total_node_num = [0 for _ in range(batch_size)]
        action_list = [[] for _ in range(batch_size)]
        ret_top_node = [[] for _ in range(batch_size)]
        ret_concept_node = [[] for _ in range(batch_size)]
        # push the tokens onto the buffer (tokens is in reverse order)
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    self.buffer.push(sent_idx,
                                     input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                                     extra={'token': sent_len[sent_idx] - token_idx - 1})

        # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.stack.push(sent_idx,
                            input=self.proot_stack_emb,
                            extra={'token': 'protection_symbol'})

        action_id = {
            action_: [self.vocab.get_token_index(a, namespace='actions') for a in
                      self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)]
            for action_ in
            ["SHIFT", "REDUCE", "LEFT-EDGE", "RIGHT-EDGE", "SELF-EDGE", "DROP", "TOP", "PASS", "START", "END", "FINISH"]
        }

        # compute probability of each of the actions and choose an action
        # either from the oracle or if there is no oracle, based on the model
        trans_not_fin = True

        action_tag_for_terminate = [False] * batch_size
        action_sequence_length = [0] * batch_size

        concept_node = {}
        for sent_idx in range(batch_size):
            concept_node[sent_idx] = {}

        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):

                if (len(concept_node[sent_idx]) > 50 * sent_len[sent_idx] or action_sequence_length[sent_idx] > 50 *
                    sent_len[sent_idx]) and oracle_actions is None:
                    continue
                total_node_num[sent_idx] = sent_len[sent_idx] + len(concept_node[sent_idx])
                # if self.buffer.get_len(sent_idx) != 0:

                if not (self.buffer.get_len(sent_idx) == 0 and self.stack.get_len(sent_idx) == 1):
                    trans_not_fin = True
                    valid_actions = []
                    # given the buffer and stack, conclude the valid action list
                    if self.buffer.get_len(sent_idx) == 0:
                        valid_actions += action_id['FINISH']

                    if self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['SHIFT']
                        valid_actions += action_id['DROP']
                        valid_actions += action_id['TOP']

                        buffer_token = self.buffer.get_stack(sent_idx)[-1]["token"]
                        if buffer_token < sent_len[sent_idx]:
                            valid_actions += action_id['START']

                    if self.buffer.get_len(sent_idx) > 0 and self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['LEFT-EDGE']
                        valid_actions += action_id['RIGHT-EDGE']
                        valid_actions += action_id['SELF-EDGE']

                    if self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['REDUCE']
                        valid_actions += action_id['PASS']

                    if self.buffer.get_len(sent_idx) > 0 and self.stack.get_len(sent_idx) > 1:
                        concept_node_token = self.stack.get_stack(sent_idx)[-1]['token']
                        concept_alignment_end_token = self.buffer.get_stack(sent_idx)[-1]["token"]
                        if not (concept_node_token not in concept_node[sent_idx] or concept_alignment_end_token >=
                                sent_len[sent_idx]):
                            valid_actions += action_id['END']

                    log_probs = None
                    action = valid_actions[0]

                    if len(valid_actions) > 1:
                        stack_emb = self.stack.get_output(sent_idx)
                        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                            else self.buffer.get_output(sent_idx)

                        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                            else self.action_stack.get_output(sent_idx)

                        deque_emb = self.pempty_deque_emb if self.deque.get_len(sent_idx) == 0 \
                            else self.deque.get_output(sent_idx)

                        p_t = torch.cat((buffer_emb, stack_emb, action_emb, deque_emb))
                        h = torch.tanh(self.p_s2h(p_t))
                        logits = self.p_act(h)[torch.tensor(valid_actions, dtype=torch.long, device=h.device)]
                        valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                        log_probs = torch.log_softmax(logits, dim=0)

                        action_idx = torch.max(log_probs, 0)[1].item()
                        action = valid_actions[action_idx]

                    if oracle_actions is not None:
                        action = oracle_actions[sent_idx].pop(0)

                    if log_probs is not None:
                        # append the action-specific loss
                        losses[sent_idx].append(log_probs[valid_action_tbl[action]])

                    # generate concept node, push it into buffer and align it with the second item in buffer
                    if action in action_id["START"]:

                        # get concept label and corresponding embedding
                        concept_node_token = len(concept_node[sent_idx]) + sent_len[sent_idx]
                        stack_emb = self.stack.get_output(sent_idx)
                        concept_node_label_token = self.vocab.get_token_from_index(action, namespace='actions') \
                            .split('#SPLIT_TAG#', maxsplit=1)[1]
                        concept_node_label = self.vocab.get_token_index(concept_node_label_token,
                                                                        namespace='concept_label')
                        concept_node_label_emb = self.concept_label_embedding(
                            torch.tensor(concept_node_label, device=embedded_text_input.device))

                        # init
                        concept_alignment_begin = self.buffer.get_stack(sent_idx)[-1]["token"]
                        concept_node[sent_idx][concept_node_token] = {"label": concept_node_label_token, \
                                                                      "start": concept_alignment_begin}

                        # insert comp_rep into buffer
                        comp_rep = torch.tanh(self.start_concept_node(torch.cat((stack_emb, concept_node_label_emb))))
                        self.buffer.push(sent_idx,
                                         input=comp_rep,
                                         extra={'token': concept_node_token})

                        # update total_node_num for early-stopping
                        total_node_num[sent_idx] = sent_len[sent_idx] + len(concept_node[sent_idx])

                    # predice the span end of the node in S0
                    elif action in action_id["END"]:

                        # get label embedding of concept node
                        concept_node_token = self.stack.get_stack(sent_idx)[-1]['token']
                        concept_alignment_end_token = self.buffer.get_stack(sent_idx)[-1]["token"]

                        # if concept_node_token not in concept_node[sent_idx] or concept_alignment_end_token>=sent_len[sent_idx]:
                        #     continue

                        concept_node_label_token = concept_node[sent_idx][concept_node_token]["label"]
                        concept_node_label = self.vocab.get_token_index(concept_node_label_token,
                                                                        namespace='concept_label')
                        concept_node_label_emb = self.concept_label_embedding(
                            torch.tensor(concept_node_label, device=embedded_text_input.device))

                        # update concept info via inserting the span end of concept node
                        concept_node[sent_idx][concept_node_token]["end"] = concept_alignment_end_token

                        # update node representation using a)begin compositioned embedding b)end embedding
                        stack_emb = self.stack.get_output(sent_idx)
                        buffer_emb = self.buffer.get_output(sent_idx)

                        comp_rep = torch.tanh(
                            self.end_concept_node(torch.cat((stack_emb, buffer_emb, concept_node_label_emb))))
                        self.stack.pop(sent_idx)
                        self.stack.push(sent_idx,
                                        input=comp_rep,
                                        extra={'token': concept_node_token})


                    elif action in action_id["LEFT-EDGE"] + action_id["RIGHT-EDGE"] + action_id["SELF-EDGE"]:

                        if action in action_id["LEFT-EDGE"]:
                            head = self.buffer.get_stack(sent_idx)[-1]
                            modifier = self.stack.get_stack(sent_idx)[-1]
                        elif action in action_id["RIGHT-EDGE"]:
                            head = self.stack.get_stack(sent_idx)[-1]
                            modifier = self.buffer.get_stack(sent_idx)[-1]
                        else:
                            head = self.stack.get_stack(sent_idx)[-1]
                            modifier = self.stack.get_stack(sent_idx)[-1]

                        (head_rep, head_tok) = (head['stack_rnn_output'], head['token'])
                        (mod_rep, mod_tok) = (modifier['stack_rnn_output'], modifier['token'])

                        edge_list[sent_idx].append((mod_tok,
                                                    head_tok,
                                                    self.vocab.get_token_from_index(action,
                                                                                    namespace='actions').split(
                                                        '#SPLIT_TAG#', maxsplit=1)[1]))

                        # compute composed representation
                        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                            else self.action_stack.get_output(sent_idx)

                        stack_emb = self.pempty_stack_emb if self.stack.get_len(sent_idx) == 0 \
                            else self.stack.get_output(sent_idx)

                        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                            else self.buffer.get_output(sent_idx)

                        deque_emb = self.pempty_deque_emb if self.deque.get_len(sent_idx) == 0 \
                            else self.deque.get_output(sent_idx)

                        comp_rep = torch.cat((head_rep, mod_rep, action_emb, buffer_emb, stack_emb, deque_emb))
                        comp_rep = torch.tanh(self.p_comp(comp_rep))

                        if action in action_id["LEFT-EDGE"]:
                            self.buffer.pop(sent_idx)
                            self.buffer.push(sent_idx,
                                             input=comp_rep,
                                             extra={'token': head_tok})

                        elif action in action_id["RIGHT-EDGE"] + action_id["SELF-EDGE"]:
                            self.stack.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=comp_rep,
                                            extra={'token': head_tok})

                    elif action in action_id["REDUCE"]:
                        self.stack.pop(sent_idx)

                    elif action in action_id["TOP"]:
                        ret_top_node[sent_idx] = self.buffer.get_stack(sent_idx)[-1]["token"]

                    elif action in action_id["DROP"]:
                        self.buffer.pop(sent_idx)
                        while self.deque.get_len(sent_idx) > 0:
                            deque_top = self.deque.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=deque_top['stack_rnn_input'],
                                            extra={'token': deque_top['token']})

                    elif action in action_id["PASS"]:
                        stack_top = self.stack.pop(sent_idx)
                        self.deque.push(sent_idx,
                                        input=stack_top['stack_rnn_input'],
                                        extra={'token': stack_top['token']})

                    elif action in action_id["SHIFT"]:
                        while self.deque.get_len(sent_idx) > 0:
                            deque_top = self.deque.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=deque_top['stack_rnn_input'],
                                            extra={'token': deque_top['token']})

                        buffer_top = self.buffer.pop(sent_idx)
                        self.stack.push(sent_idx,
                                        input=buffer_top['stack_rnn_input'],
                                        extra={'token': buffer_top['token']})

                    # push action into action_stack
                    self.action_stack.push(sent_idx,
                                           input=self.action_embedding(
                                               torch.tensor(action, device=embedded_text_input.device)),
                                           extra={
                                               'token': self.vocab.get_token_from_index(action, namespace='actions')})

                    action_list[sent_idx].append(self.vocab.get_token_from_index(action, namespace='actions'))

                    action_sequence_length[sent_idx] += 1

        # categorical cross-entropy
        _loss_CCE = -torch.sum(
            torch.stack([torch.sum(torch.stack(cur_loss)) for cur_loss in losses if len(cur_loss) > 0])) / \
                    sum([len(cur_loss) for cur_loss in losses])

        _loss = _loss_CCE

        ret = {
            'loss': _loss,
            'losses': losses,
        }

        # extract concept node list in batchmode
        for sent_idx in range(batch_size):
            ret_concept_node[sent_idx] = concept_node[sent_idx]

        ret["total_node_num"] = total_node_num
        ret['edge_list'] = edge_list
        ret['action_sequence'] = action_list
        ret['top_node'] = ret_top_node
        ret["concept_node"] = ret_concept_node

        return ret

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                lemmas: Dict[str, torch.LongTensor] = None,
                pos_tags: torch.LongTensor = None,
                arc_tags: torch.LongTensor = None,
                concept_label: torch.LongTensor = None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)
        sent_len = [len(d['tokens']) for d in metadata]
        meta_tokens = [d['tokens'] for d in metadata]
        meta_info = [d['meta_info'] for d in metadata]

        oracle_actions = None
        if gold_actions is not None:
            oracle_actions = [d['gold_actions'] for d in metadata]
            oracle_actions = [[self.vocab.get_token_index(s, namespace='actions') for s in l] for l in oracle_actions]

        embedded_text_input = self.text_field_embedder(tokens)
        embedded_text_input = self._input_dropout(embedded_text_input)

        if self.training:
            ret_train = self._greedy_decode(batch_size=batch_size,
                                            sent_len=sent_len,
                                            embedded_text_input=embedded_text_input,
                                            oracle_actions=oracle_actions)

            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            return output_dict

        training_mode = self.training

        self.eval()
        with torch.no_grad():
            ret_eval = self._greedy_decode(batch_size=batch_size,
                                           sent_len=sent_len,
                                           embedded_text_input=embedded_text_input)
        self.train(training_mode)

        edge_list = ret_eval['edge_list']
        top_node_list = ret_eval['top_node']
        _loss = ret_eval['loss']

        output_dict = {
            'tokens': [d['tokens'] for d in metadata],
            'loss': _loss,
            'edge_list': edge_list,
            'meta_info': meta_info,
            'top_node': top_node_list,
            'concept_node': ret_eval['concept_node'],
            'tokens_range': [d['tokens_range'] for d in metadata]
        }

        # prediction-mode
        # compute the mrp accuracy when gold actions exists
        if gold_actions is not None:
            gold_mrps = [x["gold_mrps"] for x in metadata]
            predicted_mrps = []

            for sent_idx in range(batch_size):
                predicted_mrps.append(eds_trans_outputs_into_mrp({
                    'tokens': output_dict['tokens'][sent_idx],
                    'edge_list': output_dict['edge_list'][sent_idx],
                    'meta_info': output_dict['meta_info'][sent_idx],
                    'top_node': output_dict['top_node'][sent_idx],
                    'concept_node': output_dict['concept_node'][sent_idx],
                    'tokens_range': output_dict['tokens_range'][sent_idx],
                }))

            self._mces_metric(predicted_mrps, gold_mrps)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._mces_metric is not None and not self.training:
            all_metrics.update(self._mces_metric.get_metric(reset=reset))
        return all_metrics
