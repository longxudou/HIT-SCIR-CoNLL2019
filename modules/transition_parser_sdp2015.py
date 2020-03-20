import logging
from typing import Dict, Optional, Any, List

import torch
from allennlp.data import Vocabulary
from allennlp.models import SimpleTagger
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Metric
from torch.nn.modules import Dropout

from modules import StackRnn, SimpleTagger
from utils import sdp_trans_outputs_into_mrp

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transition_parser_sdp2015")
class TransitionParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 num_layers: int,
                 mces_metric: Metric = None,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 lemma_text_field_embedder: TextFieldEmbedder = None,
                 pos_tag_embedding: Embedding = None,
                 action_embedding: Embedding = None,
                 frame_tagger_encoder: Seq2SeqEncoder = None,
                 pos_tagger_encoder: Seq2SeqEncoder = None,
                 node_label_tagger_encoder: Seq2SeqEncoder = None,
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
        self.text_field_embedder = text_field_embedder
        self.pos_tag_embedding = pos_tag_embedding
        self._mces_metric = mces_metric

        self.action_embedding = action_embedding

        if action_embedding is None:
            self.action_embedding = Embedding(num_embeddings=self.num_actions,
                                              embedding_dim=action_dim,
                                              trainable=False)
        # syntactic composition
        self.p_comp = torch.nn.Linear(hidden_dim * 4, word_dim)
        # parser state to hidden
        self.p_s2h = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        # hidden to action

        self.p_act = torch.nn.Linear(hidden_dim, self.num_actions)
        self.pempty_buffer_emb = torch.nn.Parameter(torch.randn(hidden_dim))
        self.proot_stack_emb = torch.nn.Parameter(torch.randn(word_dim))
        self.pempty_action_emb = torch.nn.Parameter(torch.randn(hidden_dim))
        self.pempty_deque_emb = torch.nn.Parameter(torch.randn(hidden_dim))

        self._input_dropout = Dropout(input_dropout)

        self.frame_tagger_encoder = frame_tagger_encoder
        self.pos_tagger_encoder = pos_tagger_encoder
        self.node_label_tagger_encoder = node_label_tagger_encoder

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

        self.frame_tagger = SimpleTagger(vocab=vocab,
                                         text_field_embedder=text_field_embedder,
                                         encoder=self.frame_tagger_encoder,
                                         label_namespace='frame')

        self.pos_tagger = SimpleTagger(vocab=vocab,
                                       text_field_embedder=text_field_embedder,
                                       encoder=self.pos_tagger_encoder,
                                       label_namespace='pos_tag')

        self.node_label_tagger = SimpleTagger(vocab=vocab,
                                              text_field_embedder=text_field_embedder,
                                              encoder=self.node_label_tagger_encoder,
                                              label_namespace='node_label')

        initializer(self)

    def _greedy_decode(self,
                       batch_size: int,
                       sent_len: List[int],
                       embedded_text_input: torch.Tensor,
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
        edge_list = [[] for _ in range(batch_size)]

        # push the tokens onto the buffer (tokens is in reverse order)
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    self.buffer.push(sent_idx,
                                     input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                                     extra={'token': sent_len[sent_idx] - token_idx})

        # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.stack.push(sent_idx,
                            input=self.proot_stack_emb,
                            extra={'token': 0})

        action_id = {
            action_: [self.vocab.get_token_index(a, namespace='actions') for a in
                      self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)]
            for action_ in ["LR", "LP", "RS", "RP", "NS", "NR", "NP"]
        }

        # compute probability of each of the actions and choose an action
        # either from the oracle or if there is no oracle, based on the model
        trans_not_fin = True
        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):
                if self.buffer.get_len(sent_idx) != 0:
                    trans_not_fin = True
                    valid_actions = []

                    # given the buffer and stack, conclude the valid action list
                    if self.stack.get_len(sent_idx) > 1 and self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['LR']
                        valid_actions += action_id['LP']
                        valid_actions += action_id['RP']

                    if self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['NS']
                        valid_actions += action_id['RS']  # ROOT,NULL

                    if self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['NR']
                        valid_actions += action_id['NP']

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

                        p_t = torch.cat([buffer_emb, stack_emb, action_emb, deque_emb])
                        h = torch.tanh(self.p_s2h(p_t))
                        logits = self.p_act(h)[torch.tensor(valid_actions, dtype=torch.long, device=h.device)]
                        valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                        log_probs = torch.log_softmax(logits, dim=0)

                        action_idx = torch.max(log_probs, 0)[1].item()
                        action = valid_actions[action_idx]

                    if oracle_actions is not None:
                        action = oracle_actions[sent_idx].pop(0)

                    # push action into action_stack
                    self.action_stack.push(sent_idx,
                                           input=self.action_embedding(
                                               torch.tensor(action, device=embedded_text_input.device)),
                                           extra={
                                               'token': self.vocab.get_token_from_index(action, namespace='actions')})

                    if log_probs is not None:
                        # append the action-specific loss
                        losses[sent_idx].append(log_probs[valid_action_tbl[action]])

                    if action in action_id["LR"] or action in action_id["LP"] or \
                            action in action_id["RS"] or action in action_id["RP"]:
                        # figure out which is the head and which is the modifier
                        if action in action_id["RS"] or action in action_id["RP"]:
                            head = self.stack.get_stack(sent_idx)[-1]
                            modifier = self.buffer.get_stack(sent_idx)[-1]
                        else:
                            head = self.buffer.get_stack(sent_idx)[-1]
                            modifier = self.stack.get_stack(sent_idx)[-1]

                        (head_rep, head_tok) = (head['stack_rnn_output'], head['token'])
                        (mod_rep, mod_tok) = (modifier['stack_rnn_output'], modifier['token'])

                        if oracle_actions is None:
                            edge_list[sent_idx].append((mod_tok,
                                                        head_tok,
                                                        self.vocab.get_token_from_index(action, namespace='actions')
                                                        .split(':', maxsplit=1)[1]))

                    # Execute the action to update the parser state
                    # reduce
                    if action in action_id["LR"] or action in action_id["NR"]:
                        self.stack.pop(sent_idx)
                    # pass
                    elif action in action_id["LP"] or action in action_id["NP"] or action in action_id["RP"]:
                        stack_top = self.stack.pop(sent_idx)
                        self.deque.push(sent_idx,
                                        input=stack_top['stack_rnn_input'],
                                        extra={'token': stack_top['token']})
                    # shift
                    elif action in action_id["RS"] or action in action_id["NS"]:
                        while self.deque.get_len(sent_idx) > 0:
                            deque_top = self.deque.pop(sent_idx)
                            self.stack.push(sent_idx,
                                            input=deque_top['stack_rnn_input'],
                                            extra={'token': deque_top['token']})

                        buffer_top = self.buffer.pop(sent_idx)
                        self.stack.push(sent_idx,
                                        input=buffer_top['stack_rnn_input'],
                                        extra={'token': buffer_top['token']})

        _loss = -torch.sum(
            torch.stack([torch.sum(torch.stack(cur_loss)) for cur_loss in losses if len(cur_loss) > 0])) / \
                sum([len(cur_loss) for cur_loss in losses])
        ret = {
            'loss': _loss,
            'losses': losses,
        }
        if oracle_actions is None:
            ret['edge_list'] = edge_list
        return ret

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                lemmas: Dict[str, torch.LongTensor] = None,
                mrp_pos_tags: torch.LongTensor = None,
                frame: torch.LongTensor = None,
                pos_tag: torch.LongTensor = None,
                node_label: torch.LongTensor = None,
                arc_tags: torch.LongTensor = None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)
        sent_len = [len(d['tokens']) for d in metadata]
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

            frame_tagger_train_outputs = self.frame_tagger(tokens=tokens, tags=frame)
            frame_tagger_train_outputs = self.frame_tagger.decode(frame_tagger_train_outputs)

            pos_tagger_train_outputs = self.pos_tagger(tokens=tokens, tags=pos_tag)
            pos_tagger_train_outputs = self.pos_tagger.decode(pos_tagger_train_outputs)

            node_label_tagger_train_outputs = self.node_label_tagger(tokens=tokens, tags=node_label)
            node_label_tagger_train_outputs = self.node_label_tagger.decode(node_label_tagger_train_outputs)

            _loss = ret_train['loss'] + \
                    frame_tagger_train_outputs['loss'] + \
                    pos_tagger_train_outputs['loss'] + \
                    node_label_tagger_train_outputs['loss']
            output_dict = {'loss': _loss}
            return output_dict

        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_eval = self._greedy_decode(batch_size=batch_size,
                                           sent_len=sent_len,
                                           embedded_text_input=embedded_text_input)
            if frame is not None:
                frame_tagger_eval_outputs = self.frame_tagger(tokens, tags=frame)
            else:
                frame_tagger_eval_outputs = self.frame_tagger(tokens)
            frame_tagger_eval_outputs = self.frame_tagger.decode(frame_tagger_eval_outputs)

            if pos_tag is not None:
                pos_tagger_eval_outputs = self.pos_tagger(tokens, tags=pos_tag)
            else:
                pos_tagger_eval_outputs = self.pos_tagger(tokens)
            pos_tagger_eval_outputs = self.pos_tagger.decode(pos_tagger_eval_outputs)

            if node_label is not None:
                node_label_tagger_eval_outputs = self.node_label_tagger(tokens, tags=node_label)
            else:
                node_label_tagger_eval_outputs = self.node_label_tagger(tokens)
            node_label_tagger_eval_outputs = self.node_label_tagger.decode(node_label_tagger_eval_outputs)

        self.train(training_mode)

        edge_list = ret_eval['edge_list']

        if 'loss' in frame_tagger_eval_outputs and 'loss' in pos_tagger_eval_outputs:
            _loss = ret_eval['loss'] + \
                    frame_tagger_eval_outputs['loss'] + \
                    pos_tagger_eval_outputs['loss'] + \
                    node_label_tagger_eval_outputs['loss']
        else:
            _loss = ret_eval['loss']

        # prediction-mode
        output_dict = {
            'tokens': [d['tokens'] for d in metadata],
            'edge_list': edge_list,
            'meta_info': meta_info,
            'tokens_range': [d['tokens_range'] for d in metadata],
            'frame': frame_tagger_eval_outputs["tags"],
            'pos_tag': pos_tagger_eval_outputs["tags"],
            'node_label': node_label_tagger_eval_outputs["tags"],
            'loss': _loss
        }

        # prediction-mode
        # compute the mrp accuracy when gold actions exists
        if gold_actions is not None:
            gold_mrps = [x["gold_mrps"] for x in metadata]
            predicted_mrps = []

            for sent_idx in range(batch_size):
                if len(output_dict['edge_list'][sent_idx]) <= 5 * len(output_dict['tokens'][sent_idx]):
                    predicted_mrps.append(sdp_trans_outputs_into_mrp({
                        'tokens': output_dict['tokens'][sent_idx],
                        'edge_list': output_dict['edge_list'][sent_idx],
                        'meta_info': output_dict['meta_info'][sent_idx],
                        'frame': output_dict['frame'][sent_idx],
                        'pos_tag': output_dict['pos_tag'][sent_idx],
                        "node_label": output_dict['node_label'][sent_idx],
                        'tokens_range': output_dict['tokens_range'][sent_idx],
                    }))

            self._mces_metric(predicted_mrps, gold_mrps)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._mces_metric is not None and not self.training:
            all_metrics.update(self._mces_metric.get_metric(reset=reset))
        return all_metrics
