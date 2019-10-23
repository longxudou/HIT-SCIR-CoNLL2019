"""
Based on augmented_lstm.py and stacked_bidirectional_lstm.py from AllenNLP 0.8.3
"""
from typing import Optional, Tuple, Type, List, Dict, Any

import torch
from allennlp.nn.util import get_dropout_mask

from modules.stack_rnn_cell import StackRnnCellBase, StackLstmCell


class StackRnn(torch.nn.Module):
    """
    A standard stacked LSTM where the LSTM layers
    are concatenated between each layer. The only difference between
    this and a regular LSTM is the application of
    variational dropout to the hidden states and outputs of each layer apart
    from the last layer of the LSTM. Note that this will be slower, as it
    doesn't use CUDNN.

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked Bidirectional LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The recurrent dropout probability to be used in a dropout scheme as
        stated in `A Theoretically Grounded Application of Dropout in Recurrent
        Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
    layer_dropout_probability: float, optional (default = 0.0)
        The layer wise dropout probability to be used in a dropout scheme as
        stated in  `A Theoretically Grounded Application of Dropout in
        Recurrent Neural Networks <https://arxiv.org/abs/1512.05287>`_ .
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 use_highway: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 cell: Type[StackRnnCellBase] = StackLstmCell) -> None:
        super(StackRnn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.layer_dropout_probability = layer_dropout_probability
        self.same_dropout_mask_per_instance = same_dropout_mask_per_instance

        layers = []
        rnn_input_size = input_size
        for layer_index in range(num_layers):
            layer = cell(rnn_input_size,
                         hidden_size,
                         use_highway=use_highway,
                         use_input_projection_bias=True)
            rnn_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.rnn_layers = layers

        self.stacks: List[List[Dict[str, Any]]] = []
        self.push_buffer: List[Optional[Dict[str, Any]]] = []
        if self.same_dropout_mask_per_instance:
            self.layer_dropout_mask: Optional[torch.Tensor] = None
            self.recurrent_dropout_mask: Optional[torch.Tensor] = None

    def reset_stack(self,
                    num_stacks: int) -> None:
        self.stacks = [[] for _ in range(num_stacks)]
        self.push_buffer = [None for _ in range(num_stacks)]
        if self.same_dropout_mask_per_instance:
            if 0.0 < self.layer_dropout_probability < 1.0:
                self.layer_dropout_mask = [[get_dropout_mask(self.layer_dropout_probability,
                                                             torch.ones(layer.hidden_size,
                                                                        device=self.layer_0.input_linearity.weight.device))
                                            for _ in range(num_stacks)] for layer in self.rnn_layers]
                self.layer_dropout_mask = torch.stack([torch.stack(l) for l in self.layer_dropout_mask])
            else:
                self.layer_dropout_mask = None
            if 0.0 < self.recurrent_dropout_probability < 1.0:
                self.recurrent_dropout_mask = [[get_dropout_mask(self.recurrent_dropout_probability,
                                                                 torch.ones(self.hidden_size,
                                                                            device=self.layer_0.input_linearity.weight.device))
                                                for _ in range(num_stacks)] for _ in range(self.num_layers)]
                self.recurrent_dropout_mask = torch.stack([torch.stack(l) for l in self.recurrent_dropout_mask])
            else:
                self.recurrent_dropout_mask = None

    def push(self,
             stack_index: int,
             input: torch.Tensor,
             extra: Optional[Dict[str, Any]] = None) -> None:
        if self.push_buffer[stack_index] is not None:
            self._apply_push()
        self.push_buffer[stack_index] = {'stack_rnn_input': input}
        if extra is not None:
            self.push_buffer[stack_index].update(extra)

    def pop(self,
            stack_index: int) -> Dict[str, Any]:
        if self.push_buffer[stack_index] is not None:
            self._apply_push()
        return self.stacks[stack_index].pop(-1)

    def pop_penult(self,
                   stack_index: int) -> Dict[str, Any]:
        if self.push_buffer[stack_index] is not None:
            self._apply_push()
        stack_0 = self.get_stack(stack_index)[-1]
        stack_0_emb, stack_0_token = stack_0['stack_rnn_input'], stack_0['token']
        self.stacks[stack_index].pop(-1)
        rnt = self.stacks[stack_index].pop(-1)
        self.push(stack_index, stack_0_emb, {'token': stack_0_token})

        return rnt

    def get_stack(self,
                  stack_index: int) -> List[Dict[str, Any]]:
        if self.push_buffer[stack_index] is not None:
            self._apply_push()
        return self.stacks[stack_index]

    def get_stacks(self) -> List[List[Dict[str, Any]]]:
        self._apply_push()
        return self.stacks

    def get_len(self,
                stack_index: int) -> int:
        return len(self.get_stack(stack_index))

    def get_output(self,
                   stack_index: int) -> torch.Tensor:
        return self.get_stack(stack_index)[-1]['stack_rnn_output']

    def _apply_push(self) -> None:
        index_list = []
        inputs = []
        initial_state = []
        layer_dropout_mask = []
        recurrent_dropout_mask = []
        for i, (stack, buffer) in enumerate(zip(self.stacks, self.push_buffer)):
            if buffer is not None:
                index_list.append(i)
                inputs.append(buffer['stack_rnn_input'].unsqueeze(0))
                if len(stack) > 0:
                    initial_state.append(
                        (stack[-1]['stack_rnn_state'].unsqueeze(1), stack[-1]['stack_rnn_memory'].unsqueeze(1)))
                else:
                    initial_state.append(
                        (buffer['stack_rnn_input'].new_zeros(self.num_layers, 1, self.hidden_size),) * 2)
                if self.same_dropout_mask_per_instance:
                    if self.layer_dropout_mask is not None:
                        layer_dropout_mask.append(self.layer_dropout_mask[:, i].unsqueeze(1))
                    if self.recurrent_dropout_mask is not None:
                        recurrent_dropout_mask.append(self.recurrent_dropout_mask[:, i].unsqueeze(1))
                else:
                    if 0.0 < self.layer_dropout_probability < 1.0:
                        layer_dropout_mask.append(get_dropout_mask(self.layer_dropout_probability,
                                                                   torch.ones(self.num_layers, 1, self.hidden_size,
                                                                              device=self.layer_0.input_linearity.weight.device)))
                    if 0.0 < self.recurrent_dropout_probability < 1.0:
                        recurrent_dropout_mask.append(get_dropout_mask(self.recurrent_dropout_probability,
                                                                       torch.ones(self.num_layers, 1, self.hidden_size,
                                                                                  device=self.layer_0.input_linearity.weight.device)))
        if len(layer_dropout_mask) == 0:
            layer_dropout_mask = None
        if len(recurrent_dropout_mask) == 0:
            recurrent_dropout_mask = None
        if len(index_list) > 0:
            inputs = torch.cat(inputs, 0)
            initial_state = list(torch.cat(t, 1) for t in zip(*initial_state))
            if layer_dropout_mask is not None:
                layer_dropout_mask = torch.cat(layer_dropout_mask, 1)
            if recurrent_dropout_mask is not None:
                recurrent_dropout_mask = torch.cat(recurrent_dropout_mask, 1)
            output_state, output_memory = self._forward(inputs, initial_state, layer_dropout_mask,
                                                        recurrent_dropout_mask)
            for i, stack_index in enumerate(index_list):
                output = {
                    'stack_rnn_state': output_state[:, i, :],
                    'stack_rnn_memory': output_memory[:, i, :],
                    'stack_rnn_output': output_state[-1, i, :]
                }
                output.update(self.push_buffer[stack_index])
                self.stacks[stack_index].append(output)
                self.push_buffer[stack_index] = None

    def _forward(self,  # pylint: disable=arguments-differ
                 inputs: torch.Tensor,
                 initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
                 layer_dropout_mask: Optional[torch.Tensor] = None,
                 recurrent_dropout_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : torch.Tensor, required.
            A batch first torch.Tensor to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (num_layers, batch_size, hidden_size).

        Returns
        -------
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size).
        """
        # if not initial_state:
        #     hidden_states = [None] * len(self.lstm_layers)
        # elif initial_state[0].size()[0] != len(self.lstm_layers):
        #     raise ConfigurationError("Initial states were passed to forward() but the number of "
        #                              "initial states does not match the number of layers.")
        # else:
        hidden_states = list(zip(initial_state[0].split(1, 0),
                                 initial_state[1].split(1, 0)))

        previous_output = inputs
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, 'layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            if self.training:
                if self.same_dropout_mask_per_instance:
                    if layer_dropout_mask is not None and i > 0:
                        previous_output = previous_output * layer_dropout_mask[i - 1]
                    if recurrent_dropout_mask is not None:
                        state = (state[0] * recurrent_dropout_mask[i], state[1])
                else:
                    pass
            final_state = layer(previous_output, state)
            previous_output = final_state[0].squeeze(0)

            final_h.append(final_state[0])
            final_c.append(final_state[1])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)
        final_state_tuple = (final_h, final_c)
        return final_state_tuple
