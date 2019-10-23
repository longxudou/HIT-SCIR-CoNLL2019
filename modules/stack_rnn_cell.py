"""
Based on augmented_lstm.py from AllenNLP 0.8.3
"""
from typing import Optional, Tuple

import torch
from allennlp.nn.initializers import block_orthogonal


class StackRnnCellBase(torch.nn.Module):
    pass


class StackLstmCell(StackRnnCellBase):
    """
    An LSTM with Recurrent Dropout and the option to use highway
    connections between layers. Note: this implementation is slower
    than the native Pytorch LSTM because it cannot make use of CUDNN
    optimizations for stacked RNNs due to the highway layers and
    variational dropout.

    Parameters
    ----------
    input_size : int, required.
        The dimension of the inputs to the LSTM.
    hidden_size : int, required.
        The dimension of the outputs of the LSTM.
    use_highway: bool, optional (default = True)
        Whether or not to use highway connections between layers. This effectively involves
        reparameterising the normal output of an LSTM as::

            gate = sigmoid(W_x1 * x_t + W_h * h_t)
            output = gate * h_t  + (1 - gate) * (W_x2 * x_t)
    use_input_projection_bias : bool, optional (default = True)
        Whether or not to use a bias on the input projection layer. This is mainly here
        for backwards compatibility reasons and will be removed (and set to False)
        in future releases.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 use_highway: bool = True,
                 use_input_projection_bias: bool = True) -> None:
        super(StackLstmCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.use_highway = use_highway

        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if use_highway:
            self.input_linearity = torch.nn.Linear(input_size, 6 * hidden_size, bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 5 * hidden_size, bias=True)
        else:
            self.input_linearity = torch.nn.Linear(input_size, 4 * hidden_size, bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : torch.Tensor, required.
            A tensor of shape (batch_size, input_size)
            to apply the LSTM over.

        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, hidden_size).
        """
        batch_size = inputs.size(0)

        if initial_state is None:
            previous_memory = inputs.new_zeros(batch_size, self.hidden_size)
            previous_state = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(inputs)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Mimic the pytorch API by returning state in the following shape:
        # (num_layers * num_directions, batch_size, hidden_size). As this
        # LSTM cannot be stacked, the first dimension here is just 1.
        final_state = (timestep_output.unsqueeze(0),
                       memory.unsqueeze(0))

        return final_state
