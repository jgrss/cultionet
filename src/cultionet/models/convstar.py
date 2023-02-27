"""Source: https://github.com/0zgur0/ms-convSTAR
"""
import typing as T

from .base_layers import Softmax, ResidualConv

import torch
from torch.autograd import Variable


class ConvSTARCell(torch.nn.Module):
    """Generates a convolutional STAR cell
    """
    def __init__(
        self, input_size: int, hidden_size: int, kernel_size: int
    ):
        super(ConvSTARCell, self).__init__()

        padding = int(kernel_size / 2.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = torch.nn.Conv2d(
            input_size + hidden_size,
            hidden_size,
            kernel_size,
            padding=padding
        )
        self.update = torch.nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size,
            padding=padding
        )

        torch.nn.init.orthogonal(self.update.weight)
        torch.nn.init.orthogonal(self.gate.weight)
        torch.nn.init.constant(self.update.bias, 0.0)
        torch.nn.init.constant(self.gate.bias, 1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        prev_state: T.Union[None, torch.Tensor]
    ) -> torch.Tensor:
        # get batch and spatial sizes
        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([inputs, prev_state], dim=1)
        gain = self.sigmoid(self.gate(stacked_inputs))
        update = self.tanh(self.update(inputs))
        new_state = gain * prev_state + (1.0 - gain) * update

        return new_state


class ConvSTAR(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: int,
        kernel_sizes: int,
        n_layers: int
    ):
        """Generates a multi-layer convolutional GRU. Preserves spatial dimensions across
        cells, only altering depth.

        :param input_size: integer. depth dimension of input tensors.
        :param hidden_sizes: integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        :param kernel_sizes: integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        :param n_layers: integer. number of chained `ConvSTARCell`.
        """
        super(ConvSTAR, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvSTARCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = f"ConvSTARCell_{str(i).zfill(2)}"

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(
        self,
        x: torch.Tensor,
        hidden: T.Union[None, T.List[torch.Tensor]]
    ) -> T.List[torch.Tensor]:
        """
        :param x: 4D input tensor. (batch, channels, height, width).
        :param hidden: list of 4D hidden state representations. (batch, channels, height, width).
        :returns upd_hidden: 5D hidden representation. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None] * self.n_layers

        input_ = x
        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class StarRNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_classes_l2: int = 2,
        num_classes_last: int = 3,
        n_stage: int = 3,
        kernel_size: int = 3,
        n_layers: int = 6,
        cell: str = 'star',
        crop_type_layer: bool = False,
        activation_type: str = 'LeakyReLU',
        final_activation: str = Softmax(dim=1)
    ):
        super(StarRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_stage = n_stage
        self.cell = cell
        self.crop_type_layer = crop_type_layer

        self.rnn = ConvSTAR(
            input_size=input_dim,
            hidden_sizes=hidden_dim,
            kernel_sizes=kernel_size,
            n_layers=n_layers
        )

        # Level 2 level (non-crop; crop)
        self.final_l2 = torch.nn.Sequential(
            ResidualConv(
                in_channels=int(hidden_dim * 2),
                out_channels=hidden_dim,
                dilations=2,
                activation_type=activation_type
            ),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=num_classes_l2,
                kernel_size=1,
                padding=0
            ),
            final_activation
        )
        # Last level (non-crop; crop; edges)
        self.final_last = torch.nn.Sequential(
            ResidualConv(
                in_channels=int(hidden_dim * 3),
                out_channels=hidden_dim,
                dilations=2,
                activation_type=activation_type
            ),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=num_classes_last,
                kernel_size=1,
                padding=0
            ),
            Softmax(dim=1)
        )

    def forward(
        self,
        x,
        hidden_s: T.Optional[torch.Tensor] = None
    ) -> T.Sequence[torch.Tensor]:
        # input shape = (B x C x T x H x W)
        batch_size, __, time_size, height, width = x.shape

        # convRNN step
        # hidden_s is a list (number of layer) of hidden states of size [B x C x H x W]
        if hidden_s is None:
            hidden_s = [
                torch.zeros(
                    (batch_size, self.hidden_dim, height, width),
                    dtype=x.dtype,
                    device=x.device
                )
            ] * self.n_layers

        for iter_ in range(0, time_size):
            hidden_s = self.rnn(x[:, :, iter_, :, :], hidden_s)

        if self.n_layers == 3:
            local_1 = hidden_s[0]
            local_2 = hidden_s[1]
        elif self.n_stage == 3:
            local_1 = hidden_s[1]
            local_2 = hidden_s[3]
        elif self.n_stage == 2:
            local_1 = hidden_s[1]
            local_2 = hidden_s[2]
        elif self.n_stage == 1:
            local_1 = hidden_s[-1]
            local_2 = hidden_s[-1]

        h_last = hidden_s[-1]
        if self.crop_type_layer:
            last_l2 = self.final_l2(local_2)
            last = self.final_last(h)
            h = torch.cat([local_2, h_last], dim=1)

            return h, last_l2, last
        else:
            h = torch.cat(
                [local_1, local_2], dim=1
            )
            last_l2 = self.final_l2(h)
            h = torch.cat([h, h_last], dim=1)
            last = self.final_last(h)

            # The output is (B x C x H x W)
            return h, last_l2, last
