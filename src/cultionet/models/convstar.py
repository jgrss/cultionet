"""Source: https://github.com/0zgur0/ms-convSTAR
"""
import typing as T

from . import model_utils

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable


class ConvSTARCell(nn.Module):
    """Generates a convolutional STAR cell
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvSTARCell, self).__init__()

        padding = int(kernel_size / 2.0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.update.weight)
        init.orthogonal(self.gate.weight)
        init.constant(self.update.bias, 0.0)
        init.constant(self.gate.bias, 1.0)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).to(input_.device)
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        gain = torch.sigmoid(self.gate(stacked_inputs))
        update = torch.tanh(self.update(input_))
        new_state = gain * prev_state + (1 - gain) * update

        return new_state


class ConvSTAR(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
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
            name = 'ConvSTARCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
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
        num_classes_last: int = 2,
        nstage: int = 3,
        kernel_size: int = 3,
        n_layers: int = 6,
        cell: str = 'star'
    ):
        super(StarRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.cell = cell

        self.rnn = ConvSTAR(
            input_size=input_dim,
            hidden_sizes=hidden_dim,
            kernel_sizes=kernel_size,
            n_layers=n_layers
        )

        self.final_local = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )
        self.final_last = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim, num_classes_last, 3, padding=1),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x,
        hidden_s: T.Optional[torch.Tensor] = None
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        # input shape = (B x C x T x H x W)
        batch_size, __, time_size, height, width = x.shape

        # convRNN step
        # hidden_s is a list (number of layer) of hidden states of size [b x c x h x w]
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
            # local_1 = hidden_s[0]
            local_2 = hidden_s[1]
        elif self.nstage == 3:
            # local_1 = hidden_s[1]
            local_2 = hidden_s[3]
        elif self.nstage == 2:
            # local_1 = hidden_s[1]
            local_2 = hidden_s[2]
        elif self.nstage == 1:
            # local_1 = hidden_s[-1]
            local_2 = hidden_s[-1]
        local_2 = self.final_local(local_2)
        last = self.final_last(hidden_s[-1])

        # The output is (B x C x H x W)
        return local_2, last


class Refine(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(Refine, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        self.relu = torch.nn.ReLU(inplace=False)

        self.conv2 = torch.nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode='replicate'
        )
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        self.drop2 = torch.nn.Dropout2d(dropout)

        self.conv3 = torch.nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode='replicate'
        )
        self.bn3 = torch.nn.BatchNorm2d(mid_channels)
        self.drop3 = torch.nn.Dropout2d(dropout)

        self.conv4 = torch.nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out = self.conv2(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += out1
        out = self.relu(out)
        out = self.drop3(out)
        out = self.conv4(out) + x[:, -2:]

        return out
