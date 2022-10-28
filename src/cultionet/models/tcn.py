import typing as T

from .graph import GraphMid

import torch
from torch.nn.utils import weight_norm
from torch_geometric import nn


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # N, C, D
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size1d: int,
        kernel_size2d: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float
    ):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            torch.nn.Conv3d(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=(kernel_size1d, kernel_size2d, kernel_size2d),
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )
        chomp = Chomp1d(padding)
        self.activate_layer = torch.nn.ReLU()
        batchnorm_layer = torch.nn.BatchNorm3d(n_outputs)
        # dropout = torch.nn.Dropout3d(dropout)

        self.conv2 = weight_norm(
            torch.nn.Conv3d(
                in_channels=n_outputs,
                out_channels=n_outputs,
                kernel_size=(kernel_size1d, kernel_size2d, kernel_size2d),
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )

        self.net = torch.nn.Sequential(
            self.conv1,
            chomp,
            batchnorm_layer,
            self.activate_layer,
            self.conv2,
            chomp,
            batchnorm_layer,
            self.activate_layer
        )
        self.downsample = torch.nn.Conv3d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.activate_layer(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: T.Sequence[int],
        num_time: int,
        out_channels: int,
        kernel_size1d: int = 2,
        kernel_size2d: int = 3,
        dropout: float = 0.1
    ):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            block_out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    block_out_channels,
                    kernel_size1d,
                    kernel_size2d,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size1d-1) * dilation_size,
                    dropout=dropout
                )
            ]

        self.network = torch.nn.Sequential(*layers)
        self.final = torch.nn.Conv2d(
            in_channels=num_channels[-1]*num_time,
            out_channels=out_channels,
            kernel_size=kernel_size2d,
            padding=1,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        x = self.network(x)
        # (B x C x T x H x W)
        batch_size, __, __, height, width = x.shape
        # (B x C x H x W)
        x = x.reshape(batch_size, -1, height, width)

        return self.final(x)



class GraphTransformer(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.1
    ):
        super(GraphTransformer, self).__init__()

        self.num_features = num_features
        self.in_channels = in_channels
        conv = nn.TransformerConv(
            in_channels, mid_channels, heads=heads, edge_dim=2, dropout=dropout
        )
        self.net = GraphMid(conv)
        self.final = nn.GCNConv(
            heads*mid_channels*int(num_features / in_channels), out_channels,
            improved=True
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor
    ) -> torch.Tensor:
        h = []
        # Iterate over all data features, stepping over time chunks
        for band in range(0, self.num_features, self.in_channels):
            # Get the current band through all time steps
            t = self.net(
                x[:, band:band+self.in_channels],
                edge_index,
                edge_attrs
            )
            h.append(t)
        h = torch.cat(h, dim=1)
        # Reshape to GNN 2d
        h = self.final(h, edge_index, edge_attrs[:, 1])

        return h
