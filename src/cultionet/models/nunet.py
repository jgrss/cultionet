"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
from . import model_utils
from .base_layers import DoubleConv, ResConv

import torch
from torch_geometric import nn
from torchvision import transforms


class VGGBlock(torch.nn.Module):
    """A UNet block for graphs
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int
    ):
        super(VGGBlock, self).__init__()

        conv1 = nn.GCNConv(in_channels, mid_channels, improved=True)
        conv2 = nn.GCNConv(mid_channels, out_channels, improved=True)

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight',
            [
                (conv1, 'x, edge_index, edge_weight -> x'),
                (nn.BatchNorm(mid_channels), 'x -> x'),
                (conv2, 'x, edge_index, edge_weight -> x'),
                (nn.BatchNorm(out_channels), 'x -> x'),
                (model_utils.max_pool_neighbor_x, 'x, edge_index -> x'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'x -> x')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.seq(x, edge_index, edge_weight)


class PoolConv(torch.nn.Module):
    """Max pooling followed by a double convolution
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(PoolConv, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, mid_channels, out_channels)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet2(torch.nn.Module):
    """UNet++ with residual convolutional dilation

    References:
        https://arxiv.org/pdf/1807.10165.pdf
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
    """
    def __init__(self, in_channels: int, out_channels: int, init_filter: int = 32):
        super(NestedUNet2, self).__init__()

        nb_filter = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()
        self.up = model_utils.UpSample()

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = PoolConv(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_0 = PoolConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = PoolConv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = PoolConv(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final2d = torch.nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final = nn.GCNConv(out_channels, out_channels, improved=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)

        x0_0 = self.conv0_0(x, edge_index, edge_weight)
        # Reshape to CNN 4d
        x0_0 = self.gc(x0_0, nbatch, nrows, ncols)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, size=x0_0.shape[2:])], dim=1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, size=x1_0.shape[2:])], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, size=x0_1.shape[2:])], dim=1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, size=x2_0.shape[2:])], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, size=x1_1.shape[2:])], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, size=x0_2.shape[2:])], dim=1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, size=x3_0.shape[2:])], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, size=x2_1.shape[2:])], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, size=x1_2.shape[2:])], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, size=x0_3.shape[2:])], dim=1))

        output = self.final2d(x0_4)
        # Reshape to GNN 2d
        output = self.cg(output)

        output = self.final(output, edge_index, edge_weight)

        return output
