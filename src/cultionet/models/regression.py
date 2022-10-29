from . import model_utils

import torch
import torch.nn.functional as F
from torch_geometric import nn


class RegressionLayer(torch.nn.Module):
    """A regression layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(RegressionLayer, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        conv2d_1 = torch.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        # conv1 = nn.TransformerConv(
        #     mid_channels, mid_channels, heads=1, edge_dim=2, dropout=dropout
        # )
        # pool_resample_down = nn.Sequential(
        #     'x, nrows, ncols, mode',
        #     [
        #         (torch.nn.MaxPool2d(2), 'x -> h'),
        #         (
        #             F.interpolate,
        #             "h, size=(nrows, ncols), mode=mode, align_corners=False -> h"
        #         ),
        #         (self.add_res, 'x, h -> h')
        #     ]
        # )
        conv1 = nn.GCNConv(mid_channels, mid_channels, improved=True)
        conv2 = nn.TransformerConv(
            mid_channels, mid_channels, heads=1, edge_dim=2, dropout=dropout
        )
        lin_layer = torch.nn.Linear(mid_channels, out_channels)

        batchnorm2d_layer = torch.nn.BatchNorm2d(mid_channels)
        batchnorm_layer = nn.BatchNorm(mid_channels)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)
        dropout2d_layer = torch.nn.Dropout2d(dropout)
        dropout_layer = torch.nn.Dropout(dropout)

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight, edge_weight2d, nbatch, nrows, ncols',
            [
                (conv2d_1, 'x -> h'),
                (batchnorm2d_layer, 'h -> h'),
                (dropout2d_layer, 'h -> h'),
                (activate_layer, 'h -> h'),
                # (pool_resample_down, "h1, nrows, ncols, 'bilinear' -> h"),
                # (self.cg, 'h -> h'),
                # (batchnorm_layer, 'h -> h'),
                # (dropout_layer, 'h -> h'),
                # (activate_layer, 'h -> h'),
                (self.cg, 'h -> h'),
                (conv1, 'h, edge_index, edge_weight -> h'),
                (batchnorm_layer, 'h -> h'),
                (dropout_layer, 'h -> h'),
                (activate_layer, 'h -> h'),
                (conv2, 'h, edge_index, edge_weight2d -> h'),
                (batchnorm_layer, 'h -> h'),
                (lin_layer, 'h -> h')
            ]
        )

    def add_res(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return x + h

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
        x = self.gc(x, nbatch, nrows, ncols)

        return self.seq(x, edge_index, edge_weight[:, 1], edge_weight, nbatch, nrows, ncols)


class GraphRegressionLayer(torch.nn.Module):
    """A regression layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(GraphRegressionLayer, self).__init__()

        conv2d_1 = torch.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        conv1 = nn.GCNConv(mid_channels, mid_channels, improved=True)
        conv2 = nn.TransformerConv(
            mid_channels, mid_channels, heads=1, edge_dim=2, dropout=0.1
        )
        batchnorm2d_layer = torch.nn.BatchNorm2d(mid_channels)
        batchnorm_layer = nn.BatchNorm(in_channels=mid_channels)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)
        dropout2d_layer = torch.nn.Dropout2d(dropout)
        dropout_layer = torch.nn.Dropout(dropout)
        lin_layer = torch.nn.Linear(mid_channels, out_channels)

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight, edge_weight2d',
            [
                (conv2d_1, 'x -> h'),
                (batchnorm2d_layer, 'h -> h'),
                (dropout2d_layer, 'h -> h'),
                (activate_layer, 'h -> h'),
                (self.cg, 'h -> h'),
                (conv1, 'h, edge_index, edge_weight -> h'),
                (batchnorm_layer, 'h -> h'),
                (dropout_layer, 'h -> h'),
                (activate_layer, 'h -> h'),
                (conv2, 'h, edge_index, edge_weight2d -> h'),
                (batchnorm_layer, 'h -> h'),
                (lin_layer, 'h -> h')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)
        x = self.gc(x, nbatch, nrows, ncols)

        return self.seq(
            x, edge_index, edge_attrs[:, 1], edge_attrs
        )
