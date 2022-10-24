from . import model_utils
from .nunet import ResConv

import torch
from torch_geometric import nn
# from torchvision import transforms


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
