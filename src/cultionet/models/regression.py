from . import model_utils
from .base_layers import DoubleConv
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
        dropout: float = 0.1,
        alpha: float = 0.1
    ):
        super(GraphRegressionLayer, self).__init__()

        # conv2d = DoubleConv(
        #     in_channels=in_channels,
        #     mid_channels=mid_channels,
        #     out_channels=mid_channels,
        #     alpha=alpha
        # )
        conv2d_1 = torch.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        conv2d_2 = torch.nn.Conv2d(
            mid_channels,
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
        dropout_layer = torch.nn.Dropout(dropout)
        activate_layer1 = torch.nn.ReLU(inplace=False)
        activate_layer2 = torch.nn.ELU(alpha=alpha, inplace=False)
        lin_layer = torch.nn.Linear(mid_channels, out_channels)

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        # Added extra dropout and converted 1d to
        self.seq = nn.Sequential(
            'x, edge_index, edge_weight, edge_weight2d',
            [
                (conv2d_1, 'x -> x'),
                (batchnorm2d_layer, 'x -> x'),
                (activate_layer1, 'x -> x'),
                (conv2d_2, 'x -> x'),
                (batchnorm2d_layer, 'x -> x'),
                (activate_layer1, 'x -> x'),
                (dropout_layer, 'x -> x'),
                (self.cg, 'x -> x'),
                (conv1, 'x, edge_index, edge_weight -> x'),
                (batchnorm_layer, 'x -> x'),
                (activate_layer2, 'x -> x'),
                (conv2, 'x, edge_index, edge_weight2d -> x'),
                (batchnorm_layer, 'x -> x'),
                (activate_layer2, 'x -> x'),
                (lin_layer, 'x -> x')
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


class RegressionLayer(torch.nn.Module):
    """A regression layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        rheight: int = None,
        rwidth: int = None
    ):
        super(RegressionLayer, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()
        # self.resize_up = transforms.Resize((rheight, rwidth))
        self.lin = torch.nn.Linear(mid_channels, out_channels)
        self.act = torch.nn.Sigmoid()

        self.seq = nn.Sequential(
            'x',
            [
                (
                    torch.nn.Conv2d(
                        in_channels,
                        mid_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                    'x -> x'
                ),
                (torch.nn.BatchNorm2d(mid_channels), 'x -> x'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'x -> x'),
                (
                    ResConv(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        padding=2,
                        dilation=2
                    ),
                    'x -> x'
                ),
                (torch.nn.BatchNorm2d(mid_channels), 'x -> x')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)

        # resize_down = transforms.Resize((nrows, ncols))
        # Graph to Conv
        x = self.gc(x, nbatch, nrows, ncols)
        # Resample
        # x = self.resize_up(x)
        # Forward pass
        h = self.seq(x)
        # Resize back
        # h = resize_down(h)
        # Conv to Graph
        h = self.cg(h)
        h = self.lin(h)
        h = self.act(h)

        return h
