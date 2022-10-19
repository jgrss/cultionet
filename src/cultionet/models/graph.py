import typing as T

import torch
from torch_geometric import nn


class GraphMid(torch.nn.Module):
    def __init__(self, conv: T.Callable):
        super(GraphMid, self).__init__()

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight',
            [
                (conv, 'x, edge_index, edge_weight -> x'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'x -> x')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor
    ) -> torch.Tensor:
        return self.seq(
            x, edge_index, edge_attrs
        )


class GraphFinal(torch.nn.Module):
    def __init__(
        self,
        conv1: T.Callable,
        conv2: T.Callable,
        mid_channels: int,
        out_channels: int
    ):
        super(GraphFinal, self).__init__()

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight, edge_weight2d',
            [
                (conv1, 'x, edge_index, edge_weight -> x'),
                (nn.BatchNorm(in_channels=mid_channels), 'x -> x'),
                (conv2, 'x, edge_index, edge_weight2d -> x'),
                (nn.BatchNorm(in_channels=out_channels), 'x -> x'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'x -> x')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor
    ) -> torch.Tensor:
        return self.seq(
            x, edge_index, edge_attrs[:, 1], edge_attrs
        )
