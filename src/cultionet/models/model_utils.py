import typing as T

import torch
from torch_geometric import nn
from torch_geometric.data import Data


def get_batch_count(batch: torch.Tensor) -> int:
    return batch.unique().size(0)


class UpSample(torch.nn.Module):
    """Up-samples a tensor."""

    def __init__(self):
        super(UpSample, self).__init__()

    def forward(
        self, x: torch.Tensor, size: T.Sequence[int], mode: str = "bilinear"
    ) -> torch.Tensor:
        upsampler = torch.nn.Upsample(size=size, mode=mode, align_corners=True)

        return upsampler(x)


class GraphToConv(torch.nn.Module):
    """Reshapes a 2d tensor to a 4d tensor."""

    def __init__(self):
        super(GraphToConv, self).__init__()

    def forward(
        self, x: torch.Tensor, nbatch: int, nrows: int, ncols: int
    ) -> torch.Tensor:
        n_channels = x.shape[1]
        return x.reshape(nbatch, nrows, ncols, n_channels).permute(0, 3, 1, 2)


class ConvToGraph(torch.nn.Module):
    """Reshapes a 4d tensor to a 2d tensor."""

    def __init__(self):
        super(ConvToGraph, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nbatch, n_channels, nrows, ncols = x.shape

        return x.permute(0, 2, 3, 1).reshape(
            nbatch * nrows * ncols, n_channels
        )


class ConvToTime(torch.nn.Module):
    """Reshapes a 4d tensor to a 5d tensor."""

    def __init__(self):
        super(ConvToTime, self).__init__()

    def forward(
        self, x: torch.Tensor, nbands: int, ntime: int
    ) -> torch.Tensor:
        nbatch, __, height, width = x.shape

        return x.reshape(nbatch, nbands, ntime, height, width)


def max_pool_neighbor_x(
    x: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    return nn.max_pool_neighbor_x(Data(x=x, edge_index=edge_index)).x


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return nn.global_max_pool(x=x, batch=batch, size=x.shape[0])
