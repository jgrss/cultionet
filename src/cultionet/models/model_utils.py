import typing as T

import einops
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
        return einops.rearrange(
            x,
            '(b h w) c -> b c h w',
            b=nbatch,
            c=x.shape[1],
            h=nrows,
            w=ncols,
        )


class ConvToGraph(torch.nn.Module):
    """Reshapes a 4d tensor to a 2d tensor."""

    def __init__(self):
        super(ConvToGraph, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, 'b c h w -> (b h w) c')


class ConvToTime(torch.nn.Module):
    """Reshapes a 4d tensor to a 5d tensor."""

    def __init__(self):
        super(ConvToTime, self).__init__()

    def forward(
        self, x: torch.Tensor, nbands: int, ntime: int
    ) -> torch.Tensor:
        nbatch, __, height, width = x.shape

        return einops.rearrange(
            x,
            'b (bands t) h w -> b bands t h w',
            b=nbatch,
            bands=nbands,
            t=ntime,
            h=height,
            w=width,
        )


def max_pool_neighbor_x(
    x: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    return nn.max_pool_neighbor_x(Data(x=x, edge_index=edge_index)).x


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return nn.global_max_pool(x=x, batch=batch, size=x.shape[0])
