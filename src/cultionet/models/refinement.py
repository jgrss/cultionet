from . import model_utils

import torch
from torch_geometric import nn


class RefineConv(torch.nn.Module):
    """A refinement convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(RefineConv, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        conv2d_1 = torch.nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
        )
        conv2d_2 = torch.nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate'
        )
        conv2d_3 = torch.nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
        )
        batchnorm_layer = torch.nn.BatchNorm2d(mid_channels)
        dropout_layer = torch.nn.Dropout(dropout)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)

        self.seq = nn.Sequential(
            'x',
            [
                (conv2d_1, 'x -> x'),
                (batchnorm_layer, 'x -> x'),
                (activate_layer, 'x -> x'),
                (conv2d_2, 'x -> x'),
                (batchnorm_layer, 'x -> x'),
                (activate_layer, 'x -> x'),
                (dropout_layer, 'x -> x'),
                (conv2d_2, 'x -> x'),
                (batchnorm_layer, 'x -> x'),
                (activate_layer, 'x -> x'),
                (dropout_layer, 'x -> x'),
                (conv2d_3, 'x -> x')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        batch_size: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        x = self.gc(
            x,
            batch_size,
            height,
            width
        )
        x = self.seq(x)

        return self.cg(x)
