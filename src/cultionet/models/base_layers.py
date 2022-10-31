import typing as T

import torch
from torch_geometric import nn


class Permute(torch.nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class ResAdd(torch.nn.Module):
    def __init__(self):
        super(ResAdd, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class DoubleResConv(torch.nn.Module):
    """A double residual convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(DoubleResConv, self).__init__()

        conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)
        add_layer = ResAdd()

        self.seq = nn.Sequential(
            'x',
            [
                (conv1, 'x -> h1'),
                (batchnorm_layer, 'h1 -> h'),
                (activate_layer, 'h -> h'),
                (conv2, 'h -> h'),
                (batchnorm_layer, 'h -> h'),
                (add_layer, 'h, h1 -> h'),
                (activate_layer, 'h -> h')
            ]
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DoubleConv(torch.nn.Module):
    """A double convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(DoubleConv, self).__init__()

        conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)

        self.seq = torch.nn.Sequential(
            conv1,
            batchnorm_layer,
            activate_layer,
            conv2,
            batchnorm_layer,
            activate_layer
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
