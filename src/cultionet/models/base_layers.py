import typing as T

import torch
from torch_geometric import nn


class PoolConvSingle(torch.nn.Module):
    """Max pooling followed by a double convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2
    ):
        super(PoolConvSingle, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(pool_size),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling followed by a double convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None
    ):
        super(PoolConv, self).__init__()

        if dropout is not None:
            self.seq = torch.nn.Sequential(
                torch.nn.MaxPool2d(pool_size),
                torch.nn.Dropout(dropout),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.seq = torch.nn.Sequential(
                torch.nn.MaxPool2d(pool_size),
                DoubleConv(in_channels, out_channels)
            )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
        activate_layer = torch.nn.ELU(inplace=False)
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


class SingleConv(torch.nn.Module):
    """A single convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(SingleConv, self).__init__()

        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        batchnorm_layer = torch.nn.BatchNorm2d(out_channels)
        activate_layer = torch.nn.ELU(inplace=False)

        self.seq = torch.nn.Sequential(
            conv,
            batchnorm_layer,
            activate_layer
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
        activate_layer = torch.nn.ELU(inplace=False)

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
