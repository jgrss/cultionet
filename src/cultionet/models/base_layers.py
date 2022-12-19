import typing as T

import torch
from torch_geometric import nn


class PoolConvSingle(torch.nn.Module):
    """Max pooling followed by convolution
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
            torch.nn.LeakyReLU(inplace=False)
        )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolResidualConv(torch.nn.Module):
    """Max pooling followed by a residual convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None
    ):
        super(PoolResidualConv, self).__init__()

        if dropout is not None:
            self.seq = torch.nn.Sequential(
                torch.nn.MaxPool2d(pool_size),
                torch.nn.Dropout(dropout),
                ResidualConv(in_channels, out_channels)
            )
        else:
            self.seq = torch.nn.Sequential(
                torch.nn.MaxPool2d(pool_size),
                ResidualConv(in_channels, out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Permute(torch.nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class ResidualConv(torch.nn.Module):
    """A residual convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(ResidualConv, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=2, dilation=2
            )
        )
        self.final = nn.Sequential(
            'h, x',
            [
                (torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1
                ), 'x -> x'),
                (torch.nn.BatchNorm2d(out_channels), 'x -> x'),
                (Add(), 'x, h -> h')
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.seq(x), x)


class SingleConv(torch.nn.Module):
    """A single convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(SingleConv, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False)
        )

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

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
