import typing as T

from . import model_utils

import torch
from torch_geometric import nn


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


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


def conv_batchnorm_activate2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    dilation: int = 1,
    add_activation: bool = True
):
    layers = [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ),
        torch.nn.BatchNorm2d(out_channels)
    ]
    if add_activation:
        layers += [torch.nn.LeakyReLU(inplace=False)]

    return torch.nn.Sequential(*layers)


def conv_batchnorm_activate3d(
    in_channels: int,
    in_time: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    dilation: int = 1,
    add_activation: bool = True,
    squeeze: bool = True
):
    layers = [
        torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
    ]
    if squeeze:
        layers += [
            Squeeze(),
            torch.nn.BatchNorm2d(in_time)
        ]
    else:
        layers += [torch.nn.BatchNorm3d(out_channels)]

    if add_activation:
        layers += [torch.nn.LeakyReLU(inplace=False)]

    return torch.nn.Sequential(*layers)


class AttentionAdd(torch.nn.Module):
    def __init__(self):
        super(AttentionAdd, self).__init__()

        self.up = model_utils.UpSample()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != g.shape[-2:]:
            x = self.up(x, size=g.shape[-2:], mode='bilinear')

        return x + g


class AttentionGate(torch.nn.Module):
    def __init__(
        self,
        high_channels: int,
        low_channels: int
    ):
        super(AttentionGate, self).__init__()

        conv_x = torch.nn.Conv2d(
            high_channels,
            high_channels,
            kernel_size=1,
            padding=0
        )
        conv_g = torch.nn.Conv2d(
            low_channels,
            high_channels,
            kernel_size=1,
            padding=0,
        )
        conv1d = torch.nn.Conv2d(
            high_channels,
            1,
            kernel_size=1,
            padding=0
        )
        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'x, g',
            [
                (conv_x, 'x -> x'),
                (conv_g, 'g -> g'),
                (AttentionAdd(), 'x, g -> x'),
                (torch.nn.LeakyReLU(inplace=False), 'x -> x'),
                (conv1d, 'x -> x'),
                (torch.nn.Sigmoid(), 'x -> x')
            ]
        )
        self.final = conv_batchnorm_activate2d(
            in_channels=high_channels,
            out_channels=high_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Higher dimension
            g: Lower dimension
        """
        h = self.seq(x, g)
        if h.shape[-2:] != x.shape[-2:]:
            h = self.up(h, size=x.shape[-2:], mode='bilinear')

        return self.final(x * h)


class ChannelAttention(torch.nn.Module):
    """Residual Channel Attention Block

    References:
        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
    """
    def __init__(self, channels: int):
        super(ChannelAttention, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.module(x)


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
            conv_batchnorm_activate2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            conv_batchnorm_activate2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
            conv_batchnorm_activate2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling with dropout
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


class ResidualConv(torch.nn.Module):
    """A residual convolution layer with (optional) RCAB
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention: bool = False
    ):
        super(ResidualConv, self).__init__()

        layers = [
            conv_batchnorm_activate2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            conv_batchnorm_activate2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2
            )
        ]
        if channel_attention:
            layers += [ChannelAttention(channels=out_channels)]

        self.seq = torch.nn.Sequential(*layers)
        self.expand = conv_batchnorm_activate2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.expand(x)


class ResidualConvRCAB(torch.nn.Module):
    """A group of residual convolution layers with (optional) RCAB
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention: bool = False,
        res_blocks: int = 2
    ):
        super(ResidualConvRCAB, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        for __ in range(0, res_blocks):
            layers += [
                ResidualConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    channel_attention=channel_attention
                )
            ]
        layers += [
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        self.seq = torch.nn.Sequential(*layers)
        self.expand = conv_batchnorm_activate2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.expand(x)


class PoolResidualConv(torch.nn.Module):
    """Max pooling followed by a residual convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None,
        channel_attention: bool = False,
        res_blocks: int = 0
    ):
        super(PoolResidualConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]

        if dropout is not None:
            assert isinstance(dropout, float), 'The dropout arg must be a float.'
            layers += [torch.nn.Dropout(dropout)]

        if res_blocks > 0:
            layers += [
                ResidualConvRCAB(
                    in_channels,
                    out_channels,
                    channel_attention=channel_attention,
                    res_blocks=res_blocks
                ),
                torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ]

        else:
            layers += [
                ResidualConv(
                    in_channels,
                    out_channels,
                    channel_attention=channel_attention
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

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

        self.seq = conv_batchnorm_activate2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TemporalConv(torch.nn.Module):
    """A temporal convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int
    ):
        super(TemporalConv, self).__init__()

        layers = [
            conv_batchnorm_activate3d(
                in_channels=in_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                squeeze=False
            ),
            conv_batchnorm_activate3d(
                in_channels=out_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                squeeze=True
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
