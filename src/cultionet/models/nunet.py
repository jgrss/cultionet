"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

from . import model_utils
from .base_layers import DoubleConv

import torch


class VGGBlock(torch.nn.Module):
    """A UNet block for graphs
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int
    ):
        super(VGGBlock, self).__init__()

        activate_layer = torch.nn.SiLU(inplace=False)
        conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        batchnorm_layer1 = torch.nn.BatchNorm2d(mid_channels)
        conv2 = torch.nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        batchnorm_layer2 = torch.nn.BatchNorm2d(out_channels)

        self.seq = torch.nn.Sequential(
            conv1,
            batchnorm_layer1,
            activate_layer,
            conv2,
            batchnorm_layer2,
            activate_layer
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling followed by a double convolution
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(PoolConv, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, mid_channels, out_channels)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class BoundaryStream(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BoundaryStream, self).__init__()

        conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.up = model_utils.UpSample()
        self.seq = torch.nn.Sequential(
            conv,
            torch.nn.Sigmoid()
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        s = self.up(s, size=m.shape[-2:])
        x = torch.cat([s, m], dim=1)
        x = self.seq(x)

        if s.size(1) == 2:
            return x * s
        else:
            return x


class Permute(torch.nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class NestedUNet2(torch.nn.Module):
    """UNet++ with residual convolutional dilation

    References:
        https://arxiv.org/pdf/1807.10165.pdf
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_side_channels: int,
        init_filter: int = 64,
        boundary_layer: bool = True,
        linear_fc: bool = False,
        dropout: float = 0.1
    ):
        super(NestedUNet2, self).__init__()

        self.linear_fc = linear_fc
        init_filter = int(init_filter)
        nb_filter = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.up = model_utils.UpSample()

        if boundary_layer:
            self.bound_0 = BoundaryStream(
                in_channels=nb_filter[3]+nb_filter[4],
                out_channels=out_side_channels
            )
            self.bound_1 = BoundaryStream(
                in_channels=out_side_channels+nb_filter[2],
                out_channels=out_side_channels
            )
            self.bound_2 = BoundaryStream(
                in_channels=out_side_channels+nb_filter[1],
                out_channels=out_side_channels
            )
            self.bound_3 = BoundaryStream(
                in_channels=out_side_channels+nb_filter[0],
                out_channels=out_side_channels
            )

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = PoolConv(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = PoolConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = PoolConv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = PoolConv(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.linear_fc:
            self.net_final = torch.nn.Sequential(
                torch.nn.Conv2d(
                    nb_filter[0],
                    nb_filter[0],
                    kernel_size=3,
                    padding=1
                ),
                torch.nn.BatchNorm2d(nb_filter[0]),
                torch.nn.ReLU(inplace=False),
                Permute((0, 2, 3, 1)),
                torch.nn.Linear(
                    nb_filter[0], out_channels
                ),
                Permute((0, 3, 1, 2))
            )
        else:
            self.net_final = torch.nn.Sequential(
                torch.nn.Conv2d(
                    nb_filter[0]+out_side_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                ),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ELU(alpha=0.1, inplace=False),
            )

        if boundary_layer:
            self.side_final = torch.nn.Sequential(
                torch.nn.Conv2d(
                    out_side_channels, out_side_channels, kernel_size=3, padding=1
                ),
                torch.nn.BatchNorm2d(out_side_channels),
                torch.nn.ELU(alpha=0.1, inplace=False)
            )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        net = None
        side = None

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)

        # 1/2
        x1_0 = self.conv1_0(x0_0)
        # 1/1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, size=x0_0.shape[-2:])], dim=1))

        # 1/4
        x2_0 = self.conv2_0(x1_0)
        # 1/2
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, size=x1_0.shape[-2:])], dim=1))
        # 1/1
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, size=x0_1.shape[-2:])], dim=1))

        # 1/8
        x3_0 = self.conv3_0(x2_0)
        # 1/4
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, size=x2_0.shape[-2:])], dim=1))
        # 1/2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, size=x1_1.shape[-2:])], dim=1))
        # 1/1
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, size=x0_2.shape[-2:])], dim=1))

        # 1/16
        x4_0 = self.conv4_0(x3_0)
        # 1/8
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, size=x3_0.shape[-2:])], dim=1))
        # 1/4
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, size=x2_1.shape[-2:])], dim=1))
        # 1/2
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, size=x1_2.shape[-2:])], dim=1))
        # 1/1
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, size=x0_3.shape[-2:])], dim=1))
        if self.linear_fc:
            net = self.net_final(x0_4)
        else:
            # Side stream
            b_0 = self.bound_0(x4_0, x3_0)
            b_1 = self.bound_1(b_0, x2_0)
            b_2 = self.bound_2(b_1, x1_0)
            side = self.bound_3(b_2, x0_0)
            side = self.side_final(side)
            net = self.net_final(torch.cat([x0_4, side], dim=1))

        return {
            'net': net,
            'side': side
        }


class TemporalNestedUNet2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_side_channels: int,
        init_filter: int,
        boundary_layer: bool = True,
        linear_fc: bool = False,
        dropout: float = 0.1
    ):
        super(TemporalNestedUNet2, self).__init__()

        self.nunet = NestedUNet2(
            in_channels=in_channels,
            out_channels=out_channels,
            out_side_channels=out_side_channels,
            init_filter=init_filter,
            boundary_layer=boundary_layer,
            linear_fc=linear_fc,
            dropout=dropout
        )

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        return self.nunet(x)
