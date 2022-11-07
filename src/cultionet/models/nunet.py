"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

from . import model_utils
from .base_layers import (
    SingleConv,
    DoubleConv,
    Permute,
    PoolConv,
    PoolConvSingle
)

import torch
import torch.nn.functional as F
from torch_geometric import nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class AttentionAdd(torch.nn.Module):
    def __init__(self):
        super(AttentionAdd, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if x.shape != g.shape:
            x = F.interpolate(
                x, size=g.shape[-2:], mode='bilinear', align_corners=True
            )

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
            kernel_size=3,
            padding=1,
            stride=2
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
        add_layer = AttentionAdd()
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)
        sigmoid_layer = torch.nn.Sigmoid()

        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'x, g',
            [
                (conv_x, 'x -> x'),
                (conv_g, 'g -> g'),
                (add_layer, 'x, g -> x'),
                (activate_layer, 'x -> x'),
                (conv1d, 'x -> x'),
                (sigmoid_layer, 'x -> x')
            ]
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(high_channels, high_channels, 1, padding=0),
            torch.nn.BatchNorm2d(high_channels)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Higher dimension
            x: Lower dimension
        """
        h = self.seq(x, g)
        h = self.up(h, size=x.shape[-2:], mode='bilinear')

        return self.final(x * h)


class NestedUNet2(torch.nn.Module):
    """UNet++

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
        deep_supervision: bool = False
    ):
        super(NestedUNet2, self).__init__()

        self.linear_fc = linear_fc
        self.boundary_layer = boundary_layer
        self.deep_supervision = deep_supervision

        init_filter = int(init_filter)
        nb_filter = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.up = model_utils.UpSample()

        self.attention_0 = AttentionGate(
            high_channels=nb_filter[3],
            low_channels=nb_filter[4]
        )
        self.attention_1 = AttentionGate(
            high_channels=nb_filter[2],
            low_channels=nb_filter[3]
        )
        self.attention_2 = AttentionGate(
            high_channels=nb_filter[1],
            low_channels=nb_filter[2]
        )
        self.attention_3 = AttentionGate(
            high_channels=nb_filter[0],
            low_channels=nb_filter[1]
        )

        if boundary_layer:
            # Right stream
            self.bound4_1 = DoubleConv(nb_filter[4]+nb_filter[3], nb_filter[3])
            self.bound3_1 = DoubleConv(nb_filter[3]+nb_filter[2], nb_filter[2])
            self.bound2_1 = DoubleConv(nb_filter[2]+nb_filter[1], nb_filter[1])
            self.bound1_1 = DoubleConv(nb_filter[1]+nb_filter[0], nb_filter[0])
            # Left stream
            self.bound4_0 = DoubleConv(nb_filter[4]+nb_filter[3], nb_filter[3])
            self.bound3_0 = DoubleConv(nb_filter[3]+nb_filter[2], nb_filter[2])
            self.bound2_0 = DoubleConv(nb_filter[2]+nb_filter[1], nb_filter[1])
            self.bound1_0 = DoubleConv(nb_filter[1]+nb_filter[0], nb_filter[0])
            # Top stream
            self.bound0_1 = DoubleConv(nb_filter[0]+nb_filter[0], nb_filter[0])
            self.bound0_2 = DoubleConv(nb_filter[0]+nb_filter[0], nb_filter[0])
            self.bound0_3 = DoubleConv(nb_filter[0]+nb_filter[0], nb_filter[0])
            self.bound0_4 = DoubleConv(nb_filter[0]+nb_filter[0], nb_filter[0])

            if self.deep_supervision:
                final_bound_in_channels = out_side_channels
                self.bound_final_1 = torch.nn.Conv2d(
                    nb_filter[0], out_side_channels, kernel_size=1, padding=0
                )
                self.bound_final_2 = torch.nn.Conv2d(
                    nb_filter[0], out_side_channels, kernel_size=1, padding=0
                )
                self.bound_final_3 = torch.nn.Conv2d(
                    nb_filter[0], out_side_channels, kernel_size=1, padding=0
                )
            else:
                final_bound_in_channels = nb_filter[0] + nb_filter[0] + nb_filter[0]
            self.bound_final = torch.nn.Conv2d(
                final_bound_in_channels, out_side_channels,
                kernel_size=1,
                padding=0
            )

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = PoolConv(nb_filter[0], nb_filter[1], dropout=0.25)
        self.conv2_0 = PoolConv(nb_filter[1], nb_filter[2], dropout=0.5)
        self.conv3_0 = PoolConv(nb_filter[2], nb_filter[3], dropout=0.5)
        self.conv4_0 = PoolConv(nb_filter[3], nb_filter[4], dropout=0.5)

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.linear_fc:
            self.net_final = torch.nn.Sequential(
                torch.nn.ELU(alpha=0.1, inplace=False),
                Permute((0, 2, 3, 1)),
                torch.nn.Linear(
                    nb_filter[0], out_channels
                ),
                Permute((0, 3, 1, 2))
            )
        else:
            if self.deep_supervision:
                in_final_layers = out_channels

                self.final_1 = torch.nn.Conv2d(
                    nb_filter[0], out_channels, kernel_size=1, padding=0
                )
                self.final_2 = torch.nn.Conv2d(
                    nb_filter[0], out_channels, kernel_size=1, padding=0
                )
                self.final_3 = torch.nn.Conv2d(
                    nb_filter[0], out_channels, kernel_size=1, padding=0
                )
                self.final_4 = torch.nn.Conv2d(
                    nb_filter[0], out_channels, kernel_size=1, padding=0
                )
            else:
                in_final_layers = nb_filter[0]

            if boundary_layer:
                in_final_layers += out_side_channels

            self.net_final = torch.nn.Conv2d(
                in_final_layers,
                out_channels,
                kernel_size=1,
                padding=0
            )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        mask = None
        boundary = None

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
        x3_0 = self.attention_0(x3_0, x4_0)
        # 1/8
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, size=x3_0.shape[-2:])], dim=1))
        x2_1 = self.attention_1(x2_1, x3_1)
        # 1/4
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, size=x2_1.shape[-2:])], dim=1))
        x1_2 = self.attention_2(x1_2, x2_2)
        # 1/2
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, size=x1_2.shape[-2:])], dim=1))
        x0_3 = self.attention_3(x0_3, x1_3)
        # 1/1
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, size=x0_3.shape[-2:])], dim=1))

        if self.boundary_layer:
            # Left stream
            b4_0 = self.bound4_0(torch.cat([x3_0, self.up(x4_0, size=x3_0.shape[-2:])], dim=1))
            b3_0 = self.bound3_0(torch.cat([x2_0, self.up(b4_0, size=x2_0.shape[-2:])], dim=1))
            b2_0 = self.bound2_0(torch.cat([x1_0, self.up(b3_0, size=x1_0.shape[-2:])], dim=1))
            # Right stream
            b4_1 = self.bound4_1(torch.cat([x3_1, self.up(x4_0, size=x3_1.shape[-2:])], dim=1))
            b3_1 = self.bound3_1(torch.cat([x2_2, self.up(b4_1, size=x2_2.shape[-2:])], dim=1))
            b2_1 = self.bound2_1(torch.cat([x1_3, self.up(b3_1, size=x1_3.shape[-2:])], dim=1))

        if self.boundary_layer:
            # End of left stream
            b1_0 = self.bound1_0(torch.cat([x0_0, self.up(b2_0, size=x0_0.shape[-2:])], dim=1))
            # End of right stream
            b1_1 = self.bound1_1(torch.cat([x0_4, self.up(b2_1, size=x0_4.shape[-2:])], dim=1))
            # Top stream
            b0_1 = self.bound0_1(torch.cat([x0_0, x0_1], dim=1))
            b0_2 = self.bound0_2(torch.cat([b0_1, x0_2], dim=1))
            b0_3 = self.bound0_3(torch.cat([b0_2, x0_3], dim=1))
            b0_4 = self.bound0_4(torch.cat([b0_3, x0_4], dim=1))

        if self.linear_fc:
            mask = self.net_final(x0_4)
        else:
            if self.boundary_layer:
                if self.deep_supervision:
                    # Connect boundary stream
                    b1_0 = self.bound_final_1(b1_0)
                    b0_4 = self.bound_final_2(b0_4)
                    b1_1 = self.bound_final_3(b1_1)
                    boundary = (b1_0 + b0_4 + b1_1) / 3.0
                    # Average over skip connections
                    x0_1 = self.final_1(x0_1)
                    x0_2 = self.final_2(x0_2)
                    x0_3 = self.final_3(x0_3)
                    x0_4 = self.final_4(x0_4)
                    x0_4 = (x0_1 + x0_2 + x0_3 + x0_4) / 4.0
                else:
                    # Connect boundary stream
                    boundary = self.bound_final(torch.cat([b1_0, b0_4, b1_1], dim=1))
                mask = self.net_final(torch.cat([x0_4, boundary], dim=1))
            else:
                mask = self.net_final(x0_4)

        return {
            'mask': mask,
            'boundary': boundary
        }


class TemporalNestedUNet2(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        in_channels: int,
        out_channels: int,
        out_side_channels: int,
        init_filter: int,
        boundary_layer: bool = True,
        linear_fc: bool = False
    ):
        super(TemporalNestedUNet2, self).__init__()

        self.num_features = num_features
        self.in_channels = in_channels

        self.nunet = NestedUNet2(
            in_channels=in_channels,
            out_channels=out_channels,
            out_side_channels=out_side_channels,
            init_filter=init_filter,
            boundary_layer=boundary_layer,
            linear_fc=linear_fc
        )

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        nunet_stream = []
        for band in range(0, self.num_features, self.in_channels):
            t = self.nunet(x[:, band:band+self.in_channels])['mask']
            nunet_stream.append(t)
        nunet_stream = torch.cat(nunet_stream, dim=1)

        return nunet_stream


class NestedUNet3(torch.nn.Module):
    """UNet+++

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_filter: int = 64,
        deep_supervision: bool = False,
        linear_fc: bool = False
    ):
        super(NestedUNet3, self).__init__()

        self.deep_supervision = deep_supervision

        init_filter = int(init_filter)
        nb_filter = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]
        up_channels = int(nb_filter[0] * 5)

        self.up = model_utils.UpSample()

        self.conv0_0 = SingleConv(in_channels, nb_filter[0])
        self.conv1_0 = PoolConv(nb_filter[0], nb_filter[1], dropout=0.1)
        self.conv2_0 = PoolConv(nb_filter[1], nb_filter[2], dropout=0.1)
        self.conv3_0 = PoolConv(nb_filter[2], nb_filter[3], dropout=0.1)
        self.conv4_0 = PoolConv(nb_filter[3], nb_filter[4], dropout=0.1)

        # Connect 3
        self.conv0_0_3_1_con = PoolConvSingle(nb_filter[0], nb_filter[0], pool_size=8)
        self.conv1_0_3_1_con = PoolConvSingle(nb_filter[1], nb_filter[0], pool_size=4)
        self.conv2_0_3_1_con = PoolConvSingle(nb_filter[2], nb_filter[0], pool_size=2)
        self.conv3_0_3_1_con = SingleConv(nb_filter[3], nb_filter[0])
        self.conv4_0_3_1_con = SingleConv(nb_filter[4], nb_filter[0])
        self.conv3_1 = SingleConv(up_channels, up_channels)

        # Connect 2
        self.conv0_0_2_2_con = PoolConvSingle(nb_filter[0], nb_filter[0], pool_size=4)
        self.conv1_0_2_2_con = PoolConvSingle(nb_filter[1], nb_filter[0], pool_size=2)
        self.conv2_0_2_2_con = SingleConv(nb_filter[2], nb_filter[0])
        self.conv3_1_2_2_con = SingleConv(up_channels, nb_filter[0])
        self.conv4_0_2_2_con = SingleConv(nb_filter[4], nb_filter[0])
        self.conv2_2 = SingleConv(up_channels, up_channels)

        # Connect 3
        self.conv0_0_1_3_con = PoolConvSingle(nb_filter[0], nb_filter[0], pool_size=2)
        self.conv1_0_1_3_con = SingleConv(nb_filter[1], nb_filter[0])
        self.conv2_2_1_3_con = SingleConv(up_channels, nb_filter[0])
        self.conv3_1_1_3_con = SingleConv(up_channels, nb_filter[0])
        self.conv4_0_1_3_con = SingleConv(nb_filter[4], nb_filter[0])
        self.conv1_3 = SingleConv(up_channels, up_channels)

        # Connect 4
        self.conv0_0_0_4_con = SingleConv(nb_filter[0], nb_filter[0])
        self.conv1_3_0_4_con = SingleConv(up_channels, nb_filter[0])
        self.conv2_2_0_4_con = SingleConv(up_channels, nb_filter[0])
        self.conv3_1_0_4_con = SingleConv(up_channels, nb_filter[0])
        self.conv4_0_0_4_con = SingleConv(nb_filter[4], nb_filter[0])
        self.conv0_4 = SingleConv(up_channels, up_channels)

        if self.deep_supervision:
            if linear_fc:
                self.final_0 = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_1 = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_2 = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_3 = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_4 = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        nb_filter[4], out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
            else:
                self.final_0 = torch.nn.Conv2d(
                    up_channels, out_channels, kernel_size=3, padding=1
                )
                self.final_1 = torch.nn.Conv2d(
                    up_channels, out_channels, kernel_size=3, padding=1
                )
                self.final_2 = torch.nn.Conv2d(
                    up_channels, out_channels, kernel_size=3, padding=1
                )
                self.final_3 = torch.nn.Conv2d(
                    up_channels, out_channels, kernel_size=3, padding=1
                )
                self.final_4 = torch.nn.Conv2d(
                    nb_filter[4], out_channels, kernel_size=3, padding=1
                )
        else:
            if linear_fc:
                self.final = torch.nn.Sequential(
                    torch.nn.ELU(alpha=0.1, inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
            else:
                self.final = torch.nn.Conv2d(
                    up_channels, out_channels, kernel_size=3, padding=1
                )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Backbone
        # 1/1
        x0_0 = self.conv0_0(x)
        # 1/2
        x1_0 = self.conv1_0(x0_0)
        # 1/4
        x2_0 = self.conv2_0(x1_0)
        # 1/8
        x3_0 = self.conv3_0(x2_0)
        # 1/16
        x4_0 = self.conv4_0(x3_0)

        # 1/8 connection
        x0_0_x3_1_con = self.conv0_0_3_1_con(x0_0)
        x1_0_x3_1_con = self.conv1_0_3_1_con(x1_0)
        x2_0_x3_1_con = self.conv2_0_3_1_con(x2_0)
        x3_0_x3_1_con = self.conv3_0_3_1_con(x3_0)
        x4_0_x3_1_con = self.conv4_0_3_1_con(self.up(x4_0, size=x3_0.shape[-2:]))
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x0_0_x3_1_con,
                    x1_0_x3_1_con,
                    x2_0_x3_1_con,
                    x3_0_x3_1_con,
                    x4_0_x3_1_con
                ],
                dim=1
            )
        )

        # 1/4 connection
        x0_0_x2_2_con = self.conv0_0_2_2_con(x0_0)
        x1_0_x2_2_con = self.conv1_0_2_2_con(x1_0)
        x2_0_x2_2_con = self.conv2_0_2_2_con(x2_0)
        x3_1_x2_2_con = self.conv3_1_2_2_con(self.up(x3_1, size=x2_0.shape[-2:]))
        x4_0_x2_2_con = self.conv4_0_2_2_con(self.up(x4_0, size=x2_0.shape[-2:]))
        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x0_0_x2_2_con,
                    x1_0_x2_2_con,
                    x2_0_x2_2_con,
                    x3_1_x2_2_con,
                    x4_0_x2_2_con
                ],
                dim=1
            )
        )

        # 1/2 connection
        x0_0_x1_3_con = self.conv0_0_1_3_con(x0_0)
        x1_0_x1_3_con = self.conv1_0_1_3_con(x1_0)
        x2_2_x1_3_con = self.conv2_2_1_3_con(self.up(x2_2, size=x1_0.shape[-2:]))
        x3_1_x1_3_con = self.conv3_1_1_3_con(self.up(x3_1, size=x1_0.shape[-2:]))
        x4_0_x1_3_con = self.conv4_0_1_3_con(self.up(x4_0, size=x1_0.shape[-2:]))
        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x0_0_x1_3_con,
                    x1_0_x1_3_con,
                    x2_2_x1_3_con,
                    x3_1_x1_3_con,
                    x4_0_x1_3_con
                ],
                dim=1
            )
        )

        # 1/1 connection
        x0_0_x0_4_con = self.conv0_0_0_4_con(x0_0)
        x1_3_x0_4_con = self.conv1_3_0_4_con(self.up(x1_3, size=x0_0.shape[-2:]))
        x2_2_x0_4_con = self.conv2_2_0_4_con(self.up(x2_2, size=x0_0.shape[-2:]))
        x3_1_x0_4_con = self.conv3_1_0_4_con(self.up(x3_1, size=x0_0.shape[-2:]))
        x4_0_x0_4_con = self.conv4_0_0_4_con(self.up(x4_0, size=x0_0.shape[-2:]))
        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0_x0_4_con,
                    x1_3_x0_4_con,
                    x2_2_x0_4_con,
                    x3_1_x0_4_con,
                    x4_0_x0_4_con
                ],
                dim=1
            )
        )

        if self.deep_supervision:
            mask_0 = self.final_0(x0_4)
            mask_1 = self.final_1(self.up(x1_3, size=x0_0.shape[-2:]))
            mask_2 = self.final_2(self.up(x2_2, size=x0_0.shape[-2:]))
            mask_3 = self.final_3(self.up(x3_1, size=x0_0.shape[-2:]))
            mask_4 = self.final_4(self.up(x4_0, size=x0_0.shape[-2:]))

            out = {
                'mask_0': mask_0,
                'mask_1': mask_1,
                'mask_2': mask_2,
                'mask_3': mask_3,
                'mask_4': mask_4
            }
        else:
            mask = self.final(x0_4)
            out = {'mask': mask}

        return out
