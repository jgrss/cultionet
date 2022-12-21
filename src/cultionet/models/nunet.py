"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

from . import model_utils
from .base_layers import (
    AttentionGate,
    ConvBlock2d,
    DoubleConv,
    Permute,
    PoolConv,
    PoolResidualConv,
    PoolConvSingle,
    ResidualConv,
    SingleConv
)

import torch


def weights_init_kaiming(m):
    """
    Source:
        https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/init_weights.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNet2(torch.nn.Module):
    """UNet++

    References:
        https://arxiv.org/pdf/1807.10165.pdf
        https://arxiv.org/pdf/1804.03999.pdf
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_filter: int = 64,
        boundary_layer: bool = False,
        out_side_channels: int = 2,
        linear_fc: bool = False,
        deep_supervision: bool = False
    ):
        super(UNet2, self).__init__()

        self.linear_fc = linear_fc
        self.boundary_layer = boundary_layer
        self.deep_supervision = deep_supervision

        init_filter = int(init_filter)
        channels = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.up = model_utils.UpSample()

        self.attention_0 = AttentionGate(
            high_channels=channels[3],
            low_channels=channels[4]
        )
        self.attention_1 = AttentionGate(
            high_channels=channels[2],
            low_channels=channels[3]
        )
        self.attention_2 = AttentionGate(
            high_channels=channels[1],
            low_channels=channels[2]
        )
        self.attention_3 = AttentionGate(
            high_channels=channels[0],
            low_channels=channels[1]
        )

        if boundary_layer:
            # Right stream
            self.bound4_1 = DoubleConv(channels[4]+channels[4], channels[0])
            self.bound3_1 = DoubleConv(channels[0]+channels[3]*2, channels[0])
            self.bound2_1 = DoubleConv(channels[0]+channels[2]*2, channels[0])
            self.bound1_1 = DoubleConv(channels[0]+channels[1]*2, channels[0])
            self.bound0_1 = DoubleConv(channels[0]+channels[0]*2, channels[0])
            # Left stream
            self.bound0_0 = ResidualConv(channels[0], channels[0])
            self.bound0_0_pool = PoolConv(channels[0], channels[1])
            self.bound1_0 = DoubleConv(channels[1]*2, channels[1])
            self.bound1_0_pool = PoolConv(channels[1], channels[2])
            self.bound2_0 = DoubleConv(channels[2]*2, channels[2])
            self.bound2_0_pool = PoolConv(channels[2], channels[3])
            self.bound3_0 = DoubleConv(channels[3]*2, channels[3])
            self.bound3_0_pool = PoolConv(channels[3], channels[4])
            self.bound4_0 = DoubleConv(channels[4]*2, channels[4])

            self.bound_final = torch.nn.Conv2d(
                channels[0],
                out_side_channels,
                kernel_size=1,
                padding=0
            )

        self.conv0_0 = ResidualConv(in_channels, channels[0])
        self.conv1_0 = PoolConv(channels[0], channels[1], dropout=0.25)
        self.conv2_0 = PoolConv(channels[1], channels[2], dropout=0.5)
        self.conv3_0 = PoolConv(channels[2], channels[3], dropout=0.5)
        self.conv4_0 = PoolConv(channels[3], channels[4], dropout=0.5)

        self.conv0_1 = ResidualConv(channels[0]+channels[1], channels[0])
        self.conv1_1 = DoubleConv(channels[1]+channels[2], channels[1])
        self.conv2_1 = DoubleConv(channels[2]+channels[3], channels[2])
        self.conv3_1 = DoubleConv(channels[3]+channels[4], channels[3])

        self.conv0_2 = ResidualConv(channels[0]*2+channels[1], channels[0])
        self.conv1_2 = DoubleConv(channels[1]*2+channels[2], channels[1])
        self.conv2_2 = DoubleConv(channels[2]*2+channels[3], channels[2])

        self.conv0_3 = ResidualConv(channels[0]*3+channels[1], channels[0])
        self.conv1_3 = DoubleConv(channels[1]*3+channels[2], channels[1])

        self.conv0_4 = ResidualConv(channels[0]*4+channels[1], channels[0])

        if self.linear_fc:
            self.net_final = torch.nn.Sequential(
                torch.nn.LeakyReLU(inplace=False),
                Permute((0, 2, 3, 1)),
                torch.nn.Linear(
                    channels[0], out_channels
                ),
                Permute((0, 3, 1, 2))
            )
        else:
            if self.deep_supervision:
                in_final_layers = out_channels

                self.final_1 = torch.nn.Conv2d(
                    channels[0], out_channels, kernel_size=1, padding=0
                )
                self.final_2 = torch.nn.Conv2d(
                    channels[0], out_channels, kernel_size=1, padding=0
                )
                self.final_3 = torch.nn.Conv2d(
                    channels[0], out_channels, kernel_size=1, padding=0
                )
                self.final_4 = torch.nn.Conv2d(
                    channels[0], out_channels, kernel_size=1, padding=0
                )
            else:
                in_final_layers = channels[0]

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
            b0_0 = self.bound0_0(x0_0)
            b1_0 = self.bound1_0(torch.cat([x1_0, self.bound0_0_pool(b0_0)], dim=1))
            b2_0 = self.bound2_0(torch.cat([x2_0, self.bound1_0_pool(b1_0)], dim=1))
            b3_0 = self.bound3_0(torch.cat([x3_0, self.bound2_0_pool(b2_0)], dim=1))
            b4_0 = self.bound4_0(torch.cat([x4_0, self.bound3_0_pool(b3_0)], dim=1))
            # Right stream
            b4_1 = self.bound4_1(torch.cat([b4_0, x4_0], dim=1))
            b3_1 = self.bound3_1(torch.cat([x3_1, b3_0, self.up(b4_1, size=x3_1.shape[-2:])], dim=1))
            b2_1 = self.bound2_1(torch.cat([x2_2, b2_0, self.up(b3_1, size=x2_2.shape[-2:])], dim=1))
            b1_1 = self.bound1_1(torch.cat([x1_3, b1_0, self.up(b2_1, size=x1_3.shape[-2:])], dim=1))
            boundary = self.bound0_1(torch.cat([x0_4, b0_0, self.up(b1_1, size=x0_4.shape[-2:])], dim=1))

        if self.linear_fc:
            mask = self.net_final(x0_4)
        else:
            if self.deep_supervision:
                # Average over skip connections
                x0_1 = self.final_1(x0_1)
                x0_2 = self.final_2(x0_2)
                x0_3 = self.final_3(x0_3)
                x0_4 = self.final_4(x0_4)
                x0_4 = (x0_1 + x0_2 + x0_3 + x0_4) / 4.0
            if self.boundary_layer:
                boundary = self.bound_final(boundary)
                mask = self.net_final(torch.cat([x0_4, boundary], dim=1))
            else:
                mask = self.net_final(x0_4)

        return {
            'mask': mask,
            'boundary': boundary
        }


class TemporalUNet2(torch.nn.Module):
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
        super(TemporalUNet2, self).__init__()

        self.num_features = num_features
        self.in_channels = in_channels

        self.nunet = UNet2(
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


class UNet3(torch.nn.Module):
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
        linear_fc: bool = False,
        side_stream: bool = False
    ):
        super(UNet3, self).__init__()

        self.deep_supervision = deep_supervision

        init_filter = int(init_filter)
        channels = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]
        up_channels = int(channels[0] * 5)

        self.side_stream = side_stream
        self.up = model_utils.UpSample()

        self.conv0_0 = SingleConv(in_channels, channels[0])
        self.conv1_0 = PoolConv(channels[0], channels[1], dropout=0.25)
        self.conv2_0 = PoolConv(channels[1], channels[2], dropout=0.5)
        self.conv3_0 = PoolConv(channels[2], channels[3], dropout=0.5)
        self.conv4_0 = PoolConv(channels[3], channels[4], dropout=0.5)

        # Connect 3
        self.conv0_0_3_1_con = PoolConvSingle(channels[0], channels[0], pool_size=8)
        self.conv1_0_3_1_con = PoolConvSingle(channels[1], channels[0], pool_size=4)
        self.conv2_0_3_1_con = PoolConvSingle(channels[2], channels[0], pool_size=2)
        self.conv3_0_3_1_con = SingleConv(channels[3], channels[0])
        self.conv4_0_3_1_con = SingleConv(channels[4], channels[0])
        self.conv3_1 = SingleConv(up_channels, up_channels)

        # Connect 2
        self.conv0_0_2_2_con = PoolConvSingle(channels[0], channels[0], pool_size=4)
        self.conv1_0_2_2_con = PoolConvSingle(channels[1], channels[0], pool_size=2)
        self.conv2_0_2_2_con = SingleConv(channels[2], channels[0])
        self.conv3_1_2_2_con = SingleConv(up_channels, channels[0])
        self.conv4_0_2_2_con = SingleConv(channels[4], channels[0])
        self.conv2_2 = SingleConv(up_channels, up_channels)

        # Connect 3
        self.conv0_0_1_3_con = PoolConvSingle(channels[0], channels[0], pool_size=2)
        self.conv1_0_1_3_con = SingleConv(channels[1], channels[0])
        self.conv2_2_1_3_con = SingleConv(up_channels, channels[0])
        self.conv3_1_1_3_con = SingleConv(up_channels, channels[0])
        self.conv4_0_1_3_con = SingleConv(channels[4], channels[0])
        self.conv1_3 = SingleConv(up_channels, up_channels)

        # Connect 4
        self.conv0_0_0_4_con = SingleConv(channels[0], channels[0])
        self.conv1_3_0_4_con = SingleConv(up_channels, channels[0])
        self.conv2_2_0_4_con = SingleConv(up_channels, channels[0])
        self.conv3_1_0_4_con = SingleConv(up_channels, channels[0])
        self.conv4_0_0_4_con = SingleConv(channels[4], channels[0])
        self.conv0_4 = SingleConv(up_channels, up_channels)

        if self.side_stream:
            self.convs4_0 = SingleConv(channels[4]+up_channels, channels[0])
            self.convs3_1 = SingleConv(up_channels+channels[0], channels[0])
            self.convs2_2 = SingleConv(up_channels+channels[0], channels[0])
            self.convs1_3 = SingleConv(up_channels+channels[0], channels[0])
            self.side_final = torch.nn.Conv2d(
                channels[0], out_channels, kernel_size=3, padding=1
            )

        if linear_fc:
            self.final = torch.nn.Sequential(
                torch.nn.LeakyReLU(inplace=False),
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

        if self.deep_supervision:
            if linear_fc:
                self.final_1 = torch.nn.Sequential(
                    torch.nn.LeakyReLU(inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_2 = torch.nn.Sequential(
                    torch.nn.LeakyReLU(inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_3 = torch.nn.Sequential(
                    torch.nn.LeakyReLU(inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        up_channels, out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
                self.final_4 = torch.nn.Sequential(
                    torch.nn.LeakyReLU(inplace=False),
                    Permute((0, 2, 3, 1)),
                    torch.nn.Linear(
                        channels[4], out_channels
                    ),
                    Permute((0, 3, 1, 2))
                )
            else:
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
                    channels[4], out_channels, kernel_size=3, padding=1
                )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

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

        mask = self.final(x0_4)
        if self.deep_supervision:
            mask_1 = self.final_1(self.up(x1_3, size=x0_0.shape[-2:]))
            mask_2 = self.final_2(self.up(x2_2, size=x0_0.shape[-2:]))
            mask_3 = self.final_3(self.up(x3_1, size=x0_0.shape[-2:]))
            mask_4 = self.final_4(self.up(x4_0, size=x0_0.shape[-2:]))

            out = {
                'mask_0': mask,
                'mask_1': mask_1,
                'mask_2': mask_2,
                'mask_3': mask_3,
                'mask_4': mask_4
            }
        else:
            out = {'mask': mask}

        if self.side_stream:
            s4_0 = self.convs4_0(
                torch.cat([x3_1, self.up(x4_0, size=x3_1.shape[-2:])], dim=1)
            )
            s3_1 = self.convs3_1(
                torch.cat([x2_2, self.up(s4_0, size=x2_2.shape[-2:])], dim=1)
            )
            s2_2 = self.convs2_2(
                torch.cat([x1_3, self.up(s3_1, size=x1_3.shape[-2:])], dim=1)
            )
            s1_3 = self.convs1_3(
                torch.cat([x0_4, self.up(s2_2, size=x0_4.shape[-2:])], dim=1)
            )
            out['side'] = self.side_final(s1_3)

        return out


class UNet3Psi(torch.nn.Module):
    """UNet+++ with Psi-Net

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
        https://arxiv.org/abs/1902.04099
        https://github.com/Bala93/Multi-task-deep-network
    """
    def __init__(
        self,
        in_channels: int,
        out_dist_channels: int = 1,
        out_edge_channels: int = 2,
        out_mask_channels: int = 2,
        init_filter: int = 64
    ):
        super(UNet3Psi, self).__init__()

        init_filter = int(init_filter)
        channels = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]
        up_channels = int(channels[0] * 5)

        self.up = model_utils.UpSample()

        self.conv0_0 = SingleConv(in_channels, channels[0])
        self.conv1_0 = PoolConv(channels[0], channels[1])
        self.conv2_0 = PoolConv(channels[1], channels[2])
        self.conv3_0 = PoolConv(channels[2], channels[3])
        self.conv4_0 = PoolConv(channels[3], channels[4])

        # Connect 3
        self.conv0_0_3_1_con = PoolConv(channels[0], channels[0], pool_size=8)
        self.conv1_0_3_1_con = PoolConv(channels[1], channels[0], pool_size=4)
        self.conv2_0_3_1_con = PoolConv(channels[2], channels[0], pool_size=2)
        self.conv3_0_3_1_con = DoubleConv(channels[3], channels[0])
        self.conv4_0_3_1_con = DoubleConv(channels[4], channels[0])
        self.conv3_1 = DoubleConv(up_channels, up_channels)
        self.conv3_1_skip_edge = DoubleConv(up_channels*2, up_channels)
        self.conv3_1_skip_mask = DoubleConv(up_channels*2, up_channels)

        # Connect 2
        self.conv0_0_2_2_con = PoolConv(channels[0], channels[0], pool_size=4)
        self.conv1_0_2_2_con = PoolConv(channels[1], channels[0], pool_size=2)
        self.conv2_0_2_2_con = DoubleConv(channels[2], channels[0])
        self.conv3_1_2_2_con = DoubleConv(up_channels, channels[0])
        self.conv4_0_2_2_con = DoubleConv(channels[4], channels[0])
        self.conv2_2 = DoubleConv(up_channels, up_channels)
        self.conv2_2_skip_edge = DoubleConv(up_channels*2, up_channels)
        self.conv2_2_skip_mask = DoubleConv(up_channels*2, up_channels)

        # Connect 3
        self.conv0_0_1_3_con = PoolConv(channels[0], channels[0], pool_size=2)
        self.conv1_0_1_3_con = DoubleConv(channels[1], channels[0])
        self.conv2_2_1_3_con = DoubleConv(up_channels, channels[0])
        self.conv3_1_1_3_con = DoubleConv(up_channels, channels[0])
        self.conv4_0_1_3_con = DoubleConv(channels[4], channels[0])
        self.conv1_3 = DoubleConv(up_channels, up_channels)
        self.conv1_3_skip_edge = DoubleConv(up_channels*2, up_channels)
        self.conv1_3_skip_mask = DoubleConv(up_channels*2, up_channels)

        # Connect 4
        self.conv0_0_0_4_con = DoubleConv(channels[0], channels[0])
        self.conv1_3_0_4_con = DoubleConv(up_channels, channels[0])
        self.conv2_2_0_4_con = DoubleConv(up_channels, channels[0])
        self.conv3_1_0_4_con = DoubleConv(up_channels, channels[0])
        self.conv4_0_0_4_con = DoubleConv(channels[4], channels[0])
        self.conv0_4 = DoubleConv(up_channels, up_channels)
        self.conv0_4_skip_edge = DoubleConv(up_channels*2, up_channels)
        self.conv0_4_skip_mask = DoubleConv(up_channels*2, up_channels)

        self.final_dist = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                out_dist_channels,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.final_edge = torch.nn.Conv2d(
            up_channels,
            out_edge_channels,
            kernel_size=1,
            padding=0
        )
        self.final_mask = torch.nn.Conv2d(
            up_channels,
            out_mask_channels,
            kernel_size=1,
            padding=0
        )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

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
        h3_1 = torch.cat(
            [
                x0_0_x3_1_con,
                x1_0_x3_1_con,
                x2_0_x3_1_con,
                x3_0_x3_1_con,
                x4_0_x3_1_con
            ],
            dim=1
        )
        x3_1_dist = self.conv3_1(h3_1)
        x3_1_edge = self.conv3_1_skip_edge(
            torch.cat(
                [
                    h3_1,
                    x3_1_dist
                ],
                dim=1
            )
        )
        x3_1_mask = self.conv3_1_skip_mask(
            torch.cat(
                [
                    h3_1,
                    x3_1_edge
                ],
                dim=1
            )
        )

        # 1/4 connection
        x0_0_x2_2_con = self.conv0_0_2_2_con(x0_0)
        x1_0_x2_2_con = self.conv1_0_2_2_con(x1_0)
        x2_0_x2_2_con = self.conv2_0_2_2_con(x2_0)
        x3_1_x2_2_con_dist = self.conv3_1_2_2_con(self.up(x3_1_dist, size=x2_0.shape[-2:]))
        x3_1_x2_2_con_edge = self.conv3_1_2_2_con(self.up(x3_1_edge, size=x2_0.shape[-2:]))
        x3_1_x2_2_con_mask = self.conv3_1_2_2_con(self.up(x3_1_mask, size=x2_0.shape[-2:]))
        x4_0_x2_2_con = self.conv4_0_2_2_con(self.up(x4_0, size=x2_0.shape[-2:]))
        h2_2 = torch.cat(
            [
                x0_0_x2_2_con,
                x1_0_x2_2_con,
                x2_0_x2_2_con,
                x4_0_x2_2_con
            ],
            dim=1
        )
        x2_2_dist = self.conv2_2(
            torch.cat(
                [
                    h2_2,
                    x3_1_x2_2_con_dist
                ],
                dim=1
            )
        )
        x2_2_edge = self.conv2_2_skip_edge(
            torch.cat(
                [
                    h2_2,
                    x3_1_x2_2_con_edge,
                    x2_2_dist
                ],
                dim=1
            )
        )
        x2_2_mask = self.conv2_2_skip_mask(
            torch.cat(
                [
                    h2_2,
                    x3_1_x2_2_con_mask,
                    x2_2_edge
                ],
                dim=1
            )
        )

        # 1/2 connection
        x0_0_x1_3_con = self.conv0_0_1_3_con(x0_0)
        x1_0_x1_3_con = self.conv1_0_1_3_con(x1_0)
        x2_2_x1_3_con_dist = self.conv2_2_1_3_con(self.up(x2_2_dist, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_dist = self.conv3_1_1_3_con(self.up(x3_1_dist, size=x1_0.shape[-2:]))
        x2_2_x1_3_con_edge = self.conv2_2_1_3_con(self.up(x2_2_edge, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_edge = self.conv3_1_1_3_con(self.up(x3_1_edge, size=x1_0.shape[-2:]))
        x2_2_x1_3_con_mask = self.conv2_2_1_3_con(self.up(x2_2_mask, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_mask = self.conv3_1_1_3_con(self.up(x3_1_mask, size=x1_0.shape[-2:]))
        x4_0_x1_3_con = self.conv4_0_1_3_con(self.up(x4_0, size=x1_0.shape[-2:]))
        h1_3 = torch.cat(
            [
                x0_0_x1_3_con,
                x1_0_x1_3_con,
                x4_0_x1_3_con
            ],
            dim=1
        )
        x1_3_dist = self.conv1_3(
            torch.cat(
                [
                    h1_3,
                    x2_2_x1_3_con_dist,
                    x3_1_x1_3_con_dist
                ],
                dim=1
            )
        )
        x1_3_edge = self.conv1_3_skip_edge(
            torch.cat(
                [
                    h1_3,
                    x2_2_x1_3_con_edge,
                    x3_1_x1_3_con_edge,
                    x1_3_dist
                ],
                dim=1
            )
        )
        x1_3_mask = self.conv1_3_skip_mask(
            torch.cat(
                [
                    h1_3,
                    x2_2_x1_3_con_mask,
                    x3_1_x1_3_con_mask,
                    x1_3_edge
                ],
                dim=1
            )
        )

        # 1/1 connection
        x0_0_x0_4_con = self.conv0_0_0_4_con(x0_0)
        # Distance
        x1_3_x0_4_con_dist = self.conv1_3_0_4_con(self.up(x1_3_dist, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_dist = self.conv2_2_0_4_con(self.up(x2_2_dist, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_dist = self.conv3_1_0_4_con(self.up(x3_1_dist, size=x0_0.shape[-2:]))
        # Edge
        x1_3_x0_4_con_edge = self.conv1_3_0_4_con(self.up(x1_3_edge, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_edge = self.conv2_2_0_4_con(self.up(x2_2_edge, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_edge = self.conv3_1_0_4_con(self.up(x3_1_edge, size=x0_0.shape[-2:]))
        # Mask
        x1_3_x0_4_con_mask = self.conv1_3_0_4_con(self.up(x1_3_mask, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_mask = self.conv2_2_0_4_con(self.up(x2_2_mask, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_mask = self.conv3_1_0_4_con(self.up(x3_1_mask, size=x0_0.shape[-2:]))
        x4_0_x0_4_con = self.conv4_0_0_4_con(self.up(x4_0, size=x0_0.shape[-2:]))
        h0_4 = torch.cat(
            [
                x0_0_x0_4_con,
                x4_0_x0_4_con
            ],
            dim=1
        )
        x0_4_dist = self.conv0_4(
            torch.cat(
                [
                    h0_4,
                    x1_3_x0_4_con_dist,
                    x2_2_x0_4_con_dist,
                    x3_1_x0_4_con_dist
                ],
                dim=1
            )
        )
        x0_4_edge = self.conv0_4_skip_edge(
            torch.cat(
                [
                    h0_4,
                    x1_3_x0_4_con_edge,
                    x2_2_x0_4_con_edge,
                    x3_1_x0_4_con_edge,
                    x0_4_dist
                ],
                dim=1
            )
        )
        x0_4_mask = self.conv0_4_skip_mask(
            torch.cat(
                [
                    h0_4,
                    x1_3_x0_4_con_mask,
                    x2_2_x0_4_con_mask,
                    x3_1_x0_4_con_mask,
                    x0_4_edge
                ],
                dim=1
            )
        )

        dist = self.final_dist(x0_4_dist)
        edge = self.final_edge(x0_4_edge)
        mask = self.final_mask(x0_4_mask)

        out = {
            'dist': dist,
            'edge': edge,
            'mask': mask
        }

        return out


class ResUNet3Psi(torch.nn.Module):
    """Residual UNet+++ with Psi-Net (Multi-head streams) and Attention

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
        https://arxiv.org/abs/1902.04099
        https://github.com/Bala93/Multi-task-deep-network
    """
    def __init__(
        self,
        in_channels: int,
        out_dist_channels: int = 1,
        out_edge_channels: int = 2,
        out_mask_channels: int = 2,
        init_filter: int = 64,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNet3Psi, self).__init__()

        self.attention = attention

        init_filter = int(init_filter)
        channels = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]
        up_channels = int(channels[0] * 5)
        if dilations is None:
            dilations = [2]

        self.up = model_utils.UpSample()

        self.conv0_0 = ResidualConv(
            in_channels,
            channels[0],
            init_conv=True,
            fractal_attention=self.attention
        )
        self.conv1_0 = PoolResidualConv(
            channels[0], channels[1],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_0 = PoolResidualConv(
            channels[1], channels[2],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_0 = PoolResidualConv(
            channels[2], channels[3],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv4_0 = PoolResidualConv(
            channels[3], channels[4],
            fractal_attention=self.attention,
            dilations=dilations
        )

        # Connect 3
        self.conv0_0_3_1_con = PoolResidualConv(
            channels[0], channels[0], pool_size=8,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_0_3_1_con = PoolResidualConv(
            channels[1], channels[0], pool_size=4,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_0_3_1_con = PoolResidualConv(
            channels[2], channels[0], pool_size=2,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_0_3_1_con = ResidualConv(
            channels[3], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv4_0_3_1_con = ResidualConv(
            channels[4], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1 = ResidualConv(
            up_channels, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1_skip1 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1_skip2 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )

        # Connect 2
        self.conv0_0_2_2_con = PoolResidualConv(
            channels[0], channels[0], pool_size=4,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_0_2_2_con = PoolResidualConv(
            channels[1], channels[0], pool_size=2,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_0_2_2_con = ResidualConv(
            channels[2], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1_2_2_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv4_0_2_2_con = ResidualConv(
            channels[4], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_2 = ResidualConv(
            up_channels, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_2_skip1 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_2_skip2 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )

        # Connect 3
        self.conv0_0_1_3_con = PoolResidualConv(
            channels[0], channels[0], pool_size=2,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_0_1_3_con = ResidualConv(
            channels[1], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_2_1_3_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1_1_3_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv4_0_1_3_con = ResidualConv(
            channels[4], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_3 = ResidualConv(
            up_channels, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_3_skip1 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_3_skip2 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )

        # Connect 4
        self.conv0_0_0_4_con = ResidualConv(
            channels[0], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv1_3_0_4_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv2_2_0_4_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv3_1_0_4_con = ResidualConv(
            up_channels, channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv4_0_0_4_con = ResidualConv(
            channels[4], channels[0],
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv0_4 = ResidualConv(
            up_channels, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv0_4_skip1 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )
        self.conv0_4_skip2 = ResidualConv(
            up_channels*2, up_channels,
            fractal_attention=self.attention,
            dilations=dilations
        )

        self.final_dist = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                out_dist_channels,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.final_edge = torch.nn.Conv2d(
            up_channels,
            out_edge_channels,
            kernel_size=1,
            padding=0
        )
        self.final_mask = torch.nn.Conv2d(
            up_channels,
            out_mask_channels,
            kernel_size=1,
            padding=0
        )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

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
        h3_1 = torch.cat(
            [
                x0_0_x3_1_con,
                x1_0_x3_1_con,
                x2_0_x3_1_con,
                x3_0_x3_1_con,
                x4_0_x3_1_con
            ],
            dim=1
        )
        x3_1_dist = self.conv3_1(h3_1)
        x3_1_edge = self.conv3_1_skip1(
            torch.cat(
                [
                    h3_1,
                    x3_1_dist
                ],
                dim=1
            )
        )
        x3_1_mask = self.conv3_1_skip2(
            torch.cat(
                [
                    h3_1,
                    x3_1_edge
                ],
                dim=1
            )
        )

        # 1/4 connection
        x0_0_x2_2_con = self.conv0_0_2_2_con(x0_0)
        x1_0_x2_2_con = self.conv1_0_2_2_con(x1_0)
        x2_0_x2_2_con = self.conv2_0_2_2_con(x2_0)
        x3_1_x2_2_con_dist = self.conv3_1_2_2_con(self.up(x3_1_dist, size=x2_0.shape[-2:]))
        x3_1_x2_2_con_edge = self.conv3_1_2_2_con(self.up(x3_1_edge, size=x2_0.shape[-2:]))
        x3_1_x2_2_con_mask = self.conv3_1_2_2_con(self.up(x3_1_mask, size=x2_0.shape[-2:]))
        x4_0_x2_2_con = self.conv4_0_2_2_con(self.up(x4_0, size=x2_0.shape[-2:]))
        h2_2 = torch.cat(
            [
                x0_0_x2_2_con,
                x1_0_x2_2_con,
                x2_0_x2_2_con,
                x4_0_x2_2_con
            ],
            dim=1
        )
        x2_2_dist = self.conv2_2(
            torch.cat(
                [
                    h2_2,
                    x3_1_x2_2_con_dist
                ],
                dim=1
            )
        )
        x2_2_edge = self.conv2_2_skip1(
            torch.cat(
                [
                    h2_2,
                    x2_2_dist,
                    x3_1_x2_2_con_edge
                ],
                dim=1
            )
        )
        x2_2_mask = self.conv2_2_skip2(
            torch.cat(
                [
                    h2_2,
                    x2_2_edge,
                    x3_1_x2_2_con_mask,
                ],
                dim=1
            )
        )

        # 1/2 connection
        x0_0_x1_3_con = self.conv0_0_1_3_con(x0_0)
        x1_0_x1_3_con = self.conv1_0_1_3_con(x1_0)
        x2_2_x1_3_con_dist = self.conv2_2_1_3_con(self.up(x2_2_dist, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_dist = self.conv3_1_1_3_con(self.up(x3_1_dist, size=x1_0.shape[-2:]))
        x2_2_x1_3_con_edge = self.conv2_2_1_3_con(self.up(x2_2_edge, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_edge = self.conv3_1_1_3_con(self.up(x3_1_edge, size=x1_0.shape[-2:]))
        x2_2_x1_3_con_mask = self.conv2_2_1_3_con(self.up(x2_2_mask, size=x1_0.shape[-2:]))
        x3_1_x1_3_con_mask = self.conv3_1_1_3_con(self.up(x3_1_mask, size=x1_0.shape[-2:]))
        x4_0_x1_3_con = self.conv4_0_1_3_con(self.up(x4_0, size=x1_0.shape[-2:]))
        h1_3 = torch.cat(
            [
                x0_0_x1_3_con,
                x1_0_x1_3_con,
                x4_0_x1_3_con
            ],
            dim=1
        )
        x1_3_dist = self.conv1_3(
            torch.cat(
                [
                    h1_3,
                    x3_1_x1_3_con_dist,
                    x2_2_x1_3_con_dist
                ],
                dim=1
            )
        )
        x1_3_edge = self.conv1_3_skip1(
            torch.cat(
                [
                    h1_3,
                    x1_3_dist,
                    x3_1_x1_3_con_edge,
                    x2_2_x1_3_con_edge
                ],
                dim=1
            )
        )
        x1_3_mask = self.conv1_3_skip2(
            torch.cat(
                [
                    h1_3,
                    x1_3_edge,
                    x3_1_x1_3_con_mask,
                    x2_2_x1_3_con_mask
                ],
                dim=1
            )
        )

        # 1/1 connection
        x0_0_x0_4_con = self.conv0_0_0_4_con(x0_0)
        # Distance
        x1_3_x0_4_con_dist = self.conv1_3_0_4_con(self.up(x1_3_dist, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_dist = self.conv2_2_0_4_con(self.up(x2_2_dist, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_dist = self.conv3_1_0_4_con(self.up(x3_1_dist, size=x0_0.shape[-2:]))
        # Edge
        x1_3_x0_4_con_edge = self.conv1_3_0_4_con(self.up(x1_3_edge, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_edge = self.conv2_2_0_4_con(self.up(x2_2_edge, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_edge = self.conv3_1_0_4_con(self.up(x3_1_edge, size=x0_0.shape[-2:]))
        # Mask
        x1_3_x0_4_con_mask = self.conv1_3_0_4_con(self.up(x1_3_mask, size=x0_0.shape[-2:]))
        x2_2_x0_4_con_mask = self.conv2_2_0_4_con(self.up(x2_2_mask, size=x0_0.shape[-2:]))
        x3_1_x0_4_con_mask = self.conv3_1_0_4_con(self.up(x3_1_mask, size=x0_0.shape[-2:]))
        x4_0_x0_4_con = self.conv4_0_0_4_con(self.up(x4_0, size=x0_0.shape[-2:]))
        h0_4 = torch.cat(
            [
                x0_0_x0_4_con,
                x4_0_x0_4_con
            ],
            dim=1
        )
        x0_4_dist = self.conv0_4(
            torch.cat(
                [
                    h0_4,
                    x3_1_x0_4_con_dist,
                    x2_2_x0_4_con_dist,
                    x1_3_x0_4_con_dist
                ],
                dim=1
            )
        )
        x0_4_edge = self.conv0_4_skip1(
            torch.cat(
                [
                    h0_4,
                    x0_4_dist,
                    x3_1_x0_4_con_edge,
                    x2_2_x0_4_con_edge,
                    x1_3_x0_4_con_edge
                ],
                dim=1
            )
        )
        x0_4_mask = self.conv0_4_skip2(
            torch.cat(
                [
                    h0_4,
                    x0_4_edge,
                    x3_1_x0_4_con_mask,
                    x2_2_x0_4_con_mask,
                    x1_3_x0_4_con_mask
                ],
                dim=1
            )
        )

        dist = self.final_dist(x0_4_dist)
        edge = self.final_edge(x0_4_edge)
        mask = self.final_mask(x0_4_mask)

        out = {
            'dist': dist,
            'edge': edge,
            'mask': mask
        }

        return out
