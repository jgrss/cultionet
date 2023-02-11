"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

from . import model_utils
from .base_layers import (
    AttentionGate,
    DoubleConv,
    DoubleConv3d,
    Mean,
    Permute,
    PoolConv,
    PoolConv3d,
    PoolResidualConv,
    ResidualConvInit,
    ResidualConv,
    SingleConv,
    SingleConv3d,
    Softmax,
    Squeeze,
    Unsqueeze
)
from .unet_parts import (
    UNet3P_3_1,
    UNet3P_2_2,
    UNet3P_1_3,
    UNet3P_0_4,
    UNet3_3_1,
    UNet3_2_2,
    UNet3_1_3,
    UNet3_0_4,
    ResUNet3_3_1,
    ResUNet3_2_2,
    ResUNet3_1_3,
    ResUNet3_0_4
)

import torch
import torch.nn.functional as F


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


class UNet3(torch.nn.Module):
    """UNet+++

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_filter: int = 64
    ):
        super(UNet3, self).__init__()

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

        self.conv0_0 = SingleConv(
            in_channels,
            channels[0]
        )
        self.conv1_0 = PoolConv(
            channels[0],
            channels[1]
        )
        self.conv2_0 = PoolConv(
            channels[1],
            channels[2]
        )
        self.conv3_0 = PoolConv(
            channels[2],
            channels[3]
        )
        self.conv4_0 = PoolConv(
            channels[3],
            channels[4]
        )

        # Connect 3
        self.convs_3_1 = UNet3P_3_1(
            channels=channels,
            up_channels=up_channels
        )
        self.convs_2_2 = UNet3P_2_2(
            channels=channels,
            up_channels=up_channels
        )
        self.convs_1_3 = UNet3P_1_3(
            channels=channels,
            up_channels=up_channels
        )
        self.convs_0_4 = UNet3P_0_4(
            channels=channels,
            up_channels=up_channels
        )

        self.final = torch.nn.Conv2d(
            up_channels,
            out_channels,
            kernel_size=1,
            padding=0
        )

        # Initialise weights
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                m.apply(weights_init_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        out_3_1 = self.convs_3_1(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            x3_0=x3_0,
            x4_0=x4_0
        )
        # 1/4 connection
        out_2_2 = self.convs_2_2(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            h3_1=out_3_1,
            x4_0=x4_0
        )
        # 1/2 connection
        out_1_3 = self.convs_1_3(
            x0_0=x0_0,
            x1_0=x1_0,
            h2_2=out_2_2,
            h3_1=out_3_1,
            x4_0=x4_0
        )
        # 1/1 connection
        out_0_4 = self.convs_0_4(
            x0_0=x0_0,
            h1_3=out_1_3,
            h2_2=out_2_2,
            h3_1=out_3_1,
            x4_0=x4_0
        )

        out = self.final(out_0_4)

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
        in_time: int,
        init_filter: int = 64,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        attention: bool = False
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

        self.time_conv0 = DoubleConv3d(
            in_channels=in_channels,
            out_channels=channels[0],
            double_dilation=2
        )
        self.time_conv1 = torch.nn.Sequential(
            DoubleConv3d(
                in_channels=channels[0],
                out_channels=channels[0],
                double_dilation=2
            ),
            # Reduce channels to 1, leaving time
            torch.nn.Conv3d(
                channels[0],
                1,
                kernel_size=1,
                padding=0
            ),
            Squeeze()
        )
        self.final_time_dist = torch.nn.Sequential(
            # Reduce channels to 1, leaving time
            torch.nn.Conv3d(
                channels[0],
                1,
                kernel_size=1,
                padding=0
            ),
            Squeeze(),
            # Take the mean over time
            Mean(dim=1, keepdim=True)
        )
        self.final_time_edge = torch.nn.Sequential(
            # Reduce channels to 1, leaving time
            torch.nn.Conv3d(
                channels[0],
                1,
                kernel_size=1,
                padding=0
            ),
            Squeeze(),
            # Take the mean over time
            Mean(dim=1, keepdim=True)
        )
        self.final_time_mask = torch.nn.Sequential(
            # Reduce channels to 1, leaving time
            torch.nn.Conv3d(
                channels[0],
                1,
                kernel_size=1,
                padding=0
            ),
            Squeeze(),
            # Take the mean over time
            Mean(dim=1, keepdim=True)
        )

        self.conv0_0 = SingleConv(
            in_time,
            channels[0]
        )
        self.conv1_0 = PoolConv(
            channels[0],
            channels[1]
        )
        self.conv2_0 = PoolConv(
            channels[1],
            channels[2]
        )
        self.conv3_0 = PoolConv(
            channels[2],
            channels[3]
        )
        self.conv4_0 = PoolConv(
            channels[3],
            channels[4]
        )

        # Connect 3
        self.convs_3_1 = UNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            attention=attention,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_2_2 = UNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            attention=attention,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_1_3 = UNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            attention=attention,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_0_4 = UNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            attention=attention,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )

        self.final_dist = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                1,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.final_edge = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                1,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.final_mask = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                1,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )

        # Initialise weights
        for m in self.modules():
            if isinstance(
                m,
                (
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.Conv3d,
                    torch.nn.BatchNorm3d
                )
            ):
                m.apply(weights_init_kaiming)

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # __, __, __, h, w = x.shape
        # x = F.interpolate(
        #     x[:, 1:4],
        #     size=(6, h, w),
        #     mode='trilinear'
        # )
        # Inputs shape is (B x C X T|D x H x W)
        x = self.time_conv0(x)
        h = self.time_conv1(x)
        # h shape is (B x C x H x W)
        # Backbone
        # 1/1
        x0_0 = self.conv0_0(h)
        # 1/2
        x1_0 = self.conv1_0(x0_0)
        # 1/4
        x2_0 = self.conv2_0(x1_0)
        # 1/8
        x3_0 = self.conv3_0(x2_0)
        # 1/16
        x4_0 = self.conv4_0(x3_0)

        # 1/8 connection
        out_3_1 = self.convs_3_1(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            x3_0=x3_0,
            x4_0=x4_0
        )
        # 1/4 connection
        out_2_2 = self.convs_2_2(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            h3_1_dist=out_3_1['dist'],
            h3_1_edge=out_3_1['edge'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )
        # 1/2 connection
        out_1_3 = self.convs_1_3(
            x0_0=x0_0,
            x1_0=x1_0,
            h2_2_dist=out_2_2['dist'],
            h3_1_dist=out_3_1['dist'],
            h2_2_edge=out_2_2['edge'],
            h3_1_edge=out_3_1['edge'],
            h2_2_mask=out_2_2['mask'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )
        # 1/1 connection
        out_0_4 = self.convs_0_4(
            x0_0=x0_0,
            h1_3_dist=out_1_3['dist'],
            h2_2_dist=out_2_2['dist'],
            h3_1_dist=out_3_1['dist'],
            h1_3_edge=out_1_3['edge'],
            h2_2_edge=out_2_2['edge'],
            h3_1_edge=out_3_1['edge'],
            h1_3_mask=out_1_3['mask'],
            h2_2_mask=out_2_2['mask'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )

        dist = out_0_4['dist'] + self.final_time_dist(x)
        dist = self.final_dist(dist)
        edge = out_0_4['edge'] + self.final_time_edge(x)
        edge = self.final_edge(edge)
        mask = out_0_4['mask'] + self.final_time_mask(x)
        mask = self.final_mask(mask)

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
        https://github.com/hamidriasat/UNet-3-Plus
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

        self.conv0_0 = ResidualConvInit(
            in_channels,
            channels[0]
        )
        self.conv1_0 = PoolResidualConv(
            channels[0],
            channels[1],
            dilations=dilations
        )
        self.conv2_0 = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations
        )
        self.conv3_0 = PoolResidualConv(
            channels[2],
            channels[3],
            dilations=dilations
        )
        self.conv4_0 = PoolResidualConv(
            channels[3],
            channels[4],
            dilations=dilations
        )

        # Connect 3
        self.convs_3_1 = ResUNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention=attention
        )
        self.convs_2_2 = ResUNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention=attention
        )
        self.convs_1_3 = ResUNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention=attention
        )
        self.convs_0_4 = ResUNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention=attention
        )

        edge_activation = torch.nn.Sigmoid() if out_edge_channels == 1 else Softmax(dim=1)

        self.final_dist = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                out_dist_channels,
                kernel_size=1,
                padding=0
            ),
            torch.nn.Sigmoid()
        )
        self.final_edge = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                out_edge_channels,
                kernel_size=1,
                padding=0
            ),
            edge_activation
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
        out_3_1 = self.convs_3_1(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            x3_0=x3_0,
            x4_0=x4_0
        )
        # 1/4 connection
        out_2_2 = self.convs_2_2(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            h3_1_dist=out_3_1['dist'],
            h3_1_edge=out_3_1['edge'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )
        # 1/2 connection
        out_1_3 = self.convs_1_3(
            x0_0=x0_0,
            x1_0=x1_0,
            h2_2_dist=out_2_2['dist'],
            h3_1_dist=out_3_1['dist'],
            h2_2_edge=out_2_2['edge'],
            h3_1_edge=out_3_1['edge'],
            h2_2_mask=out_2_2['mask'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )
        # 1/1 connection
        out_0_4 = self.convs_0_4(
            x0_0=x0_0,
            h1_3_dist=out_1_3['dist'],
            h2_2_dist=out_2_2['dist'],
            h3_1_dist=out_3_1['dist'],
            h1_3_edge=out_1_3['edge'],
            h2_2_edge=out_2_2['edge'],
            h3_1_edge=out_3_1['edge'],
            h1_3_mask=out_1_3['mask'],
            h2_2_mask=out_2_2['mask'],
            h3_1_mask=out_3_1['mask'],
            x4_0=x4_0
        )

        dist = self.final_dist(out_0_4['dist'])
        edge = self.final_edge(out_0_4['edge'])
        mask = self.final_mask(out_0_4['mask'])

        out = {
            'dist': dist,
            'edge': edge,
            'mask': mask
        }

        return out
