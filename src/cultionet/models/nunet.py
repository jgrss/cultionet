"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

from . import model_utils
from . import kernels
from .base_layers import (
    AttentionGate,
    DoubleConv,
    SpatioTemporalConv3d,
    Min,
    Max,
    Mean,
    Std,
    Permute,
    PoolConv,
    PoolResidualConv,
    ResidualConv,
    ResidualAConv,
    SingleConv,
    Softmax,
    SigmoidCrisp,
    Squeeze,
    SetActivation
)
from .enums import ResBlockTypes
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
        init_filter: int = 64,
        init_point_conv: bool = False,
        double_dilation: int = 1
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
            in_channels=channels[0],
            out_channels=channels[1],
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.conv2_0 = PoolConv(
            in_channels=channels[1],
            out_channels=channels[2],
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.conv3_0 = PoolConv(
            in_channels=channels[2],
            out_channels=channels[3],
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.conv4_0 = PoolConv(
            in_channels=channels[3],
            out_channels=channels[4],
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )

        # Connect 3
        self.convs_3_1 = UNet3P_3_1(
            channels=channels,
            up_channels=up_channels,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_2_2 = UNet3P_2_2(
            channels=channels,
            up_channels=up_channels,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_1_3 = UNet3P_1_3(
            channels=channels,
            up_channels=up_channels,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )
        self.convs_0_4 = UNet3P_0_4(
            channels=channels,
            up_channels=up_channels,
            init_point_conv=init_point_conv,
            double_dilation=double_dilation
        )

        self.final = torch.nn.Conv2d(
            in_channels=up_channels,
            out_channels=out_channels,
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


class PreUnet3Psi(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: T.Sequence[int],
        activation_type: str,
        trend_kernel_size: int = 5
    ):
        super(PreUnet3Psi, self).__init__()

        self.cg = model_utils.ConvToGraph()
        self.gc = model_utils.GraphToConv()

        self.peak_kernel = kernels.Peaks(kernel_size=trend_kernel_size)
        self.pos_trend_kernel = kernels.Trend(
            kernel_size=trend_kernel_size, direction='positive'
        )
        self.neg_trend_kernel = kernels.Trend(
            kernel_size=trend_kernel_size, direction='negative'
        )
        self.reduce_trend_to_time = torch.nn.Sequential(
            SpatioTemporalConv3d(
                in_channels=int(in_channels * 3),
                out_channels=1,
                activation_type=activation_type
            ),
            Squeeze(dim=1)
        )

        self.time_conv0 = SpatioTemporalConv3d(
            in_channels=in_channels,
            out_channels=channels[0],
            activation_type=activation_type
        )
        self.reduce_to_time = torch.nn.Sequential(
            SpatioTemporalConv3d(
                in_channels=channels[0],
                out_channels=1,
                activation_type=activation_type
            ),
            Squeeze(dim=1)
        )
        # (B x C x T|D x H x W)
        # Temporal reductions
        # Reduce to 2d (B x C x H x W)
        self.reduce_to_channels_min = torch.nn.Sequential(
            Min(dim=2),
            torch.nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type)
        )
        self.reduce_to_channels_max = torch.nn.Sequential(
            Max(dim=2),
            torch.nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type)
        )
        self.reduce_to_channels_mean = torch.nn.Sequential(
            Mean(dim=2),
            torch.nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type)
        )
        self.reduce_to_channels_std = torch.nn.Sequential(
            Std(dim=2),
            torch.nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type)
        )

    def forward(self, x: torch.Tensor, rnn_h: torch.Tensor) -> torch.Tensor:
        peak_kernels = []
        pos_trend_kernels = []
        neg_trend_kernels = []
        for bidx in range(0, x.shape[1]):
            # (B x C x T x H x W) -> (B x T x H x W)
            band_input = x[:, bidx]
            # (B x T x H x W) -> (B*H*W x T) -> (B*H*W x 1(C) x T)
            band_input = self.cg(band_input).unsqueeze(1)
            peak_res = self.peak_kernel(band_input)
            pos_trend_res = self.pos_trend_kernel(band_input)
            neg_trend_res = self.neg_trend_kernel(band_input)
            # Reshape (B*H*W x 1(C) x T) -> (B x C X T x H x W)
            peak_kernels += [
                self.gc(
                    # (B*H*W x T)
                    peak_res.squeeze(),
                    nbatch=x.shape[0],
                    nrows=x.shape[-2],
                    ncols=x.shape[-1]
                ).unsqueeze(1)
            ]
            pos_trend_kernels += [
                self.gc(
                    # (B*H*W x T)
                    pos_trend_res.squeeze(),
                    nbatch=x.shape[0],
                    nrows=x.shape[-2],
                    ncols=x.shape[-1]
                ).unsqueeze(1)
            ]
            neg_trend_kernels += [
                self.gc(
                    # (B*H*W x T)
                    neg_trend_res.squeeze(),
                    nbatch=x.shape[0],
                    nrows=x.shape[-2],
                    ncols=x.shape[-1]
                ).unsqueeze(1)
            ]
        # Concatentate along the channels
        trend_kernels = torch.cat(
            peak_kernels + pos_trend_kernels + neg_trend_kernels, dim=1
        )

        # Inputs shape is (B x C X T|D x H x W)
        h = self.time_conv0(x)
        h = torch.cat(
            [
                self.reduce_to_time(h),
                self.reduce_to_channels_min(h),
                self.reduce_to_channels_max(h),
                self.reduce_to_channels_mean(h),
                self.reduce_to_channels_std(h),
                rnn_h,
                self.reduce_trend_to_time(trend_kernels)
            ],
            dim=1
        )

        return h


class PostUNet3Psi(torch.nn.Module):
    def __init__(
        self,
        up_channels: int,
        num_classes: int,
        mask_activation: T.Callable,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False
    ):
        super(PostUNet3Psi, self).__init__()

        self.deep_sup_dist = deep_sup_dist
        self.deep_sup_edge = deep_sup_edge
        self.deep_sup_mask = deep_sup_mask

        self.up = model_utils.UpSample()

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
            SigmoidCrisp()
        )
        self.final_mask = torch.nn.Sequential(
            torch.nn.Conv2d(
                up_channels,
                num_classes,
                kernel_size=1,
                padding=0
            ),
            mask_activation
        )
        if self.deep_sup_dist:
            self.final_dist_3_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                torch.nn.Sigmoid()
            )
            self.final_dist_2_2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                torch.nn.Sigmoid()
            )
            self.final_dist_1_3 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                torch.nn.Sigmoid()
            )
        if self.deep_sup_edge:
            self.final_edge_3_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                SigmoidCrisp()
            )
            self.final_edge_2_2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                SigmoidCrisp()
            )
            self.final_edge_1_3 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    1,
                    kernel_size=1,
                    padding=0
                ),
                SigmoidCrisp()
            )
        if self.deep_sup_mask:
            self.final_mask_3_1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    num_classes,
                    kernel_size=1,
                    padding=0
                ),
                mask_activation
            )
            self.final_mask_2_2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    num_classes,
                    kernel_size=1,
                    padding=0
                ),
                mask_activation
            )
            self.final_mask_1_3 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    up_channels,
                    num_classes,
                    kernel_size=1,
                    padding=0
                ),
                mask_activation
            )

    def forward(
        self,
        out_0_4: T.Dict[str, torch.Tensor],
        out_3_1: T.Dict[str, torch.Tensor],
        out_2_2: T.Dict[str, torch.Tensor],
        out_1_3: T.Dict[str, torch.Tensor],
    ) -> T.Dict[str, torch.Tensor]:
        dist = self.final_dist(out_0_4['dist'])
        edge = self.final_edge(out_0_4['edge'])
        mask = self.final_mask(out_0_4['mask'])

        out = {
            'dist': dist,
            'edge': edge,
            'mask': mask,
            'dist_3_1': None,
            'dist_2_2': None,
            'dist_1_3': None,
            'edge_3_1': None,
            'edge_2_2': None,
            'edge_1_3': None,
            'mask_3_1': None,
            'mask_2_2': None,
            'mask_1_3': None
        }

        if self.deep_sup_dist:
            out['dist_3_1'] = self.final_dist_3_1(
                self.up(out_3_1['dist'], size=dist.shape[-2:], mode='bilinear')
            )
            out['dist_2_2'] = self.final_dist_2_2(
                self.up(out_2_2['dist'], size=dist.shape[-2:], mode='bilinear')
            )
            out['dist_1_3'] = self.final_dist_1_3(
                self.up(out_1_3['dist'], size=dist.shape[-2:], mode='bilinear')
            )
        if self.deep_sup_edge:
            out['edge_3_1'] = self.final_edge_3_1(
                self.up(out_3_1['edge'], size=edge.shape[-2:], mode='bilinear')
            )
            out['edge_2_2'] = self.final_edge_2_2(
                self.up(out_2_2['edge'], size=edge.shape[-2:], mode='bilinear')
            )
            out['edge_1_3'] = self.final_edge_1_3(
                self.up(out_1_3['edge'], size=edge.shape[-2:], mode='bilinear')
            )
        if self.deep_sup_mask:
            out['mask_3_1'] = self.final_mask_3_1(
                self.up(out_3_1['mask'], size=mask.shape[-2:], mode='bilinear')
            )
            out['mask_2_2'] = self.final_mask_2_2(
                self.up(out_2_2['mask'], size=mask.shape[-2:], mode='bilinear')
            )
            out['mask_1_3'] = self.final_mask_1_3(
                self.up(out_1_3['mask'], size=mask.shape[-2:], mode='bilinear')
            )

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
        in_rnn_channels: int,
        init_filter: int = 32,
        num_classes: int = 2,
        dilation: int = 2,
        activation_type: str = 'SiLU',
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[Softmax, torch.nn.Sigmoid] = Softmax(dim=1)
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

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            channels=channels,
            activation_type=activation_type
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for RNN hidden logits
        self.conv0_0 = SingleConv(
            in_channels=(
                in_time
                + int(channels[0] * 4)
                + in_rnn_channels
                # Peak kernels and Trend kernels
                + in_time
            ),
            out_channels=channels[0],
            activation_type=activation_type
        )
        self.conv1_0 = PoolConv(
            channels[0],
            channels[1],
            double_dilation=dilation,
            activation_type=activation_type
        )
        self.conv2_0 = PoolConv(
            channels[1],
            channels[2],
            double_dilation=dilation,
            activation_type=activation_type
        )
        self.conv3_0 = PoolConv(
            channels[2],
            channels[3],
            double_dilation=dilation,
            activation_type=activation_type
        )
        self.conv4_0 = PoolConv(
            channels[3],
            channels[4],
            double_dilation=dilation,
            activation_type=activation_type
        )

        # Connect 3
        self.convs_3_1 = UNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type
        )
        self.convs_2_2 = UNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type
        )
        self.convs_1_3 = UNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type
        )
        self.convs_0_4 = UNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type
        )

        self.post_unet = PostUNet3Psi(
            up_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            deep_sup_dist=deep_sup_dist,
            deep_sup_edge=deep_sup_edge,
            deep_sup_mask=deep_sup_mask
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
        self, x: torch.Tensor, rnn_h: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Inputs shape is (B x C X T|D x H x W)
        h = self.pre_unet(x, rnn_h)
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

        out = self.post_unet(
            out_0_4=out_0_4,
            out_3_1=out_3_1,
            out_2_2=out_2_2,
            out_1_3=out_1_3
        )

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
        in_time: int,
        in_rnn_channels: int,
        init_filter: int = 32,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = 'LeakyReLU',
        res_block_type: str = 'resa',
        attention_weights: T.Optional[str] = None,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[Softmax, torch.nn.Sigmoid] = Softmax(dim=1)
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

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            channels=channels,
            activation_type=activation_type
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for RNN hidden logits
        if res_block_type.lower() == 'res':
            self.conv0_0 = ResidualConv(
                in_channels=(
                    in_time
                    + int(channels[0] * 4)
                    + in_rnn_channels
                    # Peak kernels and Trend kernels
                    + in_time
                ),
                out_channels=channels[0],
                dilation=dilations[0],
                activation_type=activation_type,
                attention_weights=attention_weights
            )
        else:
            self.conv0_0 = ResidualAConv(
                in_channels=(
                    in_time
                    + int(channels[0] * 4)
                    + in_rnn_channels
                    # Peak kernels and Trend kernels
                    + in_time
                ),
                out_channels=channels[0],
                dilations=dilations,
                activation_type=activation_type,
                attention_weights=attention_weights
            )
        self.conv1_0 = PoolResidualConv(
            channels[0],
            channels[1],
            dilations=dilations,
            attention_weights=attention_weights,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.conv2_0 = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.conv3_0 = PoolResidualConv(
            channels[2],
            channels[3],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.conv4_0 = PoolResidualConv(
            channels[3],
            channels[4],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )

        # Connect 3
        self.convs_3_1 = ResUNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.convs_2_2 = ResUNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.convs_1_3 = ResUNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )
        self.convs_0_4 = ResUNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=ResBlockTypes[res_block_type.upper()]
        )

        self.post_unet = PostUNet3Psi(
            up_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            deep_sup_dist=deep_sup_dist,
            deep_sup_edge=deep_sup_edge,
            deep_sup_mask=deep_sup_mask
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
        self, x: torch.Tensor, rnn_h: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Inputs shape is (B x C X T|D x H x W)
        h = self.pre_unet(x, rnn_h)
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

        out = self.post_unet(
            out_0_4=out_0_4,
            out_3_1=out_3_1,
            out_2_2=out_2_2,
            out_1_3=out_1_3
        )

        return out
