"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet.

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from cultionet.enums import ResBlockTypes
from cultionet.layers import kernels
from cultionet.layers.base_layers import (
    PoolConv,
    PoolResidualConv,
    ResidualAConv,
    ResidualConv,
    SetActivation,
    SigmoidCrisp,
    SingleConv,
    Softmax,
)
from cultionet.layers.weights import init_conv_weights
from cultionet.models import model_utils
from cultionet.models.unet_parts import (
    ResELUNetPsiBlock,
    ResUNet3_0_4,
    ResUNet3_1_3,
    ResUNet3_2_2,
    ResUNet3_3_1,
    UNet3_0_4,
    UNet3_1_3,
    UNet3_2_2,
    UNet3_3_1,
)


class Encoding3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation_type: str
    ):
        super(Encoding3d, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            SetActivation(activation_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PreUnet3Psi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        channels: T.Sequence[int],
        out_channels: int,
        activation_type: str,
        trend_kernel_size: int = 5,
        num_layers: int = 1,
    ):
        super(PreUnet3Psi, self).__init__()

        self.cg = model_utils.ConvToGraph()
        self.gc = model_utils.GraphToConv()

        self.peak_kernel = kernels.Peaks(kernel_size=trend_kernel_size)
        self.pos_trend_kernel = kernels.Trend(
            kernel_size=trend_kernel_size, direction="positive"
        )
        self.neg_trend_kernel = kernels.Trend(
            kernel_size=trend_kernel_size, direction="negative"
        )
        self.time_conv0 = Encoding3d(
            in_channels=in_channels,
            out_channels=channels[0],
            activation_type=activation_type,
        )
        self.reduce_trend_to_time = nn.Sequential(
            Encoding3d(
                in_channels=3,
                out_channels=1,
                activation_type=activation_type,
            ),
            Rearrange('b c t h w -> b (c t) h w'),
        )
        self.reduce_to_time = nn.Sequential(
            Encoding3d(
                in_channels=channels[0],
                out_channels=1,
                activation_type=activation_type,
            ),
            Rearrange('b c t h w -> b (c t) h w'),
        )
        self.time_to_hidden = nn.Conv2d(
            in_channels=in_time,
            out_channels=channels[0],
            kernel_size=1,
            padding=0,
        )

        # (B x C x T|D x H x W)
        # Temporal reductions
        # Reduce to 2d (B x C x H x W)
        self.reduce_to_channels_min = nn.Sequential(
            Reduce('b c t h w -> b c h w', 'min'),
            nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type),
        )
        self.reduce_to_channels_max = nn.Sequential(
            Reduce('b c t h w -> b c h w', 'max'),
            nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type),
        )
        self.reduce_to_channels_mean = nn.Sequential(
            Reduce('b c t h w -> b c h w', 'max'),
            nn.BatchNorm2d(channels[0]),
            SetActivation(activation_type=activation_type),
        )
        self.instance_norm = nn.InstanceNorm2d(channels[0], affine=False)

    def forward(
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

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
                    ncols=x.shape[-1],
                ).unsqueeze(1)
            ]
            pos_trend_kernels += [
                self.gc(
                    # (B*H*W x T)
                    pos_trend_res.squeeze(),
                    nbatch=x.shape[0],
                    nrows=x.shape[-2],
                    ncols=x.shape[-1],
                ).unsqueeze(1)
            ]
            neg_trend_kernels += [
                self.gc(
                    # (B*H*W x T)
                    neg_trend_res.squeeze(),
                    nbatch=x.shape[0],
                    nrows=x.shape[-2],
                    ncols=x.shape[-1],
                ).unsqueeze(1)
            ]

        # B x 3 x T x H x W
        trend_kernels = (
            torch.cat(peak_kernels, dim=1)
            + torch.cat(pos_trend_kernels, dim=1)
            + torch.cat(neg_trend_kernels, dim=1)
        )

        # Inputs shape is (B x C X T|D x H x W)
        x = self.time_conv0(x)

        # B x T x H x W
        time_logits = self.time_to_hidden(
            self.reduce_to_time(x) + self.reduce_trend_to_time(trend_kernels)
        )

        # B x C x H x W
        channel_logits = (
            self.reduce_to_channels_min(x)
            + self.reduce_to_channels_max(x)
            + self.reduce_to_channels_mean(x)
        )

        # B x C x T x H x W
        encoded = time_logits + channel_logits

        if temporal_encoding is not None:
            encoded = encoded + temporal_encoding

        # Normalize the channels
        encoded = self.instance_norm(encoded)

        return encoded


class PostUNet3Psi(nn.Module):
    def __init__(
        self,
        up_channels: int,
        num_classes: int,
        mask_activation: T.Callable,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
    ):
        super(PostUNet3Psi, self).__init__()

        self.deep_sup_dist = deep_sup_dist
        self.deep_sup_edge = deep_sup_edge
        self.deep_sup_mask = deep_sup_mask

        self.up = model_utils.UpSample()

        self.final_dist = nn.Sequential(
            nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.final_edge = nn.Sequential(
            nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
            SigmoidCrisp(),
        )
        self.final_mask = nn.Sequential(
            nn.Conv2d(up_channels, num_classes, kernel_size=1, padding=0),
            mask_activation,
        )
        if self.deep_sup_dist:
            self.final_dist_3_1 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
            self.final_dist_2_2 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
            self.final_dist_1_3 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        if self.deep_sup_edge:
            self.final_edge_3_1 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                SigmoidCrisp(),
            )
            self.final_edge_2_2 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                SigmoidCrisp(),
            )
            self.final_edge_1_3 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                SigmoidCrisp(),
            )
        if self.deep_sup_mask:
            self.final_mask_3_1 = nn.Sequential(
                nn.Conv2d(up_channels, num_classes, kernel_size=1, padding=0),
                mask_activation,
            )
            self.final_mask_2_2 = nn.Sequential(
                nn.Conv2d(up_channels, num_classes, kernel_size=1, padding=0),
                mask_activation,
            )
            self.final_mask_1_3 = nn.Sequential(
                nn.Conv2d(up_channels, num_classes, kernel_size=1, padding=0),
                mask_activation,
            )

    def forward(
        self,
        out_0_4: T.Dict[str, torch.Tensor],
        out_3_1: T.Dict[str, torch.Tensor],
        out_2_2: T.Dict[str, torch.Tensor],
        out_1_3: T.Dict[str, torch.Tensor],
    ) -> T.Dict[str, torch.Tensor]:
        dist = self.final_dist(out_0_4["dist"])
        edge = self.final_edge(out_0_4["edge"])
        mask = self.final_mask(out_0_4["mask"])

        out = {
            "dist": dist,
            "edge": edge,
            "mask": mask,
            "dist_3_1": None,
            "dist_2_2": None,
            "dist_1_3": None,
            "edge_3_1": None,
            "edge_2_2": None,
            "edge_1_3": None,
            "mask_3_1": None,
            "mask_2_2": None,
            "mask_1_3": None,
        }

        if self.deep_sup_dist:
            out["dist_3_1"] = self.final_dist_3_1(
                self.up(out_3_1["dist"], size=dist.shape[-2:], mode="bilinear")
            )
            out["dist_2_2"] = self.final_dist_2_2(
                self.up(out_2_2["dist"], size=dist.shape[-2:], mode="bilinear")
            )
            out["dist_1_3"] = self.final_dist_1_3(
                self.up(out_1_3["dist"], size=dist.shape[-2:], mode="bilinear")
            )
        if self.deep_sup_edge:
            out["edge_3_1"] = self.final_edge_3_1(
                self.up(out_3_1["edge"], size=edge.shape[-2:], mode="bilinear")
            )
            out["edge_2_2"] = self.final_edge_2_2(
                self.up(out_2_2["edge"], size=edge.shape[-2:], mode="bilinear")
            )
            out["edge_1_3"] = self.final_edge_1_3(
                self.up(out_1_3["edge"], size=edge.shape[-2:], mode="bilinear")
            )
        if self.deep_sup_mask:
            out["mask_3_1"] = self.final_mask_3_1(
                self.up(out_3_1["mask"], size=mask.shape[-2:], mode="bilinear")
            )
            out["mask_2_2"] = self.final_mask_2_2(
                self.up(out_2_2["mask"], size=mask.shape[-2:], mode="bilinear")
            )
            out["mask_1_3"] = self.final_mask_1_3(
                self.up(out_1_3["mask"], size=mask.shape[-2:], mode="bilinear")
            )

        return out


class UNet3Psi(nn.Module):
    """UNet+++ with Psi-Net.

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
        https://arxiv.org/abs/1902.04099
        https://github.com/Bala93/Multi-task-deep-network
    """

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        in_encoding_channels: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dilation: int = 2,
        activation_type: str = "SiLU",
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[Softmax, nn.Sigmoid] = Softmax(dim=1),
    ):
        super(UNet3Psi, self).__init__()

        channels = [
            hidden_channels,
            hidden_channels * 2,
            hidden_channels * 4,
            hidden_channels * 8,
            hidden_channels * 16,
        ]
        up_channels = int(channels[0] * 5)

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            channels=channels,
            activation_type=activation_type,
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for transformer hidden logits
        self.conv0_0 = SingleConv(
            in_channels=(
                in_time
                + int(channels[0] * 4)
                + in_encoding_channels
                # Peak kernels and Trend kernels
                + in_time
            ),
            out_channels=channels[0],
            activation_type=activation_type,
        )
        self.conv1_0 = PoolConv(
            channels[0],
            channels[1],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv2_0 = PoolConv(
            channels[1],
            channels[2],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv3_0 = PoolConv(
            channels[2],
            channels[3],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv4_0 = PoolConv(
            channels[3],
            channels[4],
            double_dilation=dilation,
            activation_type=activation_type,
        )

        # Connect 3
        self.convs_3_1 = UNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_2_2 = UNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_1_3 = UNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_0_4 = UNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )

        self.post_unet = PostUNet3Psi(
            up_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            deep_sup_dist=deep_sup_dist,
            deep_sup_edge=deep_sup_edge,
            deep_sup_mask=deep_sup_mask,
        )

        # Initialise weights
        self.apply(init_conv_weights)

    def forward(
        self, x: torch.Tensor, temporal_encoding: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Inputs shape is (B x C X T|D x H x W)
        h = self.pre_unet(x, temporal_encoding)
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
            x0_0=x0_0, x1_0=x1_0, x2_0=x2_0, x3_0=x3_0, x4_0=x4_0
        )
        # 1/4 connection
        out_2_2 = self.convs_2_2(
            x0_0=x0_0,
            x1_0=x1_0,
            x2_0=x2_0,
            h3_1_dist=out_3_1["dist"],
            h3_1_edge=out_3_1["edge"],
            h3_1_mask=out_3_1["mask"],
            x4_0=x4_0,
        )
        # 1/2 connection
        out_1_3 = self.convs_1_3(
            x0_0=x0_0,
            x1_0=x1_0,
            h2_2_dist=out_2_2["dist"],
            h3_1_dist=out_3_1["dist"],
            h2_2_edge=out_2_2["edge"],
            h3_1_edge=out_3_1["edge"],
            h2_2_mask=out_2_2["mask"],
            h3_1_mask=out_3_1["mask"],
            x4_0=x4_0,
        )
        # 1/1 connection
        out_0_4 = self.convs_0_4(
            x0_0=x0_0,
            h1_3_dist=out_1_3["dist"],
            h2_2_dist=out_2_2["dist"],
            h3_1_dist=out_3_1["dist"],
            h1_3_edge=out_1_3["edge"],
            h2_2_edge=out_2_2["edge"],
            h3_1_edge=out_3_1["edge"],
            h1_3_mask=out_1_3["mask"],
            h2_2_mask=out_2_2["mask"],
            h3_1_mask=out_3_1["mask"],
            x4_0=x4_0,
        )

        out = self.post_unet(
            out_0_4=out_0_4, out_3_1=out_3_1, out_2_2=out_2_2, out_1_3=out_1_3
        )

        return out


class ResUNet3Psi(nn.Module):
    """Residual UNet+++ with Psi-Net (Multi-head streams) and Attention.

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
        in_encoding_channels: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: T.Optional[str] = None,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[Softmax, nn.Sigmoid] = Softmax(dim=1),
    ):
        super(ResUNet3Psi, self).__init__()

        if dilations is None:
            dilations = [2]
        if attention_weights is None:
            attention_weights = "spatial_channel"

        channels = [
            hidden_channels,
            hidden_channels * 2,
            hidden_channels * 4,
            hidden_channels * 8,
            hidden_channels * 16,
        ]
        up_channels = int(channels[0] * 5)

        pre_concat_channels = (
            in_time
            + int(channels[0] * 4)
            + in_encoding_channels
            # Peak kernels and Trend kernels
            + in_time
        )

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            channels=channels,
            concat_channels=pre_concat_channels,
            out_channels=channels[0],
            activation_type=activation_type,
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for RNN hidden logits
        if res_block_type.lower() == ResBlockTypes.RES:
            self.conv0_0 = ResidualConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilation=dilations[0],
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        else:
            self.conv0_0 = ResidualAConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilations=dilations,
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        self.conv1_0 = PoolResidualConv(
            channels[0],
            channels[1],
            dilations=dilations,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv2_0 = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv3_0 = PoolResidualConv(
            channels[2],
            channels[3],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv4_0 = PoolResidualConv(
            channels[3],
            channels[4],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )

        # Connect 3
        self.convs_3_1 = ResUNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_2_2 = ResUNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_1_3 = ResUNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_0_4 = ResUNet3_0_4(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )

        self.post_unet = PostUNet3Psi(
            up_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            deep_sup_dist=deep_sup_dist,
            deep_sup_edge=deep_sup_edge,
            deep_sup_mask=deep_sup_mask,
        )

        # Initialise weights
        self.apply(init_conv_weights)

    def forward(
        self, x: torch.Tensor, temporal_encoding: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Inputs shape is (B x C X T|D x H x W)
        h = self.pre_unet(x, temporal_encoding=temporal_encoding)
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
            side=x3_0,
            down=x4_0,
            pools=[x0_0, x1_0, x2_0],
        )
        # 1/4 connection
        out_2_2 = self.convs_2_2(
            side=x2_0,
            dist_down=[out_3_1["dist"]],
            edge_down=[out_3_1["edge"]],
            mask_down=[out_3_1["mask"]],
            down=x4_0,
            pools=[x0_0, x1_0],
        )
        # 1/2 connection
        out_1_3 = self.convs_1_3(
            side=x1_0,
            dist_down=[out_3_1["dist"], out_2_2["dist"]],
            edge_down=[out_3_1["edge"], out_2_2["edge"]],
            mask_down=[out_3_1["mask"], out_2_2["mask"]],
            down=x4_0,
            pools=[x0_0],
        )
        # 1/1 connection
        out_0_4 = self.convs_0_4(
            side=x0_0,
            dist_down=[out_3_1["dist"], out_2_2["dist"], out_1_3['dist']],
            edge_down=[out_3_1["edge"], out_2_2["edge"], out_1_3['edge']],
            mask_down=[out_3_1["mask"], out_2_2["mask"], out_1_3['mask']],
            down=x4_0,
        )

        out = self.post_unet(
            out_0_4=out_0_4,
            out_3_1=out_3_1,
            out_2_2=out_2_2,
            out_1_3=out_1_3,
        )

        return out


class ResELUNetPsi(nn.Module):
    """Residual efficient and lightweight U-Net (ELU-Net) with Psi-Net (Multi-
    head streams) and Attention.

    References:
        https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf
        https://github.com/Bala93/Multi-task-deep-network
        https://ieeexplore.ieee.org/document/9745574
    """

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: T.Optional[str] = None,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[Softmax, nn.Sigmoid] = Softmax(dim=1),
    ):
        super(ResELUNetPsi, self).__init__()

        if dilations is None:
            dilations = [2]
        if attention_weights is None:
            attention_weights = "spatial_channel"

        channels = [
            hidden_channels,
            hidden_channels * 2,
            hidden_channels * 4,
            hidden_channels * 8,
            hidden_channels * 16,
        ]
        up_channels = int(channels[0] * 5)

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            in_time=in_time,
            channels=channels,
            out_channels=channels[0],
            activation_type=activation_type,
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for RNN hidden logits
        if res_block_type.lower() == ResBlockTypes.RES:
            self.conv0_0 = ResidualConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilation=dilations[0],
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        else:
            self.conv0_0 = ResidualAConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilations=dilations,
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        self.conv1_0 = PoolResidualConv(
            channels[0],
            channels[1],
            dilations=dilations,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv2_0 = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv3_0 = PoolResidualConv(
            channels[2],
            channels[3],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv4_0 = PoolResidualConv(
            channels[3],
            channels[4],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )

        self.convs_3_1 = ResELUNetPsiBlock(
            out_channels=up_channels,
            side_in={
                'dist': {'backbone_3_0': channels[3]},
                'edge': {'out_dist_3_1': up_channels},
                'mask': {'out_edge_3_1': up_channels},
            },
            down_in={
                'dist': {'backbone_4_0': channels[4]},
                'edge': {'backbone_4_0': channels[4]},
                'mask': {'backbone_4_0': channels[4]},
            },
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )
        self.convs_2_2 = ResELUNetPsiBlock(
            out_channels=up_channels,
            side_in={
                'dist': {'backbone_2_0': channels[2]},
                'edge': {'out_dist_2_2': up_channels},
                'mask': {'out_edge_2_2': up_channels},
            },
            down_in={
                'dist': {
                    'backbone_3_0': channels[3],
                    'out_dist_3_1': up_channels,
                },
                'edge': {
                    'out_dist_3_1': up_channels,
                    'out_edge_3_1': up_channels,
                },
                'mask': {
                    'out_edge_3_1': up_channels,
                    'out_mask_3_1': up_channels,
                },
            },
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )
        self.convs_1_3 = ResELUNetPsiBlock(
            out_channels=up_channels,
            side_in={
                'dist': {'backbone_1_0': channels[1]},
                'edge': {'out_dist_1_3': up_channels},
                'mask': {'out_edge_1_3': up_channels},
            },
            down_in={
                'dist': {
                    'backbone_3_0': channels[3],
                    'backbone_2_0': channels[2],
                    'out_dist_2_2': up_channels,
                },
                'edge': {
                    'out_dist_2_2': up_channels,
                    'out_edge_2_2': up_channels,
                },
                'mask': {
                    'out_edge_2_2': up_channels,
                    'out_mask_2_2': up_channels,
                },
            },
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )
        self.convs_0_4 = ResELUNetPsiBlock(
            out_channels=up_channels,
            side_in={
                'dist': {'backbone_0_0': channels[0]},
                'edge': {'out_dist_0_4': up_channels},
                'mask': {'out_edge_0_4': up_channels},
            },
            down_in={
                'dist': {
                    'backbone_3_0': channels[3],
                    'backbone_2_0': channels[2],
                    'backbone_1_0': channels[1],
                    'out_dist_1_3': up_channels,
                },
                'edge': {
                    'out_dist_1_3': up_channels,
                    'out_edge_1_3': up_channels,
                },
                'mask': {
                    'out_edge_1_3': up_channels,
                    'out_mask_1_3': up_channels,
                },
            },
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )

        self.post_unet = PostUNet3Psi(
            up_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            deep_sup_dist=deep_sup_dist,
            deep_sup_edge=deep_sup_edge,
            deep_sup_mask=deep_sup_mask,
        )

        # Initialise weights
        self.apply(init_conv_weights)

    def forward(
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:

        """x Shaped (B x C X T|D x H x W) temporal_encoding Shaped (B x C x H X
        W)"""
        embeddings = self.pre_unet(x, temporal_encoding=temporal_encoding)

        # embeddings shape is (B x C x H x W)
        # Backbone
        # 1/1
        x0_0 = self.conv0_0(embeddings)
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
            side={
                'dist': {'backbone_3_0': x3_0},
                'edge': {'out_dist_3_1': None},
                'mask': {'out_edge_3_1': None},
            },
            down={
                'dist': {'backbone_4_0': x4_0},
                'edge': {'backbone_4_0': x4_0},
                'mask': {'backbone_4_0': x4_0},
            },
            shape=x3_0.shape[-2:],
        )
        out_2_2 = self.convs_2_2(
            side={
                'dist': {'backbone_2_0': x2_0},
                'edge': {'out_dist_2_2': None},
                'mask': {'out_edge_2_2': None},
            },
            down={
                'dist': {
                    'backbone_3_0': x3_0,
                    'out_dist_3_1': out_3_1['dist'],
                },
                'edge': {
                    'out_dist_3_1': out_3_1['dist'],
                    'out_edge_3_1': out_3_1['edge'],
                },
                'mask': {
                    'out_edge_3_1': out_3_1['edge'],
                    'out_mask_3_1': out_3_1['mask'],
                },
            },
            shape=x2_0.shape[-2:],
        )
        out_1_3 = self.convs_1_3(
            side={
                'dist': {'backbone_1_0': x1_0},
                'edge': {'out_dist_1_3': None},
                'mask': {'out_edge_1_3': None},
            },
            down={
                'dist': {
                    'backbone_3_0': x3_0,
                    'backbone_2_0': x2_0,
                    'out_dist_2_2': out_2_2['dist'],
                },
                'edge': {
                    'out_dist_2_2': out_2_2['dist'],
                    'out_edge_2_2': out_2_2['edge'],
                },
                'mask': {
                    'out_edge_2_2': out_2_2['edge'],
                    'out_mask_2_2': out_2_2['mask'],
                },
            },
            shape=x1_0.shape[-2:],
        )
        out_0_4 = self.convs_0_4(
            side={
                'dist': {'backbone_0_0': x0_0},
                'edge': {'out_dist_0_4': None},
                'mask': {'out_edge_0_4': None},
            },
            down={
                'dist': {
                    'backbone_3_0': x3_0,
                    'backbone_2_0': x2_0,
                    'backbone_1_0': x1_0,
                    'out_dist_1_3': out_1_3['dist'],
                },
                'edge': {
                    'out_dist_1_3': out_1_3['dist'],
                    'out_edge_1_3': out_1_3['edge'],
                },
                'mask': {
                    'out_edge_1_3': out_1_3['edge'],
                    'out_mask_1_3': out_1_3['mask'],
                },
            },
            shape=x0_0.shape[-2:],
        )

        out = self.post_unet(
            out_0_4=out_0_4,
            out_3_1=out_3_1,
            out_2_2=out_2_2,
            out_1_3=out_1_3,
        )

        return out


if __name__ == '__main__':
    batch_size = 2
    num_channels = 3
    in_encoding_channels = 64
    num_head = 8
    num_time = 12
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, num_time, height, width),
        dtype=torch.float32,
    )
    logits_hidden = torch.rand(
        (batch_size, in_encoding_channels, height, width), dtype=torch.float32
    )

    model = ResUNet3Psi(
        in_channels=num_channels,
        in_time=num_time,
        in_encoding_channels=in_encoding_channels,
        activation_type="SiLU",
        res_block_type=ResBlockTypes.RES,
    )
    logits = model(x, temporal_encoding=logits_hidden)

    assert logits['dist'].shape == (batch_size, 1, height, width)
    assert logits['edge'].shape == (batch_size, 1, height, width)
    assert logits['mask'].shape == (batch_size, 2, height, width)
