"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet.

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .. import nn as cunn
from ..enums import AttentionTypes, ResBlockTypes
from ..layers.weights import init_conv_weights
from .field_of_junctions import FieldOfJunctions


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ):
        super(DepthwiseSeparableConv, self).__init__()

        self.separable = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.separable(x)


class ReduceTimeToOne(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_time: int,
        activation_type: str = 'SiLU',
    ):
        super(ReduceTimeToOne, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(num_time, 1, 1),
                padding=0,
                bias=False,
            ),
            Rearrange('b c t h w -> b (c t) h w'),
            nn.BatchNorm2d(out_channels),
            cunn.SetActivation(activation_type=activation_type),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                hidden_channels=out_channels,
                out_channels=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            cunn.SetActivation(activation_type=activation_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PreUnet3Psi(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        channels: T.Sequence[int],
        activation_type: str,
        trend_kernel_size: int = 5,
    ):
        super(PreUnet3Psi, self).__init__()

        self.reduce_time_init = ReduceTimeToOne(
            in_channels=in_channels,
            out_channels=channels[0],
            num_time=in_time,
        )
        self.peak_kernel = nn.Sequential(
            cunn.Peaks3d(kernel_size=trend_kernel_size),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=channels[0],
                num_time=in_time,
                activation_type=activation_type,
            ),
        )
        self.pos_trend_kernel = nn.Sequential(
            cunn.Trend3d(kernel_size=trend_kernel_size, direction="positive"),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=channels[0],
                num_time=in_time,
                activation_type=activation_type,
            ),
        )
        self.neg_trend_kernel = nn.Sequential(
            cunn.Trend3d(kernel_size=trend_kernel_size, direction="negative"),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=channels[0],
                num_time=in_time,
                activation_type=activation_type,
            ),
        )

        self.layer_norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(channels[0]),
            Rearrange('b h w c -> b c h w'),
        )

    def forward(
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        encoded = self.reduce_time_init(x)
        encoded = (
            encoded
            + self.peak_kernel(x)
            + self.pos_trend_kernel(x)
            + self.neg_trend_kernel(x)
        )

        if temporal_encoding is not None:
            encoded = encoded + temporal_encoding

        # Normalize the channels
        encoded = self.layer_norm(encoded)

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

        self.up = cunn.UpSample()

        self.final_dist = nn.Sequential(
            nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.final_edge = nn.Sequential(
            nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
            cunn.SigmoidCrisp(),
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
                cunn.SigmoidCrisp(),
            )
            self.final_edge_2_2 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                cunn.SigmoidCrisp(),
            )
            self.final_edge_1_3 = nn.Sequential(
                nn.Conv2d(up_channels, 1, kernel_size=1, padding=0),
                cunn.SigmoidCrisp(),
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
        mask_activation: T.Union[nn.Softmax, nn.Sigmoid] = nn.Softmax(dim=1),
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
        self.conv0_0 = cunn.SingleConv(
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
        self.conv1_0 = cunn.PoolConv(
            channels[0],
            channels[1],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv2_0 = cunn.PoolConv(
            channels[1],
            channels[2],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv3_0 = cunn.PoolConv(
            channels[2],
            channels[3],
            double_dilation=dilation,
            activation_type=activation_type,
        )
        self.conv4_0 = cunn.PoolConv(
            channels[3],
            channels[4],
            double_dilation=dilation,
            activation_type=activation_type,
        )

        # Connect 3
        self.convs_3_1 = cunn.UNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_2_2 = cunn.UNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_1_3 = cunn.UNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=[dilation],
            activation_type=activation_type,
        )
        self.convs_0_4 = cunn.UNet3_0_4(
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
        hidden_channels: int = 32,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: T.Optional[str] = None,
        deep_sup_dist: T.Optional[bool] = False,
        deep_sup_edge: T.Optional[bool] = False,
        deep_sup_mask: T.Optional[bool] = False,
        mask_activation: T.Union[nn.Softmax, nn.Sigmoid] = nn.Softmax(dim=1),
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

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            in_time=in_time,
            channels=channels,
            activation_type=activation_type,
        )

        # Inputs =
        # Reduced time dimensions
        # Reduced channels (x2) for mean and max
        # Input filters for RNN hidden logits
        if res_block_type.lower() == ResBlockTypes.RES:
            self.conv0_0 = cunn.ResidualConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilation=dilations[0],
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        else:
            self.conv0_0 = cunn.ResidualAConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilations=dilations,
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        self.conv1_0 = cunn.PoolResidualConv(
            channels[0],
            channels[1],
            dilations=dilations,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv2_0 = cunn.PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv3_0 = cunn.PoolResidualConv(
            channels[2],
            channels[3],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )
        self.conv4_0 = cunn.PoolResidualConv(
            channels[3],
            channels[4],
            dilations=dilations,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
        )

        # Connect 3
        self.convs_3_1 = cunn.ResUNet3_3_1(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_2_2 = cunn.ResUNet3_2_2(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_1_3 = cunn.ResUNet3_1_3(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
        )
        self.convs_0_4 = cunn.ResUNet3_0_4(
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
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
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


class TowerFinal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        mask_activation: T.Callable,
        resample_factor: int = 0,
    ):
        super(TowerFinal, self).__init__()

        self.up = cunn.UpSample()

        if resample_factor > 1:
            self.up_conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=resample_factor,
                padding=1,
            )

        self.expand = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, padding=0
        )
        self.final_dist = nn.Sequential(
            cunn.ConvBlock2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                add_activation=True,
                activation_type="SiLU",
            ),
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.final_edge = nn.Sequential(
            cunn.ConvBlock2d(
                in_channels=in_channels + 1,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                add_activation=True,
                activation_type="SiLU",
            ),
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            cunn.SigmoidCrisp(),
        )
        self.final_mask = nn.Sequential(
            cunn.ConvBlock2d(
                in_channels=in_channels + 2,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                add_activation=True,
                activation_type="SiLU",
            ),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=0),
            mask_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        shape: T.Optional[tuple] = None,
        suffix: str = "",
        foj_boundaries: T.Optional[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:
        if shape is not None:
            x = self.up(
                self.up_conv(x),
                size=shape,
                mode="bilinear",
            )

        dist_connect, edge_connect, mask_connect = torch.chunk(
            self.expand(x), 3, dim=1
        )

        # if foj_boundaries is not None:
        #     edge = edge * foj_boundaries

        dist = self.final_dist(dist_connect)
        edge = self.final_edge(torch.cat((edge_connect, dist), dim=1))
        mask = self.final_mask(torch.cat((mask_connect, dist, edge), dim=1))

        return {
            f"dist{suffix}": dist,
            f"edge{suffix}": edge,
            f"mask{suffix}": mask,
        }


class TowerUNet(nn.Module):
    """Tower U-Net."""

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        mask_activation: T.Union[nn.Softmax, nn.Sigmoid] = nn.Softmax(dim=1),
        deep_supervision: bool = False,
        get_junctions: bool = False,
    ):
        super(TowerUNet, self).__init__()

        self.deep_supervision = deep_supervision

        channels = [
            hidden_channels,
            hidden_channels * 2,
            hidden_channels * 4,
            hidden_channels * 8,
        ]
        up_channels = int(hidden_channels * len(channels))

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            in_time=in_time,
            channels=channels,
            activation_type=activation_type,
        )

        # Backbone layers
        if res_block_type.lower() == ResBlockTypes.RES:
            self.down_a = cunn.ResidualConv(
                in_channels=channels[0],
                out_channels=channels[0],
                num_blocks=2,
                activation_type=activation_type,
                attention_weights=attention_weights,
            )
        else:
            self.down_a = cunn.ResidualAConv(
                in_channels=channels[0],
                out_channels=channels[0],
                dilations=dilations,
                activation_type=activation_type,
                attention_weights=attention_weights,
            )

        self.down_b = cunn.PoolResidualConv(
            channels[0],
            channels[1],
            dropout=dropout,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
            dilations=dilations,
        )
        self.down_c = cunn.PoolResidualConv(
            channels[1],
            channels[2],
            dropout=dropout,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
            dilations=dilations,
        )
        self.down_d = cunn.PoolResidualConv(
            channels[2],
            channels[3],
            dropout=dropout,
            kernel_size=1,
            num_blocks=1,
            activation_type=activation_type,
            attention_weights=attention_weights,
            res_block_type=res_block_type,
            dilations=[1],
        )

        # Up layers
        self.up_du = cunn.TowerUNetUpLayer(
            in_channels=channels[3],
            out_channels=up_channels,
            num_blocks=1,
            kernel_size=1,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=[1],
            resample_up=False,
        )
        self.up_cu = cunn.TowerUNetUpLayer(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )
        self.up_bu = cunn.TowerUNetUpLayer(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )
        self.up_au = cunn.TowerUNetUpLayer(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )

        # Towers
        self.tower_c = cunn.TowerUNetBlock(
            backbone_side_channels=channels[2],
            backbone_down_channels=channels[3],
            up_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )

        self.tower_b = cunn.TowerUNetBlock(
            backbone_side_channels=channels[1],
            backbone_down_channels=channels[2],
            up_channels=up_channels,
            out_channels=up_channels,
            tower=True,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )

        self.tower_a = cunn.TowerUNetBlock(
            backbone_side_channels=channels[0],
            backbone_down_channels=channels[1],
            up_channels=up_channels,
            out_channels=up_channels,
            tower=True,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            dilations=dilations,
        )

        self.field_of_junctions = None
        if get_junctions:
            self.field_of_junctions = FieldOfJunctions(
                in_channels=hidden_channels,
                # NOTE: setup for padding of 5 x 5
                # TODO: set this as a parameter
                height=110,
                width=110,
            )

        self.final_a = TowerFinal(
            in_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
        )

        if self.deep_supervision:
            self.final_b = TowerFinal(
                in_channels=up_channels,
                num_classes=num_classes,
                mask_activation=mask_activation,
                resample_factor=2,
            )
            self.final_c = TowerFinal(
                in_channels=up_channels,
                num_classes=num_classes,
                mask_activation=mask_activation,
                resample_factor=4,
            )

        # Initialise weights
        self.apply(init_conv_weights)

    def forward(
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:

        """Forward pass.

        Parameters
        ==========
        x
            Shaped (B x C X T|D x H x W) temporal_encoding Shaped (B x C x H X W)
        """

        embeddings = self.pre_unet(x, temporal_encoding=temporal_encoding)

        # Backbone
        x_a = self.down_a(embeddings)
        x_b = self.down_b(x_a)
        x_c = self.down_c(x_b)
        x_d = self.down_d(x_c)

        # Over
        x_du = self.up_du(x_d, shape=x_d.shape[-2:])

        # Up
        x_cu = self.up_cu(x_du, shape=x_c.shape[-2:])
        x_bu = self.up_bu(x_cu, shape=x_b.shape[-2:])
        x_au = self.up_au(x_bu, shape=x_a.shape[-2:])

        x_tower_c = self.tower_c(
            backbone_side=x_c,
            backbone_down=x_d,
            side=x_cu,
            down=x_du,
        )
        x_tower_b = self.tower_b(
            backbone_side=x_b,
            backbone_down=x_c,
            side=x_bu,
            down=x_cu,
            down_tower=x_tower_c,
        )
        x_tower_a = self.tower_a(
            backbone_side=x_a,
            backbone_down=x_b,
            side=x_au,
            down=x_bu,
            down_tower=x_tower_b,
        )

        foj_output = {}
        if self.field_of_junctions is not None:
            foj_output = self.field_of_junctions(embeddings)

        out = self.final_a(
            x_tower_a,
            foj_boundaries=foj_output.get("boundaries"),
        )

        if foj_output:
            out.update(
                {
                    "foj_image_patches": foj_output["image_patches"],
                    "foj_patches": foj_output["patches"],
                }
            )

        if self.deep_supervision:
            out_c = self.final_c(
                x_tower_c,
                shape=x_tower_a.shape[-2:],
                suffix="_c",
            )
            out_b = self.final_b(
                x_tower_b,
                shape=x_tower_a.shape[-2:],
                suffix="_b",
            )

            out.update(out_b)
            out.update(out_c)

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
