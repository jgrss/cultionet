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


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ):
        super().__init__()

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
        super().__init__()

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
        out_channels: int,
        activation_type: str,
        trend_kernel_size: int = 5,
    ):
        super().__init__()

        self.reduce_time_init = ReduceTimeToOne(
            in_channels=in_channels,
            out_channels=out_channels,
            num_time=in_time,
        )
        self.peak_kernel = nn.Sequential(
            cunn.Peaks3d(kernel_size=trend_kernel_size),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=out_channels,
                num_time=in_time,
                activation_type=activation_type,
            ),
        )
        self.pos_trend_kernel = nn.Sequential(
            cunn.Trend3d(kernel_size=trend_kernel_size, direction="positive"),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=out_channels,
                num_time=in_time,
                activation_type=activation_type,
            ),
        )
        self.neg_trend_kernel = nn.Sequential(
            cunn.Trend3d(kernel_size=trend_kernel_size, direction="negative"),
            ReduceTimeToOne(
                in_channels=in_channels,
                out_channels=out_channels,
                num_time=in_time,
                activation_type=activation_type,
            ),
        )

        self.layer_norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(out_channels),
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


class TowerUNet(nn.Module):
    """Tower U-Net."""

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        mask_activation: T.Union[nn.Softmax, nn.Sigmoid] = nn.Softmax(dim=1),
        deep_supervision: bool = False,
        pool_attention: bool = False,
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2]

        self.deep_supervision = deep_supervision

        channels = [
            hidden_channels,  # a
            hidden_channels * 2,  # b
            hidden_channels * 4,  # c
            hidden_channels * 8,  # d
        ]
        up_channels = int(hidden_channels * len(channels))

        self.pre_unet = PreUnet3Psi(
            in_channels=in_channels,
            in_time=in_time,
            out_channels=channels[0],
            activation_type=activation_type,
        )

        self.encoder = cunn.TowerUNetEncoder(
            channels=channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            pool_attention=pool_attention,
            pool_by_max=pool_by_max,
            repeat_resa_kernel=repeat_resa_kernel,
            batchnorm_first=batchnorm_first,
            concat_resid=concat_resid,
        )

        self.decoder = cunn.TowerUNetDecoder(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            repeat_resa_kernel=repeat_resa_kernel,
            batchnorm_first=batchnorm_first,
            concat_resid=concat_resid,
        )

        self.tower_fusion = cunn.TowerUNetFusion(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            repeat_resa_kernel=repeat_resa_kernel,
            batchnorm_first=batchnorm_first,
            concat_resid=concat_resid,
        )

        self.final_a = cunn.TowerUNetFinal(
            in_channels=up_channels,
            num_classes=num_classes,
            mask_activation=mask_activation,
            activation_type=activation_type,
        )

        if self.deep_supervision:
            self.final_b = cunn.TowerUNetFinal(
                in_channels=up_channels,
                num_classes=num_classes,
                mask_activation=mask_activation,
                activation_type=activation_type,
                resample_factor=2,
            )
            self.final_c = cunn.TowerUNetFinal(
                in_channels=up_channels,
                num_classes=num_classes,
                mask_activation=mask_activation,
                activation_type=activation_type,
                resample_factor=4,
            )

        # Initialize weights
        self.apply(init_conv_weights)

    def forward(
        self,
        x: torch.Tensor,
        temporal_encoding: T.Optional[torch.Tensor] = None,
        latlon_coords: T.Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> T.Dict[str, torch.Tensor]:

        """Forward pass.

        Parameters
        ==========
        x
            Shaped (B x C x T x H x W)
        temporal_encoding
            Shaped (B x C x H X W)
        """

        # Initial temporal reduction and convolutions to
        # hidden dimensions
        embeddings = self.pre_unet(x, temporal_encoding=temporal_encoding)

        encoded = self.encoder(embeddings)
        decoded = self.decoder(encoded)
        towers_fused = self.tower_fusion(encoded=encoded, decoded=decoded)

        # Final outputs
        out = self.final_a(
            towers_fused["x_tower_a"],
            latlon_coords=latlon_coords,
        )

        if training and self.deep_supervision:
            out_c = self.final_c(
                towers_fused["x_tower_c"],
                latlon_coords=latlon_coords,
                size=towers_fused["x_tower_a"].shape[-2:],
                suffix="_c",
            )
            out_b = self.final_b(
                towers_fused["x_tower_b"],
                latlon_coords=latlon_coords,
                size=towers_fused["x_tower_a"].shape[-2:],
                suffix="_b",
            )

            out.update(out_b)
            out.update(out_c)

        return out
