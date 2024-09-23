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


class PreTimeReduction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int,
        activation_type: str,
        batchnorm_first: bool,
    ):
        super().__init__()

        time_kernel_size = 5
        remaining_time = (
            (in_time - time_kernel_size + 1) - time_kernel_size + 1
        )

        self.seq = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(time_kernel_size, 1, 1),
                padding=0,
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(time_kernel_size, 1, 1),
                padding=0,
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(remaining_time, 1, 1),
                padding=0,
            ),
            # c = channels; t = 1
            Rearrange('b c t h w -> b (c t) h w'),
            cunn.ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=True,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
            ),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(out_channels),
            Rearrange('b h w c -> b c h w'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ==========
        x
            Input, shaped (B, C, T, H, W).
        """
        return self.seq(x)


class TowerUNet(nn.Module):
    """Tower U-Net."""

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 64,
        num_classes: int = 1,
        dilations: T.Optional[T.Sequence[int]] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.NATTEN,
        pool_by_max: bool = False,
        batchnorm_first: bool = False,
        edge_activation: bool = True,
        mask_activation: bool = True,
        use_latlon: bool = False,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2]

        channels = [
            hidden_channels,  # a
            hidden_channels * 2,  # b
            hidden_channels * 4,  # c
            hidden_channels * 8,  # d
        ]
        up_channels = int(hidden_channels * len(channels))

        self.pre_unet = torch.compile(
            PreTimeReduction(
                in_channels=in_channels,
                in_time=in_time,
                out_channels=channels[0],
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
            )
        )

        self.encoder = cunn.TowerUNetEncoder(
            channels=channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            pool_by_max=pool_by_max,
            batchnorm_first=batchnorm_first,
        )

        self.decoder = cunn.TowerUNetDecoder(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            batchnorm_first=batchnorm_first,
        )

        self.tower_fusion = cunn.TowerUNetFusion(
            channels=channels,
            up_channels=up_channels,
            dilations=dilations,
            activation_type=activation_type,
            dropout=dropout,
            res_block_type=res_block_type,
            attention_weights=attention_weights,
            batchnorm_first=batchnorm_first,
            use_latlon=use_latlon,
        )

        self.final_a = cunn.TowerUNetFinal(
            in_channels=up_channels,
            num_classes=num_classes,
            activation_type=activation_type,
        )

        self.final_b = cunn.TowerUNetFinal(
            in_channels=up_channels,
            num_classes=num_classes,
            activation_type=activation_type,
            resample_factor=2,
        )

        self.final_c = cunn.TowerUNetFinal(
            in_channels=up_channels,
            num_classes=num_classes,
            activation_type=activation_type,
            resample_factor=4,
        )

        self.final_combine = cunn.TowerUNetFinalCombine(
            num_classes=num_classes,
            edge_activation=edge_activation,
            mask_activation=mask_activation,
        )

        # Initialize weights
        self.apply(init_conv_weights)

    def forward(
        self,
        x: torch.Tensor,
        latlon_coords: T.Optional[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:

        """Forward pass.

        Parameters
        ==========
        x
            The input image time series, shaped (B, C, T, H, W).
        """

        # Initial temporal reduction and convolutions to
        # hidden dimensions
        embeddings = self.pre_unet(x)

        encoded = self.encoder(embeddings)
        decoded = self.decoder(encoded)
        towers_fused = self.tower_fusion(
            encoded=encoded,
            decoded=decoded,
            latlon_coords=latlon_coords,
        )

        # Final outputs

        # -> {InferenceNames.DISTANCE_a, InferenceNames.EDGE_a, InferenceNames.CROP_a}
        out_a = self.final_a(
            towers_fused["x_tower_a"],
            suffix="_a",
        )

        # -> {InferenceNames.DISTANCE_b, InferenceNames.EDGE_b, InferenceNames.CROP_b}
        out_b = self.final_b(
            towers_fused["x_tower_b"],
            size=towers_fused["x_tower_a"].shape[-2:],
            suffix="_b",
        )

        # -> {InferenceNames.DISTANCE_c, InferenceNames.EDGE_c, InferenceNames.CROP_c}
        out_c = self.final_c(
            towers_fused["x_tower_c"],
            size=towers_fused["x_tower_a"].shape[-2:],
            suffix="_c",
        )

        out = self.final_combine(
            out_a, out_b, out_c, suffixes=["_a", "_b", "_c"]
        )

        return out
