"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet.

MIT License

Copyright (c) 2018 Takato Kimura
"""
import typing as T

import torch
import torch.nn as nn
from einops import rearrange
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


class PreTimeReduction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int,
        activation_type: str,
        batchnorm_first: bool,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Reduce time to 1
        self.reduce_time = nn.Sequential(
            # Randomly drop time steps
            Rearrange('b c t h w -> b t c h w'),
            nn.Dropout3d(p=dropout),
            Rearrange('b t c h w -> b c t h w'),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(in_time, 1, 1),
                padding=0,
                bias=False,
            ),
            Rearrange('b c t h w -> b (c t) h w'),
            nn.BatchNorm2d(in_channels),
            cunn.SetActivation(activation_type),
        )

        # Reduce channels to 1
        self.reduce_channels = nn.Sequential(
            # Randomly drop channels
            nn.Dropout3d(p=dropout),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            Rearrange('b c t h w -> b (c t) h w'),
            nn.BatchNorm2d(in_time),
            cunn.SetActivation(activation_type),
        )

        self.channels_to_stack = cunn.ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels * in_time,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.time_to_stack = cunn.ConvBlock2d(
            in_channels=in_time,
            out_channels=in_channels * in_time,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.channels_stack_to_hidden = cunn.ConvBlock2d(
            in_channels=in_channels * in_time,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.time_stack_to_hidden = cunn.ConvBlock2d(
            in_channels=in_channels * in_time,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.stream_to_hidden = cunn.ConvBlock2d(
            in_channels=in_channels * in_time,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.layer_norm = nn.Sequential(
            cunn.SqueezeAndExcitation(
                in_channels=out_channels,
                squeeze_channels=out_channels // 2,
            ),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(out_channels),
            Rearrange('b h w c -> b c h w'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shaped (B, C, T, H, W).
        """

        # Reduce time to 1 and squeeze
        # -> (B, C, H, W)
        x_channels = self.reduce_time(x)

        # Reduce channels to 1 and squeeze
        # -> (B, T, H, W)
        x_time = self.reduce_channels(x)

        # C -> (C x T)
        x_channels = self.channels_to_stack(x_channels)
        # T -> (C x T)
        x_time = self.time_to_stack(x_time)

        # Stack time and channels
        x_stream = rearrange(x, 'b c t h w -> b (c t) h w')

        # Fuse
        x_stream = x_stream + x_channels + x_time

        # (C x T) -> H
        x_stream = self.stream_to_hidden(x_stream)
        x_channels = self.channels_stack_to_hidden(x_channels)
        x_time = self.time_stack_to_hidden(x_time)

        # Fuse
        x_stream = x_stream + x_channels + x_time

        return self.layer_norm(x_stream)


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


class TowerUNet(nn.Module):
    """Tower U-Net."""

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 64,
        num_classes: int = 2,
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

        # self.pre_unet = torch.compile(
        #     PreUnet3Psi(
        #         in_channels=in_channels,
        #         in_time=in_time,
        #         out_channels=channels[0],
        #         activation_type=activation_type,
        #     )
        # )

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

        if torch.isnan(embeddings).any():
            import ipdb

            ipdb.set_trace()

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
