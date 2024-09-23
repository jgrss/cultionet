import logging
import typing as T

import natten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from cultionet.enums import AttentionTypes, ResBlockTypes

from ..functional import check_upsample
from .activations import SetActivation
from .attention import SpatialChannelAttention

# logging.getLogger("natten").setLevel(logging.ERROR)
# natten.use_fused_na(True)
# natten.use_kv_parallelism_in_fused_na(True)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.separable = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.separable(x)


class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        return check_upsample(
            self.up_conv(x),
            size=size,
        )


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
        add_activation: bool = True,
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
    ):
        super().__init__()

        layers = []

        if batchnorm_first:
            layers += [
                nn.BatchNorm2d(in_channels),
                SetActivation(activation_type),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                ),
            ]
        else:
            layers += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
            if add_activation:
                layers += [SetActivation(activation_type)]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResConvBlock2d(nn.Module):
    """Convolution layer designed for a residual activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation_type: str = "SiLU",
        num_blocks: int = 2,
        batchnorm_first: bool = False,
    ):
        super().__init__()

        assert num_blocks > 0, "There must be at least one block."

        conv_layers = []

        conv_layers.append(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0 if kernel_size == 1 else kernel_size // 2,
                dilation=1,
                activation_type=activation_type,
                add_activation=True,
                batchnorm_first=batchnorm_first,
            )
        )

        for _ in range(num_blocks - 1):
            conv_layers.append(
                ConvBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=0 if kernel_size == 1 else max(1, dilation - 1),
                    dilation=1 if kernel_size == 1 else max(1, dilation - 1),
                    activation_type=activation_type,
                    add_activation=True,
                    batchnorm_first=batchnorm_first,
                )
            )

        self.block = nn.ModuleList(conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.block:
            x = layer(x)

        return x


class ResidualConv(nn.Module):
    """A residual convolution layer with (optional) attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_blocks: int = 2,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
    ):
        super().__init__()

        self.attention_weights = attention_weights

        if self.attention_weights is not None:
            assert self.attention_weights in [
                AttentionTypes.SPATIAL_CHANNEL,
            ], "The attention method is not supported."

            self.gamma = nn.Parameter(torch.ones(1, requires_grad=True))

            self.attention_conv = SpatialChannelAttention(
                out_channels=out_channels, activation_type=activation_type
            )

        self.seq = ResConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            activation_type=activation_type,
            batchnorm_first=batchnorm_first,
        )

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

        if self.attention_weights is not None:
            self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip is not None:
            # Align channels
            out = self.skip(x)
        else:
            out = x

        out = out + self.seq(x)

        if self.attention_weights is not None:
            # Get weights from the residual
            attention = self.attention_conv(out)

            # 1 + γA
            attention = 1.0 + self.gamma * attention
            out = out * attention

            out = self.final_act(out)

        return out


class ResidualAConv(nn.Module):
    r"""Residual convolution with atrous/dilated convolutions.

    Residual convolutions:

        CSIRO BSTD/MIT LICENSE

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that
        the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
            following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
            the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
            promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
        USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Citation:
            @article{diakogiannis_etal_2020,
                title={ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data},
                author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter and Wu, Chen},
                journal={ISPRS Journal of Photogrammetry and Remote Sensing},
                volume={162},
                pages={94--114},
                year={2020},
                publisher={Elsevier}
            }

        References:
            https://www.sciencedirect.com/science/article/abs/pii/S0924271620300149
            https://arxiv.org/abs/1904.00592
            https://arxiv.org/pdf/1904.00592.pdf

    Attention with NATTEN:
        MIT License
        Copyright (c) 2022 - 2024 Ali Hassani.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_blocks: int = 2,
        dilations: T.Optional[T.List[int]] = None,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2]

        self.attention_weights = attention_weights

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.skip = nn.Identity()

        if self.attention_weights is not None:

            assert self.attention_weights in [
                AttentionTypes.NATTEN,
                AttentionTypes.SPATIAL_CHANNEL,
            ], "The attention method is not supported."

            if self.attention_weights == AttentionTypes.NATTEN:

                self.attention_conv = nn.Sequential(
                    Rearrange('b c h w -> b h w c'),
                    nn.LayerNorm(out_channels),
                    natten.NeighborhoodAttention2D(
                        dim=out_channels,
                        num_heads=natten_num_heads,
                        kernel_size=natten_kernel_size,
                        dilation=natten_dilation,
                        rel_pos_bias=False,
                        qkv_bias=True,
                        attn_drop=natten_attn_drop,
                        proj_drop=natten_proj_drop,
                    ),
                    nn.LayerNorm(out_channels),
                    Rearrange('b h w c -> b c h w'),
                )

            else:

                self.attention_conv = SpatialChannelAttention(
                    in_channels=out_channels,
                    activation_type=activation_type,
                )

        self.res_modules = nn.ModuleList(
            [
                ResConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation_type=activation_type,
                    num_blocks=num_blocks,
                    batchnorm_first=batchnorm_first,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.skip(x)

        if self.attention_weights is not None:
            skip = out

        # Resunet-a block takes the same input and
        # sums multiple outputs with varying dilations.
        for layer in self.res_modules:
            out = out + layer(x)

        if self.attention_weights is not None:
            attention_out = self.attention_conv(skip)
            if self.attention_weights == AttentionTypes.NATTEN:
                out = out + attention_out
            else:
                out = out * attention_out

        return out


class PoolResidualConv(nn.Module):
    """Residual convolution with down-sampling.

    Default:
        1) Convolution block
        2) Down-sampling by adaptive max pooling

    If pool_first=True:
        1) Down-sampling by adaptive max pooling
        2) Convolution block
        If dropout > 0
        3) Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
        num_blocks: int = 2,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RESA,
        dilations: T.Sequence[int] = None,
        pool_first: bool = True,
        pool_by_max: bool = False,
        batchnorm_first: bool = False,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
    ):
        super().__init__()

        assert res_block_type in (
            ResBlockTypes.RES,
            ResBlockTypes.RESA,
        )

        self.pool_first = pool_first
        self.pool_by_max = pool_by_max
        if self.pool_first:
            if not self.pool_by_max:
                if batchnorm_first:
                    self.pool_conv = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                    )
                else:
                    self.pool_conv = ConvBlock2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        add_activation=False,
                        batchnorm_first=False,
                    )

                in_channels = out_channels

        if res_block_type == ResBlockTypes.RES:

            self.res_conv = ResidualConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                attention_weights=attention_weights,
                num_blocks=num_blocks,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
            )

        else:

            self.res_conv = ResidualAConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilations=dilations,
                num_blocks=num_blocks,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
                natten_num_heads=natten_num_heads,
                natten_kernel_size=natten_kernel_size,
                natten_dilation=natten_dilation,
                natten_attn_drop=natten_attn_drop,
                natten_proj_drop=natten_proj_drop,
            )

        self.dropout_layer = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        if self.pool_first:
            if self.pool_by_max:
                x = F.adaptive_max_pool2d(
                    x, output_size=(height // 2, width // 2)
                )
            else:
                x = self.pool_conv(x)

        # Residual convolution
        x = self.res_conv(x)

        # Dropout
        x = self.dropout_layer(x)

        return x
