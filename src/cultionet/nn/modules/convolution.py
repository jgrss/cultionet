import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from cultionet.enums import AttentionTypes, ResBlockTypes

from .activations import SetActivation
from .attention import FractalAttention, SpatialChannelAttention
from .reshape import Squeeze, UpSample


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = "SiLU",
    ):
        super(ConvBlock2d, self).__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if add_activation:
            layers += [
                SetActivation(activation_type, channels=out_channels, dims=2)
            ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        in_time: int = 0,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        squeeze: bool = False,
        activation_type: str = "SiLU",
    ):
        super(ConvBlock3d, self).__init__()

        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        if squeeze:
            layers += [Squeeze(), nn.BatchNorm2d(in_time)]
            dims = 2
        else:
            layers += [nn.BatchNorm3d(out_channels)]
            dims = 3
        if add_activation:
            layers += [
                SetActivation(
                    activation_type, channels=out_channels, dims=dims
                )
            ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResSpatioTemporalConv3d(nn.Module):
    """A spatio-temporal convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = "SiLU",
    ):
        super(ResSpatioTemporalConv3d, self).__init__()

        layers = [
            # Conv -> Batchnorm -> Activation
            ConvBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type=activation_type,
            ),
            # Conv -> Batchnorm
            ConvBlock3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                add_activation=False,
            ),
        ]

        self.seq = nn.Sequential(*layers)
        # Conv -> Batchnorm
        self.skip = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            add_activation=False,
        )
        self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x) + self.skip(x)

        return self.final_act(x)


class SpatioTemporalConv3d(nn.Module):
    """A spatio-temporal convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        activation_type: str = "SiLU",
    ):
        super(SpatioTemporalConv3d, self).__init__()

        layers = [
            # Conv -> Batchnorm -> Activation
            ConvBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type=activation_type,
            ),
        ]
        if num_layers > 1:
            for _ in range(1, num_layers):
                # Conv -> Batchnorm -> Activation
                layers += [
                    ConvBlock3d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=2,
                        dilation=2,
                        activation_type=activation_type,
                    )
                ]

        self.skip = nn.Sequential(
            Rearrange('b c t h w -> b t h w c'),
            nn.Linear(in_channels, out_channels),
            Rearrange('b t h w c -> b c t h w'),
        )
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.skip(x)


class DoubleConv(nn.Module):
    """A double convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(DoubleConv, self).__init__()

        layers = []

        init_channels = in_channels
        if init_point_conv:
            layers += [
                ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    activation_type=activation_type,
                )
            ]
            init_channels = out_channels

        layers += [
            ConvBlock2d(
                in_channels=init_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type=activation_type,
            ),
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=double_dilation,
                dilation=double_dilation,
                activation_type=activation_type,
            ),
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(ResBlock2d, self).__init__()

        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            SetActivation(activation_type, channels=in_channels, dims=2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AtrousPyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_b: int = 2,
        dilation_c: int = 3,
        dilation_d: int = 4,
    ):
        super(AtrousPyramidPooling, self).__init__()

        self.up = UpSample()

        self.pool_a = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_b = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_c = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_d = nn.AdaptiveAvgPool2d((8, 8))

        self.conv_a = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            add_activation=False,
        )
        self.conv_b = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation_b,
            dilation=dilation_b,
            add_activation=False,
        )
        self.conv_c = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation_c,
            dilation=dilation_c,
            add_activation=False,
        )
        self.conv_d = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation_d,
            dilation=dilation_d,
            add_activation=False,
        )
        self.final = ConvBlock2d(
            in_channels=int(in_channels * 4) + int(out_channels * 4),
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_pa = self.up(self.pool_a(x), size=x.shape[-2:], mode="bilinear")
        out_pb = self.up(self.pool_b(x), size=x.shape[-2:], mode="bilinear")
        out_pc = self.up(self.pool_c(x), size=x.shape[-2:], mode="bilinear")
        out_pd = self.up(self.pool_d(x), size=x.shape[-2:], mode="bilinear")
        out_ca = self.conv_a(x)
        out_cb = self.conv_b(x)
        out_cc = self.conv_c(x)
        out_cd = self.conv_d(x)
        out = torch.cat(
            [out_pa, out_pb, out_pc, out_pd, out_ca, out_cb, out_cc, out_cd],
            dim=1,
        )
        out = self.final(out)

        return out


class PoolConvSingle(nn.Module):
    """Max pooling followed by convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, pool_size: int = 2
    ):
        super(PoolConvSingle, self).__init__()

        self.seq = nn.Sequential(
            nn.MaxPool2d(pool_size),
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(nn.Module):
    """Max pooling with (optional) dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
        dropout: T.Optional[float] = None,
    ):
        super(PoolConv, self).__init__()

        layers = [nn.MaxPool2d(pool_size)]
        if dropout is not None:
            layers += [nn.Dropout(dropout)]
        layers += [
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                init_point_conv=init_point_conv,
                double_dilation=double_dilation,
                activation_type=activation_type,
            )
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResidualConvInit(nn.Module):
    """A residual convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = "SiLU",
    ):
        super(ResidualConvInit, self).__init__()

        self.seq = nn.Sequential(
            # Conv -> Batchnorm -> Activation
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type=activation_type,
            ),
            # Conv -> Batchnorm
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                add_activation=False,
            ),
        )
        # Conv -> Batchnorm
        self.skip = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            add_activation=False,
        )
        self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x) + self.skip(x)

        return self.final_act(x)


class ResConvLayer(nn.Module):
    """Convolution layer designed for a residual activation.

    if num_blocks [Conv2d-BatchNorm-Activation -> Conv2dAtrous-BatchNorm]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilations: T.List[int] = None,
        activation_type: str = "SiLU",
        num_blocks: int = 1,
    ):
        super(ResConvLayer, self).__init__()

        assert num_blocks > 0, "There must be at least one block."

        if dilations is None:
            dilations = list(range(1, num_blocks + 1))

        # Block 1
        layers = [
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0 if kernel_size == 1 else dilations[0],
                dilation=dilations[0],
                activation_type=activation_type,
                add_activation=True,
            )
        ]

        if num_blocks > 1:
            # Blocks 2:N-1
            layers += [
                ConvBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=0 if kernel_size == 1 else dilations[blk_idx],
                    dilation=dilations[blk_idx],
                    activation_type=activation_type,
                    add_activation=True,
                )
                for blk_idx in range(1, num_blocks)
            ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
    ):
        super(ResidualConv, self).__init__()

        self.attention_weights = attention_weights

        if self.attention_weights is not None:
            assert self.attention_weights in [
                AttentionTypes.FRACTAL,
                AttentionTypes.SPATIAL_CHANNEL,
            ], "The attention method is not supported."

            self.gamma = nn.Parameter(torch.ones(1))

            if self.attention_weights == AttentionTypes.FRACTAL:
                self.attention_conv = FractalAttention(
                    in_channels=in_channels, out_channels=out_channels
                )
            elif self.attention_weights == AttentionTypes.SPATIAL_CHANNEL:
                self.attention_conv = SpatialChannelAttention(
                    out_channels=out_channels, activation_type=activation_type
                )

        # Ends with Conv2d -> BatchNorm2d
        self.seq = ResConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            activation_type=activation_type,
        )

        self.skip = None
        if in_channels != out_channels:
            # Conv2d -> BatchNorm2d
            self.skip = ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=False,
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
            # Get the attention weights
            if self.attention_weights == AttentionTypes.SPATIAL_CHANNEL:
                # Get weights from the residual
                attention = self.attention_conv(out)
            elif self.attention_weights == AttentionTypes.FRACTAL:
                # Get weights from the input
                attention = self.attention_conv(x)

            # 1 + γA
            attention = 1.0 + self.gamma * attention
            out = out * attention

            out = self.final_act(out)

        return out


class ResidualAConv(nn.Module):
    r"""Residual convolution with atrous/dilated convolutions.

    Adapted from publication below:

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

    Modules:
        module1: [Conv2dAtrous-BatchNorm]
        ...
        moduleN: [Conv2dAtrous-BatchNorm]

    Dilation sum:
        sum = [module1 + module2 + ... + moduleN]
        out = sum + skip

    Attention:
        out = out * attention
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilations: T.List[int] = None,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
    ):
        super(ResidualAConv, self).__init__()

        self.attention_weights = attention_weights

        if self.attention_weights is not None:
            assert self.attention_weights in [
                AttentionTypes.FRACTAL,
                AttentionTypes.SPATIAL_CHANNEL,
            ], "The attention method is not supported."

            self.gamma = nn.Parameter(torch.ones(1))

            if self.attention_weights == AttentionTypes.FRACTAL:
                self.attention_conv = FractalAttention(
                    in_channels=in_channels, out_channels=out_channels
                )
            elif self.attention_weights == AttentionTypes.SPATIAL_CHANNEL:
                self.attention_conv = SpatialChannelAttention(
                    out_channels=out_channels, activation_type=activation_type
                )

        self.res_modules = nn.ModuleList(
            [
                # Conv2dAtrous -> Batchnorm
                ResConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilations=[dilation] * 2,
                    activation_type=activation_type,
                    num_blocks=2,
                )
                for dilation in dilations
            ]
        )

        self.skip = None
        if in_channels != out_channels:
            # Conv2d -> BatchNorm2d
            self.skip = ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=False,
            )

        if self.attention_weights is not None:
            self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip is not None:
            # Align channels
            out = self.skip(x)
        else:
            out = x

        for seq in self.res_modules:
            out = out + seq(x)

        if self.attention_weights is not None:
            # Get the attention weights
            if self.attention_weights == AttentionTypes.SPATIAL_CHANNEL:
                # Get weights from the residual
                attention = self.attention_conv(out)
            elif self.attention_weights == AttentionTypes.FRACTAL:
                # Get weights from the input
                attention = self.attention_conv(x)

            # 1 + γA
            attention = 1.0 + self.gamma * attention
            out = out * attention

            out = self.final_act(out)

        return out


class PoolResidualConv(nn.Module):
    """Max pooling followed by a residual convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
        num_blocks: int = 2,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RES,
        dilations: T.Sequence[int] = None,
        pool_first: bool = False,
    ):
        super(PoolResidualConv, self).__init__()

        assert res_block_type in (
            ResBlockTypes.RES,
            ResBlockTypes.RESA,
        )

        self.pool_first = pool_first

        if res_block_type == ResBlockTypes.RES:
            self.conv = ResidualConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                attention_weights=attention_weights,
                num_blocks=num_blocks,
                activation_type=activation_type,
            )
        else:
            self.conv = ResidualAConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilations=dilations,
                attention_weights=attention_weights,
                activation_type=activation_type,
            )

        self.dropout_layer = None
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        if self.pool_first:
            # Max pooling
            x = F.adaptive_max_pool2d(x, output_size=(height // 2, width // 2))

        # Apply convolutions
        x = self.conv(x)

        if not self.pool_first:
            x = F.adaptive_max_pool2d(x, output_size=(height // 2, width // 2))

        # Optional dropout
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        return x


class SingleConv3d(nn.Module):
    """A single convolution layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConv3d, self).__init__()

        self.seq = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class SingleConv(nn.Module):
    """A single convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = "SiLU",
    ):
        super(SingleConv, self).__init__()

        self.seq = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation_type=activation_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TemporalConv(nn.Module):
    """A temporal convolution layer."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ):
        super(TemporalConv, self).__init__()

        layers = [
            ConvBlock3d(
                in_channels=in_channels,
                in_time=0,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBlock3d(
                in_channels=hidden_channels,
                in_time=0,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            ConvBlock3d(
                in_channels=hidden_channels,
                in_time=0,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class FinalConv2dDropout(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim_factor: int,
        activation_type: str,
        final_activation: T.Callable,
        num_classes: int,
    ):
        super(FinalConv2dDropout, self).__init__()

        self.net = nn.Sequential(
            ResidualConv(
                in_channels=int(hidden_dim * dim_factor),
                out_channels=hidden_dim,
                activation_type=activation_type,
            ),
            nn.Dropout(0.1),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=num_classes,
                kernel_size=1,
                padding=0,
            ),
            final_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
