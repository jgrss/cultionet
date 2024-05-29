import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from cultionet.enums import AttentionTypes, ResBlockTypes

from .activations import SetActivation
from .attention import FractalAttention, SpatialChannelAttention
from .reshape import UpSample
from .utils import check_upsample


class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super(ConvTranspose2d, self).__init__()

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
        super(ConvBlock2d, self).__init__()

        layers = []

        if batchnorm_first:
            layers += [
                nn.BatchNorm2d(in_channels),
                SetActivation(activation_type, channels=in_channels, dims=2),
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
                layers += [
                    SetActivation(
                        activation_type, channels=out_channels, dims=2
                    )
                ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
        repeat_kernel: bool = False,
        batchnorm_first: bool = False,
    ):
        super(ResConvBlock2d, self).__init__()

        assert (
            0 < num_blocks < 3
        ), "There must be at least one block but no more than 3."

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                dilation=1,
                activation_type=activation_type,
                add_activation=True,
                batchnorm_first=batchnorm_first,
            )

        conv_layers = [
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=0 if kernel_size == 1 else dilation,
                dilation=1 if kernel_size == 1 else dilation,
                activation_type=activation_type,
                add_activation=True,
                batchnorm_first=batchnorm_first,
            )
        ]

        if (kernel_size > 1) and (num_blocks > 1):
            conv_layers += [
                ConvBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=max(1, dilation - 1),
                    dilation=max(1, dilation - 1),
                    activation_type=activation_type,
                    add_activation=True,
                    batchnorm_first=batchnorm_first,
                )
            ]

        self.block = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x

        x = residual + self.block(x)

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
        num_blocks: int = 2,
        repeat_kernel: bool = False,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super(ResidualAConv, self).__init__()

        self.attention_weights = attention_weights
        self.concat_resid = concat_resid

        if self.attention_weights is not None:
            assert self.attention_weights in [
                AttentionTypes.FRACTAL,
                AttentionTypes.SPATIAL_CHANNEL,
            ], "The attention method is not supported."

            if self.attention_weights == AttentionTypes.FRACTAL:
                self.attention_conv = FractalAttention(
                    in_channels=in_channels, out_channels=out_channels
                )
            elif self.attention_weights == AttentionTypes.SPATIAL_CHANNEL:
                self.attention_conv = SpatialChannelAttention(
                    out_channels=out_channels,
                    activation_type=activation_type,
                )

            self.gamma = nn.Parameter(torch.ones(1))
            self.act = SetActivation(activation_type=activation_type)

        self.res_modules = nn.ModuleList(
            [
                ResConvBlock2d(
                    in_channels=in_channels,
                    out_channels=in_channels
                    if self.concat_resid
                    else out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation_type=activation_type,
                    num_blocks=num_blocks,
                    repeat_kernel=repeat_kernel,
                    batchnorm_first=batchnorm_first,
                )
                for dilation in dilations
            ]
        )

        self.skip = None
        self.resid_connect = None
        if self.concat_resid:
            if in_channels != out_channels:
                # Follows the FishNet block
                self.resid_connect = ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    activation_type=activation_type,
                    batchnorm_first=batchnorm_first,
                )
        else:
            if in_channels != out_channels:
                self.skip = ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    add_activation=False,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x if self.skip is None else self.skip(x)

        for layer in self.res_modules:
            out = out + layer(x)

        if self.resid_connect is not None:
            out = self.resid_connect(out)

        if self.attention_weights is not None:
            attention = self.attention_conv(out)
            attention = 1.0 + self.gamma * attention
            out = self.act(out * attention)

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
        repeat_resa_kernel: bool = False,
        pool_first: bool = True,
        pool_by_max: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super(PoolResidualConv, self).__init__()

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
                repeat_kernel=repeat_resa_kernel,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
                concat_resid=concat_resid,
            )

        self.dropout_layer = None
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        if self.pool_first:
            if self.pool_by_max:
                x = F.adaptive_max_pool2d(
                    x, output_size=(height // 2, width // 2)
                )
            else:
                x = self.pool_conv(x)

        # Residual convolution
        x = self.res_conv(x)

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        return x


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
