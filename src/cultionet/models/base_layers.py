import typing as T

from . import model_utils

import torch
from torch_geometric import nn


class Permute(torch.nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class ConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU'
    ):
        super(ConvBlock2d, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        ]
        if add_activation:
            layers += [
                getattr(torch.nn, activation_type)(inplace=False)
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1
    ):
        super(ResBlock2d, self).__init__()

        layers = [
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        squeeze: bool = True
    ):
        super(ConvBlock3d, self).__init__()

        self.seq = [
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )
        ]
        if squeeze:
            self.seq += [
                Squeeze(),
                torch.nn.BatchNorm2d(in_time)
            ]
        else:
            self.seq += [torch.nn.BatchNorm3d(out_channels)]
        if add_activation:
            self.seq += [torch.nn.LeakyReLU(inplace=False)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AttentionAdd(torch.nn.Module):
    def __init__(self):
        super(AttentionAdd, self).__init__()

        self.up = model_utils.UpSample()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != g.shape[-2:]:
            x = self.up(x, size=g.shape[-2:], mode='bilinear')

        return x + g


class AttentionGate(torch.nn.Module):
    def __init__(
        self,
        high_channels: int,
        low_channels: int
    ):
        super(AttentionGate, self).__init__()

        conv_x = torch.nn.Conv2d(
            high_channels,
            high_channels,
            kernel_size=1,
            padding=0
        )
        conv_g = torch.nn.Conv2d(
            low_channels,
            high_channels,
            kernel_size=1,
            padding=0,
        )
        conv1d = torch.nn.Conv2d(
            high_channels,
            1,
            kernel_size=1,
            padding=0
        )
        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'x, g',
            [
                (conv_x, 'x -> x'),
                (conv_g, 'g -> g'),
                (AttentionAdd(), 'x, g -> x'),
                (torch.nn.LeakyReLU(inplace=False), 'x -> x'),
                (conv1d, 'x -> x'),
                (torch.nn.Sigmoid(), 'x -> x')
            ]
        )
        self.final = ConvBlock2d(
            in_channels=high_channels,
            out_channels=high_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Higher dimension
            g: Lower dimension
        """
        h = self.seq(x, g)
        if h.shape[-2:] != x.shape[-2:]:
            h = self.up(h, size=x.shape[-2:], mode='bilinear')

        return self.final(x * h)


class TanimotoComplement(torch.nn.Module):
    """Tanimoto distance with complement

    Adapted from publications and source code below:

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

        References:
            https://www.mdpi.com/2072-4292/14/22/5738
            https://arxiv.org/abs/2009.02062
            https://github.com/waldnerf/decode/blob/main/FracTAL_ResUNet/nn/layers/ftnmt.py
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        depth: int = 5,
        dim: T.Union[int, T.Sequence[int]] = 0,
        targets_are_labels: bool = True
    ):
        super(TanimotoComplement, self).__init__()

        self.smooth = smooth
        self.depth = depth
        self.dim = dim
        self.targets_are_labels = targets_are_labels

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        if self.depth == 1:
            scale = 1.0
        else:
            scale = 1.0 / self.depth

        def tanimoto(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
            tpl = torch.sum(y * yhat, dim=self.dim, keepdim=True)
            numerator = tpl + self.smooth
            sq_sum = torch.sum(y**2 + yhat**2, dim=self.dim, keepdim=True)
            denominator = torch.zeros(1, dtype=inputs.dtype).to(device=inputs.device)
            for d in range(0, self.depth):
                a = 2**d
                b = -(2.0 * a - 1.0)
                denominator = denominator + torch.reciprocal((a * sq_sum) + (b * tpl) + self.smooth)

            return numerator * denominator * scale

        l1 = tanimoto(targets, inputs)
        l2 = tanimoto(1.0 - targets, 1.0 - inputs)
        score = (l1 + l2) * 0.5

        return score


class FractalAttention(torch.nn.Module):
    """Fractal Tanimoto Attention Layer (FracTAL)

    Adapted from publications and source code below:

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

        Reference:
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/units/fractal_resnet.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 5
    ):
        super(FractalAttention, self).__init__()

        self.query = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )
        self.key = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )
        self.values = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False
            ),
            torch.nn.Sigmoid()
        )

        self.spatial_sim = TanimotoComplement(depth=depth, dim=1)
        self.channel_sim = TanimotoComplement(depth=depth, dim=[2, 3])
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        attention_spatial = self.spatial_sim(q, k)
        v_spatial = attention_spatial * v

        attention_channel = self.channel_sim(q, k)
        v_channel = attention_channel * v

        v_channel_spatial = (v_spatial + v_channel) * 0.5
        v_channel_spatial = self.norm(v_channel_spatial)

        return v_channel_spatial


class ChannelAttention(torch.nn.Module):
    """Residual Channel Attention Block

    References:
        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
    """
    def __init__(self, channels: int):
        super(ChannelAttention, self).__init__()

        self.module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.module(x)


class DoubleConv(torch.nn.Module):
    """A double convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(DoubleConv, self).__init__()

        self.seq = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConvSingle(torch.nn.Module):
    """Max pooling followed by convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2
    ):
        super(PoolConvSingle, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(pool_size),
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling with (optional) dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None
    ):
        super(PoolConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]
        if dropout is not None:
            layers += [torch.nn.Dropout(dropout)]
        layers += [DoubleConv(in_channels, out_channels)]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResidualConv(torch.nn.Module):
    """A residual convolution layer with (optional) attention
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_conv: bool = False,
        fractal_attention: bool = False,
        channel_attention: bool = False,
        dilations: T.List[int] = None
    ):
        super(ResidualConv, self).__init__()

        assert not all([fractal_attention, channel_attention]), \
            'Only one attention method should be used.'

        init_in_channels = in_channels

        layers = []
        if init_conv:
            layers = [
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ]
            in_channels = out_channels

        layers += [
            ResBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        ]
        if dilations is not None:
            for dilation in dilations:
                layers += [
                    ResBlock2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation
                    )
                ]

        self.fractal_weights = None
        if fractal_attention:
            self.fractal_weights = FractalAttention(
                in_channels=init_in_channels,
                out_channels=out_channels
            )
            self.gamma = torch.nn.Parameter(torch.ones(1))
        if channel_attention:
            layers += [ChannelAttention(channels=out_channels)]

        self.seq = torch.nn.Sequential(*layers)
        self.skip = ConvBlock2d(
            in_channels=init_in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x) + self.skip(x)

        if self.fractal_weights is not None:
            # Fractal attention
            attention= self.fractal_weights(x)
            # 1 + γA
            attention = attention * self.gamma + 1.0
            out *= attention

        return out


class ResidualConvRCAB(torch.nn.Module):
    """A group of residual convolution layers with (optional) RCAB
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_attention: bool = False,
        res_blocks: int = 2,
        dilations: T.List[int] = None
    ):
        super(ResidualConvRCAB, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        for __ in range(0, res_blocks):
            layers += [
                ResidualConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    channel_attention=channel_attention,
                    dilations=dilations
                )
            ]
        layers += [
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            )
        ]

        self.seq = torch.nn.Sequential(*layers)
        self.expand = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            add_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + self.expand(x)


class PoolResidualConv(torch.nn.Module):
    """Max pooling followed by a residual convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None,
        dilations: T.List[int] = None,
        fractal_attention: bool = False,
        channel_attention: bool = False,
        res_blocks: int = 0
    ):
        super(PoolResidualConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]

        if dropout is not None:
            assert isinstance(dropout, float), 'The dropout arg must be a float.'
            layers += [torch.nn.Dropout(dropout)]

        if res_blocks > 0:
            layers += [
                ResidualConvRCAB(
                    in_channels,
                    out_channels,
                    fractal_attention=fractal_attention,
                    channel_attention=channel_attention,
                    res_blocks=res_blocks
                ),
                torch.nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ]

        else:
            layers += [
                ResidualConv(
                    in_channels,
                    out_channels,
                    fractal_attention=fractal_attention,
                    channel_attention=channel_attention,
                    dilations=dilations
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class SingleConv(torch.nn.Module):
    """A single convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super(SingleConv, self).__init__()

        self.seq = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TemporalConv(torch.nn.Module):
    """A temporal convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        out_channels: int
    ):
        super(TemporalConv, self).__init__()

        layers = [
            ConvBlock3d(
                in_channels=in_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                squeeze=False
            ),
            ConvBlock3d(
                in_channels=out_channels,
                in_time=in_time,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                squeeze=True
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
