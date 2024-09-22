import typing as T

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import SetActivation
from .reshape import UpSample


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
        super().__init__()

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


class AttentionAdd(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = UpSample()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != g.shape[-2:]:
            x = self.up(x, size=g.shape[-2:], mode="bilinear")

        return x + g


class AttentionGate(nn.Module):
    def __init__(self, high_channels: int, low_channels: int):
        super().__init__()

        conv_x = nn.Conv2d(
            high_channels, high_channels, kernel_size=1, padding=0
        )
        conv_g = nn.Conv2d(
            low_channels,
            high_channels,
            kernel_size=1,
            padding=0,
        )
        conv1d = nn.Conv2d(high_channels, 1, kernel_size=1, padding=0)
        self.up = UpSample()

        self.seq = nn.Sequential(
            "x, g",
            [
                (conv_x, "x -> x"),
                (conv_g, "g -> g"),
                (AttentionAdd(), "x, g -> x"),
                (SetActivation("SiLU"), 'x -> x'),
                (conv1d, "x -> x"),
                (nn.Sigmoid(), "x -> x"),
            ],
        )
        self.final = ConvBlock2d(
            in_channels=high_channels,
            out_channels=high_channels,
            kernel_size=1,
            add_activation=False,
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Higher dimension
            g: Lower dimension
        """
        h = self.seq(x, g)
        if h.shape[-2:] != x.shape[-2:]:
            h = self.up(h, size=x.shape[-2:], mode="bilinear")

        return self.final(x * h)


class FractalTanimoto(nn.Module):
    """Average Tanimoto.

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
            https://github.com/waldnerf/decode/blob/bed6f6b4173f49362a5113207155cc103e8fd139/FracTAL_ResUNet/nn/layers/ftnmt.py
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        depth: int = 5,
        dim: T.Optional[T.Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.smooth = smooth
        self.depth = depth
        self.dim = dim

        if self.dim is None:
            self.dim = (1, 2, 3)

    def tanimoto_distance(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / self.depth

        tpl = y * yhat
        sq_sum = y**2 + yhat**2

        tpl = tpl.sum(dim=self.dim, keepdims=True)
        sq_sum = sq_sum.sum(dim=self.dim, keepdims=True)

        denominator = 0.0
        for d in range(0, self.depth):
            a = 2.0**d
            b = -(2.0 * a - 1.0)
            denominator = denominator + torch.reciprocal(
                ((a * sq_sum) + (b * tpl)) + self.smooth
            )

        numerator = tpl + self.smooth

        distance = (numerator * denominator) * scale

        return distance

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model, shaped (B, C, H, W).
            targets: Ground truth values, shaped (B, C, H, W).

        Returns:
            Tanimoto distance
        """

        c1 = self.tanimoto_distance(targets, inputs)
        c2 = self.tanimoto_distance(1.0 - targets, 1.0 - inputs)
        c_mean = (c1 + c2) * 0.5

        return c_mean


class FractalAttention(nn.Module):
    r"""Fractal Tanimoto Attention Layer (FracTAL)

    Adapted from publication and source code below:

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
            @article{diakogiannis_etal_2021,
                title={Looking for change? Roll the dice and demand attention},
                author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter},
                journal={Remote Sensing},
                volume={13},
                number={18},
                pages={3707},
                year={2021},
                publisher={MDPI},
                url={https://www.mdpi.com/2072-4292/13/18/3707},
            }

            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/units/fractal_resnet.py
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/layers/attention.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        smooth: float = 1e-5,
        depth: int = 5,
    ):
        super().__init__()

        self.query = nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            nn.Sigmoid(),
        )
        self.key = nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            nn.Sigmoid(),
        )
        self.value = nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            nn.Sigmoid(),
        )

        self.spatial_sim = FractalTanimoto(smooth=smooth, depth=depth, dim=1)
        self.channel_sim = FractalTanimoto(
            smooth=smooth, depth=depth, dim=(2, 3)
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_spatial = self.spatial_sim(q, k)
        v_spatial = attention_spatial * v

        attention_channel = self.channel_sim(q, k)
        v_channel = attention_channel * v

        attention = (v_spatial + v_channel) * 0.5
        attention = self.norm(attention)

        return attention


class ChannelAttention(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation_type: str
    ):
        super().__init__()

        # Channel attention
        self.channel_adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_adaptive_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            SetActivation(activation_type=activation_type),
            nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            SetActivation(activation_type=activation_type),
            nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        avg_attention = self.fc1(self.channel_adaptive_avg(x))
        max_attention = self.fc2(self.channel_adaptive_max(x))
        attention = avg_attention + max_attention
        attention = F.sigmoid(attention)

        return attention.expand(-1, -1, height, width)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        avg_attention = einops.reduce(x, 'b c h w -> b 1 h w', 'mean')
        max_attention = einops.reduce(x, 'b c h w -> b 1 h w', 'max')
        attention = torch.cat([avg_attention, max_attention], dim=1)
        attention = self.conv(attention)
        attention = F.sigmoid(attention)

        return attention.expand(-1, -1, height, width)


class SpatialChannelAttention(nn.Module):
    """Spatial-Channel Attention Block.

    References:
        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
        https://www.mdpi.com/2072-4292/14/9/2253
        https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """

    def __init__(
        self, in_channels: int, out_channels: int, activation_type: str
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            activation_type=activation_type,
        )
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        attention = (channel_attention + spatial_attention) * 0.5

        return attention
