import typing as T
import enum

import torch
import torch.nn.functional as F
from torch_geometric import nn

from . import model_utils
from .enums import ResBlockTypes


class Swish(torch.nn.Module):
    def __init__(self, channels: int, dims: int):
        super(Swish, self).__init__()

        self.sigmoid = torch.nn.Sigmoid()
        self.beta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.beta * x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.beta)


class SetActivation(torch.nn.Module):
    def __init__(
        self,
        activation_type: str,
        channels: T.Optional[int] = None,
        dims: T.Optional[int] = None,
    ):
        """
        Examples:
            >>> act = SetActivation('ReLU')
            >>> act(x)
            >>>
            >>> act = SetActivation('LeakyReLU')
            >>> act(x)
            >>>
            >>> act = SetActivation('Swish', channels=32)
            >>> act(x)
        """
        super(SetActivation, self).__init__()

        if activation_type == 'Swish':
            assert isinstance(
                channels, int
            ), 'Swish requires the input channels.'
            assert isinstance(
                dims, int
            ), 'Swish requires the tensor dimension.'
            self.activation = Swish(channels=channels, dims=dims)
        else:
            self.activation = getattr(torch.nn, activation_type)(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super(LogSoftmax, self).__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim, dtype=x.dtype)


class Softmax(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super(Softmax, self).__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim, dtype=x.dtype)


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


class Min(torch.nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Min, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.min(dim=self.dim, keepdim=self.keepdim)[0]


class Max(torch.nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Max, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]


class Mean(torch.nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Mean, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class Var(torch.nn.Module):
    def __init__(
        self, dim: int, keepdim: bool = False, unbiased: bool = False
    ):
        super(Var, self).__init__()

        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.var(
            dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased
        )


class Std(torch.nn.Module):
    def __init__(
        self, dim: int, keepdim: bool = False, unbiased: bool = False
    ):
        super(Std, self).__init__()

        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.std(
            dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased
        )


class Squeeze(torch.nn.Module):
    def __init__(self, dim: T.Optional[int] = None):
        super(Squeeze, self).__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int):
        super(Unsqueeze, self).__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class SigmoidCrisp(torch.nn.Module):
    r"""Sigmoid crisp.

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

        Citation:
            @article{diakogiannis_etal_2021,
                title={Looking for change? Roll the dice and demand attention},
                author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter},
                journal={Remote Sensing},
                volume={13},
                number={18},
                pages={3707},
                year={2021},
                publisher={MDPI}
            }

        Reference:
            https://www.mdpi.com/2072-4292/13/18/3707
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/main/FracTAL_ResUNet/nn/activations/sigmoid_crisp.py
    """

    def __init__(self, smooth: float = 1e-2):
        super(SigmoidCrisp, self).__init__()

        self.smooth = smooth
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.smooth + self.sigmoid(self.gamma)
        out = torch.reciprocal(out)
        out = x * out
        out = self.sigmoid(out)

        return out


class ConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU',
    ):
        super(ConvBlock2d, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
        ]
        if add_activation:
            layers += [
                SetActivation(activation_type, channels=out_channels, dims=2)
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
        dilation: int = 1,
        activation_type: str = 'LeakyReLU',
    ):
        super(ResBlock2d, self).__init__()

        layers = [
            torch.nn.BatchNorm2d(in_channels),
            SetActivation(activation_type, channels=in_channels, dims=2),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
        ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock3d(torch.nn.Module):
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
        activation_type: str = 'LeakyReLU',
    ):
        super(ConvBlock3d, self).__init__()

        layers = [
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            )
        ]
        if squeeze:
            layers += [Squeeze(), torch.nn.BatchNorm2d(in_time)]
            dims = 2
        else:
            layers += [torch.nn.BatchNorm3d(out_channels)]
            dims = 3
        if add_activation:
            layers += [
                SetActivation(
                    activation_type, channels=out_channels, dims=dims
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

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


class AttentionGate3d(torch.nn.Module):
    def __init__(self, high_channels: int, low_channels: int):
        super(AttentionGate3d, self).__init__()

        conv_x = torch.nn.Conv3d(
            high_channels, high_channels, kernel_size=1, padding=0
        )
        conv_g = torch.nn.Conv3d(
            low_channels,
            high_channels,
            kernel_size=1,
            padding=0,
        )
        conv1d = torch.nn.Conv3d(high_channels, 1, kernel_size=1, padding=0)
        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'x, g',
            [
                (conv_x, 'x -> x'),
                (conv_g, 'g -> g'),
                (AttentionAdd(), 'x, g -> x'),
                (torch.nn.LeakyReLU(inplace=False), 'x -> x'),
                (conv1d, 'x -> x'),
                (torch.nn.Sigmoid(), 'x -> x'),
            ],
        )
        self.final = ConvBlock3d(
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
            h = self.up(h, size=x.shape[-2:], mode='bilinear')

        return self.final(x * h)


class AttentionGate(torch.nn.Module):
    def __init__(self, high_channels: int, low_channels: int):
        super(AttentionGate, self).__init__()

        conv_x = torch.nn.Conv2d(
            high_channels, high_channels, kernel_size=1, padding=0
        )
        conv_g = torch.nn.Conv2d(
            low_channels,
            high_channels,
            kernel_size=1,
            padding=0,
        )
        conv1d = torch.nn.Conv2d(high_channels, 1, kernel_size=1, padding=0)
        self.up = model_utils.UpSample()

        self.seq = nn.Sequential(
            'x, g',
            [
                (conv_x, 'x -> x'),
                (conv_g, 'g -> g'),
                (AttentionAdd(), 'x, g -> x'),
                (torch.nn.LeakyReLU(inplace=False), 'x -> x'),
                (conv1d, 'x -> x'),
                (torch.nn.Sigmoid(), 'x -> x'),
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
            h = self.up(h, size=x.shape[-2:], mode='bilinear')

        return self.final(x * h)


class TanimotoComplement(torch.nn.Module):
    """Tanimoto distance with complement.

    THIS IS NOT CURRENTLY USED ANYWHERE IN THIS REPOSITORY

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
        targets_are_labels: bool = True,
    ):
        super(TanimotoComplement, self).__init__()

        self.smooth = smooth
        self.depth = depth
        self.dim = dim
        self.targets_are_labels = targets_are_labels

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

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
            denominator = torch.zeros(1, dtype=inputs.dtype).to(
                device=inputs.device
            )
            for d in range(0, self.depth):
                a = 2**d
                b = -(2.0 * a - 1.0)
                denominator = denominator + torch.reciprocal(
                    (a * sq_sum) + (b * tpl) + self.smooth
                )

            return numerator * denominator * scale

        l1 = tanimoto(targets, inputs)
        l2 = tanimoto(1.0 - targets, 1.0 - inputs)
        score = (l1 + l2) * 0.5

        return score


class TanimotoDist(torch.nn.Module):
    r"""Tanimoto distance.

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

        Citation:
            @article{diakogiannis_etal_2021,
                title={Looking for change? Roll the dice and demand attention},
                author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter},
                journal={Remote Sensing},
                volume={13},
                number={18},
                pages={3707},
                year={2021},
                publisher={MDPI}
            }

        References:
            https://www.mdpi.com/2072-4292/13/18/3707
            https://arxiv.org/abs/2009.02062
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/layers/ftnmt.py

    Adapted from source code below:

        MIT License

        Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Matic Lubej, Grega Milčinski (Sinergise)
        Copyright (c) 2017-2020 Devis Peressutti, Jernej Puc, Anže Zupanc, Lojze Žust, Jovan Višnjić (Sinergise)

        Reference:
            https://github.com/sentinel-hub/eo-flow/blob/master/eoflow/models/losses.py
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        weight: T.Optional[torch.Tensor] = None,
        dim: T.Union[int, T.Sequence[int]] = 0,
    ):
        super(TanimotoDist, self).__init__()

        self.smooth = smooth
        self.weight = weight
        self.dim = dim

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """

        def _tanimoto(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            tpl = torch.sum(yhat * y, dim=self.dim, keepdim=True)
            sq_sum = torch.sum(yhat**2 + y**2, dim=self.dim, keepdim=True)
            numerator = tpl + self.smooth
            denominator = (sq_sum - tpl) + self.smooth
            tanimoto_score = numerator / denominator

            return tanimoto_score

        score = _tanimoto(inputs, targets)
        compl_score = _tanimoto(1.0 - inputs, 1.0 - targets)
        score = (score + compl_score) * 0.5

        return score


class FractalAttention(torch.nn.Module):
    """Fractal Tanimoto Attention Layer (FracTAL)

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
            https://www.mdpi.com/2072-4292/13/18/3707
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/units/fractal_resnet.py
            https://github.com/waldnerf/decode/blob/9e922a2082e570e248eaee10f7a1f2f0bd852b42/FracTAL_ResUNet/nn/layers/attention.py
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(FractalAttention, self).__init__()

        self.query = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            torch.nn.Sigmoid(),
        )
        self.key = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            torch.nn.Sigmoid(),
        )
        self.value = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                add_activation=False,
            ),
            torch.nn.Sigmoid(),
        )

        self.spatial_sim = TanimotoDist(dim=1)
        self.channel_sim = TanimotoDist(dim=[2, 3])
        self.norm = torch.nn.BatchNorm2d(out_channels)

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


class ChannelAttention(torch.nn.Module):
    def __init__(self, out_channels: int, activation_type: str):
        super(ChannelAttention, self).__init__()

        # Channel attention
        self.channel_adaptive_avg = torch.nn.AdaptiveAvgPool2d(1)
        self.channel_adaptive_max = torch.nn.AdaptiveMaxPool2d(1)
        self.sigmoid = torch.nn.Sigmoid()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=int(out_channels / 2),
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            SetActivation(activation_type=activation_type),
            torch.nn.Conv2d(
                in_channels=int(out_channels / 2),
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attention = self.seq(self.channel_adaptive_avg(x))
        max_attention = self.seq(self.channel_adaptive_max(x))
        attention = avg_attention + max_attention
        attention = self.sigmoid(attention)

        return attention.expand_as(x)


class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False
        )
        self.channel_mean = Mean(dim=1, keepdim=True)
        self.channel_max = Max(dim=1, keepdim=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_attention = self.channel_mean(x)
        max_attention = self.channel_max(x)
        attention = torch.cat([avg_attention, max_attention], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return attention.expand_as(x)


class SpatialChannelAttention(torch.nn.Module):
    """Spatial-Channel Attention Block.

    References:
        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
        https://www.mdpi.com/2072-4292/14/9/2253
        https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """

    def __init__(self, out_channels: int, activation_type: str):
        super(SpatialChannelAttention, self).__init__()

        self.channel_attention = ChannelAttention(
            out_channels=out_channels, activation_type=activation_type
        )
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        attention = (channel_attention + spatial_attention) * 0.5

        return attention


class ResSpatioTemporalConv3d(torch.nn.Module):
    """A spatio-temporal convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = 'LeakyReLU',
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

        self.seq = torch.nn.Sequential(*layers)
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


class SpatioTemporalConv3d(torch.nn.Module):
    """A spatio-temporal convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = 'LeakyReLU',
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
            # Conv -> Batchnorm
            ConvBlock3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                activation_type=activation_type,
            ),
        ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DoubleConv(torch.nn.Module):
    """A double convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = 'LeakyReLU',
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

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class AtrousPyramidPooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_b: int = 2,
        dilation_c: int = 3,
        dilation_d: int = 4,
    ):
        super(AtrousPyramidPooling, self).__init__()

        self.up = model_utils.UpSample()

        self.pool_a = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.pool_b = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.pool_c = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.pool_d = torch.nn.AdaptiveAvgPool2d((8, 8))

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
        out_pa = self.up(self.pool_a(x), size=x.shape[-2:], mode='bilinear')
        out_pb = self.up(self.pool_b(x), size=x.shape[-2:], mode='bilinear')
        out_pc = self.up(self.pool_c(x), size=x.shape[-2:], mode='bilinear')
        out_pd = self.up(self.pool_d(x), size=x.shape[-2:], mode='bilinear')
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


class PoolConvSingle(torch.nn.Module):
    """Max pooling followed by convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, pool_size: int = 2
    ):
        super(PoolConvSingle, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(pool_size),
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class PoolConv(torch.nn.Module):
    """Max pooling with (optional) dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = 'LeakyReLU',
        dropout: T.Optional[float] = None,
    ):
        super(PoolConv, self).__init__()

        layers = [torch.nn.MaxPool2d(pool_size)]
        if dropout is not None:
            layers += [torch.nn.Dropout(dropout)]
        layers += [
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                init_point_conv=init_point_conv,
                double_dilation=double_dilation,
                activation_type=activation_type,
            )
        ]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResidualConvInit(torch.nn.Module):
    """A residual convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = 'LeakyReLU',
    ):
        super(ResidualConvInit, self).__init__()

        self.seq = torch.nn.Sequential(
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


class ResConvLayer(torch.nn.Module):
    """Convolution layer designed for a residual activation.

    if num_blocks [Conv2d-BatchNorm-Activation -> Conv2dAtrous-BatchNorm]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        activation_type: str = 'LeakyReLU',
        num_blocks: int = 2,
    ):
        super(ResConvLayer, self).__init__()

        assert num_blocks > 0

        if num_blocks == 1:
            layers = [
                ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    add_activation=False,
                )
            ]
        else:
            # Block 1
            layers = [
                ConvBlock2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    activation_type=activation_type,
                )
            ]
            if num_blocks > 2:
                # Blocks 2:N-1
                layers += [
                    ConvBlock2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        activation_type=activation_type,
                    )
                    for __ in range(num_blocks - 2)
                ]
            # Block N
            layers += [
                ConvBlock2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    add_activation=False,
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ResidualConv(torch.nn.Module):
    """A residual convolution layer with (optional) attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 2,
        attention_weights: str = None,
        activation_type: str = 'LeakyReLU',
    ):
        super(ResidualConv, self).__init__()

        self.attention_weights = attention_weights

        if self.attention_weights is not None:
            assert self.attention_weights in [
                'fractal',
                'spatial_channel',
            ], 'The attention method is not supported.'

            self.gamma = torch.nn.Parameter(torch.ones(1))

            if self.attention_weights == 'fractal':
                self.attention_conv = FractalAttention(
                    in_channels=in_channels, out_channels=out_channels
                )
            elif self.attention_weights == 'spatial_channel':
                self.attention_conv = SpatialChannelAttention(
                    out_channels=out_channels, activation_type=activation_type
                )

        # Ends with Conv2d -> BatchNorm2d
        self.seq = ResConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            activation_type=activation_type,
            num_blocks=2,
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
        self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.skip is not None:
            # Align channels
            residual = self.skip(x)
        residual = residual + self.seq(x)

        if self.attention_weights is not None:
            # Get the attention weights
            if self.attention_weights == 'spatial_channel':
                # Get weights from the residual
                attention = self.attention_conv(residual)
            elif self.attention_weights == 'fractal':
                # Get weights from the input
                attention = self.attention_conv(x)

            # 1 + γA
            attention = 1.0 + self.gamma * attention
            residual = residual * attention

        out = self.final_act(residual)

        return out


class ResidualAConv(torch.nn.Module):
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
        dilations: T.List[int] = None,
        attention_weights: str = None,
        activation_type: str = 'LeakyReLU',
    ):
        super(ResidualAConv, self).__init__()

        self.attention_weights = attention_weights

        if self.attention_weights is not None:
            assert self.attention_weights in [
                'fractal',
                'spatial_channel',
            ], 'The attention method is not supported.'

            self.gamma = torch.nn.Parameter(torch.ones(1))

            if self.attention_weights == 'fractal':
                self.attention_conv = FractalAttention(
                    in_channels=in_channels, out_channels=out_channels
                )
            elif self.attention_weights == 'spatial_channel':
                self.attention_conv = SpatialChannelAttention(
                    out_channels=out_channels, activation_type=activation_type
                )

        self.res_modules = torch.nn.ModuleList(
            [
                # Conv2dAtrous -> Batchnorm
                ResConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    activation_type=activation_type,
                    num_blocks=1,
                )
                for dilation in dilations
            ]
        )
        self.skip = None
        if in_channels != out_channels:
            # Conv2dAtrous -> BatchNorm2d
            self.skip = ConvBlock2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=False,
            )
        self.final_act = SetActivation(activation_type=activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.skip is not None:
            # Align channels
            residual = self.skip(x)

        for seq in self.res_modules:
            residual = residual + seq(x)

        if self.attention_weights is not None:
            # Get the attention weights
            if self.attention_weights == 'spatial_channel':
                # Get weights from the residual
                attention = self.attention_conv(residual)
            elif self.attention_weights == 'fractal':
                # Get weights from the input
                attention = self.attention_conv(x)

            # 1 + γA
            attention = 1.0 + self.gamma * attention
            residual = residual * attention

        out = self.final_act(residual)

        return out


class PoolResidualConv(torch.nn.Module):
    """Max pooling followed by a residual convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 2,
        dropout: T.Optional[float] = None,
        dilations: T.List[int] = None,
        attention_weights: str = None,
        activation_type: str = 'LeakyReLU',
        res_block_type: enum = ResBlockTypes.RESA,
    ):
        super(PoolResidualConv, self).__init__()

        assert res_block_type in (ResBlockTypes.RES, ResBlockTypes.RESA)

        layers = [torch.nn.MaxPool2d(pool_size)]

        if dropout is not None:
            assert isinstance(
                dropout, float
            ), 'The dropout arg must be a float.'
            layers += [torch.nn.Dropout(dropout)]

        if res_block_type == ResBlockTypes.RES:
            layers += [
                ResidualConv(
                    in_channels,
                    out_channels,
                    attention_weights=attention_weights,
                    dilation=dilations[0],
                    activation_type=activation_type,
                )
            ]
        else:
            layers += [
                ResidualAConv(
                    in_channels,
                    out_channels,
                    attention_weights=attention_weights,
                    dilations=dilations,
                    activation_type=activation_type,
                )
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class SingleConv3d(torch.nn.Module):
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


class SingleConv(torch.nn.Module):
    """A single convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: str = 'LeakyReLU',
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


class TemporalConv(torch.nn.Module):
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
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
