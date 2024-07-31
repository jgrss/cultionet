from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSoftmax(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim, dtype=x.dtype)


class Softmax(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim, dtype=x.dtype)


class Swish(nn.Module):
    def __init__(self, channels: int, dims: int):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.beta * x)

    def reset_parameters(self):
        nn.init.ones_(self.beta)


class SetActivation(nn.Module):
    def __init__(
        self,
        activation_type: str,
        channels: Optional[int] = None,
        dims: Optional[int] = None,
    ):
        """
        Examples:
            >>> act = SetActivation('ReLU')
            >>> act(x)
            >>>
            >>> act = SetActivation('SiLU')
            >>> act(x)
            >>>
            >>> act = SetActivation('Swish', channels=32)
            >>> act(x)
        """
        super().__init__()

        if activation_type == "Swish":
            assert isinstance(
                channels, int
            ), "Swish requires the input channels."
            assert isinstance(
                dims, int
            ), "Swish requires the tensor dimension."
            self.activation = Swish(channels=channels, dims=dims)
        else:
            self.activation = getattr(torch.nn, activation_type)(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class SigmoidCrisp(nn.Module):
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
        super().__init__()

        self.smooth = smooth
        self.gamma = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.smooth + self.sigmoid(self.gamma)
        out = torch.reciprocal(out)
        out = x * out
        out = self.sigmoid(out)

        return out
