from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
