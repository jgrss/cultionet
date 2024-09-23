import torch
import torch.nn as nn


class SetActivation(nn.Module):
    """
    Examples:
        >>> act = SetActivation('ReLU')
        >>> act(x)
        >>>
        >>> act = SetActivation('SiLU')
        >>> act(x)
    """

    def __init__(self, activation_type: str):
        super().__init__()

        try:
            self.activation = getattr(torch.nn, activation_type)(inplace=False)
        except TypeError:
            self.activation = getattr(torch.nn, activation_type)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)
