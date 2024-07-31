import typing as T

import torch
import torch.nn as nn


class UpSample(nn.Module):
    """Up-samples a tensor."""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, size: T.Sequence[int], mode: str = "bilinear"
    ) -> torch.Tensor:
        upsampler = nn.Upsample(size=size, mode=mode, align_corners=True)

        return upsampler(x)
