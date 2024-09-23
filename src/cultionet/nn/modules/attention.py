import typing as T

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import SetActivation


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
