import typing as T

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from natten.functional import na2d, na2d_av, na2d_qk

from .activations import SetActivation


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, activation_type: str):
        super().__init__()

        # Channel attention
        self.channel_adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_adaptive_max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            SetActivation(activation_type=activation_type),
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            SetActivation(activation_type=activation_type),
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=in_channels,
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
        @inproceedings{woo_etal_2018,
            title={Cbam: Convolutional block attention module},
            author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
            booktitle={Proceedings of the European conference on computer vision (ECCV)},
            pages={3--19},
            year={2018},
            url={https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf},
        }

        https://arxiv.org/abs/1807.02758
        https://github.com/yjn870/RCAN-pytorch
        https://www.mdpi.com/2072-4292/14/9/2253
        https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """

    def __init__(self, in_channels: int, activation_type: str):
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            activation_type=activation_type,
        )
        self.spatial_attention = SpatialAttention()
        self.gamma = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)
        attention = (channel_attention + spatial_attention) * 0.5
        attention = 1.0 + self.gamma * attention

        return attention


class NeighborhoodAttention2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        self.query = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        self.key = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        self.value = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = einops.rearrange(q, 'b c h w -> b h w 1 c')
        k = einops.rearrange(k, 'b c h w -> b h w 1 c')
        v = einops.rearrange(v, 'b c h w -> b h w 1 c')

        output = na2d(
            q, k, v, kernel_size=self.kernel_size, dilation=self.dilation
        )

        output = einops.rearrange(output, 'b h w 1 c -> b c h w')

        return output
