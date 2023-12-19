"""
Source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py

TODO:
    https://www.sciencedirect.com/science/article/pii/S0893608023005361
    https://github.com/AzadDeihim/STTRE/blob/main/STTRE.ipynb
"""
from typing import Callable, Optional, Tuple, Sequence, Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from cultionet.models.base_layers import Softmax, FinalConv2dDropout
from cultionet.models.encodings import cartesian, get_sinusoid_encoding_table


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(
        self,
        scale: float,
        dropout: float = 0.1,
    ):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prev_attention: Optional[torch.Tensor] = None,
    ):
        scores = torch.einsum('hblk, hbtk -> hblt', [query, key]) * self.scale
        if prev_attention is not None:
            scores = scores + prev_attention
        attention = self.softmax(scores)
        output = torch.einsum('hblt, hbtv -> hblv', [attention, value])
        output = self.dropout(output)

        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module Modified from
    github.com/jadore801120/attention-is-all-you-need-pytorch."""

    def __init__(self, num_head: int, d_in: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_head = num_head
        d_k = d_in // num_head
        scale = 1.0 / d_k**0.5

        self.proj_query = nn.Linear(d_in, d_in, bias=False)
        self.proj_key = nn.Linear(d_in, d_in, bias=False)
        self.proj_value = nn.Linear(d_in, d_in, bias=False)

        self.scaled_attention = ScaledDotProductAttention(
            scale, dropout=dropout
        )
        self.final = nn.Sequential(
            Rearrange('head b t c -> b t (head c)'),
            nn.LayerNorm(d_in),
        )

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            x, 'b t (num_head k) -> num_head b t k', num_head=self.num_head
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prev_attention: Optional[torch.Tensor] = None,
    ):
        # batch_size, num_time, n_channels = query.shape
        residual = query
        query = self.proj_query(query)
        key = self.proj_key(key)
        value = self.proj_value(value)
        # Split heads
        query = self.split(query)
        key = self.split(key)
        value = self.split(value)

        output, attention = self.scaled_attention(
            query, key, value, prev_attention=prev_attention
        )
        output = self.final(output)
        output = output + residual

        return output, attention


class InLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InLayer, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(),
        )
        self.skip = nn.Sequential(
            Rearrange('b c t h w -> b t h w c'),
            nn.Linear(in_channels, out_channels),
            Rearrange('b t h w c -> b c t h w'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        return self.seq(x) + residual


class InBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ):
        super(InBlock, self).__init__()

        self.seq = nn.Sequential(
            InLayer(in_channels=in_channels, out_channels=hidden_channels),
            InLayer(in_channels=hidden_channels, out_channels=out_channels),
        )
        self.skip = nn.Sequential(
            Rearrange('b c t h w -> b t h w c'),
            nn.Linear(in_channels, out_channels),
            Rearrange('b t h w c -> b c t h w'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        return self.seq(x) + residual


class TemporalAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_head: int = 8,
        num_time: int = 1,
        d_model: int = 256,
        dropout: float = 0.1,
        time_scaler: int = 1_000,
        num_classes_l2: int = 2,
        num_classes_last: int = 3,
        activation_type: str = "SiLU",
        final_activation: Callable = Softmax(dim=1),
    ):
        """Transformer Self-Attention.

        Args:
            in_channels (int): Number of channels of the inputs.
            hidden_channels (int): Number of hidden layers.
            num_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            time_scaler (int): Period to use for the positional encoding.
        """
        super(TemporalAttention, self).__init__()

        self.init_conv = nn.Sequential(
            InBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=d_model,
            ),
            Rearrange('b c t h w -> (b h w) t c'),
        )

        # Absolute positional embeddings
        self.positional_encoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                positions=num_time,
                d_hid=d_model,
                time_scaler=time_scaler,
            ),
            freeze=True,
        )
        # Coordinate embeddings
        self.coordinate_encoder = nn.Linear(3, d_model)
        # self.channel_embed = nn.Embedding(
        #     num_embeddings=in_channels,
        #     embedding_dim=d_model,
        # )

        # Attention
        self.attention_a = MultiHeadAttention(
            num_head=num_head, d_in=d_model, dropout=dropout
        )
        self.attention_b = MultiHeadAttention(
            num_head=num_head, d_in=d_model, dropout=dropout
        )
        # Level 2 level (non-crop; crop)
        self.final_l2 = FinalConv2dDropout(
            hidden_dim=d_model,
            dim_factor=1,
            activation_type=activation_type,
            final_activation=final_activation,
            num_classes=num_classes_l2,
        )
        # Last level (non-crop; crop; edges)
        self.final_last = FinalConv2dDropout(
            hidden_dim=d_model,
            dim_factor=1,
            activation_type=activation_type,
            final_activation=Softmax(dim=1),
            num_classes=num_classes_last,
        )

    def forward(
        self,
        x: torch.Tensor,
        longitude: torch.Tensor,
        latitude: torch.Tensor,
    ) -> tuple:
        batch_size, num_channels, num_time, height, width = x.shape

        out = self.init_conv(x)

        # Positional embedding
        src_pos = (
            torch.arange(0, out.shape[1], dtype=torch.long)
            .expand(out.shape[0], out.shape[1])
            .to(x.device)
        )
        position_tokens = self.positional_encoder(src_pos)
        # Coordinate embedding
        coordinate_tokens = self.coordinate_encoder(
            cartesian(
                einops.rearrange(
                    torch.tile(longitude[:, None], (1, height * width)),
                    'b (h w) -> (b h w) 1',
                    b=batch_size,
                    h=height,
                    w=width,
                ),
                einops.rearrange(
                    torch.tile(latitude[:, None], (1, height * width)),
                    'b (h w) -> (b h w) 1',
                    b=batch_size,
                    h=height,
                    w=width,
                ),
            )
        )
        out = out + position_tokens + coordinate_tokens

        # Attention
        out, attention = self.attention_a(out, out, out)
        # Concatenate heads
        last_l2 = einops.rearrange(
            out, '(b h w) t c -> b c t h w', b=batch_size, h=height, w=width
        )
        last_l2 = einops.reduce(last_l2, 'b c t h w -> b c h w', 'mean')
        last_l2 = self.final_l2(last_l2)

        # Attention
        out, attention = self.attention_b(
            out, out, out, prev_attention=attention
        )
        # Concatenate heads
        out = einops.rearrange(
            out, '(b h w) t c -> b c t h w', b=batch_size, h=height, w=width
        )
        out = einops.reduce(out, 'b c t h w -> b c h w', 'mean')
        last = self.final_last(out)

        return out, last_l2, last


if __name__ == '__main__':
    batch_size = 2
    num_channels = 3
    hidden_size = 64
    num_head = 8
    d_model = 128
    num_time = 12
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, num_time, height, width),
        dtype=torch.float32,
    )
    lon = torch.distributions.uniform.Uniform(-180, 180).sample([batch_size])
    lat = torch.distributions.uniform.Uniform(-90, 90).sample([batch_size])

    model = TemporalAttention(
        in_channels=num_channels,
        hidden_size=hidden_size,
        num_head=num_head,
        d_model=d_model,
        num_time=num_time,
    )
    logits_hidden, classes_l2, classes_last = model(x, lon, lat)

    assert logits_hidden.shape == (batch_size, d_model, height, width)
    assert classes_l2.shape == (batch_size, 2, height, width)
    assert classes_last.shape == (batch_size, 3, height, width)
