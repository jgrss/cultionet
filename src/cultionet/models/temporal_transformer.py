"""
Source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py

TODO:
    https://www.sciencedirect.com/science/article/pii/S0893608023005361
    https://github.com/AzadDeihim/STTRE/blob/main/STTRE.ipynb
"""
from typing import Callable, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .. import nn as cunn
from ..layers.encodings import get_sinusoid_encoding_table
from ..layers.weights import init_attention_weights


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

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.scale = scale

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

        attention = F.softmax(scores, dim=-1, dtype=scores.dtype)
        if self.dropout is not None:
            attention = self.dropout(attention)

        output = torch.einsum('hblt, hbtv -> hblv', [attention, value])

        return output, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.add()

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, d_model: int, num_head: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_head = num_head
        d_k = d_model // num_head
        scale = 1.0 / d_k**0.5

        self.proj_query = nn.Linear(d_model, d_model)
        self.proj_key = nn.Linear(d_model, d_model)
        self.proj_value = nn.Linear(d_model, d_model)

        self.scaled_attention = ScaledDotProductAttention(
            scale, dropout=dropout
        )

        self.final = nn.Sequential(
            Rearrange('head b t c -> b t (head c)'),
            nn.LayerNorm(d_model),
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

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        dropout: float = 0.1,
    ):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model, num_head=num_head, dropout=dropout
        )
        self.feed_forward = PositionWiseFeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_head=num_head,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)

        x = self.norm(residual + self.dropout(x))

        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
        return self.seq(x) + self.skip(x)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_head: int = 8,
        in_time: int = 1,
        d_model: int = 256,
        dropout: float = 0.1,
        num_layers: int = 1,
        time_scaler: int = 100,
        num_classes_l2: int = 2,
        num_classes_last: int = 3,
        activation_type: str = "SiLU",
        final_activation: Callable = nn.Softmax(dim=1),
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
        super(TemporalTransformer, self).__init__()

        self.init_conv = nn.Sequential(
            InBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=d_model,
            ),
            Rearrange('b c t h w -> (b h w) t c'),
        )

        # Absolute positional embeddings
        self.positions = torch.arange(0, in_time, dtype=torch.long)
        self.positional_encoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                positions=in_time,
                d_hid=d_model,
                time_scaler=time_scaler,
            ),
            freeze=True,
        )

        self.layernorm = nn.LayerNorm(d_model)

        # BUG: https://github.com/Lightning-AI/pytorch-lightning/issues/15006
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=num_head,
        #     dim_feedforward=d_model * 2,
        #     dropout=dropout,
        #     activation='gelu',
        #     batch_first=True,
        #     norm_first=False,
        #     bias=True,
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        # )
        self.transformer_encoder = Transformer(
            d_model=d_model,
            num_head=num_head,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.final = nn.Conv2d(
            in_channels=d_model,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        # Level 2 level (non-crop; crop)
        self.final_l2 = cunn.FinalConv2dDropout(
            hidden_dim=d_model,
            dim_factor=1,
            activation_type=activation_type,
            final_activation=final_activation,
            num_classes=num_classes_l2,
        )
        # Last level (non-crop; crop; edges)
        self.final_l3 = cunn.FinalConv2dDropout(
            hidden_dim=d_model + num_classes_l2,
            dim_factor=1,
            activation_type=activation_type,
            final_activation=nn.Softmax(dim=1),
            num_classes=num_classes_last,
        )

        self.apply(init_attention_weights)

    def forward(self, x: torch.Tensor) -> dict:
        batch_size, num_channels, num_time, height, width = x.shape

        x = self.init_conv(x)

        # Positional embedding
        src_pos = (
            self.positions.expand(batch_size * height * width, num_time)
        ).to(x.device)
        position_tokens = self.positional_encoder(src_pos)

        x = x + position_tokens
        x = self.layernorm(x)

        # Transformer self-attention
        encoded = self.transformer_encoder(x)

        # Reshape output
        encoded = einops.rearrange(
            encoded,
            '(b h w) t c -> b c t h w',
            b=batch_size,
            h=height,
            w=width,
        )
        # Reduce the time dimension
        encoded = einops.reduce(encoded, 'b c t h w -> b c h w', 'mean')

        # Get the target classes
        l2 = self.final_l2(encoded)
        l3 = self.final_l3(torch.cat([encoded, l2], dim=1))

        encoded = self.final(encoded)

        return {
            'encoded': encoded,
            'l2': l2,
            'l3': l3,
        }


if __name__ == '__main__':
    batch_size = 2
    num_channels = 3
    hidden_channels = 64
    num_head = 8
    d_model = 128
    in_time = 12
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, in_time, height, width),
        dtype=torch.float32,
    )
    lon = torch.distributions.uniform.Uniform(-180, 180).sample([batch_size])
    lat = torch.distributions.uniform.Uniform(-90, 90).sample([batch_size])

    model = TemporalTransformer(
        in_channels=num_channels,
        hidden_channels=hidden_channels,
        num_head=num_head,
        d_model=d_model,
        in_time=in_time,
    )
    logits_hidden, classes_l2, classes_last = model(x, lon, lat)

    assert logits_hidden.shape == (batch_size, d_model, height, width)
    assert classes_l2.shape == (batch_size, 2, height, width)
    assert classes_last.shape == (batch_size, 3, height, width)
