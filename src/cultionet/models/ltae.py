"""
Source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py

TODO:
    https://www.sciencedirect.com/science/article/pii/S0893608023005361
    https://github.com/AzadDeihim/STTRE/blob/main/STTRE.ipynb
"""
import copy
import math
from typing import Callable, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from .base_layers import Softmax, FinalConv2dDropout
from .encodings import cartesian, get_sinusoid_encoding_table


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(
        self,
        dropout: float = 0.1,
        scale: Optional[float] = None,
    ):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_comp: bool = False,
    ):
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        L, S = query.size(-2), key.size(-2)
        scale_factor = (
            1.0 / math.sqrt(query.size(-1))
            if self.scale is None
            else self.scale
        )
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(
                L, S, dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias + attn_mask

        attn = (query.unsqueeze(1) @ key.transpose(1, 2)) * scale_factor
        attn = attn + attn_bias.unsqueeze(1)

        if return_comp:
            comp = attn

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = attn @ value

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module Modified from
    github.com/jadore801120/attention-is-all-you-need-pytorch."""

    def __init__(self, n_head: int, d_k: int, d_in: int):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention()
        # self.attention = nn.MultiheadAttention(
        #     n_head,
        #     d_k,
        #     dropout=0.1,
        #     # (batch x seq x feature)
        #     batch_first=False,
        # )

    def forward(
        self,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_comp: bool = False,
    ):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        batch_size, time_size, _ = value.size()

        # (n*b) x d_k
        query = self.Q.repeat(batch_size, 1)
        key = self.fc1_k(value).view(batch_size, time_size, n_head, d_k)
        # (n*b) x lk x dk
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, time_size, d_k)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(
                (n_head, 1)
            )  # replicate attn_mask for each head (nxb) x lk

        value = torch.stack(
            value.split(value.shape[-1] // n_head, dim=-1)
        ).view(n_head * batch_size, time_size, -1)

        if return_comp:
            output, attn, comp = self.attention(
                query, key, value, attn_mask=attn_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                query, key, value, attn_mask=attn_mask, return_comp=return_comp
            )

        attn = attn.view(n_head, batch_size, 1, time_size)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, batch_size, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class MLPBlock(nn.Module):
    def __init__(self, idx: int, dimensions: Sequence[int]):
        super(MLPBlock, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(dimensions[idx], dimensions[idx]),
            nn.BatchNorm1d(dimensions[idx]),
            nn.GELU(),
            nn.Linear(dimensions[idx], dimensions[idx + 1]),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class LightweightTemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 128,
        n_head: int = 8,
        n_time: int = 1,
        d_k: int = 4,
        mlp: Sequence[int] = [256, 128],
        dropout: float = 0.2,
        d_model: int = 256,
        time_scaler: int = 1_000,
        return_att: bool = False,
        num_classes_l2: int = 2,
        num_classes_last: int = 3,
        activation_type: str = "SiLU",
        final_activation: Callable = Softmax(dim=1),
    ):
        """Lightweight Temporal Attention Encoder (L-TAE) for image time
        series. Attention-based sequence encoding that maps a sequence of
        images to a single feature map. A shared L-TAE is applied to all pixel
        positions of the image sequence.

        Args:
            in_channels (int): Number of channels of the inputs.
            hidden_size (int): Number of hidden layers.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            time_scaler (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
        """
        super(LightweightTemporalAttentionEncoder, self).__init__()

        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head
        mlp = copy.deepcopy(mlp)

        self.init_conv = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(hidden_size, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert mlp[0] == self.d_model

        # Absolute positional embeddings
        self.positional_encoder = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                positions=n_time,
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
        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=hidden_size,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = [MLPBlock(i, mlp) for i in range(len(mlp) - 1)]
        layers += [nn.Dropout(dropout)]
        self.mlp_seq = nn.Sequential(*layers)

        # Level 2 level (non-crop; crop)
        self.final_l2 = FinalConv2dDropout(
            hidden_dim=n_head * n_time,
            dim_factor=1,
            activation_type=activation_type,
            final_activation=final_activation,
            num_classes=num_classes_l2,
        )
        # Last level (non-crop; crop; edges)
        self.final_last = FinalConv2dDropout(
            hidden_dim=mlp[-1],
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
        mask_padded: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, channel_size, time_size, height, width = x.shape
        # TODO: Channel embedding

        # input shape = (B x C x T x H x W)
        # permuted shape = (B x T x C x H x W)
        x = self.init_conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        # x shape = (batch_size, time_size, channel_size, height, width)

        attn_mask = None
        if mask_padded:
            attn_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)
            attn_mask = (
                attn_mask.unsqueeze(-1)
                .repeat((1, 1, height))
                .unsqueeze(-1)
                .repeat((1, 1, 1, width))
            )  # BxTxHxW
            attn_mask = (
                attn_mask.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * height * width, time_size)
            )

        out = (
            x.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(batch_size * height * width, time_size, x.shape[-3])
        )
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

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
                torch.tile(longitude[:, None], (1, height * width)).view(
                    batch_size * height * width, 1
                ),
                torch.tile(latitude[:, None], (1, height * width)).view(
                    batch_size * height * width, 1
                ),
            )
        )
        # TODO: concatenate?
        out = out + position_tokens + coordinate_tokens
        # Attention
        out, attn = self.attention_heads(out, attn_mask=attn_mask)
        # Concatenate heads
        out = (
            out.permute(1, 0, 2)
            .contiguous()
            .view(batch_size * height * width, -1)
        )
        out = self.mlp_seq(out)
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        # head x b x t x h x w
        attn = attn.view(
            self.n_head, batch_size, height, width, time_size
        ).permute(0, 1, 4, 2, 3)

        # attn shape = (n_head x batch_size x time_size x height x width)
        last_l2 = self.final_l2(
            attn.permute(1, 0, 2, 3, 4).reshape(batch_size, -1, height, width)
        )
        last = self.final_last(out)

        if self.return_att:
            return out, last_l2, last, attn
        else:
            return out, last_l2, last
