"""
Source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py
"""
import copy
from typing import Callable, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from .base_layers import Softmax, FinalConv2dDropout
from .positional_encoding import PositionalEncoder


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_comp: bool = False,
    ):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

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

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5)
        )

    def forward(
        self,
        v: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_comp: bool = False,
    ):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        batch_size, time_size, _ = v.size()

        q = torch.stack([self.Q for _ in range(batch_size)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(batch_size, time_size, n_head, d_k)
        k = (
            k.permute(2, 0, 1, 3).contiguous().view(-1, time_size, d_k)
        )  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * batch_size, time_size, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, batch_size, 1, time_size)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, batch_size, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class LightweightTemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 128,
        n_head: int = 16,
        n_time: int = 1,
        d_k: int = 4,
        mlp: Sequence[int] = [256, 128],
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1_000,
        return_att: bool = False,
        positional_encoding: bool = True,
        num_classes_l2: int = 2,
        num_classes_last: int = 3,
        activation_type: str = "LeakyReLU",
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
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LightweightTemporalAttentionEncoder, self).__init__()

        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

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
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

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

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

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
        mask_padded: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, channel_size, time_size, height, width = x.shape
        batch_positions = (
            torch.arange(time_size)
            .unsqueeze(-1)
            .repeat(batch_size, 1, 1)
            .unsqueeze(-1)
            .repeat(1, 1, 1, height)
            .unsqueeze(-1)
            .repeat(1, 1, 1, 1, width)
        ).to(dtype=x.dtype, device=x.device)
        # input shape = (B x C x T x H x W)
        # permuted shape = (B x T x C x H x W)
        x = self.init_conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        # x shape = (batch_size, time_size, channel_size, height, width)

        pad_mask = None
        if mask_padded:
            pad_mask = (x == 0).all(dim=-1).all(dim=-1).all(dim=-1)
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, height))
                .unsqueeze(-1)
                .repeat((1, 1, 1, width))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1)
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

        if self.positional_encoder is not None:
            # B x T x C
            bp = batch_positions.contiguous().view(
                batch_size * height * width, time_size
            )
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2)
            .contiguous()
            .view(batch_size * height * width, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        attn = attn.view(
            self.n_head, batch_size, height, width, time_size
        ).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        # attn shape = (n_head x batch_size x time_size x height x width)
        last_l2 = self.final_l2(
            attn.permute(1, 0, 2, 3, 4).reshape(batch_size, -1, height, width)
        )
        last = self.final_last(out)
        if self.return_att:
            return out, last_l2, last, attn
        else:
            return out, last_l2, last
