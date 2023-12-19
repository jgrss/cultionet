import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from cultionet.models.base_layers import SigmoidCrisp
from cultionet.models.encodings import get_sinusoid_encoding_table


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
        stride: T.Union[int, T.Tuple[int, ...]],
        padding: T.Union[int, T.Tuple[int, ...]],
        dilation: T.Union[int, T.Tuple[int, ...]],
        bias: bool = True,
    ):
        super(ConvLayer, self).__init__()

        self.seq = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.SiLU(),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # num_batches, num_channels, num_time, height, width = x.shape
        return self.seq(x)


class AtrousConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
        stride: T.Union[int, T.Tuple[int, ...]],
        padding: T.Union[int, T.Tuple[int, ...]],
        dilation: T.Union[int, T.Tuple[int, ...]],
    ):
        super(AtrousConvLayer, self).__init__()

        self.seq = nn.Sequential(
            ConvLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # num_batches, num_channels, num_time, height, width = x.shape
        return self.seq(x)


class ResABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
        stride: T.Union[int, T.Tuple[int, ...]],
        dilations: T.Sequence[int],
    ):
        super(ResABlock, self).__init__()

        self.resa_layers = nn.ModuleList(
            [
                AtrousConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(1, dilation, dilation),
                    dilation=(1, dilation, dilation),
                )
                for dilation in dilations
            ]
        )
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # num_batches, num_channels, num_time, height, width = x.shape

        residual = x
        if self.skip is not None:
            residual = self.skip(residual)
        for layer in self.resa_layers:
            residual = residual + layer(x)

        return residual


class PSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
        stride: T.Union[int, T.Tuple[int, ...]],
    ):
        super(PSPLayer, self).__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=(1, stride, stride)
            if isinstance(kernel_size, tuple)
            else stride,
            padding=(0, 1, 1),
        )
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            self.pool(x),
            size=x.shape[-3:],
            mode="trilinear",
            align_corners=True,
        )
        x = self.conv(x)

        return x


class PyramidPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
    ):
        super(PyramidPooling, self).__init__()

        self.layer0 = PSPLayer(
            in_channels=in_channels, kernel_size=kernel_size, stride=1
        )
        self.layer1 = PSPLayer(
            in_channels=in_channels, kernel_size=kernel_size, stride=2
        )
        self.layer2 = PSPLayer(
            in_channels=in_channels, kernel_size=kernel_size, stride=4
        )
        self.layer3 = PSPLayer(
            in_channels=in_channels, kernel_size=kernel_size, stride=8
        )
        self.conv = nn.Conv3d(
            in_channels * 5,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x = torch.cat((x, x0, x1, x2, x3), dim=1)
        out = self.conv(x)

        return out


def combine(x: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    down = F.interpolate(
        down,
        size=x.shape[-3:],
        mode="trilinear",
        align_corners=True,
    )

    return torch.cat((x, down), dim=1)


class Combine(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(Combine, self).__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )

    def forward(self, x: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        x = combine(x, down)
        out = self.conv(x)

        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: T.Union[int, T.Tuple[int, ...]],
        stride: T.Union[int, T.Tuple[int, ...]],
        dilations: T.Sequence[int],
    ):
        super(UpBlock, self).__init__()

        self.combine = Combine(
            in_channels=in_channels,
            out_channels=hidden_channels,
        )
        self.conv = ResABlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
        )

    def forward(self, x: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        return self.conv(self.combine(x, down))


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: T.Sequence[int],
        stride: T.Optional[T.Union[int, T.Tuple[int, ...]]] = None,
    ):
        super(DownBlock, self).__init__()

        if stride is None:
            stride = (1, 2, 2)

        self.seq = nn.Sequential(
            ResABlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                dilations=dilations,
            ),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=stride,
                padding=(0, 1, 1),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ScaledDotProduct(nn.Module):
    def __init__(self, scale: float, dropout: float):
        super(ScaledDotProduct, self).__init__()

        self.scale = scale
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        scores = query @ key.transpose(-2, -1) / self.scale
        attention = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        attention = attention @ value

        return attention


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0):
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = self.d_k**0.5

        self.query_w = nn.Linear(d_model, d_model)
        self.key_w = nn.Linear(d_model, d_model)
        self.value_w = nn.Linear(d_model, d_model)
        self.out_w = nn.Linear(d_model, d_model)

        self.scaled_dot_product_attention = ScaledDotProduct(
            self.scale, dropout=dropout
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.shape
        return x.view(
            batch_size, seq_length, self.num_heads, self.d_k
        ).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # Apply linear transformations and split heads
        query = self.split_heads(self.query_w(query))
        key = self.split_heads(self.key_w(key))
        value = self.split_heads(self.value_w(value))
        # Perform scaled dot-product attention
        attention = self.scaled_dot_product_attention(query, key, value)
        # Combine heads and apply output transformation
        attention = self.out_w(self.combine_heads(attention))

        return attention


class TemporalResAUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_time: int,
        height: int,
        width: int,
    ):
        super(TemporalResAUNet, self).__init__()

        kernel_size = 3
        stride = 1
        dilations0 = (1, 2, 3, 4)
        dilations1 = (1, 2, 3)
        dilations2 = (1,)
        hidden_dims = [hidden_channels]
        for _ in range(4):
            hidden_dims += [hidden_dims[-1] * 2]

        self.input = nn.Conv3d(
            in_channels, hidden_dims[0], kernel_size=1, padding=0
        )
        # Down 0
        self.down_block0 = DownBlock(
            in_channels=hidden_dims[0],
            out_channels=hidden_dims[1],
            dilations=dilations0,
        )
        # self.down_skip_block0_3 = DownBlock(
        #     in_channels=hidden_dims[0],
        #     out_channels=hidden_dims[3],
        #     dilations=dilations0,
        #     stride=(1, 8, 8),
        # )
        # Down 2
        self.down_block1 = DownBlock(
            in_channels=hidden_dims[1],
            out_channels=hidden_dims[2],
            dilations=dilations0,
        )
        # self.down_skip_block1_3 = DownBlock(
        #     in_channels=hidden_dims[1],
        #     out_channels=hidden_dims[3],
        #     dilations=dilations0,
        #     stride=(1, 4, 4),
        # )
        # Down 3
        self.down_block2 = DownBlock(
            in_channels=hidden_dims[2],
            out_channels=hidden_dims[3],
            dilations=dilations1,
        )
        # self.down_skip_block2_3 = DownBlock(
        #     in_channels=hidden_dims[2],
        #     out_channels=hidden_dims[3],
        #     dilations=dilations0,
        #     stride=(1, 2, 2),
        # )
        # Down 4
        self.down_block3 = DownBlock(
            in_channels=hidden_dims[3],
            out_channels=hidden_dims[4],
            dilations=dilations2,
        )
        # Absolute positional embeddings
        # self.positional_encoder = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(
        #         positions=num_time,
        #         d_hid=hidden_dims[4],
        #         time_scaler=1_000,
        #     ),
        #     freeze=True,
        # )
        # Multi-head self-attention
        # self.attention = nn.MultiheadAttention(
        #     hidden_dims[4], num_heads=4, dropout=0.1
        # )
        # self.attention = MultiheadAttention(
        #     hidden_dims[4], num_heads=4, dropout=0.1
        # )
        # Pool
        self.u_pool = PyramidPooling(
            in_channels=hidden_dims[4],
            out_channels=hidden_dims[3],
            kernel_size=(1, 3, 3),
        )
        # Up 3
        self.up_block3 = UpBlock(
            in_channels=hidden_dims[3] * 2,
            hidden_channels=hidden_dims[3],
            out_channels=hidden_dims[2],
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations2,
        )
        # Up 2
        self.up_block2 = UpBlock(
            in_channels=hidden_dims[2] * 2,
            hidden_channels=hidden_dims[2],
            out_channels=hidden_dims[1],
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations1,
        )
        # Up 1
        self.up_block1 = UpBlock(
            in_channels=hidden_dims[1] * 2,
            hidden_channels=hidden_dims[1],
            out_channels=hidden_dims[0],
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations0,
        )
        # Up 0
        self.final_combine = Combine(
            in_channels=hidden_dims[0] * 2,
            out_channels=hidden_dims[0],
        )
        self.final_pool = nn.Sequential(
            PyramidPooling(
                in_channels=hidden_dims[0],
                out_channels=hidden_dims[0],
                kernel_size=(1, 3, 3),
            ),
        )

        self.reduce_logit_time = nn.AdaptiveAvgPool3d((1, height, width))
        self.reduce_pool_time = nn.AdaptiveAvgPool3d((1, height, width))

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_crisp = SigmoidCrisp()
        self.final_dist = nn.Conv2d(
            hidden_dims[0], 1, kernel_size=1, padding=0
        )
        self.final_boundary = nn.Conv2d(
            hidden_dims[0] + 1, 1, kernel_size=1, padding=0
        )
        self.final_mask = nn.Conv2d(
            hidden_dims[0] + 2, out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_batches, num_channels, num_time, height, width = x.shape

        x_in = self.input(x)
        # Down
        down_out_block0 = self.down_block0(x_in)
        down_out_block1 = self.down_block1(down_out_block0)
        down_out_block2 = self.down_block2(down_out_block1)
        down_out_block3 = self.down_block3(down_out_block2)
        # Self-attention
        # _, block3_dims, _, block3_height, block3_width = down_out_block3.shape
        # block3_attention = (
        #     down_out_block3
        #     .permute(0, 3, 4, 1, 2)
        #     .contiguous()
        #     .view(-1, num_time, block3_dims)
        # )
        # src_pos = (
        #     torch.arange(0, num_time, dtype=torch.long)
        #     .expand(block3_attention.shape[0], block3_attention.shape[1])
        #     .to(x.device)
        # )
        # block3_attention = block3_attention + self.positional_encoder(src_pos)
        # block3_attention = self.attention(
        #     block3_attention, block3_attention, block3_attention
        # )
        # block3_attention = (
        #     block3_attention
        #     .view(
        #         num_batches,
        #         block3_height,
        #         block3_width,
        #         block3_dims,
        #         num_time,
        #     )
        #     .permute(0, 3, 4, 1, 2)
        # )
        # Pyramid pooling
        u_pool = self.u_pool(down_out_block3)
        # self.down_skip_block0_3(x_in)
        # self.down_skip_block1_3(x_in)
        # self.down_skip_block2_3(x_in)
        # Up
        up_out_block3 = self.up_block3(down_out_block2, u_pool)
        up_out_block2 = self.up_block2(down_out_block1, up_out_block3)
        up_out_block1 = self.up_block1(down_out_block0, up_out_block2)
        # Final
        up_out_block0 = self.final_combine(x_in, up_out_block1)
        final_pool = self.final_pool(up_out_block0)
        # Reduce time to 1
        final_logits = self.reduce_logit_time(up_out_block0).squeeze(dim=2)
        pool_logits = self.reduce_pool_time(final_pool).squeeze(dim=2)
        # Final layers
        distance = self.final_dist(final_logits)
        boundary = self.final_boundary(
            torch.cat((pool_logits, distance), dim=1)
        )
        mask = self.final_mask(
            torch.cat((pool_logits, distance, boundary), dim=1)
        )

        distance = self.sigmoid(distance)
        boundary = self.sigmoid_crisp(boundary)
        mask = self.sigmoid(mask)

        return {
            "dist": distance,
            "edge": boundary,
            "mask": mask,
            "dist_3_1": None,
            "mask_3_1": None,
            "edge_3_1": None,
        }


if __name__ == "__main__":
    num_batches = 2
    num_time = 12

    in_channels = 3
    height = 100
    width = 100

    hidden_channels = 32
    out_channels = 1

    x = torch.rand(
        (num_batches, in_channels, num_time, height, width),
        dtype=torch.float32,
    )

    block = TemporalResAUNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        height=height,
        width=width,
    )
    out = block(x)
    import ipdb

    ipdb.set_trace()
