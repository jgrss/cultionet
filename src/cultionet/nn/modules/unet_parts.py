import typing as T

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from cultionet.enums import AttentionTypes, ResBlockTypes

from .activations import SigmoidCrisp
from .convolution import (
    ConvBlock2d,
    ConvTranspose2d,
    PoolResidualConv,
    ResidualAConv,
    ResidualConv,
)


class GeoEmbeddings(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.coord_embedding = nn.Linear(3, channels)

    @torch.no_grad
    def decimal_degrees_to_cartesian(
        self, degrees: torch.Tensor
    ) -> torch.Tensor:
        radians = torch.deg2rad(degrees)
        cosine = torch.cos(radians)
        sine = torch.sin(radians)
        x = cosine[:, 1] * cosine[:, 0]
        y = cosine[:, 1] * sine[:, 0]

        return torch.stack([x, y, sine[:, 1]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coord_embedding(self.decimal_degrees_to_cartesian(x))


class TowerUNetFinal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        mask_activation: T.Callable,
        activation_type: str = "SiLU",
        resample_factor: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mask_activation = mask_activation

        if resample_factor > 1:
            self.up_conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=resample_factor,
                padding=1,
            )

        self.geo_embeddings = GeoEmbeddings(in_channels)
        self.layernorm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(in_channels),
            Rearrange('b h w c -> b c h w'),
        )

        self.expand = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels * 3,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
        )
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_crisp = SigmoidCrisp()

        self.dist_alpha1 = nn.Parameter(torch.ones(1))
        self.dist_alpha2 = nn.Parameter(torch.ones(1))
        self.edge_alpha1 = nn.Parameter(torch.ones(1))

        self.final_dist = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.final_edge = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.final_mask = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, padding=1
        )

    def forward(
        self,
        x: torch.Tensor,
        latlon_coords: T.Optional[torch.Tensor],
        size: T.Optional[torch.Size] = None,
        suffix: str = "",
    ) -> T.Dict[str, torch.Tensor]:
        if size is not None:
            x = self.up_conv(x, size=size)

        # Embed coordinates
        x = x + rearrange(self.geo_embeddings(latlon_coords), 'b c -> b c 1 1')
        x = self.layernorm(x)

        # Expand into separate streams
        dist_h, edge_h, mask_h = torch.chunk(self.expand(x), 3, dim=1)

        dist = self.final_dist(dist_h)
        edge = self.final_edge(edge_h) + dist * torch.reciprocal(
            self.dist_alpha1
        )
        mask = (
            self.final_mask(mask_h)
            + edge * torch.reciprocal(self.edge_alpha1)
            + dist * torch.reciprocal(self.dist_alpha2)
        )

        return {
            f"dist{suffix}": self.sigmoid(dist),
            f"edge{suffix}": self.sigmoid_crisp(edge),
            f"mask{suffix}": self.mask_activation(mask),
        }


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_blocks: int = 2,
        attention_weights: T.Optional[str] = None,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RESA,
        dilations: T.Sequence[int] = None,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
        resample_up: bool = True,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
    ):
        super().__init__()

        if resample_up:
            self.up_conv = ConvTranspose2d(in_channels, in_channels)

        if res_block_type == ResBlockTypes.RES:
            self.res_conv = ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_blocks=num_blocks,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
            )
        else:
            self.res_conv = ResidualAConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilations=dilations,
                repeat_kernel=repeat_resa_kernel,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
                concat_resid=concat_resid,
                natten_num_heads=natten_num_heads,
                natten_kernel_size=natten_kernel_size,
                natten_dilation=natten_dilation,
                natten_attn_drop=natten_attn_drop,
                natten_proj_drop=natten_proj_drop,
            )

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        if x.shape[-2:] != size:
            x = self.up_conv(x, size=size)

        return self.res_conv(x)


class TowerUNetEncoder(nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        pool_attention: bool = False,
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super().__init__()

        # Backbone layers
        backbone_kwargs = dict(
            dropout=dropout,
            activation_type=activation_type,
            res_block_type=res_block_type,
            batchnorm_first=batchnorm_first,
            pool_by_max=pool_by_max,
            concat_resid=concat_resid,
            natten_num_heads=8,
            natten_kernel_size=3,
            natten_dilation=1,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
        )
        self.down_a = PoolResidualConv(
            in_channels=channels[0],
            out_channels=channels[0],
            dilations=dilations,
            repeat_resa_kernel=repeat_resa_kernel,
            pool_first=False,
            attention_weights=attention_weights if pool_attention else None,
            **backbone_kwargs,
        )
        self.down_b = PoolResidualConv(
            in_channels=channels[0],
            out_channels=channels[1],
            dilations=dilations,
            repeat_resa_kernel=repeat_resa_kernel,
            attention_weights=attention_weights if pool_attention else None,
            **backbone_kwargs,
        )
        self.down_c = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations[:2],
            repeat_resa_kernel=repeat_resa_kernel,
            attention_weights=attention_weights if pool_attention else None,
            **backbone_kwargs,
        )
        self.down_d = PoolResidualConv(
            channels[2],
            channels[3],
            kernel_size=1,
            num_blocks=1,
            dilations=[1],
            repeat_resa_kernel=repeat_resa_kernel,
            attention_weights=None,
            **backbone_kwargs,
        )

    def forward(self, x: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        # Backbone
        x_a = self.down_a(x)  # 1/1 of input
        x_b = self.down_b(x_a)  # 1/2 of input
        x_c = self.down_c(x_b)  # 1/4 of input
        x_d = self.down_d(x_c)  # 1/8 of input

        return {
            "x_a": x_a,
            "x_b": x_b,
            "x_c": x_c,
            "x_d": x_d,
        }


class TowerUNetDecoder(nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super().__init__()

        # Up layers
        up_kwargs = dict(
            activation_type=activation_type,
            res_block_type=res_block_type,
            repeat_resa_kernel=repeat_resa_kernel,
            batchnorm_first=batchnorm_first,
            concat_resid=concat_resid,
            natten_num_heads=8,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
        )
        self.over_d = UNetUpBlock(
            in_channels=channels[3],
            out_channels=up_channels,
            kernel_size=1,
            num_blocks=1,
            dilations=[1],
            attention_weights=None,
            resample_up=False,
            **up_kwargs,
        )
        self.up_cu = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            dilations=dilations[:2],
            natten_kernel_size=3,
            natten_dilation=1,
            **up_kwargs,
        )
        self.up_bu = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            dilations=dilations,
            natten_kernel_size=5,
            natten_dilation=2,
            **up_kwargs,
        )
        self.up_au = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            dilations=dilations,
            natten_kernel_size=7,
            natten_dilation=3,
            **up_kwargs,
        )

    def forward(
        self, x: T.Dict[str, torch.Tensor]
    ) -> T.Dict[str, torch.Tensor]:
        x_du = self.over_d(x["x_d"], size=x["x_d"].shape[-2:])

        # Up
        x_cu = self.up_cu(x_du, size=x["x_c"].shape[-2:])
        x_bu = self.up_bu(x_cu, size=x["x_b"].shape[-2:])
        x_au = self.up_au(x_bu, size=x["x_a"].shape[-2:])

        return {
            "x_au": x_au,
            "x_bu": x_bu,
            "x_cu": x_cu,
            "x_du": x_du,
        }


class TowerUNetFusion(nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
        dropout: float = 0.0,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        concat_resid: bool = False,
    ):
        super().__init__()

        # Towers
        tower_kwargs = dict(
            up_channels=up_channels,
            out_channels=up_channels,
            attention_weights=attention_weights,
            activation_type=activation_type,
            res_block_type=res_block_type,
            repeat_resa_kernel=repeat_resa_kernel,
            batchnorm_first=batchnorm_first,
            concat_resid=concat_resid,
            natten_num_heads=8,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
        )
        self.tower_c = TowerUNetBlock(
            backbone_side_channels=channels[2],
            backbone_down_channels=channels[3],
            dilations=dilations[:2],
            natten_kernel_size=3,
            natten_dilation=1,
            **tower_kwargs,
        )
        self.tower_b = TowerUNetBlock(
            backbone_side_channels=channels[1],
            backbone_down_channels=channels[2],
            tower=True,
            dilations=dilations,
            natten_kernel_size=5,
            natten_dilation=2,
            **tower_kwargs,
        )
        self.tower_a = TowerUNetBlock(
            backbone_side_channels=channels[0],
            backbone_down_channels=channels[1],
            tower=True,
            dilations=dilations,
            natten_kernel_size=7,
            natten_dilation=3,
            **tower_kwargs,
        )

    def forward(
        self,
        encoded: T.Dict[str, torch.Tensor],
        decoded: T.Dict[str, torch.Tensor],
    ) -> T.Dict[str, torch.Tensor]:
        # Central towers
        x_tower_c = self.tower_c(
            backbone_side=encoded["x_c"],
            backbone_down=encoded["x_d"],
            decode_side=decoded["x_cu"],
            decode_down=decoded["x_du"],
        )
        x_tower_b = self.tower_b(
            backbone_side=encoded["x_b"],
            backbone_down=encoded["x_c"],
            decode_side=decoded["x_bu"],
            decode_down=decoded["x_cu"],
            tower_down=x_tower_c,
        )
        x_tower_a = self.tower_a(
            backbone_side=encoded["x_a"],
            backbone_down=encoded["x_b"],
            decode_side=decoded["x_au"],
            decode_down=decoded["x_bu"],
            tower_down=x_tower_b,
        )

        return {
            "x_tower_a": x_tower_a,
            "x_tower_b": x_tower_b,
            "x_tower_c": x_tower_c,
        }


class TowerUNetBlock(nn.Module):
    def __init__(
        self,
        backbone_side_channels: int,
        backbone_down_channels: int,
        up_channels: int,
        out_channels: int,
        tower: bool = False,
        kernel_size: int = 3,
        num_blocks: int = 2,
        attention_weights: T.Optional[str] = None,
        res_block_type: str = ResBlockTypes.RESA,
        dilations: T.Sequence[int] = None,
        repeat_resa_kernel: bool = False,
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
        concat_resid: bool = False,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
    ):
        super().__init__()

        in_channels = (
            backbone_side_channels + backbone_down_channels + up_channels * 2
        )

        self.backbone_down_conv = ConvTranspose2d(
            in_channels=backbone_down_channels,
            out_channels=backbone_down_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.decode_down_conv = ConvTranspose2d(
            in_channels=up_channels,
            out_channels=up_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        if tower:
            self.tower_conv = ConvTranspose2d(
                in_channels=up_channels,
                out_channels=up_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            in_channels += up_channels

        if res_block_type == ResBlockTypes.RES:
            self.res_conv = ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_blocks=num_blocks,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
            )
        else:
            self.res_conv = ResidualAConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                num_blocks=num_blocks,
                dilations=dilations,
                repeat_kernel=repeat_resa_kernel,
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
                concat_resid=concat_resid,
                natten_num_heads=natten_num_heads,
                natten_kernel_size=natten_kernel_size,
                natten_dilation=natten_dilation,
                natten_attn_drop=natten_attn_drop,
                natten_proj_drop=natten_proj_drop,
            )

    def forward(
        self,
        backbone_side: torch.Tensor,
        backbone_down: torch.Tensor,
        decode_side: torch.Tensor,
        decode_down: torch.Tensor,
        tower_down: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backbone_down = self.backbone_down_conv(
            backbone_down,
            size=decode_side.shape[-2:],
        )
        decode_down = self.decode_down_conv(
            decode_down,
            size=decode_side.shape[-2:],
        )

        x = torch.cat(
            (backbone_side, backbone_down, decode_side, decode_down),
            dim=1,
        )

        if tower_down is not None:
            tower_down = self.tower_conv(
                tower_down,
                size=decode_side.shape[-2:],
            )

            x = torch.cat((x, tower_down), dim=1)

        return self.res_conv(x)
