import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cultionet.enums import AttentionTypes, InferenceNames, ResBlockTypes

from .convolution import (
    ConvBlock2d,
    ConvTranspose2d,
    PoolResidualConv,
    ResidualAConv,
    ResidualConv,
)
from .geo_encoding import GeoEmbeddings

NATTEN_PARAMS = {
    "a": {
        "natten_num_heads": 4,
        "natten_kernel_size": 3,
        "natten_dilation": 2,
    },
    "b": {
        "natten_num_heads": 4,
        "natten_kernel_size": 3,
        "natten_dilation": 1,
    },
    "c": {
        "natten_num_heads": 8,
        "natten_kernel_size": 3,
        "natten_dilation": 1,
    },
    "d": {
        "natten_num_heads": 8,
        "natten_kernel_size": 1,
        "natten_dilation": 1,
    },
}


class SigmoidCrisp(nn.Module):
    r"""Sigmoid crisp.

    Adapted from publication and source code below:

        CSIRO BSTD/MIT LICENSE

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that
        the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
            following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
            the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
            promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
        USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        Citation:
            @article{diakogiannis_etal_2021,
                title={Looking for change? Roll the dice and demand attention},
                author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter},
                journal={Remote Sensing},
                volume={13},
                number={18},
                pages={3707},
                year={2021},
                publisher={MDPI}
            }

        Reference:
            https://www.mdpi.com/2072-4292/13/18/3707
            https://arxiv.org/pdf/2009.02062.pdf
            https://github.com/waldnerf/decode/blob/main/FracTAL_ResUNet/nn/activations/sigmoid_crisp.py
    """

    def __init__(self, smooth: float = 1e-2):
        super().__init__()

        self.smooth = smooth
        self.gamma = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.smooth + F.sigmoid(self.gamma)
        out = torch.reciprocal(out)
        out = x * out
        out = F.sigmoid(out)

        return out


class TowerUNetFinalCombine(nn.Module):
    """Final output by fusing all tower outputs."""

    def __init__(
        self,
        num_classes: int,
        edge_activation: bool = True,
        mask_activation: bool = True,
    ):
        super().__init__()

        edge_activation_layer = (
            SigmoidCrisp() if edge_activation else nn.Identity()
        )
        mask_activation_layer = (
            nn.Sigmoid() if mask_activation else nn.Identity()
        )

        self.final_dist = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.dist_gamma1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.dist_gamma2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.dist_gamma3 = nn.Parameter(torch.ones(1, requires_grad=True))

        self.final_edge = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0),
            edge_activation_layer,
        )
        self.edge_gamma1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.edge_gamma2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.edge_gamma3 = nn.Parameter(torch.ones(1, requires_grad=True))

        self.final_crop = nn.Sequential(
            nn.Conv2d(
                in_channels=num_classes,
                out_channels=num_classes,
                kernel_size=1,
                padding=0,
            ),
            mask_activation_layer,
        )
        self.crop_gamma1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crop_gamma2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.crop_gamma3 = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(
        self,
        out_a: T.Dict[str, torch.Tensor],
        out_b: T.Dict[str, torch.Tensor],
        out_c: T.Dict[str, torch.Tensor],
        suffixes: T.Sequence[str],
    ) -> T.Dict[str, torch.Tensor]:

        distance = self.final_dist(
            (
                torch.reciprocal(self.dist_gamma1)
                * out_a[f"{InferenceNames.DISTANCE}{suffixes[0]}"]
                + torch.reciprocal(self.dist_gamma2)
                * out_b[f"{InferenceNames.DISTANCE}{suffixes[1]}"]
                + torch.reciprocal(self.dist_gamma3)
                * out_c[f"{InferenceNames.DISTANCE}{suffixes[2]}"]
            )
        )

        edge = self.final_edge(
            (
                torch.reciprocal(self.edge_gamma1)
                * out_a[f"{InferenceNames.EDGE}{suffixes[0]}"]
                + torch.reciprocal(self.edge_gamma2)
                * out_b[f"{InferenceNames.EDGE}{suffixes[1]}"]
                + torch.reciprocal(self.edge_gamma3)
                * out_c[f"{InferenceNames.EDGE}{suffixes[2]}"]
            )
        )

        crop = self.final_crop(
            (
                torch.reciprocal(self.crop_gamma1)
                * out_a[f"{InferenceNames.CROP}{suffixes[0]}"]
                + torch.reciprocal(self.crop_gamma2)
                * out_b[f"{InferenceNames.CROP}{suffixes[1]}"]
                + torch.reciprocal(self.crop_gamma3)
                * out_c[f"{InferenceNames.CROP}{suffixes[2]}"]
            )
        )

        return {
            InferenceNames.DISTANCE: distance,
            InferenceNames.EDGE: edge,
            InferenceNames.CROP: crop,
        }


class StreamConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation_type: str,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                add_activation=True,
                activation_type=activation_type,
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TowerUNetFinal(nn.Module):
    """Output of an individual tower fusion."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        activation_type: str = "SiLU",
        resample_factor: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        if resample_factor > 1:
            self.up_conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=resample_factor,
                padding=1,
            )

        # Hidden -> 3 -> 3 -> 1
        self.dist_conv = StreamConv2d(
            in_channels=in_channels,
            hidden_channels=3,
            out_channels=1,
            activation_type=activation_type,
        )
        self.edge_conv = StreamConv2d(
            in_channels=in_channels,
            hidden_channels=3,
            out_channels=1,
            activation_type=activation_type,
        )
        self.crop_conv = StreamConv2d(
            in_channels=in_channels,
            hidden_channels=3,
            out_channels=1,
            activation_type=activation_type,
        )

        # 3 -> 3
        self.fuse_conv = ConvBlock2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            padding=1,
            add_activation=True,
            activation_type=activation_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        size: T.Optional[torch.Size] = None,
        suffix: str = "",
    ) -> T.Dict[str, torch.Tensor]:
        if size is not None:
            x = self.up_conv(x, size=size)

        # Separate hidden into task streams
        # H -> 3 -> 1
        dist_h = self.dist_conv(x)
        edge_h = self.edge_conv(x)
        crop_h = self.crop_conv(x)

        # [1, 1, 1] -> 3
        h = torch.cat([dist_h, edge_h, crop_h], dim=1)
        # 3 -> 3
        h = self.fuse_conv(h)
        # -> [1, 1, 1]
        dist_out, edge_out, mask_out = torch.chunk(h, 3, dim=1)

        # x --> H(3) --> H(1) --> Concat(3) --> Fuse(3) --> Chunk(1,1,1)

        return {
            f"{InferenceNames.DISTANCE}{suffix}": dist_out,
            f"{InferenceNames.EDGE}{suffix}": edge_out,
            f"{InferenceNames.CROP}{suffix}": mask_out,
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
        batchnorm_first: bool = False,
        resample_up: bool = True,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
    ):
        super().__init__()

        assert res_block_type in (
            ResBlockTypes.RES,
            ResBlockTypes.RESA,
        )

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
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
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
        attention_weights: str = AttentionTypes.NATTEN,
        pool_by_max: bool = False,
        batchnorm_first: bool = False,
    ):
        super().__init__()

        # Backbone layers
        backbone_kwargs = dict(
            dropout=dropout,
            activation_type=activation_type,
            res_block_type=res_block_type,
            batchnorm_first=batchnorm_first,
            pool_by_max=pool_by_max,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
        )
        self.down_a = PoolResidualConv(
            in_channels=channels[0],
            out_channels=channels[0],
            dilations=dilations,
            pool_first=False,
            # Attention applied at 1/1 spatial resolution
            attention_weights=attention_weights,
            **{**backbone_kwargs, **NATTEN_PARAMS["a"]},
        )
        self.down_b = PoolResidualConv(
            in_channels=channels[0],
            out_channels=channels[1],
            dilations=dilations[:3],
            # Attention applied at 1/2 spatial resolution
            attention_weights=attention_weights,
            **{**backbone_kwargs, **NATTEN_PARAMS["b"]},
        )
        self.down_c = PoolResidualConv(
            channels[1],
            channels[2],
            dilations=dilations[:2],
            # Attention applied at 1/4 spatial resolution
            attention_weights=attention_weights,
            **{**backbone_kwargs, **NATTEN_PARAMS["c"]},
        )
        self.down_d = PoolResidualConv(
            channels[2],
            channels[3],
            kernel_size=1,
            num_blocks=1,
            dilations=[1],
            # Attention applied at 1/8 spatial resolution
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
        attention_weights: str = AttentionTypes.NATTEN,
        batchnorm_first: bool = False,
    ):
        super().__init__()

        # Up layers
        up_kwargs = dict(
            activation_type=activation_type,
            res_block_type=res_block_type,
            batchnorm_first=batchnorm_first,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
        )
        self.over_d = UNetUpBlock(
            in_channels=channels[3],
            out_channels=up_channels,
            kernel_size=1,
            num_blocks=1,
            dilations=[1],
            resample_up=False,
            # Attention applied at 1/8 spatial resolution
            attention_weights=None,
            **up_kwargs,
        )
        self.up_cu = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            dilations=dilations[:2],
            # Attention applied at 1/4 spatial resolution
            attention_weights=attention_weights,
            **{**up_kwargs, **NATTEN_PARAMS["c"]},
        )
        self.up_bu = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            dilations=dilations[:3],
            # Attention applied at 1/2 spatial resolution
            attention_weights=attention_weights,
            **{**up_kwargs, **NATTEN_PARAMS["b"]},
        )
        self.up_au = UNetUpBlock(
            in_channels=up_channels,
            out_channels=up_channels,
            dilations=dilations,
            # Attention applied at 1/1 spatial resolution
            attention_weights=attention_weights,
            **{**up_kwargs, **NATTEN_PARAMS["a"]},
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
        attention_weights: str = AttentionTypes.NATTEN,
        batchnorm_first: bool = False,
        use_latlon: bool = False,
    ):
        super().__init__()

        # Towers
        tower_kwargs = dict(
            up_channels=up_channels,
            out_channels=up_channels,
            activation_type=activation_type,
            res_block_type=res_block_type,
            batchnorm_first=batchnorm_first,
            attention_weights=attention_weights,
            natten_attn_drop=dropout,
            natten_proj_drop=dropout,
            use_latlon=use_latlon,
        )
        self.tower_c = TowerUNetBlock(
            backbone_side_channels=channels[2],
            backbone_down_channels=channels[3],
            dilations=dilations[:2],
            **{**tower_kwargs, **NATTEN_PARAMS["c"]},
        )
        self.tower_b = TowerUNetBlock(
            backbone_side_channels=channels[1],
            backbone_down_channels=channels[2],
            tower=True,
            dilations=dilations,
            **{**tower_kwargs, **NATTEN_PARAMS["b"]},
        )
        self.tower_a = TowerUNetBlock(
            backbone_side_channels=channels[0],
            backbone_down_channels=channels[1],
            tower=True,
            dilations=dilations,
            **{**tower_kwargs, **NATTEN_PARAMS["a"]},
        )

    def forward(
        self,
        encoded: T.Dict[str, torch.Tensor],
        decoded: T.Dict[str, torch.Tensor],
        latlon_coords: T.Optional[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:

        # Central towers
        x_tower_c = self.tower_c(
            backbone_side=encoded["x_c"],
            backbone_down=encoded["x_d"],
            decode_side=decoded["x_cu"],
            decode_down=decoded["x_du"],
            latlon_coords=latlon_coords,
        )
        x_tower_b = self.tower_b(
            backbone_side=encoded["x_b"],
            backbone_down=encoded["x_c"],
            decode_side=decoded["x_bu"],
            decode_down=decoded["x_cu"],
            tower_down=x_tower_c,
            latlon_coords=latlon_coords,
        )
        x_tower_a = self.tower_a(
            backbone_side=encoded["x_a"],
            backbone_down=encoded["x_b"],
            decode_side=decoded["x_au"],
            decode_down=decoded["x_bu"],
            tower_down=x_tower_b,
            latlon_coords=latlon_coords,
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
        activation_type: str = "SiLU",
        batchnorm_first: bool = False,
        natten_num_heads: int = 8,
        natten_kernel_size: int = 3,
        natten_dilation: int = 1,
        natten_attn_drop: float = 0.0,
        natten_proj_drop: float = 0.0,
        use_latlon: bool = False,
    ):
        super().__init__()

        self.use_latlon = use_latlon

        assert res_block_type in (
            ResBlockTypes.RES,
            ResBlockTypes.RESA,
        )

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

        if self.use_latlon:
            # TODO: make optional
            self.geo_embeddings = torch.compile(GeoEmbeddings(up_channels))
            # self.geo_embeddings4 = SphericalHarmonics(out_channels=in_channels, legendre_polys=4)
            # self.geo_embeddings8 = SphericalHarmonics(out_channels=in_channels, legendre_polys=8)

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
                attention_weights=attention_weights,
                activation_type=activation_type,
                batchnorm_first=batchnorm_first,
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
        latlon_coords: T.Optional[torch.Tensor] = None,
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

        # Embed coordinates
        if self.use_latlon:
            assert latlon_coords is not None, "No lat/lon coordinates given."

            latlon_coords = rearrange(
                self.geo_embeddings(latlon_coords.to(dtype=x.dtype)),
                'b c -> b c 1 1',
            )

            _, _, height, width = x.shape
            x = torch.cat(
                (x, latlon_coords.expand(-1, -1, height, width)), dim=1
            )

        if tower_down is not None:
            tower_down = self.tower_conv(
                tower_down,
                size=decode_side.shape[-2:],
            )

            x = torch.cat((x, tower_down), dim=1)

        return self.res_conv(x)
