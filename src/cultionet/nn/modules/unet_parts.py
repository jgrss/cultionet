import enum
import typing as T

import torch
import torch.nn as nn

from cultionet.enums import AttentionTypes, ModelTypes, ResBlockTypes

from .attention import AttentionGate
from .convolution import (
    DoubleConv,
    PoolConv,
    PoolResidualConv,
    ResidualAConv,
    ResidualConv,
)
from .reshape import UpSample


class TowerUNetUpLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_blocks: int = 1,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
    ):
        super(TowerUNetUpLayer, self).__init__()

        self.up = UpSample()

        self.conv = ResidualConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )

    def forward(self, x: torch.Tensor, shape: tuple) -> torch.Tensor:
        if x.shape[-2:] != shape:
            x = self.up(
                x,
                size=shape,
                mode="bilinear",
            )

        return self.conv(x)


class TowerUNetBlock(nn.Module):
    def __init__(
        self,
        backbone_side_channels: int,
        backbone_down_channels: int,
        up_channels: int,
        out_channels: int,
        tower: bool = False,
        kernel_size: int = 3,
        num_blocks: int = 1,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
    ):
        super(TowerUNetBlock, self).__init__()

        self.up = UpSample()

        in_channels = (
            backbone_side_channels + backbone_down_channels + up_channels * 2
        )

        self.backbone_down_conv = nn.ConvTranspose2d(
            in_channels=backbone_down_channels,
            out_channels=backbone_down_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.down_conv = nn.ConvTranspose2d(
            in_channels=up_channels,
            out_channels=up_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        if tower:
            self.tower_conv = nn.ConvTranspose2d(
                in_channels=up_channels,
                out_channels=up_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            in_channels += up_channels

        self.conv = ResidualConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )

    def forward(
        self,
        backbone_side: torch.Tensor,
        backbone_down: torch.Tensor,
        side: torch.Tensor,
        down: torch.Tensor,
        down_tower: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backbone_down = self.up(
            self.backbone_down_conv(backbone_down),
            size=side.shape[-2:],
            mode="bilinear",
        )
        down = self.up(
            self.down_conv(down),
            size=side.shape[-2:],
            mode="bilinear",
        )

        x = torch.cat(
            (backbone_side, backbone_down, side, down),
            dim=1,
        )

        if down_tower is not None:
            down_tower = self.up(
                self.tower_conv(down_tower),
                size=side.shape[-2:],
                mode="bilinear",
            )
            x = torch.cat((x, down_tower), dim=1)

        return self.conv(x)


class ResELUNetPsiLayer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        side_in: T.Dict[str, int] = None,
        down_in: T.Dict[str, int] = None,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
    ):
        super(ResELUNetPsiLayer, self).__init__()

        self.up = UpSample()
        if dilations is None:
            dilations = [2]

        cat_channels = 0

        module_dict = {}

        if side_in is not None:
            for name, in_channels in side_in.items():
                module_dict[name] = ResidualConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilations[0],
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )
                cat_channels += out_channels

        if down_in is not None:
            for name, in_channels in down_in.items():
                module_dict[name] = ResidualConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilations[0],
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )
                cat_channels += out_channels

        self.module_dict = nn.ModuleDict(module_dict)

        self.final = ResidualConv(
            in_channels=cat_channels,
            out_channels=out_channels,
            dilation=dilations[0],
            attention_weights=attention_weights,
            activation_type=activation_type,
        )

    def forward(
        self,
        side: T.Dict[str, torch.Tensor],
        down: T.Dict[str, torch.Tensor],
        shape: tuple,
    ) -> torch.Tensor:
        out = []
        for name, x in side.items():
            layer = self.module_dict[name]
            assert x is not None, 'A tensor must be given.'
            out += [layer(x)]

        for name, x in down.items():
            layer = self.module_dict[name]
            x = self.up(
                x,
                size=shape,
                mode="bilinear",
            )
            out += [layer(x)]

        out = torch.cat(out, dim=1)
        out = self.final(out)

        return out


class ResELUNetPsiBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        side_in: dict,
        down_in: dict,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
    ):
        super(ResELUNetPsiBlock, self).__init__()

        self.dist_layer = ResELUNetPsiLayer(
            out_channels=out_channels,
            side_in=side_in['dist'],
            down_in=down_in['dist'],
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )
        self.edge_layer = ResELUNetPsiLayer(
            out_channels=out_channels,
            side_in=side_in['edge'],
            down_in=down_in['edge'],
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )
        self.mask_layer = ResELUNetPsiLayer(
            out_channels=out_channels,
            side_in=side_in['mask'],
            down_in=down_in['mask'],
            dilations=dilations,
            attention_weights=attention_weights,
            activation_type=activation_type,
        )

    def update_data(
        self,
        data_dict: T.Dict[str, T.Union[None, torch.Tensor]],
        data: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        out = data_dict.copy()
        for key, x in data_dict.items():
            if x is None:
                out[key] = data

        return out

    def forward(
        self,
        side: T.Dict[str, T.Union[None, torch.Tensor]],
        down: T.Dict[str, T.Union[None, torch.Tensor]],
        shape: tuple,
    ) -> dict:
        dist_out = self.dist_layer(
            side=side['dist'],
            down=down['dist'],
            shape=shape,
        )

        edge_out = self.edge_layer(
            side=self.update_data(side['edge'], dist_out),
            down=down['edge'],
            shape=shape,
        )

        mask_out = self.mask_layer(
            side=self.update_data(side['mask'], edge_out),
            down=down['mask'],
            shape=shape,
        )

        return {
            "dist": dist_out,
            "edge": edge_out,
            "mask": mask_out,
        }


class UNet3Connector(torch.nn.Module):
    """Connects layers in a UNet 3+ architecture."""

    def __init__(
        self,
        channels: T.List[int],
        up_channels: int,
        prev_backbone_channel_index: int,
        use_backbone: bool = True,
        is_side_stream: bool = True,
        n_pools: int = 0,
        n_prev_down: int = 0,
        n_stream_down: int = 0,
        prev_down_is_pooled: bool = False,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        init_point_conv: bool = False,
        dilations: T.Sequence[int] = None,
        model_type: str = ModelTypes.UNET,
        res_block_type: str = ResBlockTypes.RESA,
        activation_type: str = "SiLU",
    ):
        super(UNet3Connector, self).__init__()

        assert attention_weights in [
            "gate",
            AttentionTypes.FRACTAL,
            AttentionTypes.SPATIAL_CHANNEL,
        ], "Choose from 'gate', 'fractal', or 'spatial_channel' attention weights."

        assert model_type in (
            ModelTypes.UNET,
            ModelTypes.RESUNET,
            ModelTypes.RESUNET3PSI,
            ModelTypes.RESELUNETPSI,
        )
        assert res_block_type in (
            ResBlockTypes.RES,
            ResBlockTypes.RESA,
        )

        self.n_pools = n_pools
        self.n_prev_down = n_prev_down
        self.n_stream_down = n_stream_down
        self.attention_weights = attention_weights
        self.use_backbone = use_backbone
        self.is_side_stream = is_side_stream
        self.cat_channels = 0
        self.pool4_0 = None

        self.up = UpSample()

        if dilations is None:
            dilations = [2]

        # Pool layers
        if n_pools > 0:
            if n_pools == 3:
                pool_size = 8
            elif n_pools == 2:
                pool_size = 4
            else:
                pool_size = 2

            for n in range(0, n_pools):
                if model_type == ModelTypes.UNET:
                    setattr(
                        self,
                        f"pool_{n}",
                        PoolConv(
                            in_channels=channels[n],
                            out_channels=channels[0],
                            pool_size=pool_size,
                            double_dilation=dilations[0],
                            activation_type=activation_type,
                        ),
                    )
                else:
                    setattr(
                        self,
                        f"pool_{n}",
                        PoolResidualConv(
                            in_channels=channels[n],
                            out_channels=channels[0],
                            pool_size=pool_size,
                            dilations=dilations,
                            attention_weights=attention_weights,
                            activation_type=activation_type,
                            res_block_type=res_block_type,
                        ),
                    )
                pool_size = int(pool_size / 2)
                self.cat_channels += channels[0]
        if self.use_backbone:
            if model_type == ModelTypes.UNET:
                self.prev_backbone = DoubleConv(
                    in_channels=channels[prev_backbone_channel_index],
                    out_channels=up_channels,
                    init_point_conv=init_point_conv,
                    double_dilation=dilations[0],
                    activation_type=activation_type,
                )
            else:
                if res_block_type == ResBlockTypes.RES:
                    self.prev_backbone = ResidualConv(
                        in_channels=channels[prev_backbone_channel_index],
                        out_channels=up_channels,
                        dilation=dilations[0],
                        attention_weights=attention_weights,
                        activation_type=activation_type,
                    )
                else:
                    self.prev_backbone = ResidualAConv(
                        in_channels=channels[prev_backbone_channel_index],
                        out_channels=up_channels,
                        dilations=dilations,
                        attention_weights=attention_weights,
                        activation_type=activation_type,
                    )
            self.cat_channels += up_channels
        if self.is_side_stream:
            if model_type == ModelTypes.UNET:
                # Backbone, same level
                self.prev = DoubleConv(
                    in_channels=up_channels,
                    out_channels=up_channels,
                    init_point_conv=init_point_conv,
                    double_dilation=dilations[0],
                    activation_type=activation_type,
                )
            else:
                if res_block_type == ResBlockTypes.RES:
                    self.prev = ResidualConv(
                        in_channels=up_channels,
                        out_channels=up_channels,
                        dilation=dilations[0],
                        attention_weights=attention_weights,
                        activation_type=activation_type,
                    )
                else:
                    self.prev = ResidualAConv(
                        in_channels=up_channels,
                        out_channels=up_channels,
                        dilations=dilations,
                        attention_weights=attention_weights,
                        activation_type=activation_type,
                    )
            self.cat_channels += up_channels
        # Previous output, downstream
        if self.n_prev_down > 0:
            for n in range(0, self.n_prev_down):
                if model_type == ModelTypes.UNET:
                    setattr(
                        self,
                        f"prev_{n}",
                        DoubleConv(
                            in_channels=up_channels,
                            out_channels=up_channels,
                            init_point_conv=init_point_conv,
                            double_dilation=dilations[0],
                            activation_type=activation_type,
                        ),
                    )
                else:
                    if res_block_type == ResBlockTypes.RES:
                        setattr(
                            self,
                            f"prev_{n}",
                            ResidualConv(
                                in_channels=up_channels,
                                out_channels=up_channels,
                                dilation=dilations[0],
                                attention_weights=attention_weights,
                                activation_type=activation_type,
                            ),
                        )
                    else:
                        setattr(
                            self,
                            f"prev_{n}",
                            ResidualAConv(
                                in_channels=up_channels,
                                out_channels=up_channels,
                                dilations=dilations,
                                attention_weights=attention_weights,
                                activation_type=activation_type,
                            ),
                        )
                self.cat_channels += up_channels

        # Previous output, (same) downstream
        if self.n_stream_down > 0:
            for n in range(0, self.n_stream_down):
                in_stream_channels = up_channels
                if self.attention_weights is not None and (
                    self.attention_weights == "gate"
                ):
                    attention_module = AttentionGate(up_channels, up_channels)
                    setattr(self, f"attn_stream_{n}", attention_module)
                    in_stream_channels = up_channels * 2

                # All but the last inputs are pooled
                if prev_down_is_pooled and (n + 1 < self.n_stream_down):
                    in_stream_channels = channels[
                        prev_backbone_channel_index
                        + (self.n_stream_down - 1)
                        - n
                    ]

                if model_type == ModelTypes.UNET:
                    setattr(
                        self,
                        f"stream_{n}",
                        DoubleConv(
                            in_channels=in_stream_channels,
                            out_channels=up_channels,
                            init_point_conv=init_point_conv,
                            double_dilation=dilations[0],
                            activation_type=activation_type,
                        ),
                    )
                else:
                    if res_block_type == ResBlockTypes.RES:
                        setattr(
                            self,
                            f"stream_{n}",
                            ResidualConv(
                                in_channels=in_stream_channels,
                                out_channels=up_channels,
                                dilation=dilations[0],
                                attention_weights=attention_weights,
                                activation_type=activation_type,
                            ),
                        )
                    else:
                        setattr(
                            self,
                            f"stream_{n}",
                            ResidualAConv(
                                in_channels=in_stream_channels,
                                out_channels=up_channels,
                                dilations=dilations,
                                attention_weights=attention_weights,
                                activation_type=activation_type,
                            ),
                        )
                self.cat_channels += up_channels

        self.cat_channels += channels[0]
        if model_type == ModelTypes.UNET:
            self.conv4_0 = DoubleConv(
                in_channels=channels[4],
                out_channels=channels[0],
                init_point_conv=init_point_conv,
                activation_type=activation_type,
            )
            self.final = DoubleConv(
                in_channels=self.cat_channels,
                out_channels=up_channels,
                init_point_conv=init_point_conv,
                double_dilation=dilations[0],
                activation_type=activation_type,
            )
        else:
            if res_block_type == ResBlockTypes.RES:
                self.conv4_0 = ResidualConv(
                    in_channels=channels[4],
                    out_channels=channels[0],
                    dilation=dilations[0],
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )
                self.final = ResidualConv(
                    in_channels=self.cat_channels,
                    out_channels=up_channels,
                    dilation=dilations[0],
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )
            else:
                self.conv4_0 = ResidualAConv(
                    in_channels=channels[4],
                    out_channels=channels[0],
                    dilations=dilations,
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )
                self.final = ResidualAConv(
                    in_channels=self.cat_channels,
                    out_channels=up_channels,
                    dilations=dilations,
                    attention_weights=attention_weights,
                    activation_type=activation_type,
                )

    def forward(
        self,
        prev_same: T.List[T.Tuple[str, torch.Tensor]],
        x4_0: torch.Tensor = None,
        pools: T.List[torch.Tensor] = None,
        prev_down: T.List[torch.Tensor] = None,
        stream_down: T.List[torch.Tensor] = None,
    ):
        h: T.List[torch.Tensor] = []
        # Pooling layer of the backbone
        if pools is not None:
            assert self.n_pools == len(
                pools
            ), "There are no convolutions available for the pool layers."
            for n, x in zip(range(self.n_pools), pools):
                c = getattr(self, f"pool_{n}")
                h += [c(x)]
        # Up down layers from the previous head
        if prev_down is not None:
            assert self.n_prev_down == len(
                prev_down
            ), "There are no convolutions available for the previous downstream layers."
            for n, x in zip(range(self.n_prev_down), prev_down):
                c = getattr(self, f"prev_{n}")
                h += [
                    c(
                        self.up(
                            x, size=prev_same[0][1].shape[-2:], mode="bilinear"
                        )
                    )
                ]
        assert len(prev_same) == sum(
            [self.use_backbone, self.is_side_stream]
        ), "The previous same layers do not match the setup."
        # Previous same layers from the previous head
        for conv_name, prev_inputs in prev_same:
            c = getattr(self, conv_name)
            h += [c(prev_inputs)]
        if self.attention_weights is not None and (
            self.attention_weights == "gate"
        ):
            prev_same_hidden = h[-1].clone()
        # Previous down layers from the same head
        if stream_down is not None:
            assert self.n_stream_down == len(
                stream_down
            ), "There are no convolutions available for the downstream layers."
            for n, x in zip(range(self.n_stream_down), stream_down):
                if self.attention_weights is not None and (
                    self.attention_weights == "gate"
                ):
                    # Gate
                    g = self.up(
                        x, size=prev_same[0][1].shape[-2:], mode="bilinear"
                    )
                    c_attn = getattr(self, f"attn_stream_{n}")
                    # Attention gate
                    attn_out = c_attn(g, prev_same_hidden)
                    c = getattr(self, f"stream_{n}")
                    # Concatenate attention weights
                    h += [c(torch.cat([attn_out, g], dim=1))]
                else:
                    c = getattr(self, f"stream_{n}")
                    h += [
                        c(
                            self.up(
                                x,
                                size=prev_same[0][1].shape[-2:],
                                mode="bilinear",
                            )
                        )
                    ]

        # Lowest level
        if x4_0 is not None:
            x4_0_up = self.conv4_0(
                self.up(x4_0, size=prev_same[0][1].shape[-2:], mode="bilinear")
            )
            if self.pool4_0 is not None:
                h += [self.pool4_0(x4_0_up)]
            else:
                h += [x4_0_up]
        h = torch.cat(h, dim=1)
        h = self.final(h)

        return h


class UNet3P_3_1(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 3,1."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(UNet3P_3_1, self).__init__()

        self.conv = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=3,
            init_point_conv=init_point_conv,
            dilations=[double_dilation],
            model_type=ModelTypes.UNET,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        x2_0: torch.Tensor,
        x3_0: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(
            prev_same=[("prev_backbone", x3_0)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0,
        )

        return h


class UNet3P_2_2(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 2,2."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(UNet3P_2_2, self).__init__()

        self.conv = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            init_point_conv=init_point_conv,
            dilations=[double_dilation],
            model_type=ModelTypes.UNET,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        x2_0: torch.Tensor,
        h3_1: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(
            prev_same=[("prev_backbone", x2_0)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1],
        )

        return h


class UNet3P_1_3(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 1,3."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(UNet3P_1_3, self).__init__()

        self.conv = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            init_point_conv=init_point_conv,
            dilations=[double_dilation],
            model_type=ModelTypes.UNET,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        h2_2: torch.Tensor,
        h3_1: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(
            prev_same=[("prev_backbone", x1_0)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h3_1, h2_2],
        )

        return h


class UNet3P_0_4(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 0,4."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        double_dilation: int = 1,
        activation_type: str = "SiLU",
    ):
        super(UNet3P_0_4, self).__init__()

        self.up = UpSample()

        self.conv = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            init_point_conv=init_point_conv,
            dilations=[double_dilation],
            model_type=ModelTypes.UNET,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        h1_3: torch.Tensor,
        h2_2: torch.Tensor,
        h3_1: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        h = self.conv(
            prev_same=[("prev_backbone", x0_0)],
            x4_0=x4_0,
            stream_down=[h3_1, h2_2, h1_3],
        )

        return h


class UNet3_3_1(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 3,1."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
    ):
        super(UNet3_3_1, self).__init__()

        self.up = UpSample()

        # Distance stream connection
        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        # Edge stream connection
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        # Mask stream connection
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        x2_0: torch.Tensor,
        x3_0: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        # Distance logits
        h_dist = self.conv_dist(
            prev_same=[("prev_backbone", x3_0)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0,
        )
        # Output distance logits pass to edge layer
        h_edge = self.conv_edge(
            prev_same=[("prev_backbone", x3_0), ("prev", h_dist)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0,
        )
        # Output edge logits pass to mask layer
        h_mask = self.conv_mask(
            prev_same=[("prev_backbone", x3_0), ("prev", h_edge)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0,
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class UNet3_2_2(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 2,2."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
    ):
        super(UNet3_2_2, self).__init__()

        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        x2_0: torch.Tensor,
        h3_1_dist: torch.Tensor,
        h3_1_edge: torch.Tensor,
        h3_1_mask: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        h_dist = self.conv_dist(
            prev_same=[("prev_backbone", x2_0)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_dist],
        )
        h_edge = self.conv_edge(
            prev_same=[("prev_backbone", x2_0), ("prev", h_dist)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_edge],
        )
        h_mask = self.conv_mask(
            prev_same=[("prev_backbone", x2_0), ("prev", h_edge)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_mask],
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class UNet3_1_3(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 1,3."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
    ):
        super(UNet3_1_3, self).__init__()

        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        x1_0: torch.Tensor,
        h2_2_dist: torch.Tensor,
        h3_1_dist: torch.Tensor,
        h2_2_edge: torch.Tensor,
        h3_1_edge: torch.Tensor,
        h2_2_mask: torch.Tensor,
        h3_1_mask: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        h_dist = self.conv_dist(
            prev_same=[("prev_backbone", x1_0)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h3_1_dist, h2_2_dist],
        )
        h_edge = self.conv_edge(
            prev_same=[("prev_backbone", x1_0), ("prev", h_dist)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h3_1_edge, h2_2_edge],
        )
        h_mask = self.conv_mask(
            prev_same=[("prev_backbone", x1_0), ("prev", h_edge)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h3_1_mask, h2_2_mask],
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class UNet3_0_4(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 0,4."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        init_point_conv: bool = False,
        dilations: T.Sequence[int] = None,
        activation_type: str = "SiLU",
    ):
        super(UNet3_0_4, self).__init__()

        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            init_point_conv=init_point_conv,
            dilations=dilations,
            activation_type=activation_type,
        )

    def forward(
        self,
        x0_0: torch.Tensor,
        h1_3_dist: torch.Tensor,
        h2_2_dist: torch.Tensor,
        h3_1_dist: torch.Tensor,
        h1_3_edge: torch.Tensor,
        h2_2_edge: torch.Tensor,
        h3_1_edge: torch.Tensor,
        h1_3_mask: torch.Tensor,
        h2_2_mask: torch.Tensor,
        h3_1_mask: torch.Tensor,
        x4_0: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:
        h_dist = self.conv_dist(
            prev_same=[("prev_backbone", x0_0)],
            x4_0=x4_0,
            stream_down=[h3_1_dist, h2_2_dist, h1_3_dist],
        )
        h_edge = self.conv_edge(
            prev_same=[("prev_backbone", x0_0), ("prev", h_dist)],
            x4_0=x4_0,
            stream_down=[h3_1_edge, h2_2_edge, h1_3_edge],
        )
        h_mask = self.conv_mask(
            prev_same=[("prev_backbone", x0_0), ("prev", h_edge)],
            x4_0=x4_0,
            stream_down=[h3_1_mask, h2_2_mask, h1_3_mask],
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


def get_prev_list(
    use_backbone: bool,
    x: torch.Tensor,
    prev_same: T.List[tuple],
) -> T.List[tuple]:
    prev = [
        (
            "prev",
            x,
        )
    ]
    if use_backbone:
        prev += prev_same

    return prev


class ResUNet3_3_1(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 3,1."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        n_pools: int = 3,
        use_backbone: bool = True,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RESA,
        model_type: str = ModelTypes.RESUNET,
    ):
        super(ResUNet3_3_1, self).__init__()

        self.use_backbone = use_backbone
        self.up = UpSample()

        # Distance stream connection
        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=n_pools,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        # Edge stream connection
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=3,
            n_pools=n_pools,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        # Mask stream connection
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=3,
            n_pools=n_pools,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )

    def forward(
        self,
        side: torch.Tensor,
        down: torch.Tensor,
        pools: T.Sequence[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:
        prev_same = [
            (
                "prev_backbone",
                side,
            )
        ]
        # Distance logits
        h_dist = self.conv_dist(
            prev_same=prev_same,
            pools=pools,
            x4_0=down,
        )
        # Output distance logits pass to edge layer
        h_edge = self.conv_edge(
            prev_same=get_prev_list(self.use_backbone, h_dist, prev_same),
            pools=pools,
            x4_0=down,
        )
        # Output edge logits pass to mask layer
        h_mask = self.conv_mask(
            prev_same=get_prev_list(self.use_backbone, h_edge, prev_same),
            pools=pools,
            x4_0=down,
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class ResUNet3_2_2(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 2,2."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        n_pools: int = 2,
        use_backbone: bool = True,
        n_stream_down: int = 1,
        prev_down_is_pooled: bool = False,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RESA,
        model_type: str = ModelTypes.RESUNET,
    ):
        super(ResUNet3_2_2, self).__init__()

        self.use_backbone = use_backbone
        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=2,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=False,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=2,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=False,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )

    def forward(
        self,
        side: torch.Tensor,
        dist_down: T.Sequence[torch.Tensor],
        edge_down: T.Sequence[torch.Tensor],
        mask_down: T.Sequence[torch.Tensor],
        down: torch.Tensor = None,
        pools: T.Sequence[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:
        prev_same = [
            (
                "prev_backbone",
                side,
            )
        ]

        h_dist = self.conv_dist(
            prev_same=prev_same,
            pools=pools,
            x4_0=down,
            stream_down=dist_down,
        )
        h_edge = self.conv_edge(
            prev_same=get_prev_list(self.use_backbone, h_dist, prev_same),
            pools=pools,
            x4_0=down,
            stream_down=edge_down,
        )
        h_mask = self.conv_mask(
            prev_same=get_prev_list(self.use_backbone, h_edge, prev_same),
            pools=pools,
            x4_0=down,
            stream_down=mask_down,
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class ResUNet3_1_3(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 1,3."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        n_pools: int = 1,
        use_backbone: bool = True,
        n_stream_down: int = 2,
        prev_down_is_pooled: bool = False,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
        res_block_type: enum = ResBlockTypes.RESA,
        model_type: str = ModelTypes.RESUNET,
    ):
        super(ResUNet3_1_3, self).__init__()

        self.use_backbone = use_backbone
        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=1,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=1,
            n_pools=n_pools,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )

    def forward(
        self,
        side: torch.Tensor,
        dist_down: T.Sequence[torch.Tensor],
        edge_down: T.Sequence[torch.Tensor],
        mask_down: T.Sequence[torch.Tensor],
        down: torch.Tensor = None,
        pools: T.Sequence[torch.Tensor] = None,
    ) -> T.Dict[str, torch.Tensor]:
        prev_same = [
            (
                "prev_backbone",
                side,
            )
        ]

        h_dist = self.conv_dist(
            prev_same=prev_same,
            pools=pools,
            x4_0=down,
            stream_down=dist_down,
        )
        h_edge = self.conv_edge(
            prev_same=get_prev_list(self.use_backbone, h_dist, prev_same),
            pools=pools,
            x4_0=down,
            stream_down=edge_down,
        )
        h_mask = self.conv_mask(
            prev_same=get_prev_list(self.use_backbone, h_edge, prev_same),
            pools=pools,
            x4_0=down,
            stream_down=mask_down,
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }


class ResUNet3_0_4(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 0,4."""

    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        n_stream_down: int = 3,
        use_backbone: bool = True,
        prev_down_is_pooled: bool = False,
        dilations: T.Sequence[int] = None,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        activation_type: str = "SiLU",
        res_block_type: str = ResBlockTypes.RESA,
        model_type: str = ModelTypes.RESUNET,
    ):
        super(ResUNet3_0_4, self).__init__()

        self.use_backbone = use_backbone
        self.up = UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=True,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=0,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            use_backbone=use_backbone,
            is_side_stream=True,
            prev_backbone_channel_index=0,
            n_stream_down=n_stream_down,
            prev_down_is_pooled=prev_down_is_pooled,
            dilations=dilations,
            attention_weights=attention_weights,
            model_type=model_type,
            res_block_type=res_block_type,
            activation_type=activation_type,
        )

    def forward(
        self,
        side: torch.Tensor,
        dist_down: T.Sequence[torch.Tensor],
        edge_down: T.Sequence[torch.Tensor],
        mask_down: T.Sequence[torch.Tensor],
        down: torch.Tensor = None,
    ) -> T.Dict[str, torch.Tensor]:
        prev_same = [
            (
                "prev_backbone",
                side,
            )
        ]

        h_dist = self.conv_dist(
            prev_same=prev_same,
            x4_0=down,
            stream_down=dist_down,
        )
        h_edge = self.conv_edge(
            prev_same=get_prev_list(self.use_backbone, h_dist, prev_same),
            x4_0=down,
            stream_down=edge_down,
        )
        h_mask = self.conv_mask(
            prev_same=get_prev_list(self.use_backbone, h_edge, prev_same),
            x4_0=down,
            stream_down=mask_down,
        )

        return {
            "dist": h_dist,
            "edge": h_edge,
            "mask": h_mask,
        }
