import typing as T

from .base_layers import (
    DoubleConv,
    PoolConv,
    PoolResidualConv,
    ResidualConv
)
from . import model_utils
from .rcsb import RCSB

import torch


class UNet3Connector(torch.nn.Module):
    """Connects layers in a UNet 3+ architecture
    """
    def __init__(
        self,
        channels: T.List[int],
        up_channels: int,
        prev_backbone_channel_index: int,
        is_side_stream: bool = True,
        n_pools: int = 0,
        n_prev_down: int = 0,
        n_stream_down: int = 0,
        attention: bool = False,
        attention_weights: str = 'gate'
    ):
        super(UNet3Connector, self).__init__()

        assert attention_weights in ['gate', 'fractal'], \
            "Choose from 'gate' or 'fractal' attention weights."

        self.n_pools = n_pools
        self.n_prev_down = n_prev_down
        self.n_stream_down = n_stream_down
        self.attention = attention
        self.cat_channels = 0

        self.up = model_utils.UpSample()

        # Pool layers
        if n_pools > 0:
            if n_pools == 3:
                pool_size = 8
            elif n_pools == 2:
                pool_size = 4
            else:
                pool_size = 2

            for n in range(0, n_pools):
                setattr(
                    self,
                    f'pool_{n}',
                    PoolConv(
                        channels[n],
                        channels[0],
                        pool_size=pool_size
                    )
                )
                pool_size = int(pool_size / 2)
                self.cat_channels += channels[0]
        if is_side_stream:
            self.prev = DoubleConv(
                up_channels,
                up_channels
            )
        else:
            # Backbone, same level
            self.prev_backbone = DoubleConv(
                channels[prev_backbone_channel_index],
                up_channels
            )
        if not self.attention or (self.n_stream_down == 0):
            self.cat_channels += up_channels
        # Previous output, downstream
        if self.n_prev_down > 0:
            for n in range(0, self.n_prev_down):
                setattr(
                    self,
                    f'prev_{n}',
                    DoubleConv(
                        up_channels,
                        up_channels
                    )
                )
                self.cat_channels += up_channels
        # Previous output, (same) downstream
        if self.n_stream_down > 0:
            for n in range(0, self.n_stream_down):
                in_stream_channels = up_channels
                if self.attention:
                    if attention_weights == 'gate':
                        attention_module = AttentionGate(up_channels, up_channels)
                    else:
                        attention_module = FractalAttention(up_channels, up_channels)
                    setattr(
                        self,
                        f'attn_stream_{n}',
                        attention_module
                    )
                    in_stream_channels = up_channels * 2
                setattr(
                    self,
                    f'stream_{n}',
                    DoubleConv(
                        in_stream_channels,
                        up_channels
                    )
                )
                self.cat_channels += up_channels

        self.conv4_0 = DoubleConv(
            channels[4],
            channels[0]
        )
        self.cat_channels += channels[0]

        self.final = DoubleConv(
            self.cat_channels,
            up_channels
        )

    def forward(
        self,
        prev_same: T.List[T.Tuple[str, torch.Tensor]],
        x4_0: torch.Tensor,
        pools: T.List[torch.Tensor] = None,
        prev_down: T.List[torch.Tensor] = None,
        stream_down: T.List[torch.Tensor] = None
    ):
        h = []
        if pools is not None:
            for n, x in zip(range(self.n_pools), pools):
                c = getattr(self, f'pool_{n}')
                h += [c(x)]
        if not self.attention or stream_down is None:
            for conv_name, prev_inputs in prev_same:
                c = getattr(self, conv_name)
                h += [c(prev_inputs)]
        if prev_down is not None:
            for n, x in zip(range(self.n_prev_down), prev_down):
                c = getattr(self, f'prev_{n}')
                h += [
                    c(self.up(x, size=prev_same[0][1].shape[-2:]))
                ]
        if stream_down is not None:
            for n, x in zip(range(self.n_stream_down), stream_down):
                if self.attention:
                    # Gate
                    g = self.up(x, size=prev_same[0][1].shape[-2:])
                    c_attn = getattr(self, f'attn_stream_{n}')
                    conv_name, prev_inputs = prev_same[0]
                    c = getattr(self, conv_name)
                    # Attention gate
                    attn_out = c_attn(g, c(prev_inputs))
                    c = getattr(self, f'stream_{n}')
                    # Concatenate attention weights
                    h += [c(torch.cat([attn_out, g], dim=1))]
                else:
                    c = getattr(self, f'stream_{n}')
                    h += [
                        c(self.up(x, size=prev_same[0][1].shape[-2:]))
                    ]

        h += [
            self.conv4_0(self.up(x4_0, size=prev_same[0][1].shape[-2:]))
        ]
        h = torch.cat(h, dim=1)
        h = self.final(h)

        return h


class UNet3_3_1(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 3,1
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        attention: bool = False,
        attention_weights: str = 'gate',
        depthwise_conv: bool = False
    ):
        super(UNet3_3_1, self).__init__()

        self.up = model_utils.UpSample()

        # Distance stream connection
        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=3,
            depthwise_conv=depthwise_conv
        )
        # Edge stream connection
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
        )
        # Mask stream connection
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
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
            prev_same=[('prev_backbone', x3_0)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        # Output distance logits pass to edge layer
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        # Output edge logits pass to mask layer
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class UNet3_2_2(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 2,2
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        attention: bool = False,
        attention_weights: str = 'gate',
        depthwise_conv: bool = False
    ):
        super(UNet3_2_2, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            depthwise_conv=depthwise_conv
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
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
            prev_same=[('prev_backbone', x2_0)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            prev_down=[h3_1_dist],
            stream_down=[h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            prev_down=[h3_1_edge],
            stream_down=[h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class UNet3_1_3(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 1,3
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        attention: bool = False,
        attention_weights: str = 'gate',
        depthwise_conv: bool = False
    ):
        super(UNet3_1_3, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            depthwise_conv=depthwise_conv
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
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
            prev_same=[('prev_backbone', x1_0)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0],
            x4_0=x4_0,
            prev_down=[h2_2_dist, h3_1_dist],
            stream_down=[h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0],
            x4_0=x4_0,
            prev_down=[h2_2_edge, h3_1_edge],
            stream_down=[h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class UNet3_0_4(torch.nn.Module):
    """UNet 3+ connection from backbone to upstream 0,4
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        attention: bool = False,
        attention_weights: str = 'gate',
        depthwise_conv: bool = False
    ):
        super(UNet3_0_4, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            depthwise_conv=depthwise_conv
        )
        self.conv_edge = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
        )
        self.conv_mask = UNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            attention=attention,
            attention_weights=attention_weights,
            depthwise_conv=depthwise_conv
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
            prev_same=[('prev_backbone', x0_0)],
            x4_0=x4_0,
            stream_down=[h1_3_dist, h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            x4_0=x4_0,
            prev_down=[h1_3_dist, h2_2_dist, h3_1_dist],
            stream_down=[h1_3_edge, h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            x4_0=x4_0,
            prev_down=[h1_3_edge, h2_2_edge, h3_1_edge],
            stream_down=[h1_3_mask, h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNet3Connector(torch.nn.Module):
    """Connects layers in a Residual UNet 3+ architecture
    """
    def __init__(
        self,
        channels: T.List[int],
        up_channels: int,
        prev_backbone_channel_index: int,
        is_side_stream: bool = True,
        n_pools: int = 0,
        n_prev_down: int = 0,
        n_stream_down: int = 0,
        dilations: T.List[int] = None,
        attention: bool = False,
        rcsb: bool = False,
        rcsb_recursion: int = 3,
        rcsb_g: int = 1
    ):
        super(ResUNet3Connector, self).__init__()

        self.n_pools = n_pools
        self.n_prev_down = n_prev_down
        self.n_stream_down = n_stream_down
        self.attention = attention
        self.rcsb = rcsb
        self.cat_channels = 0

        self.up = model_utils.UpSample()

        # Pool layers
        if n_pools > 0:
            if n_pools == 3:
                pool_size = 8
            elif n_pools == 2:
                pool_size = 4
            else:
                pool_size = 2

            for n in range(0, n_pools):
                setattr(
                    self,
                    f'pool_{n}',
                    PoolResidualConv(
                        channels[n],
                        channels[0],
                        pool_size=pool_size,
                        dilations=dilations
                    )
                )
                pool_size = int(pool_size / 2)
                self.cat_channels += channels[0]
        # Previous output, same level
        if is_side_stream:
            self.prev = ResidualConv(
                up_channels,
                up_channels,
                dilations=dilations
            )
        else:
            # Backbone, same level
            self.prev_backbone = ResidualConv(
                channels[prev_backbone_channel_index],
                up_channels,
                dilations=dilations
            )
        if not self.attention or (self.n_stream_down == 0):
            self.cat_channels += up_channels
        # Previous output, downstream
        if self.n_prev_down > 0:
            for n in range(0, self.n_prev_down):
                setattr(
                    self,
                    f'prev_{n}',
                    ResidualConv(
                        up_channels,
                        up_channels,
                        dilations=dilations
                    )
                )
                self.cat_channels += up_channels
        # Previous output, (same) downstream
        if self.n_stream_down > 0:
            for n in range(0, self.n_stream_down):
                setattr(
                    self,
                    f'stream_{n}',
                    ResidualConv(
                        up_channels,
                        up_channels,
                        fractal_attention=attention,
                        dilations=dilations
                    )
                )
                self.cat_channels += up_channels
        self.conv4_0 = ResidualConv(
            channels[4],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.cat_channels += channels[0]

        if self.rcsb:
            self.rcsb_net = RCSB(
                in_channels=self.cat_channels,
                recursion=rcsb_recursion,
                g=rcsb_g
            )
        self.final = ResidualConv(
            self.cat_channels,
            up_channels,
            dilations=dilations
        )

    def forward(
        self,
        prev_same: T.List[T.Tuple[str, torch.Tensor]],
        x4_0: torch.Tensor,
        pools: T.List[torch.Tensor] = None,
        prev_down: T.List[torch.Tensor] = None,
        stream_down: T.List[torch.Tensor] = None
    ):
        h = []
        if pools is not None:
            for n, x in zip(range(self.n_pools), pools):
                c = getattr(self, f'pool_{n}')
                h += [c(x)]
        if not self.attention or stream_down is None:
            for conv_name, prev_inputs in prev_same:
                c = getattr(self, conv_name)
                h += [c(prev_inputs)]
        if prev_down is not None:
            for n, x in zip(range(self.n_prev_down), prev_down):
                c = getattr(self, f'prev_{n}')
                h += [
                    c(self.up(x, size=prev_same[0][1].shape[-2:]))
                ]
        if stream_down is not None:
            for n, x in zip(range(self.n_stream_down), stream_down):
                c = getattr(self, f'stream_{n}')
                h += [
                    c(self.up(x, size=prev_same[0][1].shape[-2:]))
                ]
        h += [
            self.conv4_0(self.up(x4_0, size=prev_same[0][1].shape[-2:]))
        ]
        h = torch.cat(h, dim=1)
        if self.rcsb:
            h = self.rcsb_net(h)
        h = self.final(h)

        return h


class ResUNet3_3_1(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 3,1
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False,
        rcsb: bool = False
    ):
        super(ResUNet3_3_1, self).__init__()

        self.up = model_utils.UpSample()

        # Distance stream connection
        self.conv_dist = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=3,
            dilations=dilations,
            attention=attention
        )
        # Edge stream connection
        self.conv_edge = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            dilations=dilations,
            attention=attention,
            rcsb=rcsb,
            rcsb_recursion=3,
            rcsb_g=1
        )
        # Mask stream connection
        self.conv_mask = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            dilations=dilations,
            attention=attention
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
            prev_same=[('prev_backbone', x3_0)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        # Output distance logits pass to edge layer
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        # Output edge logits pass to mask layer
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNet3_2_2(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 2,2
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False,
        rcsb: bool = False
    ):
        super(ResUNet3_2_2, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            dilations=dilations,
            attention=attention,
            rcsb=rcsb,
            rcsb_recursion=3,
            rcsb_g=1
        )
        self.conv_mask = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            dilations=dilations,
            attention=attention
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
            prev_same=[('prev_backbone', x2_0)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNet3_1_3(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 1,3
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False,
        rcsb: bool = False
    ):
        super(ResUNet3_1_3, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            dilations=dilations,
            attention=attention,
            rcsb=rcsb,
            rcsb_recursion=3,
            rcsb_g=1
        )
        self.conv_mask = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            dilations=dilations,
            attention=attention
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
            prev_same=[('prev_backbone', x1_0)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNet3_0_4(torch.nn.Module):
    """Residual UNet 3+ connection from backbone to upstream 0,4
    """
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False,
        rcsb: bool = False
    ):
        super(ResUNet3_0_4, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            dilations=dilations,
            attention=attention,
            rcsb=rcsb,
            rcsb_recursion=3,
            rcsb_g=1
        )
        self.conv_mask = ResUNet3Connector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            dilations=dilations,
            attention=attention
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
            prev_same=[('prev_backbone', x0_0)],
            x4_0=x4_0,
            stream_down=[h1_3_dist, h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[('prev', h_dist)],
            x4_0=x4_0,
            stream_down=[h1_3_edge, h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[('prev', h_edge)],
            x4_0=x4_0,
            stream_down=[h1_3_mask, h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }
