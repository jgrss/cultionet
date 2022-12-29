import typing as T

from .base_layers import (
    PoolResidualConv,
    ResidualConv
)
from . import model_utils

import torch


class ResUNetConnector(torch.nn.Module):
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
        attention: bool = False
    ):
        super(ResUNetConnector, self).__init__()

        self.n_pools = n_pools
        self.n_prev_down = n_prev_down
        self.n_stream_down = n_stream_down
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
        # Backbone, same level
        self.prev_backbone = ResidualConv(
            channels[prev_backbone_channel_index],
            up_channels,
            dilations=dilations
        )
        self.cat_channels += up_channels
        # Previous output, same level
        if is_side_stream:
            self.prev = ResidualConv(
                up_channels,
                up_channels,
                dilations=dilations
            )
            self.cat_channels += up_channels
        # Previous output, downstream
        if n_prev_down > 0:
            for n in range(0, n_prev_down):
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
        if n_stream_down > 0:
            for n in range(0, n_stream_down):
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

        self.conv = ResidualConv(
            self.cat_channels,
            up_channels,
            dilations=dilations
        )

    def forward(
        self,
        prev_same: T.List[torch.Tensor],
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
        h += [self.prev_backbone(prev_same[0])]
        if len(prev_same) > 1:
            h += [self.prev(prev_same[1])]
        if prev_down is not None:
            for n, x in zip(range(self.n_prev_down), prev_down):
                c = getattr(self, f'prev_{n}')
                h += [
                    c(self.up(x, size=prev_same[0].shape[-2:]))
                ]
        if stream_down is not None:
            for n, x in zip(range(self.n_stream_down), stream_down):
                c = getattr(self, f'stream_{n}')
                h += [
                    c(self.up(x, size=prev_same[0].shape[-2:]))
                ]
        h += [
            self.conv4_0(self.up(x4_0, size=prev_same[0].shape[-2:]))
        ]
        h = torch.cat(h, dim=1)
        h = self.conv(h)

        return h


class ResUNETConnect3_1(torch.nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNETConnect3_1, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=3,
            n_pools=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=3,
            n_pools=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetConnector(
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
        h_dist = self.conv_dist(
            prev_same=[x3_0],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        h_edge = self.conv_edge(
            prev_same=[x3_0, h_dist],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        h_mask = self.conv_mask(
            prev_same=[x3_0, h_edge],
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNETConnect2_2(torch.nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNETConnect2_2, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=2,
            n_pools=2,
            n_stream_down=1,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetConnector(
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
            prev_same=[x2_0],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[x2_0, h_dist],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[x2_0, h_edge],
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            stream_down=[h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNETConnect1_3(torch.nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNETConnect1_3, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=1,
            n_pools=1,
            n_stream_down=2,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetConnector(
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
            prev_same=[x1_0],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[x1_0, h_dist],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[x1_0, h_edge],
            pools=[x0_0],
            x4_0=x4_0,
            stream_down=[h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }


class ResUNETConnect0_4(torch.nn.Module):
    def __init__(
        self,
        channels: T.Sequence[int],
        up_channels: int,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNETConnect0_4, self).__init__()

        self.up = model_utils.UpSample()

        self.conv_dist = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            is_side_stream=False,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_edge = ResUNetConnector(
            channels=channels,
            up_channels=up_channels,
            prev_backbone_channel_index=0,
            n_stream_down=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetConnector(
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
            prev_same=[x0_0],
            x4_0=x4_0,
            stream_down=[h1_3_dist, h2_2_dist, h3_1_dist]
        )
        h_edge = self.conv_edge(
            prev_same=[x0_0, h_dist],
            x4_0=x4_0,
            stream_down=[h1_3_edge, h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=[x0_0, h_edge],
            x4_0=x4_0,
            stream_down=[h1_3_mask, h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }
