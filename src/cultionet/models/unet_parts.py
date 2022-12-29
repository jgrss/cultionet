import typing as T

from .base_layers import (
    PoolResidualConv,
    ResidualConv
)
from . import model_utils

import torch


class ResUNetUp(torch.nn.Module):
    def __init__(
        self,
        channels: T.List[int],
        up_channels: int,
        n_pools: int = 0,
        n_prev_down: int = 0,
        n_stream_down: int = 0,
        dilations: T.List[int] = None,
        attention: bool = False
    ):
        super(ResUNetUp, self).__init__()

        self.n_pools = n_pools
        self.n_prev_down = n_prev_down
        self.n_stream_down = n_stream_down
        self.cat_channels = 0

        self.up = model_utils.UpSample()

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
        self.prev = ResidualConv(
            up_channels, up_channels,
            fractal_attention=attention,
            dilations=dilations
        )
        self.cat_channels += up_channels
        if n_prev_down > 0:
            for n in range(0, n_prev_down):
                setattr(
                    self,
                    f'prev_{n}',
                    ResidualConv(
                        up_channels, up_channels,
                        fractal_attention=attention,
                        dilations=dilations
                    )
                )
                self.cat_channels += up_channels
        if n_stream_down > 0:
            for n in range(0, n_stream_down):
                setattr(
                    self,
                    f'stream_{n}',
                    ResidualConv(
                        up_channels, up_channels,
                        fractal_attention=attention,
                        dilations=dilations
                    )
                )
                self.cat_channels += up_channels
        self.conv4_0_prev = ResidualConv(
            channels[4], channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv4_0_stream = ResidualConv(
            channels[4], channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.cat_channels += channels[0] * 2

        self.conv = ResidualConv(
            self.cat_channels, up_channels,
            fractal_attention=attention,
            dilations=dilations
        )

    def forward(
        self,
        prev_same: torch.Tensor,
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
        x_prev_same = self.prev(prev_same)
        h += [x_prev_same]
        if prev_down is not None:
            for n, x in zip(range(self.n_prev_down), prev_down):
                c = getattr(self, f'prev_{n}')
                h += [c(self.up(x, size=prev_same.shape[-2:]))]
        if stream_down is not None:
            for n, x in zip(range(self.n_stream_down), stream_down):
                c = getattr(self, f'stream_{n}')
                h += [c(self.up(x, size=prev_same.shape[-2:]))]
        x4_0_prev = self.conv4_0_prev(self.up(x4_0, size=prev_same.shape[-2:]))
        x4_0_stream = self.conv4_0_stream(self.up(x4_0, size=prev_same.shape[-2:]))
        h += [x4_0_prev, x4_0_stream]
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

        self.conv0_0_dist = PoolResidualConv(
            channels[0],
            channels[0],
            pool_size=8,
            dilations=dilations
        )
        self.conv1_0_dist = PoolResidualConv(
            channels[1],
            channels[0],
            pool_size=4,
            dilations=dilations
        )
        self.conv2_0_dist = PoolResidualConv(
            channels[2],
            channels[0],
            pool_size=2,
            dilations=dilations
        )
        self.conv3_0_dist = ResidualConv(
            channels[3],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv4_0_dist = ResidualConv(
            channels[4],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_dist = ResidualConv(
            up_channels,
            up_channels,
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_edge = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_pools=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
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
        h0_0_dist_logits = self.conv0_0_dist(x0_0) # 0,0 to 3,1
        h1_0_dist_logits = self.conv1_0_dist(x1_0) # 1,0 to 3,1
        h2_0_dist_logits = self.conv2_0_dist(x2_0) # 2,0 to 3,1
        h3_0_dist_logits = self.conv3_0_dist(x3_0) # 3,0 to 3,1
        h4_0_dist_logits = self.conv4_0_dist(self.up(x4_0, size=x3_0.shape[-2:]))
        h_dist_logits = torch.cat(
            [
                h0_0_dist_logits,
                h1_0_dist_logits,
                h2_0_dist_logits,
                h3_0_dist_logits,
                h4_0_dist_logits
            ],
            dim=1
        )
        h_dist = self.conv_dist(h_dist_logits)
        h_edge = self.conv_edge(
            prev_same=h_dist,
            pools=[x0_0, x1_0, x2_0],
            x4_0=x4_0
        )
        h_mask = self.conv_mask(
            prev_same=h_edge,
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

        self.conv0_0_dist = PoolResidualConv(
            channels[0],
            channels[0],
            pool_size=4,
            dilations=dilations
        )
        self.conv1_0_dist = PoolResidualConv(
            channels[1],
            channels[0],
            pool_size=2,
            dilations=dilations
        )
        self.conv2_0_dist = ResidualConv(
            channels[2],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv3_1_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv4_0_dist = ResidualConv(
            channels[4],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_dist = ResidualConv(
            up_channels,
            up_channels,
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_edge = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_pools=2,
            n_prev_down=1,
            n_stream_down=1,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_pools=2,
            n_prev_down=1,
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
        h0_0_dist_logits = self.conv0_0_dist(x0_0)
        h1_0_dist_logits = self.conv1_0_dist(x1_0)
        h2_0_dist_logits = self.conv2_0_dist(x2_0)
        h3_1_dist_logits = self.conv3_1_dist(self.up(h3_1_dist, size=x2_0.shape[-2:]))
        h4_0_dist_logits = self.conv4_0_dist(self.up(x4_0, size=x2_0.shape[-2:]))
        h_dist_logits = torch.cat(
            [
                h0_0_dist_logits,
                h1_0_dist_logits,
                h2_0_dist_logits,
                h3_1_dist_logits,
                h4_0_dist_logits
            ],
            dim=1
        )
        h_dist = self.conv_dist(h_dist_logits)
        h_edge = self.conv_edge(
            prev_same=h_dist,
            pools=[x0_0, x1_0],
            x4_0=x4_0,
            prev_down=[h3_1_dist],
            stream_down=[h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=h_edge,
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

        self.conv0_0_dist = PoolResidualConv(
            channels[0],
            channels[0],
            pool_size=2,
            dilations=dilations
        )
        self.conv1_0_dist = ResidualConv(
            channels[1],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv2_2_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv3_1_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv4_0_dist = ResidualConv(
            channels[4],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_dist = ResidualConv(
            up_channels,
            up_channels,
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_edge = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_pools=1,
            n_prev_down=2,
            n_stream_down=2,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_pools=1,
            n_prev_down=2,
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
        h0_0_dist_logits = self.conv0_0_dist(x0_0)
        h1_0_dist_logits = self.conv1_0_dist(x1_0)
        h2_2_dist_logits = self.conv2_2_dist(self.up(h2_2_dist, size=x1_0.shape[-2:]))
        h3_1_dist_logits = self.conv3_1_dist(self.up(h3_1_dist, size=x1_0.shape[-2:]))
        h4_0_dist_logits = self.conv4_0_dist(self.up(x4_0, size=x1_0.shape[-2:]))
        h_dist_logits = torch.cat(
            [
                h0_0_dist_logits,
                h1_0_dist_logits,
                h2_2_dist_logits,
                h3_1_dist_logits,
                h4_0_dist_logits
            ],
            dim=1
        )
        h_dist = self.conv_dist(h_dist_logits)
        h_edge = self.conv_edge(
            prev_same=h_dist,
            pools=[x0_0],
            x4_0=x4_0,
            prev_down=[h2_2_dist, h3_1_dist],
            stream_down=[h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=h_edge,
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

        self.conv0_0_dist = ResidualConv(
            channels[0],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv1_3_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv2_2_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv3_1_dist = ResidualConv(
            up_channels,
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv4_0_dist = ResidualConv(
            channels[4],
            channels[0],
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_dist = ResidualConv(
            up_channels,
            up_channels,
            fractal_attention=attention,
            dilations=dilations
        )
        self.conv_edge = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_prev_down=3,
            n_stream_down=3,
            dilations=dilations,
            attention=attention
        )
        self.conv_mask = ResUNetUp(
            channels=channels,
            up_channels=up_channels,
            n_prev_down=3,
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
        h0_0_dist_logits = self.conv0_0_dist(x0_0)
        h1_3_dist_logits = self.conv2_2_dist(self.up(h1_3_dist, size=x0_0.shape[-2:]))
        h2_2_dist_logits = self.conv2_2_dist(self.up(h2_2_dist, size=x0_0.shape[-2:]))
        h3_1_dist_logits = self.conv3_1_dist(self.up(h3_1_dist, size=x0_0.shape[-2:]))
        h4_0_dist_logits = self.conv4_0_dist(self.up(x4_0, size=x0_0.shape[-2:]))
        h_dist_logits = torch.cat(
            [
                h0_0_dist_logits,
                h1_3_dist_logits,
                h2_2_dist_logits,
                h3_1_dist_logits,
                h4_0_dist_logits
            ],
            dim=1
        )
        h_dist = self.conv_dist(h_dist_logits)
        h_edge = self.conv_edge(
            prev_same=h_dist,
            x4_0=x4_0,
            prev_down=[h1_3_dist, h2_2_dist, h3_1_dist],
            stream_down=[h1_3_edge, h2_2_edge, h3_1_edge]
        )
        h_mask = self.conv_mask(
            prev_same=h_edge,
            x4_0=x4_0,
            prev_down=[h1_3_edge, h2_2_edge, h3_1_edge],
            stream_down=[h1_3_mask, h2_2_mask, h3_1_mask]
        )

        return {
            'dist': h_dist,
            'edge': h_edge,
            'mask': h_mask,
        }
