import typing as T

from .base_layers import ConvBlock2d

import torch


class GroupConvBlock2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        add_activation: bool = True,
        activation_type: str = 'LeakyReLU'
    ):
        super(GroupConvBlock2d, self).__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            torch.nn.GroupNorm(out_channels // 2, out_channels)
        ]
        if add_activation:
            layers += [
                getattr(torch.nn, activation_type)(inplace=False)
            ]

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class CSFBUnit(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(CSFBUnit, self).__init__()

        self.conv_head_ctr = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.conv_head_sal = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.merge_sal = GroupConvBlock2d(
            in_channels=in_channels*2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0
        )
        self.merge_ctr = GroupConvBlock2d(
            in_channels=in_channels*2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0
        )
        self.conv_tail_ctr = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.conv_tail_sal = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )

    def forward(
        self, x: T.List[torch.Tensor]
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        ctr, sal = x
        ctr = self.conv_head_ctr(ctr)
        sal = self.conv_head_sal(sal)

        ctr_n_sal = torch.cat([ctr, sal], dim=1)
        ctr_sal = self.merge_sal(ctr_n_sal)
        sal_ctr = self.merge_ctr(ctr_n_sal)

        ctr = self.conv_tail_ctr(ctr_sal)
        sal = self.conv_tail_sal(sal_ctr)

        return ctr, sal


class CSFBBlock(torch.nn.Module):
    def __init__(self, in_channels: int, recursion: int):
        super(CSFBBlock, self).__init__()

        self.fbunit = CSFBUnit(in_channels=in_channels)
        self.tail_ctr = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.tail_sal = ConvBlock2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1
        )
        self.recursion = recursion

    def forward(
        self, x: T.List[torch.Tensor]
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        ctr, sal = x
        h0_ctr = ctr
        h0_sal = sal

        for __ in range(self.recursion):
            ctr, sal = self.fbunit([ctr, sal])
            ctr = ctr + h0_ctr
            sal = sal + h0_sal

        ctr = self.tail_ctr(ctr)
        sal = self.tail_sal(sal)

        ctr = ctr + h0_ctr
        sal = sal + h0_sal

        return ctr, sal


class MapAdapter(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(MapAdapter, self).__init__()

        self.conv_ctr = torch.nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.conv_sal = torch.nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
        self.conv_end = torch.nn.Conv2d(2, in_channels, kernel_size=3, padding=1)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.sal_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.ctr_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(
        self, ctr: torch.Tensor, sal: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_ctr = self.conv_ctr(ctr) * self.ctr_scale
        pred_sal = self.conv_sal(sal) * self.sal_scale

        merge = torch.cat([pred_ctr, pred_sal], dim=1)
        merge = self.sigmoid(merge)

        stage_feature = self.conv_end(merge)
        stage_feature = self.relu(stage_feature)

        return pred_ctr, pred_sal, stage_feature


class MergeAdapter(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MergeAdapter, self).__init__()

        self.merge_head = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge_head(x)


class RCSB(torch.nn.Module):
    """
    Reference:
        https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/model/rcsb.py
    """
    def __init__(self, in_channels: int, recursion: int, g: int):
        super(RCSB, self).__init__()

        csfb_layers = [
            CSFBBlock(in_channels=in_channels, recursion=recursion) for __ in range(g)
        ]
        self.csfb = torch.nn.Sequential(*csfb_layers)
        self.map_gen = torch.nn.ModuleList(
            [MapAdapter(in_channels=in_channels) for __ in range(5)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contour, saliency = self.csfb([x, x])
        __, __, sm = self.map_gen[0](contour, saliency)

        return sm
