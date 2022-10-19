"""Adapted from: https://github.com/4uiiurz1/pytorch-nested-unet

MIT License

Copyright (c) 2018 Takato Kimura
"""
from . import model_utils
from .base_layers import DoubleConv, ResConv

import torch
from torch_geometric import nn
from torchvision import transforms


class VGGBlock(torch.nn.Module):
    """A UNet block for graphs
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(VGGBlock, self).__init__()

        conv1 = nn.GCNConv(in_channels, mid_channels, improved=True)
        conv2 = nn.GCNConv(mid_channels, out_channels, improved=True)

        self.seq = nn.Sequential(
            'x, edge_index, edge_weight',
            [
                (conv1, 'x, edge_index, edge_weight -> x'),
                (nn.BatchNorm(mid_channels), 'x -> x'),
                (conv2, 'x, edge_index, edge_weight -> x'),
                (nn.BatchNorm(out_channels), 'x -> x'),
                (model_utils.max_pool_neighbor_x, 'x, edge_index -> x'),
                (torch.nn.ELU(alpha=0.1, inplace=False), 'x -> x')
                ]
            )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        return self.seq(x, edge_index, edge_weight)


class PoolConv(torch.nn.Module):
    """Max pooling followed by a double convolution
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(PoolConv, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_channels, mid_channels, out_channels)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet3Conv2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(NestedUNet3Conv2, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha=0.1, inplace=False),
            ResConv(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet3Pool(torch.nn.Module):
    def __init__(self, pool_in: int, in_channels: int, out_channels: int):
        super(NestedUNet3Pool, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.MaxPool2d(pool_in, pool_in, ceil_mode=True),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            ResConv(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet3Cat(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(NestedUNet3Cat, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet3Up(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super(NestedUNet3Up, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha=0.1, inplace=False)
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class NestedUNet2(torch.nn.Module):
    """UNet++ with residual convolutional dilation

    References:
        https://arxiv.org/pdf/1807.10165.pdf
        https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
    """
    def __init__(self, in_channels: int, out_channels: int, init_filter: int = 32):
        super(NestedUNet2, self).__init__()

        nb_filter = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()
        self.up = model_utils.UpSample()

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = PoolConv(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_0 = PoolConv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = PoolConv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = PoolConv(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final2d = torch.nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final = nn.GCNConv(out_channels, out_channels, improved=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)

        x0_0 = self.conv0_0(x, edge_index, edge_weight)
        # Reshape to CNN 4d
        x0_0 = self.gc(x0_0, nbatch, nrows, ncols)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, size=x0_0.shape[2:])], dim=1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, size=x1_0.shape[2:])], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, size=x0_1.shape[2:])], dim=1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, size=x2_0.shape[2:])], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, size=x1_1.shape[2:])], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, size=x0_2.shape[2:])], dim=1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, size=x3_0.shape[2:])], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, size=x2_1.shape[2:])], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, size=x1_2.shape[2:])], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, size=x0_3.shape[2:])], dim=1))

        output = self.final2d(x0_4)
        # Reshape to GNN 2d
        output = self.cg(output)

        output = self.final(output, edge_index, edge_weight)

        return output


class NestedUNet3(torch.nn.Module):
    """UNet+++ with residual convolutional dilation

    References:
        https://arxiv.org/abs/2004.08790
        https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_3Plus.py
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_filter: int = 32,
        rheight: int = 320,
        rwidth: int = 320
    ):
        super(NestedUNet3, self).__init__()

        self.in_channels = in_channels
        filters = [
            init_filter,
            init_filter*2,
            init_filter*4,
            init_filter*8,
            init_filter*16
        ]

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()
        self.resizer_up = transforms.Resize((rheight, rwidth))

        ## -------------Encoder--------------
        self.conv1 = NestedUNet3Conv2(in_channels, filters[0])
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = NestedUNet3Conv2(filters[0], filters[1])
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3 = NestedUNet3Conv2(filters[1], filters[2])
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4 = NestedUNet3Conv2(filters[2], filters[3])
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv5 = NestedUNet3Conv2(filters[3], filters[4])

        ## -------------Decoder--------------
        self.cat_channels = filters[0]
        self.cat_blocks = 5
        self.up_channels = self.cat_channels * self.cat_blocks

        # Stage 4d
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_pt_hd4 = NestedUNet3Pool(
            pool_in=8, in_channels=filters[0], out_channels=self.cat_channels
        )
        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_pt_hd4 = NestedUNet3Pool(
            pool_in=4, in_channels=filters[1], out_channels=self.cat_channels
        )
        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_pt_hd4 = NestedUNet3Pool(
            pool_in=2, in_channels=filters[2], out_channels=self.cat_channels
        )
        # h4->40*40, hd4->40*40, Concatenation
        self.h4_cat_hd4_conv = NestedUNet3Cat(filters[3], self.cat_channels)
        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_ut_hd4 = NestedUNet3Up(filters[4], self.cat_channels, scale_factor=2)
        # fusion(h1_pt_hd4, h2_pt_hd4, h3_pt_hd4, h4_cat_hd4, hd5_ut_hd4)
        self.conv4d_1 = NestedUNet3Cat(self.up_channels, self.up_channels)

        # Stage 3d
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_pt_hd3 = NestedUNet3Pool(
            pool_in=4, in_channels=filters[0], out_channels=self.cat_channels
        )
        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_pt_hd3 = NestedUNet3Pool(
            pool_in=2, in_channels=filters[1], out_channels=self.cat_channels
        )
        # h3->80*80, hd3->80*80, Concatenation
        self.h3_cat_hd3_conv = NestedUNet3Cat(filters[2], self.cat_channels)
        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_ut_hd3 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=2)
        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_ut_hd3 = NestedUNet3Up(filters[4], self.cat_channels, scale_factor=4)
        # fusion(h1_pt_hd3, h2_pt_hd3, h3_cat_hd3, hd4_ut_hd3, hd5_ut_hd3)
        self.conv3d_1 = NestedUNet3Cat(self.up_channels, self.up_channels)

        # Stage 2d
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_pt_hd2 = NestedUNet3Pool(
            pool_in=2, in_channels=filters[0], out_channels=self.cat_channels
        )
        # h2->160*160, hd2->160*160, Concatenation
        self.h2_cat_hd2_conv = NestedUNet3Cat(filters[1], self.cat_channels)
        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_ut_hd2 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=2)
        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_ut_hd2 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=4)
        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_ut_hd2 = NestedUNet3Up(filters[4], self.cat_channels, scale_factor=8)
        # fusion(h1_pt_hd2, h2_cat_hd2, hd3_ut_hd2, hd4_ut_hd2, hd5_ut_hd2)
        self.conv2d_1 = NestedUNet3Cat(self.up_channels, self.up_channels)

        # Stage 1d
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_cat_hd1_conv = NestedUNet3Cat(filters[0], self.cat_channels)
        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_ut_hd1 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=2)
        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_ut_hd1 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=4)
        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_ut_hd1 = NestedUNet3Up(self.up_channels, self.cat_channels, scale_factor=8)
        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_ut_hd1 = NestedUNet3Up(filters[4], self.cat_channels, scale_factor=16)
        # fusion(h1_cat_hd1, hd2_ut_hd1, hd3_ut_hd1, hd4_ut_hd1, hd5_ut_hd1)
        self.conv1d_1 = NestedUNet3Cat(self.up_channels, self.up_channels)

        # output
        self.outconv1 = torch.nn.Conv2d(self.up_channels, self.up_channels, 3, padding=1)
        self.final = nn.GCNConv(self.up_channels, out_channels, improved=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)
        resizer_down = transforms.Resize((nrows, ncols))

        # Reshape to CNN 4d
        x = self.gc(x, nbatch, nrows, ncols)
        x = self.resizer_up(x)

        ## -------------Encoder-------------
        h1 = self.conv1(x)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_pt_hd4 = self.h1_pt_hd4(h1)
        h2_pt_hd4 = self.h2_pt_hd4(h2)
        h3_pt_hd4 = self.h3_pt_hd4(h3)
        h4_cat_hd4 = self.h4_cat_hd4_conv(h4)
        hd5_ut_hd4 = self.hd5_ut_hd4(hd5)
        hd4 = self.conv4d_1(
            torch.cat(
                (h1_pt_hd4, h2_pt_hd4, h3_pt_hd4, h4_cat_hd4, hd5_ut_hd4), 1
            )
        ) # hd4->40*40*UpChannels

        h1_pt_hd3 = self.h1_pt_hd3(h1)
        h2_pt_hd3 = self.h2_pt_hd3(h2)
        h3_cat_hd3 = self.h3_cat_hd3_conv(h3)
        hd4_ut_hd3 = self.hd4_ut_hd3(hd4)
        hd5_ut_hd3 = self.hd5_ut_hd3(hd5)
        hd3 = self.conv3d_1(
            torch.cat(
                (h1_pt_hd3, h2_pt_hd3, h3_cat_hd3, hd4_ut_hd3, hd5_ut_hd3), 1
            )
        ) # hd3->80*80*UpChannels

        h1_pt_hd2 = self.h1_pt_hd2(h1)
        h2_cat_hd2 = self.h2_cat_hd2_conv(h2)
        hd3_ut_hd2 = self.hd3_ut_hd2(hd3)
        hd4_ut_hd2 = self.hd4_ut_hd2(hd4)
        hd5_ut_hd2 = self.hd5_ut_hd2(hd5)
        hd2 = self.conv2d_1(
            torch.cat(
                (h1_pt_hd2, h2_cat_hd2, hd3_ut_hd2, hd4_ut_hd2, hd5_ut_hd2), 1
            )
        ) # hd2->160*160*UpChannels

        h1_cat_hd1 = self.h1_cat_hd1_conv(h1)
        hd2_ut_hd1 = self.hd2_ut_hd1(hd2)
        hd3_ut_hd1 = self.hd3_ut_hd1(hd3)
        hd4_ut_hd1 = self.hd4_ut_hd1(hd4)
        hd5_ut_hd1 = self.hd5_ut_hd1(hd5)
        hd1 = self.conv1d_1(
            torch.cat(
                (h1_cat_hd1, hd2_ut_hd1, hd3_ut_hd1, hd4_ut_hd1, hd5_ut_hd1), 1
            )
        ) # hd1->320*320*UpChannels
        output = self.outconv1(hd1)  # d1->320*320*n_classes
        output = resizer_down(output)
        # Reshape to GNN 2d
        output = self.cg(output)
        output = self.final(output, edge_index, edge_weight)

        return output
