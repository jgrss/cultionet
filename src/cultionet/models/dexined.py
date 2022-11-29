import typing as T

from . import model_utils

import torch
import torch.nn.functional as F


def weight_init(m):
    if isinstance(m, (torch.nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    if isinstance(m, (torch.nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class _DenseLayer(torch.nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module(
            'conv1',
            torch.nn.Conv2d(
                input_features,
                out_features,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            )
        )
        self.add_module('norm1', torch.nn.BatchNorm2d(out_features))
        self.add_module('relu1', torch.nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            torch.nn.Conv2d(
                out_features,
                out_features,
                kernel_size=3,
                stride=1,
                bias=True
            )
        )
        self.add_module('norm2', torch.nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))

        return 0.5 * (new_features + x2), x2


class _DenseBlock(torch.nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class SingleConvBlock(torch.nn.Module):
    def __init__(
        self, in_features, out_features, stride, use_bs=True
    ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = torch.nn.Conv2d(
            in_features,
            out_features,
            1,
            stride=stride,
            bias=True
        )
        self.bn = torch.nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        return x


class UpConvBlock(torch.nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        self.up = model_utils.UpSample()
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = torch.nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(torch.nn.Conv2d(in_features, out_features, 1))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(
                torch.nn.ConvTranspose2d(
                    out_features, out_features, kernel_size, stride=2, padding=pad
                )
            )
            in_features = out_features

        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class DoubleConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_features,
        mid_features,
        out_features=None,
        stride=1,
        use_act=True
    ):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = torch.nn.Conv2d(
            in_features,
            mid_features,
            3,
            padding=1,
            stride=stride
        )
        self.bn1 = torch.nn.BatchNorm2d(mid_features)
        self.conv2 = torch.nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_features)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)

        return x


class DexiNed(torch.nn.Module):
    """DexiNed

    References:
        https://arxiv.org/pdf/2112.02250.pdf
        https://github.com/xavysp/DexiNed
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_filter: int = 32
    ):
        super(DexiNed, self).__init__()

        channels = [
            init_filter,    # 32
            init_filter*2,  # 64
            init_filter*4,  # 128
            init_filter*8,  # 256
            init_filter*16  # 512
        ]

        self.block_1 = DoubleConvBlock(in_channels, channels[0], channels[1])
        self.block_2 = DoubleConvBlock(channels[1], channels[2], use_act=False)
        self.dblock_3 = _DenseBlock(2, channels[2], channels[3])
        self.dblock_4 = _DenseBlock(3, channels[3], channels[4])
        self.dblock_5 = _DenseBlock(3, channels[4], channels[4])
        self.dblock_6 = _DenseBlock(3, channels[4], channels[3])
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(channels[1], channels[2], 2)
        self.side_2 = SingleConvBlock(channels[2], channels[3], 2)
        self.side_3 = SingleConvBlock(channels[3], channels[4], 2)
        self.side_4 = SingleConvBlock(channels[4], channels[4], 1)
        self.side_5 = SingleConvBlock(channels[4], channels[3], 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(channels[2], channels[3], 2)
        self.pre_dense_3 = SingleConvBlock(channels[2], channels[3], 1)
        self.pre_dense_4 = SingleConvBlock(channels[3], channels[4], 1)
        self.pre_dense_5 = SingleConvBlock(channels[4], channels[4], 1)
        self.pre_dense_6 = SingleConvBlock(channels[4], channels[3], 1)

        self.up = model_utils.UpSample()
        self.up_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.up_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.up_block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.up_block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.up_block_5 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.up_block_6 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], 1, 1),
            torch.nn.ReLU(inplace=False)
        )
        self.block_cat = SingleConvBlock(6, out_channels, stride=1, use_bs=False)

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor

        return new_tensor

    def forward(
        self, x: torch.Tensor
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(block_4_down)
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(self.up(block_2, size=x.shape[-2:], mode='bilinear'))
        out_3 = self.up_block_3(self.up(block_3, size=x.shape[-2:], mode='bilinear'))
        out_4 = self.up_block_4(self.up(block_4, size=x.shape[-2:], mode='bilinear'))
        out_5 = self.up_block_5(self.up(block_5, size=x.shape[-2:], mode='bilinear'))
        out_6 = self.up_block_6(self.up(block_6, size=x.shape[-2:], mode='bilinear'))
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # B x 6 x H x W
        final = self.block_cat(block_cat)  # B x 2 x H x W

        return {
            'blocks': block_cat,
            'final': final
        }
