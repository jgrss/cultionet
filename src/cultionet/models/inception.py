import typing as T

from .base_layers import ConvBlock2d, ConvBlock3d, model_utils

import torch


class InceptionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        med_channels: T.Dict[str, int],
        out_channels: T.Dict[str, int],
        activation_type: str
    ):
        super(InceptionBlock, self).__init__()

        self.conv1 = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels['1'],
            squeeze=False,
            kernel_size=1,
            padding=0,
            activation_type=activation_type
        )
        self.conv3 = torch.nn.Sequential(
            ConvBlock3d(
                in_channels=in_channels,
                out_channels=med_channels['3'],
                squeeze=False,
                kernel_size=1,
                padding=0,
                activation_type=activation_type
            ),
            ConvBlock3d(
                in_channels=med_channels['3'],
                out_channels=out_channels['3'],
                squeeze=False,
                kernel_size=3,
                padding=1,
                activation_type=activation_type
            )
        )
        self.conv5 = torch.nn.Sequential(
            ConvBlock3d(
                in_channels=in_channels,
                out_channels=med_channels['5'],
                squeeze=False,
                kernel_size=1,
                padding=0,
                activation_type=activation_type
            ),
            ConvBlock3d(
                in_channels=med_channels['5'],
                out_channels=out_channels['5'],
                squeeze=False,
                kernel_size=3,
                padding=2,
                dilation=2,
                activation_type=activation_type
            )
        )
        self.max_pool = torch.nn.Sequential(
            torch.nn.MaxPool3d(kernel_size=3, padding=1, stride=1),
            ConvBlock3d(
                in_channels=in_channels,
                out_channels=out_channels['max'],
                squeeze=False,
                kernel_size=1,
                padding=0,
                activation_type=activation_type
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.conv1(x)
        x_3 = self.conv3(x)
        x_5 = self.conv5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1, x_3, x_5, x_max], dim=1)

        return x_out


class InceptionNet(torch.nn.Module):
    """
    Reference:
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        num_classes_last: int = 1,
        activation_type: str = 'LeakyReLU'
    ):
        super(InceptionNet, self).__init__()

        self.up = model_utils.UpSample()

        self.activation_type = activation_type
        self.input_net = ConvBlock3d(
            in_channels=in_channels,
            out_channels=out_channels,
            squeeze=False,
            kernel_size=3,
            padding=1,
            activation_type=activation_type
        )
        out_dict1 = {'1': 16, '3': 32, '5': 8, 'max': 8}
        in_channels1 = sum(out_dict1.values())
        out_dict2 = {'1': 24, '3': 48, '5': 12, 'max': 12}
        in_channels2 = sum(out_dict2.values())
        out_dict3 = {'1': 24, '3': 48, '5': 12, 'max': 12}
        in_channels3 = sum(out_dict3.values())
        out_dict4 = {'1': 16, '3': 48, '5': 16, 'max': 16}
        in_channels4 = sum(out_dict4.values())
        out_dict5 = {'1': 16, '3': 48, '5': 16, 'max': 16}
        in_channels5 = sum(out_dict5.values())
        out_dict6 = {'1': 32, '3': 48, '5': 24, 'max': 24}
        in_channels6 = sum(out_dict6.values())
        out_dict7 = {'1': 32, '3': 64, '5': 16, 'max': 16}
        in_channels7 = sum(out_dict7.values())
        out_dict8 = {'1': 32, '3': 64, '5': 16, 'max': 16}
        end_channels = sum(out_dict8.values())

        self.inception_blocks = torch.nn.Sequential(
            InceptionBlock(
                in_channels=out_channels,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict1,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels1,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict2,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels2,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict3,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels3,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict4,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels4,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict5,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels5,
                med_channels={'3': 32, '5': 16},
                out_channels=out_dict6,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels6,
                med_channels={'3': 48, '5': 16},
                out_channels=out_dict7,
                activation_type=self.activation_type
            ),
            InceptionBlock(
                in_channels=in_channels7,
                med_channels={'3': 48, '5': 16},
                out_channels=out_dict8,
                activation_type=self.activation_type
            )
        )
        self.final_reduce = ConvBlock2d(
            in_channels=end_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            activation_type=self.activation_type
        )
        self.final_last = torch.nn.Conv2d(
            out_channels,
            num_classes_last,
            kernel_size=1,
            padding=0
        )

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity='leaky_relu' if self.activation_type.lower() == 'leakyrelu' else self.activation_type.lower()
                )
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input dimensions should be -> B x C x T x H x W
        if len(x.shape) == 4:
            # Single batch
            x = x.unsqueeze(0)
        height, width = x.shape[-2:]

        x = self.input_net(x)
        x = self.inception_blocks(x)
        # Reduce time to 1 by trilinear interpolation
        x = self.up(
            x, size=(1, height, width), mode='trilinear'
        ).squeeze()
        # Input dimensions should be -> B x C x H x W
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.final_reduce(x)
        out = self.final_last(x)

        return out
