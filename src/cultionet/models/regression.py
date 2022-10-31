from .base_layers import Permute

import torch


class RegressionConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(RegressionConv, self).__init__()

        conv1 = torch.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        batchnorm_layer1 = torch.nn.BatchNorm2d(mid_channels)
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)

        conv2 = torch.nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        batchnorm_layer2 = torch.nn.BatchNorm2d(mid_channels)

        conv3 = torch.nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        dropout_layer = torch.nn.Dropout(dropout)

        linear_layer = torch.nn.Linear(
            mid_channels,
            out_channels,
        )

        self.seq = torch.nn.Sequential(
            conv1,
            batchnorm_layer1,
            activate_layer,
            conv2,
            batchnorm_layer2,
            activate_layer,
            dropout_layer,
            conv3,
            batchnorm_layer2,
            dropout_layer,
            activate_layer,
            Permute((0, 2, 3, 1)),
            linear_layer,
            Permute((0, 3, 1, 2))
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)
