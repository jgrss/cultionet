import torch


class ResConv(torch.nn.Module):
    """2d residual convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1
    ):
        super(ResConv, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + x


class DoubleConv(torch.nn.Module):
    """A double convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int
    ):
        super(DoubleConv, self).__init__()

        conv1 = torch.nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        conv2 = ResConv(
            mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2
        )
        conv3 = ResConv(
            mid_channels, out_channels, kernel_size=3, padding=3, dilation=3
        )
        activate_layer = torch.nn.ELU(alpha=0.1, inplace=False)
        batchnorm_layer1 = torch.nn.BatchNorm2d(mid_channels)
        batchnorm_layer2 = torch.nn.BatchNorm2d(out_channels)

        self.seq = torch.nn.Sequential(
            conv1,
            batchnorm_layer1,
            activate_layer,
            conv2,
            batchnorm_layer1,
            activate_layer,
            conv3,
            batchnorm_layer2,
            activate_layer
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
