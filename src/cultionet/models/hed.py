from . import model_utils

import torch


class SingleConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SingleConv, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.net = torch.nn.Sequential(
            SingleConv(
                in_channels=in_channels,
                out_channels=out_channels
            ),
            SingleConv(
                in_channels=out_channels,
                out_channels=out_channels
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HED(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(HED, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        layer_counts = [64, 128, 256, 512]

        self.net_vgg_one = DoubleConv(
            in_channels=in_channels,
            out_channels=layer_counts[0]
        )
        self.net_vgg_two = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(
                in_channels=layer_counts[0],
                out_channels=layer_counts[1]
            )
        )
        self.net_vgg_three = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(
                in_channels=layer_counts[1],
                out_channels=layer_counts[2]
            ),
            SingleConv(
                in_channels=layer_counts[2],
                out_channels=layer_counts[2]
            )
        )
        self.net_vgg_four = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(
                in_channels=layer_counts[2],
                out_channels=layer_counts[3]
            ),
            SingleConv(
                in_channels=layer_counts[3],
                out_channels=layer_counts[3]
            )
        )
        self.net_vgg_five = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(
                in_channels=layer_counts[3],
                out_channels=layer_counts[3]
            ),
            SingleConv(
                in_channels=layer_counts[3],
                out_channels=layer_counts[3]
            )
        )

        self.net_score_one = torch.nn.Conv2d(
            in_channels=layer_counts[0], out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.net_score_two = torch.nn.Conv2d(
            in_channels=layer_counts[1], out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.net_score_three = torch.nn.Conv2d(
            in_channels=layer_counts[2], out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.net_score_four = torch.nn.Conv2d(
            in_channels=layer_counts[3], out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.net_score_five = torch.nn.Conv2d(
            in_channels=layer_counts[3], out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.net_combine = torch.nn.Conv2d(
            in_channels=5, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        nrows: int,
        ncols: int
    ) -> torch.Tensor:
        nbatch = 1 if batch is None else batch.unique().size(0)

        x = self.gc(x, nbatch, nrows, ncols)

        ten_vgg_one = self.net_vgg_one(x)
        ten_vgg_two = self.net_vgg_two(ten_vgg_one)
        ten_vgg_three = self.net_vgg_three(ten_vgg_two)
        ten_vgg_four = self.net_vgg_four(ten_vgg_three)
        ten_vgg_five = self.net_vgg_five(ten_vgg_four)

        ten_score_one = self.net_score_one(ten_vgg_one)
        ten_score_two = self.net_score_two(ten_vgg_two)
        ten_score_three = self.net_score_three(ten_vgg_three)
        ten_score_four = self.net_score_four(ten_vgg_four)
        ten_score_five = self.net_score_five(ten_vgg_five)

        ten_score_one = torch.nn.functional.interpolate(
            input=ten_score_one, size=(nrows, ncols), mode='bilinear', align_corners=False
        )
        ten_score_two = torch.nn.functional.interpolate(
            input=ten_score_two, size=(nrows, ncols), mode='bilinear', align_corners=False
        )
        ten_score_three = torch.nn.functional.interpolate(
            input=ten_score_three, size=(nrows, ncols), mode='bilinear', align_corners=False
        )
        ten_score_four = torch.nn.functional.interpolate(
            input=ten_score_four, size=(nrows, ncols), mode='bilinear', align_corners=False
        )
        ten_score_five = torch.nn.functional.interpolate(
            input=ten_score_five, size=(nrows, ncols), mode='bilinear', align_corners=False
        )

        h = self.net_combine(
            torch.cat(
                [
                    ten_score_one,
                    ten_score_two,
                    ten_score_three,
                    ten_score_four,
                    ten_score_five
                ], 1
            )
        )

        return self.cg(h)
