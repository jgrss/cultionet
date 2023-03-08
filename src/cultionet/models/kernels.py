"""
Source:
    @inproceedings{ismail-fawaz2022hccf,
        author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber, Jonathan and Forestier, Germain},
        title = {Deep Learning For Time Series Classification Using New Hand-Crafted Convolution Filters},
        booktitle = {2022 IEEE International Conference on Big Data (IEEE BigData 2022)},
        city = {Osaka},
        country = {Japan},
        pages = {972-981},
        url = {doi.org/10.1109/BigData55660.2022.10020496},
        year = {2022},
        organization = {IEEE}
    }

Paper:
    https://germain-forestier.info/publis/bigdata2022.pdf

Code:
    https://github.com/MSD-IRIMAS/CF-4-TSC
"""
import torch
import torch.nn.functional as F


class Trend(torch.nn.Module):
    def __init__(self, kernel_size: int, direction: str = "positive"):
        super(Trend, self).__init__()

        assert direction in (
            "positive",
            "negative",
        ), "The trend direction must be one of 'positive' or 'negative'."

        self.padding = int(kernel_size / 2)
        self.weights = torch.ones(kernel_size)
        indices_ = torch.arange(kernel_size)
        if direction == "positive":
            self.weights[indices_ % 2 == 0] *= -1
        elif direction == "negative":
            self.weights[indices_ % 2 > 0] *= -1

        self.weights = self.weights[(None,) * 2]
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B x C x T)
        x = F.conv1d(
            x,
            self.weights.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        x = self.relu(x)

        return x


class Peaks(torch.nn.Module):
    def __init__(self, kernel_size: int, radius: int = 9, sigma: float = 1.5):
        super(Peaks, self).__init__()

        self.padding = int(kernel_size / 2)
        x = torch.linspace(-radius, radius + 1, kernel_size)
        mu = 0.0
        gaussian = (
            1.0
            / (torch.sqrt(torch.tensor([2.0 * torch.pi])) * sigma)
            * torch.exp(-1.0 * (x - mu) ** 2 / (2.0 * sigma**2))
        )
        self.weights = gaussian * (x**2 / sigma**4 - 1.0) / sigma**2
        self.weights -= self.weights.mean()
        self.weights /= torch.sum(self.weights * x**2) / 2.0
        self.weights *= -1.0

        self.weights = self.weights[(None,) * 2]
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (B x C x T)
        x = F.conv1d(
            x,
            self.weights.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=1,
        )
        x = self.relu(x)

        return x
