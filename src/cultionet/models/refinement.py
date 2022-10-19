from . import model_utils

import torch


class RefineConv(torch.nn.Module):
    """A refinement convolution layer
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super(RefineConv, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ELU(alpha=0.1, inplace=False),
            torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate'
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate'
            ),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(
                mid_channels, out_channels, kernel_size=1, padding=0, bias=False, padding_mode='replicate'
            )
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        batch_size: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        x = self.gc(
            x,
            batch_size,
            height,
            width
        )

        x = self.seq(x)

        return self.cg(x)
