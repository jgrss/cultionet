import torch
from torch import nn


class GeoEmbeddings(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.coord_embedding = nn.Linear(3, channels)

    @torch.no_grad
    def decimal_degrees_to_cartesian(
        self, degrees: torch.Tensor
    ) -> torch.Tensor:
        radians = torch.deg2rad(degrees)
        cosine = torch.cos(radians)
        sine = torch.sin(radians)
        x = cosine[:, 1] * cosine[:, 0]
        y = cosine[:, 1] * sine[:, 0]

        return torch.stack([x, y, sine[:, 1]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cartesian_coords = self.decimal_degrees_to_cartesian(x)

        return self.coord_embedding(cartesian_coords)
