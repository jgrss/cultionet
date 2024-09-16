import torch
from torch import nn

from .spherical_harmonics_closed_form import SHClosedForm
from .spherical_harmonics_ylm import SH as SH_analytic


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()

        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SphericalHarmonics(nn.Module):
    """Sherical harmonics.

    legendre_polys: determines the number of legendre polynomials.
                    more polynomials lead more fine-grained resolutions
    calculation of spherical harmonics:
        analytic uses pre-computed equations. This is exact, but works only up to degree 50,
        closed-form uses one equation but is computationally slower (especially for high degrees)

    Reference:
        @inproceedings{russwurm_etal_2024,
            title={Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks},
            author={Marc Rußwurm and Konstantin Klemmer and Esther Rolf and Robin Zbinden and Devis Tuia},
            year={2024},
            booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
            year={2024},
            url={https://iclr.cc/virtual/2024/poster/18690},
        }

    Source:
        https://github.com/MarcCoru/locationencoder/tree/main
    """

    def __init__(
        self,
        out_channels: int,
        legendre_polys: int = 10,
        harmonics_calculation: str = "analytic",
    ):
        super().__init__()

        self.legendre_polys_l, self.legendre_polys_m = int(
            legendre_polys
        ), int(legendre_polys)
        self.embedding_dim = self.legendre_polys_l * self.legendre_polys_m

        if harmonics_calculation == "closed-form":
            self.sh_func = SHClosedForm()
        elif harmonics_calculation == "analytic":
            self.sh_func = SH_analytic

        self.final = nn.Sequential(
            nn.Linear(self.embedding_dim, out_channels),
            Sine(),
        )

    def forward(self, degrees: torch.Tensor) -> torch.Tensor:
        lon, lat = degrees[:, 0], degrees[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l_param in range(self.legendre_polys_l):
            for m_param in range(-l_param, l_param + 1):
                y = self.sh_func(m_param, l_param, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)

                Y.append(y)

        h = torch.stack(Y, dim=-1)

        return self.final(h)
