import math

import torch


class SHClosedForm:
    def associated_legendre_polynomial(
        self, l_param: int, m_param: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Spherical Harmonics utilities.

        Code copied from https://github.com/BachiLi/redner/blob/master/pyredner/utils.py
        Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
        http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf
        """
        pmm = torch.ones_like(x)

        if m_param > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m_param + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0

        if l_param == m_param:
            return pmm

        pmmp1 = x * (2.0 * m_param + 1.0) * pmm

        if l_param == m_param + 1:
            return pmmp1

        pll = torch.zeros_like(x)
        for ll in range(m_param + 2, l_param + 1):
            pll = (
                (2.0 * ll - 1.0) * x * pmmp1 - (ll + m_param - 1.0) * pmm
            ) / (ll - m_param)
            pmm = pmmp1
            pmmp1 = pll

        return pll

    def sh_renormalization(self, l_param: int, m_param: int) -> float:
        return math.sqrt(
            (2.0 * l_param + 1.0)
            * math.factorial(l_param - m_param)
            / (4 * math.pi * math.factorial(l_param + m_param))
        )

    def __call__(
        self, m_param: int, l_param: int, phi: float, theta: float
    ) -> torch.Tensor:
        if m_param == 0:
            return self.sh_renormalization(
                l_param, m_param
            ) * self.associated_legendre_polynomial(
                l_param, m_param, torch.cos(theta)
            )
        elif m_param > 0:
            return (
                math.sqrt(2.0)
                * self.sh_renormalization(l_param, m_param)
                * torch.cos(m_param * phi)
                * self.associated_legendre_polynomial(
                    l_param, m_param, torch.cos(theta)
                )
            )
        else:
            return (
                math.sqrt(2.0)
                * self.sh_renormalization(l_param, -m_param)
                * torch.sin(-m_param * phi)
                * self.associated_legendre_polynomial(
                    l_param, -m_param, torch.cos(theta)
                )
            )
