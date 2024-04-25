import typing as T

import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, axis_order: T.Sequence[int]):
        super(Permute, self).__init__()
        self.axis_order = axis_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.axis_order)


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Min(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Min, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.min(dim=self.dim, keepdim=self.keepdim)[0]


class Max(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Max, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]


class Mean(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super(Mean, self).__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class Var(nn.Module):
    def __init__(
        self, dim: int, keepdim: bool = False, unbiased: bool = False
    ):
        super(Var, self).__init__()

        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.var(
            dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased
        )


class Std(nn.Module):
    def __init__(
        self, dim: int, keepdim: bool = False, unbiased: bool = False
    ):
        super(Std, self).__init__()

        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.std(
            dim=self.dim, keepdim=self.keepdim, unbiased=self.unbiased
        )
