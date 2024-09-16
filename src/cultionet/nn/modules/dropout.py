import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop-path dropout.

    Reference:
        @article{larsson_etal_2016,
            title={Fractalnet: Ultra-deep neural networks without residuals},
            author={Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory},
            journal={arXiv preprint arXiv:1605.07648},
            year={2016},
            url={https://arxiv.org/abs/1605.07648v4},
        }
    """

    def __init__(self, p: float = 0.5):
        super().__init__()

        self.p = p
        self.keep_prob = 1.0 - self.p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and (self.p > 0):
            mask = torch.empty(
                x.shape[0], 1, 1, 1, device=x.device
            ).bernoulli_(self.keep_prob)
            x_scaled = x / self.keep_prob
            x = x_scaled * mask

        return x
