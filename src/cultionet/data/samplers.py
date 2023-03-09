import numpy as np
from torch.utils.data import Sampler

from .datasets import EdgeDataset


class EpochRandomSampler(Sampler):
    """A random sampler per epoch.

    Reference:
        Adapted from: https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    """

    def __init__(self, dataset: EdgeDataset, num_samples: int):
        super().__init__()

        self.dataset = dataset
        self.num_samples = num_samples

    @property
    def n(self) -> int:
        return len(self.dataset)

    @property
    def samples(self) -> int:
        if self.num_samples > self.n:
            samples_ = self.n
        else:
            samples_ = self.num_samples

        return samples_

    def __iter__(self):
        return iter(
            np.random.choice(range(self.n), replace=False, size=self.samples)
        )

    def __len__(self):
        return self.samples
