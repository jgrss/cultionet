import numpy as np
from torch.utils.data import Sampler

from .datasets import EdgeDataset


class EpochRandomSampler(Sampler):
    """A random sampler per epoch.

    Reference:
        Adapted from: https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    """

    def __init__(self, data_source: EdgeDataset, num_samples: int):
        super().__init__()

        self.data_source = data_source
        self._num_samples = num_samples

    @property
    def n(self) -> int:
        return len(self.data_source)

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        return iter(
            np.random.choice(
                range(self.n), replace=False, size=self.num_samples
            )
        )

    def __len__(self):
        return self.num_samples
