from typing import Iterator, Sized

from torch.utils.data import Sampler


class EpochRandomSampler(Sampler[int]):
    """A random sampler per epoch.

    Reference:
        Adapted from: https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    """

    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: int):
        self.data_source = data_source
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)

        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        return iter(
            self.data_source.rng.choice(
                range(len(self)), replace=False, size=self.num_samples
            )
        )

    def __len__(self) -> int:
        return self.num_samples
