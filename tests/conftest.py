from pathlib import Path

import pytest
import torch

from cultionet.data.data import Data
from cultionet.data.datasets import EdgeDataset


@pytest.fixture
def data_batch() -> Data:
    num_channels = 3
    num_time = 12
    height = 20
    width = 20

    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=3, size=(1, height, width))
    bdist = torch.rand(1, height, width)

    return Data(x=x, y=y, bdist=bdist)


def temporary_dataset(
    batch: Data,
    temp_dir: str,
    num_samples: int,
    **kwargs,
) -> EdgeDataset:
    train_path = Path(temp_dir)
    processed_path = train_path / 'processed'

    for i in range(num_samples):
        temp_path = processed_path / f"data_{i:06d}_2022_0_none.pt"
        batch_sample = batch.copy()
        random_batch = Data(
            **{
                key: torch.rand(*value.shape)
                for key, value in batch_sample.to_dict().items()
            }
        )
        batch_sample += random_batch
        batch_sample.to_file(temp_path)

    return EdgeDataset(train_path, **kwargs)
