from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from cultionet.data import Data
from cultionet.data.datasets import EdgeDataset

RNG = np.random.default_rng(100)


@pytest.fixture
def class_info() -> dict:
    return {'max_crop_class': 1, 'edge_class': 2}


def create_batch(
    num_channels: int = 3,
    num_time: int = 12,
    height: int = 20,
    width: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Data:
    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=-1, high=3, size=(1, height, width))
    bdist = torch.rand(1, height, width)

    if rng is None:
        rng = RNG

    idx = rng.integers(low=0, high=99_999)
    year = rng.choice([2020, 2021, 2022, 2023])

    top = rng.uniform(-90, 90, size=1)
    bottom = rng.uniform(-90, 90, size=1)
    if top < bottom:
        top, bottom = bottom, top

    left = rng.uniform(-180, 180, size=1)
    right = rng.uniform(-180, 180, size=1)
    if right < left:
        left, right = right, left

    return Data(
        x=x,
        y=y,
        bdist=bdist,
        batch_id=[f"data_{idx:06d}_{year}_none.pt"],
        left=torch.from_numpy(left),
        bottom=torch.from_numpy(bottom),
        right=torch.from_numpy(right),
        top=torch.from_numpy(top),
    )


@pytest.fixture
def data_batch() -> Data:
    return create_batch()


def temporary_dataset(
    temp_dir: str,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
    batch_kwargs: Optional[dict] = None,
    **kwargs,
) -> EdgeDataset:
    if batch_kwargs is None:
        batch_kwargs = {}

    train_path = Path(temp_dir)
    processed_path = train_path / 'processed'

    if rng is None:
        rng = np.random.default_rng(100)

    for _ in range(num_samples):
        batch = create_batch(rng=rng, **batch_kwargs)
        batch.to_file(processed_path / batch.batch_id[0])

    return EdgeDataset(train_path, **kwargs)
