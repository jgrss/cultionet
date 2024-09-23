import tempfile
from pathlib import Path

import numpy as np
import torch

from cultionet.data import Data
from cultionet.data.modules import EdgeDataModule

from .conftest import temporary_dataset


def test_assign_x():
    num_channels = 3
    num_time = 10
    height = 5
    width = 5
    x = torch.rand(1, num_channels, num_time, height, width)
    batch = Data(x=x)

    assert batch.x.shape == (1, num_channels, num_time, height, width)
    assert batch.y is None
    assert torch.allclose(x, batch.x)
    assert batch.num_samples == 1
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.height == height
    assert batch.width == width


def test_assign_xy():
    num_channels = 3
    num_time = 10
    height = 5
    width = 5
    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=2, size=(1, height, width))
    batch = Data(x=x, y=y)

    assert batch.x.shape == (1, num_channels, num_time, height, width)
    assert batch.y.shape == (1, height, width)
    assert torch.allclose(x, batch.x)
    assert torch.allclose(y, batch.y)
    assert batch.num_samples == 1
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.height == height
    assert batch.width == width


def test_assign_xy_kwargs():
    num_channels = 3
    num_time = 10
    height = 5
    width = 5
    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=2, size=(1, height, width))
    bdist = torch.rand(1, height, width)
    batch = Data(x=x, y=y, bdist=bdist)

    assert batch.x.shape == (1, num_channels, num_time, height, width)
    assert batch.y.shape == (1, height, width)
    assert batch.bdist.shape == (1, height, width)
    assert torch.allclose(x, batch.x)
    assert torch.allclose(y, batch.y)
    assert torch.allclose(bdist, batch.bdist)
    assert batch.num_samples == 1
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.height == height
    assert batch.width == width


def test_create_data():
    num_channels = 3
    num_time = 10
    height = 5
    width = 5

    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=2, size=(1, height, width))
    bdist = torch.rand(1, height, width)
    batch = Data(x=x, y=y, bdist=bdist)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / 'test_batch.pt'

        # Save and load a single batch
        batch.to_file(temp_path)
        loaded_batch = batch.from_file(temp_path)

        assert loaded_batch.x.shape == (
            1,
            num_channels,
            num_time,
            height,
            width,
        )
        assert loaded_batch.y.shape == (1, height, width)
        assert loaded_batch.bdist.shape == (1, height, width)
        assert torch.allclose(x, loaded_batch.x)
        assert torch.allclose(y, loaded_batch.y)
        assert torch.allclose(bdist, loaded_batch.bdist)
        assert loaded_batch.num_samples == 1
        assert loaded_batch.num_channels == num_channels
        assert loaded_batch.num_time == num_time
        assert loaded_batch.height == height
        assert loaded_batch.width == width


def test_copy_data(data_batch: Data):
    x_clone = data_batch.x.clone()

    batch_copy = data_batch.copy()

    for key in batch_copy.to_dict().keys():
        assert key in data_batch.to_dict().keys()

    batch_copy.x *= 10

    assert not torch.allclose(data_batch.x, batch_copy.x)
    assert torch.allclose(data_batch.x, x_clone)
    assert torch.allclose(data_batch.y, batch_copy.y)


def test_train_dataset():
    num_samples = 6
    batch_size = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
        )

        assert len(ds) == num_samples

        data_module = EdgeDataModule(
            train_ds=ds,
            batch_size=batch_size,
            num_workers=0,
        )
        for batch in data_module.train_dataloader():
            assert batch.num_samples == batch_size
            for key, value in batch.to_dict().items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    assert value.shape[0] == batch_size
                else:
                    assert len(value) == batch_size
