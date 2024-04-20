import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cultionet.data.data import Data
from cultionet.data.datasets import EdgeDataset
from cultionet.data.modules import EdgeDataModule


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
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.num_rows == height
    assert batch.num_cols == width


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
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.num_rows == height
    assert batch.num_cols == width


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
    assert batch.num_channels == num_channels
    assert batch.num_time == num_time
    assert batch.num_rows == height
    assert batch.num_cols == width


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
        assert loaded_batch.num_channels == num_channels
        assert loaded_batch.num_time == num_time
        assert loaded_batch.num_rows == height
        assert loaded_batch.num_cols == width


def create_full_batch(
    num_channels: int,
    num_time: int,
    height: int,
    width: int,
) -> Data:
    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=2, size=(1, height, width))
    bdist = torch.rand(1, height, width)

    return Data(x=x, y=y, bdist=bdist)


def test_copy_data():
    batch = create_full_batch(
        num_channels=3,
        num_time=10,
        height=5,
        width=5,
    )
    x_clone = batch.x.clone()

    batch_copy = batch.copy()
    batch_copy.x *= 10

    assert not torch.allclose(batch.x, batch_copy.x)
    assert torch.allclose(batch.x, x_clone)
    assert torch.allclose(batch.y, batch_copy.y)


def test_train_dataset():
    num_samples = 6
    batch_size = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = Path(temp_dir)
        processed_path = train_path / 'processed'

        for i in range(num_samples):
            temp_path = processed_path / f"data_{i:06d}_2022_0_none.pt"
            batch = create_full_batch(
                num_channels=3,
                num_time=10,
                height=5,
                width=5,
            )
            batch.to_file(temp_path)

        ds = EdgeDataset(train_path)

        assert len(ds) == num_samples

        data_module = EdgeDataModule(
            train_ds=ds,
            batch_size=batch_size,
            num_workers=0,
        )
        for batch in data_module.train_dataloader():
            assert batch.num_samples == batch_size
