import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cultionet.data import Data
from cultionet.data.utils import collate_fn
from cultionet.utils.normalize import NormValues

from .conftest import temporary_dataset


def test_norm():
    num_channels = 3
    shape = (1, num_channels, 1, 1, 1)
    norm_values = NormValues(
        dataset_mean=torch.zeros(shape),
        dataset_std=torch.ones(shape),
        dataset_crop_counts=None,
        dataset_edge_counts=None,
        num_channels=num_channels,
    )

    batch = Data(x=torch.ones(shape))
    assert torch.allclose(
        norm_values(batch).x,
        torch.ones(shape),
    )
    assert torch.allclose(batch.x, torch.ones(shape))

    batch = Data(x=torch.zeros(shape))
    assert torch.allclose(
        norm_values(batch).x,
        torch.zeros(shape),
    )
    assert torch.allclose(batch.x, torch.zeros(shape))

    norm_values = NormValues(
        dataset_mean=torch.zeros(shape) + 0.5,
        dataset_std=torch.ones(shape) + 0.5,
        dataset_crop_counts=None,
        dataset_edge_counts=None,
        num_channels=num_channels,
    )

    batch = Data(x=torch.ones(shape))
    assert torch.allclose(
        norm_values(batch).x,
        torch.zeros(shape) + 0.3333,
        rtol=0.01,
    )
    assert torch.allclose(batch.x, torch.ones(shape))


def test_train_dataset(class_info: dict):
    num_samples = 6
    batch_size = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
            log_transform=True,
        )

        norm_values = NormValues.from_dataset(
            ds,
            batch_size=batch_size,
            class_info=class_info,
            num_workers=0,
        )

        norm_path = Path(temp_dir) / 'data.norm'
        norm_values.to_file(norm_path)
        loaded_norm_values = NormValues.from_file(norm_path)

        assert torch.allclose(
            norm_values.dataset_mean, loaded_norm_values.dataset_mean
        )
        assert torch.allclose(
            norm_values.dataset_std, loaded_norm_values.dataset_std
        )
        assert torch.allclose(
            norm_values.dataset_crop_counts,
            loaded_norm_values.dataset_crop_counts,
        )
        assert torch.allclose(
            norm_values.dataset_edge_counts,
            loaded_norm_values.dataset_edge_counts,
        )

        assert norm_values.dataset_mean.shape == (
            1,
            norm_values.num_channels,
            1,
            1,
            1,
        )

        # Apply normalization
        norm_ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
            norm_values=norm_values,
            log_transform=True,
        )
        data_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
        )
        norm_data_loader = DataLoader(
            norm_ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # The normalization should be applied to each batch
        for batch, norm_batch in zip(data_loader, norm_data_loader):
            assert not torch.allclose(
                batch.x,
                norm_batch.x,
            )
            assert torch.allclose(
                norm_values(batch).x,
                norm_batch.x,
            )
