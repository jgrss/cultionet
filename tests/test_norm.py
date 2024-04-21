import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cultionet.data.data import Data
from cultionet.data.utils import collate_fn
from cultionet.utils.normalize import NormValues

from .conftest import temporary_dataset

PROJECT_PATH = Path(__file__).parent.absolute()
CLASS_INFO = {'max_crop_class': 1, 'edge_class': 2}


def test_train_dataset(data_batch: Data):
    num_samples = 6
    batch_size = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = temporary_dataset(
            batch=data_batch,
            temp_dir=temp_dir,
            num_samples=num_samples,
        )

        norm_values = NormValues.from_dataset(
            ds,
            batch_size=batch_size,
            class_info=CLASS_INFO,
            num_workers=0,
            centering='median',
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
            batch=data_batch,
            temp_dir=temp_dir,
            num_samples=num_samples,
            norm_values=norm_values,
        )
        norm_data_loader = DataLoader(
            norm_ds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn,
        )
        data_loader = DataLoader(
            ds,
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
