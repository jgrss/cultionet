import tempfile

import torch

from cultionet.data.modules import EdgeDataModule
from cultionet.utils.normalize import NormValues

from .conftest import temporary_dataset


def test_dataset(class_info: dict) -> EdgeDataModule:

    batch_size = 2
    num_channels = 3
    in_time = 12
    height = 20
    width = 20
    num_samples = 20
    val_frac = 0.1

    batch_kwargs = dict(
        num_channels=num_channels,
        num_time=in_time,
        height=height,
        width=width,
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
            batch_kwargs=batch_kwargs,
            processes=1,
            random_seed=100,
        )
        norm_values = NormValues.from_dataset(
            ds,
            batch_size=batch_size,
            class_info=class_info,
            num_workers=0,
        )
        ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
            batch_kwargs=batch_kwargs,
            processes=1,
            norm_values=norm_values,
            augment_prob=0.1,
            random_seed=100,
        )
        train_ds, val_ds = ds.split_train_val(
            val_frac=val_frac,
            spatial_overlap_allowed=False,
            spatial_balance=True,
        )

        generator = torch.Generator()
        generator.manual_seed(100)

        data_module = EdgeDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
        )
        first_train_batch = next(iter(data_module.train_dataloader()))
        first_val_batch = next(iter(data_module.val_dataloader()))
        assert first_train_batch.batch_id == [
            'data_002257_2022_none.pt',
            'data_012624_2023_none.pt',
        ]
        assert first_val_batch.batch_id == [
            'data_051349_2022_none.pt',
            'data_094721_2022_none.pt',
        ]
        data_module = EdgeDataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        first_train_batch = next(iter(data_module.train_dataloader()))
        first_val_batch = next(iter(data_module.val_dataloader()))
        assert first_train_batch.batch_id == [
            'data_034049_2022_none.pt',
            'data_050552_2023_none.pt',
        ]
        assert first_val_batch.batch_id == [
            'data_051349_2022_none.pt',
            'data_094721_2022_none.pt',
        ]

        assert len(ds) == num_samples
        assert len(val_ds) == int(val_frac * len(ds))
        assert len(train_ds) == len(ds) - int(val_frac * len(ds))
        assert ds.num_time == in_time
        assert train_ds.num_time == in_time
        assert val_ds.num_time == in_time

        assert ds.data_list[0].name == 'data_002257_2022_none.pt'
        assert ds.data_list[-1].name == 'data_094721_2022_none.pt'
        ds.shuffle()
        assert ds.data_list[0].name == 'data_032192_2020_none.pt'
        assert ds.data_list[-1].name == 'data_022792_2023_none.pt'

        ds.cleanup()
        assert len(ds) == 0
