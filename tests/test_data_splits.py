import tempfile

from .conftest import temporary_dataset


def test_train_dataset():
    num_samples = 6
    val_frac = 0.2

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = temporary_dataset(
            temp_dir=temp_dir,
            num_samples=num_samples,
            processes=1,
        )
        train_ds, val_ds = ds.split_train_val(
            val_frac=val_frac,
            spatial_overlap_allowed=False,
            spatial_balance=True,
        )

    assert len(train_ds) == len(ds) - int(len(ds) * val_frac)
    assert len(val_ds) == int(len(ds) * val_frac)
