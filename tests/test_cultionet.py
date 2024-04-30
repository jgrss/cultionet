import tempfile

from cultionet.data.modules import EdgeDataModule
from cultionet.enums import ModelTypes, ResBlockTypes
from cultionet.models.cultionet import CultioNet
from cultionet.utils.normalize import NormValues

from .conftest import temporary_dataset


def get_train_dataset(
    class_nums: dict,
    temp_dir: str,
    batch_kwargs: dict,
    batch_size: int,
    num_samples: int,
    val_frac: float,
) -> EdgeDataModule:

    ds = temporary_dataset(
        temp_dir=temp_dir,
        num_samples=num_samples,
        batch_kwargs=batch_kwargs,
        processes=1,
    )
    norm_values = NormValues.from_dataset(
        ds,
        batch_size=batch_size,
        class_info=class_nums,
        num_workers=0,
        centering='median',
    )
    ds = temporary_dataset(
        temp_dir=temp_dir,
        num_samples=num_samples,
        batch_kwargs=batch_kwargs,
        processes=1,
        norm_values=norm_values,
        augment_prob=0.1,
    )
    train_ds, val_ds = ds.split_train_val(
        val_frac=val_frac,
        spatial_overlap_allowed=False,
        spatial_balance=True,
    )

    return EdgeDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
    )


def test_cultionet(class_info: dict):
    num_channels = 3
    in_time = 12
    height = 50
    width = 50
    batch_size = 2
    num_samples = 12
    val_frac = 0.2

    kwargs = dict(
        in_channels=num_channels,
        in_time=in_time,
        hidden_channels=32,
        num_classes=2,
        model_type=ModelTypes.TOWERUNET,
        activation_type="SiLU",
        dilations=None,
        res_block_type=ResBlockTypes.RES,
        attention_weights="spatial_channel",
        deep_supervision=False,
    )

    model = CultioNet(**kwargs)

    with tempfile.TemporaryDirectory() as temp_dir:
        data_module = get_train_dataset(
            class_nums=class_info,
            temp_dir=temp_dir,
            batch_kwargs=dict(
                num_channels=num_channels,
                num_time=in_time,
                height=height,
                width=width,
            ),
            batch_size=batch_size,
            num_samples=num_samples,
            val_frac=val_frac,
        )

        assert data_module.train_ds.augment_prob == 0.1
        assert data_module.val_ds.augment_prob == 0.0

        for batch in data_module.train_dataloader():
            output = model(batch)

            assert output["dist"].shape == (batch_size, 1, height, width)
            assert output["edge"].shape == (batch_size, 1, height, width)
            assert output["crop"].shape == (batch_size, 2, height, width)
            assert output["classes_l2"].shape == (batch_size, 2, height, width)
            assert output["classes_l3"].shape == (
                batch_size,
                class_info["edge_class"] + 1,
                height,
                width,
            )
