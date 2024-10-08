import json
import logging
import typing as T
from pathlib import Path

import attr
import lightning as L
import numpy as np
import torch
from lightning.pytorch.tuner import Tuner
from rasterio.windows import Window
from scipy.stats import mode as sci_mode
from torchvision import transforms

from .callbacks import (
    PROGRESS_BAR_CALLBACK,
    LightningGTiffWriter,
    setup_callbacks,
)
from .data import Data
from .data.constant import SCALE_FACTOR
from .data.datasets import EdgeDataset
from .data.modules import EdgeDataModule

# from .data.samplers import EpochRandomSampler
from .enums import (
    AttentionTypes,
    LearningRateSchedulers,
    LossTypes,
    ModelNames,
    ModelTypes,
    ResBlockTypes,
)
from .models.lightning import CultionetLitModel, CultionetLitTransferModel
from .utils.logging import set_color_logger
from .utils.normalize import NormValues
from .utils.reshape import ModelOutputs

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

logger = set_color_logger(__name__)


@attr.s
class CultionetParams:
    ckpt_file: T.Union[str, Path] = attr.ib(converter=Path, default=None)
    spatial_partitions: str = attr.ib(default=None)
    dataset: EdgeDataset = attr.ib(default=None)
    test_dataset: T.Optional[EdgeDataset] = attr.ib(default=None)
    val_frac: float = attr.ib(converter=float, default=0.2)
    batch_size: int = attr.ib(converter=int, default=4)
    load_batch_workers: int = attr.ib(converter=int, default=0)
    edge_class: int = attr.ib(converter=int, default=None)
    class_counts: torch.Tensor = attr.ib(default=None)
    hidden_channels: int = attr.ib(converter=int, default=64)
    model_type: str = attr.ib(converter=str, default=ModelTypes.TOWERUNET)
    activation_type: str = attr.ib(converter=str, default="SiLU")
    dropout: float = attr.ib(converter=float, default=0.1)
    dilations: T.Union[int, T.Sequence[int]] = attr.ib(
        converter=list, default=None
    )
    res_block_type: str = attr.ib(converter=str, default=ResBlockTypes.RESA)
    attention_weights: str = attr.ib(default=None)
    optimizer: str = attr.ib(converter=str, default="AdamW")
    loss_name: str = attr.ib(
        converter=str, default=LossTypes.TANIMOTO_COMPLEMENT
    )
    learning_rate: float = attr.ib(converter=float, default=0.01)
    lr_scheduler: str = attr.ib(
        converter=str, default=LearningRateSchedulers.ONE_CYCLE_LR
    )
    steplr_step_size: int = attr.ib(converter=int, default=5)
    weight_decay: float = attr.ib(converter=float, default=1e-3)
    eps: float = attr.ib(converter=float, default=1e-4)
    ckpt_name: str = attr.ib(converter=str, default="last")
    model_name: str = attr.ib(converter=str, default="cultionet")
    pool_by_max: bool = attr.ib(default=False)
    batchnorm_first: bool = attr.ib(default=False)
    scale_pos_weight: bool = attr.ib(default=False)
    save_batch_val_metrics: bool = attr.ib(default=False)
    epochs: int = attr.ib(converter=int, default=100)
    accumulate_grad_batches: int = attr.ib(converter=int, default=1)
    gradient_clip_val: float = attr.ib(converter=float, default=1.0)
    gradient_clip_algorithm: str = attr.ib(converter=str, default="norm")
    precision: T.Union[int, str] = attr.ib(default="16-mixed")
    device: str = attr.ib(converter=str, default="gpu")
    devices: int = attr.ib(converter=int, default=1)
    reset_model: bool = attr.ib(default=False)
    auto_lr_find: bool = attr.ib(default=False)
    stochastic_weight_averaging: bool = attr.ib(default=False)
    stochastic_weight_averaging_lr: float = attr.ib(
        converter=float, default=0.05
    )
    stochastic_weight_averaging_start: float = attr.ib(
        converter=float, default=0.8
    )
    model_pruning: bool = attr.ib(default=False)
    skip_train: bool = attr.ib(default=False)
    finetune: str = attr.ib(default=None)
    strategy: str = attr.ib(converter=str, default="ddp")
    profiler: str = attr.ib(default=None)

    def check_checkpoint(self) -> None:
        if self.reset_model:
            if self.ckpt_file.is_file():
                self.ckpt_file.unlink()

            model_file = self.ckpt_file.parent / f"{self.model_name}.pt"
            if model_file.is_file():
                model_file.unlink()

    def update_channels(
        self, data_module: EdgeDataModule
    ) -> "CultionetParams":
        self.in_channels = data_module.train_ds.num_channels
        self.in_time = data_module.train_ds.num_time

        return self

    def get_callback_params(self) -> dict:
        return dict(
            ckpt_file=self.ckpt_file,
            stochastic_weight_averaging=self.stochastic_weight_averaging,
            stochastic_weight_averaging_lr=self.stochastic_weight_averaging_lr,
            stochastic_weight_averaging_start=self.stochastic_weight_averaging_start,
            model_pruning=self.model_pruning,
        )

    def get_datamodule_params(self) -> dict:
        return dict(
            dataset=self.dataset,
            test_dataset=self.test_dataset,
            val_frac=self.val_frac,
            spatial_partitions=self.spatial_partitions,
            batch_size=self.batch_size,
            load_batch_workers=self.load_batch_workers,
        )

    def get_lightning_params(self) -> dict:
        return dict(
            in_channels=self.in_channels,
            in_time=self.in_time,
            hidden_channels=self.hidden_channels,
            model_type=self.model_type,
            dropout=self.dropout,
            activation_type=self.activation_type,
            dilations=self.dilations,
            res_block_type=self.res_block_type,
            attention_weights=self.attention_weights,
            optimizer=self.optimizer,
            loss_name=self.loss_name,
            learning_rate=self.learning_rate,
            lr_scheduler=self.lr_scheduler,
            steplr_step_size=self.steplr_step_size,
            weight_decay=self.weight_decay,
            eps=self.eps,
            ckpt_name=self.ckpt_name,
            model_name=self.model_name,
            pool_by_max=self.pool_by_max,
            batchnorm_first=self.batchnorm_first,
            class_counts=self.class_counts,
            edge_class=self.edge_class,
            scale_pos_weight=self.scale_pos_weight,
            save_batch_val_metrics=self.save_batch_val_metrics,
        )

    def get_trainer_params(self) -> dict:
        return dict(
            default_root_dir=str(self.ckpt_file.parent),
            enable_checkpointing=True,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
            check_val_every_n_epoch=1,
            min_epochs=5 if self.epochs >= 5 else self.epochs,
            max_epochs=self.epochs,
            precision=self.precision,
            devices=self.devices,
            accelerator=self.device,
            log_every_n_steps=50,
            deterministic=False,
            benchmark=False,
            strategy=self.strategy,
            profiler=self.profiler,
        )


def get_data_module(
    dataset: EdgeDataset,
    test_dataset: T.Optional[EdgeDataset] = None,
    val_frac: T.Optional[float] = 0.2,
    spatial_partitions: T.Optional[T.Union[str, Path]] = None,
    batch_size: T.Optional[int] = 4,
    load_batch_workers: T.Optional[int] = 2,
) -> EdgeDataModule:
    # Split the dataset into train/validation
    if spatial_partitions is not None:
        # TODO: We removed `dataset.split_train_val_by_partition` but
        # could make it an option in future versions.
        train_ds, val_ds = dataset.split_train_val(
            val_frac=val_frac,
            spatial_overlap_allowed=False,
            spatial_balance=True,
        )
    else:
        train_ds, val_ds = dataset.split_train_val(val_frac=val_frac)

    # Setup the data module
    data_module = EdgeDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_dataset,
        batch_size=batch_size,
        num_workers=load_batch_workers,
        shuffle=True,
    )

    return data_module


def fit_transfer(cultionet_params: CultionetParams) -> None:
    """Fits a transfer model."""

    # This file should already exist
    pretrained_ckpt_file = (
        cultionet_params.ckpt_file.parent / ModelNames.CKPT_TRANSFER_NAME
    )
    assert (
        pretrained_ckpt_file.exists()
    ), "The pretrained checkpoint does not exist."

    # Remove the spatial data because there is no check upstream
    if cultionet_params.dataset.grid_gpkg_path.exists():
        cultionet_params.dataset.grid_gpkg_path.unlink()

    # Split the dataset into train/validation
    data_module: EdgeDataModule = get_data_module(
        **cultionet_params.get_datamodule_params()
    )

    # Get the channel and time dimensions from the dataset
    cultionet_params = cultionet_params.update_channels(data_module)

    # Setup the Lightning model
    lit_model = CultionetLitTransferModel(
        pretrained_ckpt_file=pretrained_ckpt_file,
        finetune=cultionet_params.finetune,
        **cultionet_params.get_lightning_params(),
    )

    # Remove the model file if requested
    cultionet_params.check_checkpoint()

    _, callbacks = setup_callbacks(**cultionet_params.get_callback_params())
    callbacks.append(PROGRESS_BAR_CALLBACK)

    # Setup the trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        **cultionet_params.get_trainer_params(),
    )

    trainer.fit(
        model=lit_model,
        datamodule=data_module,
        ckpt_path=cultionet_params.ckpt_file
        if cultionet_params.ckpt_file.exists()
        else None,
    )


def fit(cultionet_params: CultionetParams) -> None:
    """Fits a model."""

    # Split the dataset into train/validation
    data_module: EdgeDataModule = get_data_module(
        **cultionet_params.get_datamodule_params()
    )

    # Get the channel and time dimensions from the dataset
    cultionet_params = cultionet_params.update_channels(data_module)

    # Setup the Lightning model
    lit_model = CultionetLitModel(**cultionet_params.get_lightning_params())

    # Remove the model file if requested
    cultionet_params.check_checkpoint()

    lr_monitor, callbacks = setup_callbacks(
        **cultionet_params.get_callback_params()
    )
    callbacks.append(PROGRESS_BAR_CALLBACK)

    # Setup the trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        **cultionet_params.get_trainer_params(),
    )

    if cultionet_params.auto_lr_find:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model=lit_model, datamodule=data_module)
        opt_lr = lr_finder.suggestion()
        logger.info(f"The suggested learning rate is {opt_lr}")
    else:
        if not cultionet_params.skip_train:
            trainer.fit(
                model=lit_model,
                datamodule=data_module,
                ckpt_path=cultionet_params.ckpt_file
                if cultionet_params.ckpt_file.exists()
                else None,
            )

        if cultionet_params.test_dataset is not None:
            trainer.test(
                model=lit_model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path="best",
            )
            logged_metrics = trainer.logged_metrics
            for k, v in logged_metrics.items():
                logged_metrics[k] = float(v)
            with open(
                Path(trainer.logger.save_dir) / "test.metrics", mode="w"
            ) as f:
                f.write(json.dumps(logged_metrics))


def load_model(
    ckpt_file: T.Union[str, Path] = None,
    model_file: T.Union[str, Path] = None,
    num_features: T.Optional[int] = None,
    num_time_features: T.Optional[int] = None,
    filters: T.Optional[int] = None,
    device: T.Union[str, bytes] = "gpu",
    devices: T.Optional[int] = 1,
    lit_model: T.Optional[CultionetLitModel] = None,
    enable_progress_bar: T.Optional[bool] = True,
    return_trainer: T.Optional[bool] = False,
) -> T.Tuple[T.Union[None, L.Trainer], CultionetLitModel]:
    """Loads a model from file.

    Parameters
    ==========
    ckpt_file
        The model checkpoint file.
    model_file
        The model file.
    device
        The device to apply inference on.
    lit_model
        A model to predict with. If `None`, the model is loaded from file.
    enable_progress_bar
        Whether to use the progress bar.
    return_trainer
        Whether to return the `lightning` `Trainer`.
    """
    if ckpt_file is not None:
        ckpt_file = Path(ckpt_file)
    if model_file is not None:
        model_file = Path(model_file)

    trainer = None
    if return_trainer:
        trainer_kwargs = dict(
            default_root_dir=str(ckpt_file.parent),
            precision=32,
            devices=devices,
            accelerator=device,
            log_every_n_steps=0,
            logger=False,
            enable_progress_bar=enable_progress_bar,
        )

        trainer = L.Trainer(**trainer_kwargs)

    if lit_model is None:
        if model_file is not None:
            assert model_file.is_file(), "The model file does not exist."
            if not isinstance(num_features, int) or not isinstance(
                num_time_features, int
            ):
                raise TypeError(
                    "The features must be given to load the model file."
                )
            lit_model = CultionetLitModel(
                num_features=num_features,
                num_time_features=num_time_features,
                filters=filters,
            )
            lit_model.load_state_dict(state_dict=torch.load(model_file))
        else:
            assert ckpt_file.is_file(), "The checkpoint file does not exist."
            lit_model = CultionetLitModel.load_from_checkpoint(
                checkpoint_path=str(ckpt_file)
            )
        lit_model.eval()
        lit_model.freeze()

    return trainer, lit_model


def predict_lightning(
    reference_image: T.Union[str, Path],
    out_path: T.Union[str, Path],
    ckpt: Path,
    dataset: EdgeDataset,
    device: str = "gpu",
    devices: int = 1,
    strategy: str = "ddp",
    batch_size: int = 4,
    load_batch_workers: int = 0,
    precision: T.Union[int, str] = "16-mixed",
    resampling: str = "nearest",
    compression: str = "lzw",
    is_transfer_model: bool = False,
):
    reference_image = Path(reference_image)
    out_path = Path(out_path)
    ckpt_file = Path(ckpt)
    assert ckpt_file.exists(), "The checkpoint file does not exist."

    data_module = EdgeDataModule(
        predict_ds=dataset,
        batch_size=batch_size,
        num_workers=load_batch_workers,
        shuffle=False,
    )
    pred_writer = LightningGTiffWriter(
        reference_image=reference_image,
        out_path=out_path,
        resampling=resampling,
        compression=compression,
    )
    trainer_kwargs = dict(
        default_root_dir=str(ckpt_file.parent),
        callbacks=[pred_writer, PROGRESS_BAR_CALLBACK],
        precision=precision,
        devices=devices,
        accelerator=device,
        strategy=strategy,
        log_every_n_steps=0,
        logger=False,
    )

    trainer = L.Trainer(**trainer_kwargs)

    if is_transfer_model:
        pretrained_ckpt_file = ckpt.parent / ModelNames.CKPT_TRANSFER_NAME

        cultionet_lit_model = CultionetLitTransferModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_file),
            pretrained_ckpt_file=pretrained_ckpt_file,
        )
    else:
        cultionet_lit_model = CultionetLitModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_file)
        )

    # Make predictions
    trainer.predict(
        model=cultionet_lit_model,
        datamodule=data_module,
        return_predictions=False,
    )
