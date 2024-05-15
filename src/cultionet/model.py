import json
import logging
import typing as T
from pathlib import Path

import attr
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    StochasticWeightAveraging,
)
from lightning.pytorch.tuner import Tuner
from rasterio.windows import Window
from scipy.stats import mode as sci_mode
from torchvision import transforms

from .callbacks import (
    PROGRESS_BAR_CALLBACK,
    LightningGTiffWriter,
    setup_callbacks,
)
from .data.constant import SCALE_FACTOR
from .data.data import Data
from .data.datasets import EdgeDataset
from .data.modules import EdgeDataModule
from .data.samplers import EpochRandomSampler
from .enums import (
    AttentionTypes,
    LearningRateSchedulers,
    LossTypes,
    ModelNames,
    ModelTypes,
    ResBlockTypes,
)
from .models.cultionet import GeoRefinement
from .models.lightning import (
    CultionetLitModel,
    CultionetLitTransferModel,
    MaskRCNNLitModel,
    RefineLitModel,
)
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
    num_classes: int = attr.ib(converter=int, default=None)
    edge_class: int = attr.ib(converter=int, default=None)
    class_counts: torch.Tensor = attr.ib(default=None)
    hidden_channels: int = attr.ib(converter=int, default=64)
    model_type: str = attr.ib(converter=str, default=ModelTypes.TOWERUNET)
    activation_type: str = attr.ib(converter=str, default="SiLU")
    dropout: float = attr.ib(converter=float, default=0.1)
    dilations: T.Union[int, T.Sequence[int]] = attr.ib(
        converter=list, default=None
    )
    res_block_type: str = attr.ib(converter=str, default=ResBlockTypes.RES)
    attention_weights: str = attr.ib(
        converter=str, default=AttentionTypes.SPATIAL_CHANNEL
    )
    optimizer: str = attr.ib(converter=str, default="AdamW")
    loss_name: str = attr.ib(converter=str, default=LossTypes.TANIMOTO)
    learning_rate: float = attr.ib(converter=float, default=0.01)
    lr_scheduler: str = attr.ib(
        converter=str, default=LearningRateSchedulers.ONE_CYCLE_LR
    )
    steplr_step_size: int = attr.ib(converter=int, default=5)
    weight_decay: float = attr.ib(converter=float, default=1e-3)
    eps: float = attr.ib(converter=float, default=1e-4)
    ckpt_name: str = attr.ib(converter=str, default="last")
    model_name: str = attr.ib(converter=str, default="cultionet")
    deep_supervision: bool = attr.ib(default=False)
    pool_first: bool = attr.ib(default=False)
    pool_attention: bool = attr.ib(default=False)
    repeat_resa_kernel: bool = attr.ib(default=False)
    std_conv: bool = attr.ib(default=False)
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
    refine_model: bool = attr.ib(default=False)
    finetune: bool = attr.ib(default=False)
    strategy: str = attr.ib(converter=str, default="ddp")

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
            num_classes=self.num_classes,
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
            deep_supervision=self.deep_supervision,
            pool_first=self.pool_first,
            pool_attention=self.pool_attention,
            repeat_resa_kernel=self.repeat_resa_kernel,
            std_conv=self.std_conv,
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
        )


def fit_maskrcnn(
    dataset: EdgeDataset,
    ckpt_file: T.Union[str, Path],
    test_dataset: T.Optional[EdgeDataset] = None,
    val_frac: T.Optional[float] = 0.2,
    batch_size: T.Optional[int] = 4,
    accumulate_grad_batches: T.Optional[int] = 1,
    filters: T.Optional[int] = 64,
    num_classes: T.Optional[int] = 2,
    learning_rate: T.Optional[float] = 0.001,
    epochs: T.Optional[int] = 30,
    save_top_k: T.Optional[int] = 1,
    early_stopping_patience: T.Optional[int] = 7,
    early_stopping_min_delta: T.Optional[float] = 0.01,
    gradient_clip_val: T.Optional[float] = 1.0,
    reset_model: T.Optional[bool] = False,
    auto_lr_find: T.Optional[bool] = False,
    device: T.Optional[str] = "gpu",
    devices: T.Optional[int] = 1,
    weight_decay: T.Optional[float] = 1e-5,
    precision: T.Optional[int] = 32,
    stochastic_weight_averaging: T.Optional[bool] = False,
    stochastic_weight_averaging_lr: T.Optional[float] = 0.05,
    stochastic_weight_averaging_start: T.Optional[float] = 0.8,
    model_pruning: T.Optional[bool] = False,
    resize_height: T.Optional[int] = 201,
    resize_width: T.Optional[int] = 201,
    min_image_size: T.Optional[int] = 100,
    max_image_size: T.Optional[int] = 600,
    trainable_backbone_layers: T.Optional[int] = 3,
):
    """Fits a Mask R-CNN instance model.

    Args:
        dataset (EdgeDataset): The dataset to fit on.
        ckpt_file (str | Path): The checkpoint file path.
        test_dataset (Optional[EdgeDataset]): A test dataset to evaluate on. If given, early stopping
            will switch from the validation dataset to the test dataset.
        val_frac (Optional[float]): The fraction of data to use for model validation.
        batch_size (Optional[int]): The data batch size.
        filters (Optional[int]): The number of initial model filters.
        learning_rate (Optional[float]): The model learning rate.
        epochs (Optional[int]): The number of epochs.
        save_top_k (Optional[int]): The number of top-k model checkpoints to save.
        early_stopping_patience (Optional[int]): The patience (epochs) before early stopping.
        early_stopping_min_delta (Optional[float]): The minimum change threshold before early stopping.
        gradient_clip_val (Optional[float]): A gradient clip limit.
        reset_model (Optional[bool]): Whether to reset an existing model. Otherwise, pick up from last epoch of
            an existing model.
        auto_lr_find (Optional[bool]): Whether to search for an optimized learning rate.
        device (Optional[str]): The device to train on. Choices are ['cpu', 'gpu'].
        devices (Optional[int]): The number of GPU devices to use.
        weight_decay (Optional[float]): The weight decay passed to the optimizer. Default is 1e-5.
        precision (Optional[int]): The data precision. Default is 32.
        stochastic_weight_averaging (Optional[bool]): Whether to use stochastic weight averaging.
            Default is False.
        stochastic_weight_averaging_lr (Optional[float]): The stochastic weight averaging learning rate.
            Default is 0.05.
        stochastic_weight_averaging_start (Optional[float]): The stochastic weight averaging epoch start.
            Default is 0.8.
        model_pruning (Optional[bool]): Whether to prune the model. Default is False.
    """
    ckpt_file = Path(ckpt_file)

    # Split the dataset into train/validation
    train_ds, val_ds = dataset.split_train_val(val_frac=val_frac)

    # Setup the data module
    data_module = EdgeDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    lit_model = MaskRCNNLitModel(
        cultionet_model_file=ckpt_file.parent / "cultionet.pt",
        cultionet_num_features=train_ds.num_features,
        cultionet_num_time_features=train_ds.num_time_features,
        cultionet_filters=filters,
        cultionet_num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        resize_height=resize_height,
        resize_width=resize_width,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    if reset_model:
        if ckpt_file.is_file():
            ckpt_file.unlink()
        model_file = ckpt_file.parent / "maskrcnn.pt"
        if model_file.is_file():
            model_file.unlink()

    # Checkpoint
    cb_train_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode="min",
        monitor="loss",
        every_n_train_steps=0,
        every_n_epochs=1,
    )
    # Validation and test loss
    cb_val_loss = ModelCheckpoint(monitor="val_loss")
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode="min",
        check_on_train_epoch_end=False,
    )
    # Learning rate
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor, cb_train_loss, cb_val_loss, early_stop_callback]
    if stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=stochastic_weight_averaging_lr,
                swa_epoch_start=stochastic_weight_averaging_start,
            )
        )
    if 0 < model_pruning <= 1:
        callbacks.append(ModelPruning("l1_unstructured", amount=model_pruning))

    trainer = L.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=callbacks,
        enable_checkpointing=True,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="value",
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=precision,
        devices=devices,
        accelerator=device,
        log_every_n_steps=50,
        profiler=None,
        deterministic=False,
        benchmark=False,
    )

    if auto_lr_find:
        trainer.tune(model=lit_model, datamodule=data_module)
    else:
        trainer.fit(
            model=lit_model,
            datamodule=data_module,
            ckpt_path=ckpt_file if ckpt_file.is_file() else None,
        )
        if test_dataset is not None:
            trainer.test(
                model=lit_model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path="last",
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
    pretrained_ckpt_file = cultionet_params.ckpt_file
    assert (
        pretrained_ckpt_file.is_file()
    ), "The pretrained checkpoint does not exist."
    # This will be the new checkpoint for the transfer model
    ckpt_file = (
        cultionet_params.ckpt_file.parent / ModelNames.CKPT_TRANSFER_NAME
    )

    # Split the dataset into train/validation
    data_module: EdgeDataModule = get_data_module(
        **cultionet_params.get_datamodule_params()
    )

    # Setup the Lightning model
    lit_model = CultionetLitTransferModel(
        **cultionet_params.get_lightning_params()
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
        ckpt_path=ckpt_file if ckpt_file.is_file() else None,
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
                if cultionet_params.ckpt_file.is_file()
                else None,
            )

        if cultionet_params.refine_model:
            refine_data_module = EdgeDataModule(
                train_ds=cultionet_params.dataset,
                batch_size=cultionet_params.batch_size,
                num_workers=cultionet_params.load_batch_workers,
                shuffle=True,
                # For each epoch, train on a random
                # subset of 50% of the data.
                sampler=EpochRandomSampler(
                    cultionet_params.dataset,
                    num_samples=int(len(cultionet_params.dataset) * 0.5),
                ),
            )
            refine_ckpt_file = (
                cultionet_params.ckpt_file.parent
                / "refine"
                / cultionet_params.ckpt_file.name
            )
            refine_ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            # refine checkpoints
            refine_cb_train_loss = ModelCheckpoint(
                dirpath=refine_ckpt_file.parent,
                filename=refine_ckpt_file.stem,
                save_last=True,
                save_top_k=1,
                mode="min",
                monitor="loss",
                every_n_train_steps=0,
                every_n_epochs=1,
            )
            # Early stopping
            refine_early_stop_callback = EarlyStopping(
                monitor="loss",
                min_delta=0.1,
                patience=5,
                mode="min",
                check_on_train_epoch_end=False,
            )
            refine_callbacks = [
                lr_monitor,
                refine_cb_train_loss,
                refine_early_stop_callback,
            ]
            refine_trainer = L.Trainer(
                default_root_dir=str(refine_ckpt_file.parent),
                callbacks=refine_callbacks,
                enable_checkpointing=True,
                gradient_clip_val=cultionet_params.gradient_clip_val,
                gradient_clip_algorithm="value",
                check_val_every_n_epoch=1,
                min_epochs=1
                if cultionet_params.epochs >= 1
                else cultionet_params.epochs,
                max_epochs=10,
                precision=32,
                devices=cultionet_params.devices,
                accelerator=cultionet_params.device,
                log_every_n_steps=50,
                deterministic=False,
                benchmark=False,
            )
            # Calibrate the logits
            refine_model = RefineLitModel(
                in_features=data_module.train_ds.num_features,
                num_classes=cultionet_params.num_classes,
                edge_class=cultionet_params.edge_class,
                class_counts=cultionet_params.class_counts,
                cultionet_ckpt=cultionet_params.ckpt_file,
            )
            refine_trainer.fit(
                model=refine_model,
                datamodule=refine_data_module,
                ckpt_path=refine_ckpt_file
                if refine_ckpt_file.is_file()
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
    num_classes: T.Optional[int] = None,
    filters: T.Optional[int] = None,
    device: T.Union[str, bytes] = "gpu",
    devices: T.Optional[int] = 1,
    lit_model: T.Optional[CultionetLitModel] = None,
    enable_progress_bar: T.Optional[bool] = True,
    return_trainer: T.Optional[bool] = False,
) -> T.Tuple[T.Union[None, L.Trainer], CultionetLitModel]:
    """Loads a model from file.

    Args:
        ckpt_file (str | Path): The model checkpoint file.
        model_file (str | Path): The model file.
        device (str): The device to apply inference on.
        lit_model (CultionetLitModel): A model to predict with. If `None`, the model
            is loaded from file.
        enable_progress_bar (Optional[bool]): Whether to use the progress bar.
        return_trainer (Optional[bool]): Whether to return the `lightning` `Trainer`.
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
                num_classes=num_classes,
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
    num_classes: int,
    device: str = "gpu",
    devices: int = 1,
    strategy: str = "ddp",
    batch_size: int = 4,
    load_batch_workers: int = 0,
    precision: T.Union[int, str] = "16-mixed",
    resampling: str = "nearest",
    compression: str = "lzw",
    is_transfer_model: bool = False,
    refine_pt: T.Optional[Path] = None,
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
        num_classes=num_classes,
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
        cultionet_lit_model = CultionetLitTransferModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_file)
        )
    else:
        cultionet_lit_model = CultionetLitModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_file)
        )

    geo_refine_model = None
    if refine_pt is not None:
        if refine_pt.is_file():
            geo_refine_model = GeoRefinement(
                in_features=dataset.num_features, out_channels=num_classes
            )
            geo_refine_model.load_state_dict(torch.load(refine_pt))
            geo_refine_model.eval()

    setattr(cultionet_lit_model, "temperature_lit_model", geo_refine_model)

    # Make predictions
    trainer.predict(
        model=cultionet_lit_model,
        datamodule=data_module,
        return_predictions=False,
    )


def predict(
    lit_model: CultionetLitModel,
    data: Data,
    written: np.ndarray,
    norm_values: NormValues,
    w: Window = None,
    w_pad: Window = None,
    device: str = "cpu",
    include_maskrcnn: bool = False,
) -> np.ndarray:
    """Applies a model to predict image labels|values.

    Args:
        lit_model (CultionetLitModel): A model to predict with.
        data (Data): The data to predict on.
        written (ndarray)
        data_values (Tensor)
        w (Optional[int]): The ``rasterio.windows.Window`` to write to.
        w_pad (Optional[int]): The ``rasterio.windows.Window`` to predict on.
        device (Optional[str])
    """
    norm_batch = norm_values(data)

    if device == "gpu":
        norm_batch = norm_batch.to("cuda")
        lit_model = lit_model.to("cuda")
    with torch.no_grad():
        distance, dist_1, dist_2, dist_3, dist_4, edge, crop = lit_model(
            norm_batch
        )
        crop_type = torch.zeros((crop.size(0), 2), dtype=crop.dtype)

        if include_maskrcnn:
            # TODO: fix this -- separate Mask R-CNN model
            predictions = lit_model.mask_forward(
                distance=distance,
                edge=edge,
                height=norm_batch.height,
                width=norm_batch.width,
                batch=None,
            )
    instances = None
    if include_maskrcnn:
        instances = np.zeros(
            (norm_batch.height, norm_batch.width), dtype="float64"
        )
        if include_maskrcnn:
            scores = predictions[0]["scores"].squeeze()
            masks = predictions[0]["masks"].squeeze()
            resizer = transforms.Resize((norm_batch.height, norm_batch.width))
            masks = resizer(masks)
            # Filter by box scores
            masks = masks[scores > 0.5]
            scores = scores[scores > 0.5]
            # Filter by pixel scores
            masks = torch.where(masks > 0.5, masks, 0)
            masks = masks.detach().cpu().numpy()
            if masks.shape[0] > 0:
                distance_mask = (
                    distance.detach()
                    .cpu()
                    .numpy()
                    .reshape(norm_batch.height, norm_batch.width)
                )
                edge_mask = (
                    edge[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(norm_batch.height, norm_batch.width)
                )
                crop_mask = (
                    crop[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(norm_batch.height, norm_batch.width)
                )
                instances = np.zeros(
                    (norm_batch.height, norm_batch.width), dtype="float64"
                )

                uid = 1 if written.max() == 0 else written.max() + 1

                def iou(reference, targets):
                    tp = ((reference > 0.5) & (targets > 0.5)).sum()
                    fp = ((reference <= 0.5) & (targets > 0.5)).sum()
                    fn = ((reference > 0.5) & (targets <= 0.5)).sum()

                    return tp / (tp + fp + fn)

                for lyr_idx_ref, lyr_ref in enumerate(masks):
                    lyr = None
                    for lyr_idx_targ, lyr_targ in enumerate(masks):
                        if lyr_idx_targ != lyr_idx_ref:
                            if iou(lyr_ref, lyr_targ) > 0.5:
                                lyr = (
                                    lyr_ref
                                    if scores[lyr_idx_ref]
                                    > scores[lyr_idx_targ]
                                    else lyr_targ
                                )
                    if lyr is None:
                        lyr = lyr_ref
                    conditional = (
                        (lyr > 0.5)
                        & (distance_mask > 0.1)
                        & (edge_mask < 0.5)
                        & (crop_mask > 0.5)
                    )
                    if written[conditional].max() > 0:
                        uid = int(sci_mode(written[conditional]).mode)
                    instances = np.where(
                        ((instances == 0) & conditional), uid, instances
                    )
                    uid = instances.max() + 1
                instances /= SCALE_FACTOR
            else:
                logger.warning("No fields were identified.")

    mo = ModelOutputs(
        distance=distance,
        edge=edge,
        crop=crop,
        crop_type=crop_type,
        instances=instances,
        apply_softmax=False,
    )
    stack = mo.stack_outputs(w, w_pad)
    if include_maskrcnn:
        stack[:-1] = (stack[:-1] * SCALE_FACTOR).clip(0, SCALE_FACTOR)
        stack[-1] *= SCALE_FACTOR
    else:
        stack = (stack * SCALE_FACTOR).clip(0, SCALE_FACTOR)

    return stack
