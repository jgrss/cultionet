import typing as T
from pathlib import Path
import logging
import json

import numpy as np
from scipy.stats import mode as sci_mode
from rasterio.windows import Window
import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
    ModelPruning,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

from .callbacks import LightningGTiffWriter
from .data.const import SCALE_FACTOR
from .data.datasets import EdgeDataset, zscores
from .data.modules import EdgeDataModule
from .models.cultio import GeoRefinement
from .models.lightning import (
    CultioLitModel,
    MaskRCNNLitModel,
    TemperatureScaling,
)
from .utils.reshape import ModelOutputs
from .utils.logging import set_color_logger


logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

logger = set_color_logger(__name__)


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

    trainer = pl.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="value",
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=precision,
        devices=None if device == "cpu" else devices,
        num_processes=0,
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


def fit(
    dataset: EdgeDataset,
    ckpt_file: T.Union[str, Path],
    test_dataset: T.Optional[EdgeDataset] = None,
    val_frac: T.Optional[float] = 0.2,
    spatial_partitions: T.Optional[T.Union[str, Path]] = None,
    partition_name: T.Optional[str] = None,
    partition_column: T.Optional[str] = None,
    batch_size: T.Optional[int] = 4,
    load_batch_workers: T.Optional[int] = 2,
    accumulate_grad_batches: T.Optional[int] = 1,
    filters: T.Optional[int] = 32,
    num_classes: T.Optional[int] = 2,
    edge_class: T.Optional[int] = None,
    class_counts: T.Sequence[float] = None,
    model_type: str = "ResUNet3Psi",
    activation_type: str = "SiLU",
    dilations: T.Union[int, T.Sequence[int]] = None,
    res_block_type: str = "resa",
    attention_weights: str = "spatial_channel",
    deep_sup_dist: bool = False,
    deep_sup_edge: bool = False,
    deep_sup_mask: bool = False,
    optimizer: str = "AdamW",
    learning_rate: T.Optional[float] = 1e-3,
    lr_scheduler: str = "CosineAnnealingLR",
    steplr_step_size: T.Optional[T.Sequence[int]] = None,
    scale_pos_weight: T.Optional[bool] = True,
    epochs: T.Optional[int] = 30,
    save_top_k: T.Optional[int] = 1,
    early_stopping_patience: T.Optional[int] = 7,
    early_stopping_min_delta: T.Optional[float] = 0.01,
    gradient_clip_val: T.Optional[float] = 1.0,
    gradient_clip_algorithm: T.Optional[float] = "norm",
    reset_model: T.Optional[bool] = False,
    auto_lr_find: T.Optional[bool] = False,
    device: T.Optional[str] = "gpu",
    devices: T.Optional[int] = 1,
    profiler: T.Optional[str] = None,
    weight_decay: T.Optional[float] = 1e-5,
    precision: T.Optional[int] = 32,
    stochastic_weight_averaging: T.Optional[bool] = False,
    stochastic_weight_averaging_lr: T.Optional[float] = 0.05,
    stochastic_weight_averaging_start: T.Optional[float] = 0.8,
    model_pruning: T.Optional[bool] = False,
    save_batch_val_metrics: T.Optional[bool] = False,
    skip_train: T.Optional[bool] = False,
    refine_and_calibrate: T.Optional[bool] = False,
):
    """Fits a model.

    Args:
        dataset (EdgeDataset): The dataset to fit on.
        ckpt_file (str | Path): The checkpoint file path.
        test_dataset (Optional[EdgeDataset]): A test dataset to evaluate on. If given, early stopping
            will switch from the validation dataset to the test dataset.
        val_frac (Optional[float]): The fraction of data to use for model validation.
        spatial_partitions (Optional[str | Path]): A spatial partitions file.
        partition_name (Optional[str]): The spatial partition file column query name.
        partition_column (Optional[str]): The spatial partition file column name.
        batch_size (Optional[int]): The data batch size.
        load_batch_workers (Optional[int]): The number of parallel batches to load.
        filters (Optional[int]): The number of initial model filters.
        optimizer (Optional[str]): The optimizer.
        model_type (Optional[str]): The model type.
        activation_type (Optional[str]): The activation type.
        dilations (Optional[list]): The dilation size or sizes.
        res_block_type (Optional[str]): The residual block type.
        attention_weights (Optional[str]): The attention weights.
        deep_sup_dist (Optional[bool]): Whether to use deep supervision for distances.
        deep_sup_edge (Optional[bool]): Whether to use deep supervision for edges.
        deep_sup_mask (Optional[bool]): Whether to use deep supervision for masks.
        learning_rate (Optional[float]): The model learning rate.
        lr_scheduler (Optional[str]): The learning rate scheduler.
        steplr_step_size (Optional[list]): The multiplicative step size factor.
        scale_pos_weight (Optional[bool]): Whether to scale class weights (i.e., balance classes).
        epochs (Optional[int]): The number of epochs.
        save_top_k (Optional[int]): The number of top-k model checkpoints to save.
        early_stopping_patience (Optional[int]): The patience (epochs) before early stopping.
        early_stopping_min_delta (Optional[float]): The minimum change threshold before early stopping.
        gradient_clip_val (Optional[float]): The gradient clip limit.
        gradient_clip_algorithm (Optional[str]): The gradient clip algorithm.
        reset_model (Optional[bool]): Whether to reset an existing model. Otherwise, pick up from last epoch of
            an existing model.
        auto_lr_find (Optional[bool]): Whether to search for an optimized learning rate.
        device (Optional[str]): The device to train on. Choices are ['cpu', 'gpu'].
        devices (Optional[int]): The number of GPU devices to use.
        profiler (Optional[str]): A profiler level. Choices are [None, 'simple', 'advanced'].
        weight_decay (Optional[float]): The weight decay passed to the optimizer. Default is 1e-5.
        precision (Optional[int]): The data precision. Default is 32.
        stochastic_weight_averaging (Optional[bool]): Whether to use stochastic weight averaging.
            Default is False.
        stochastic_weight_averaging_lr (Optional[float]): The stochastic weight averaging learning rate.
            Default is 0.05.
        stochastic_weight_averaging_start (Optional[float]): The stochastic weight averaging epoch start.
            Default is 0.8.
        model_pruning (Optional[bool]): Whether to prune the model. Default is False.
        save_batch_val_metrics (Optional[bool]): Whether to save batch validation metrics to a parquet file.
        skip_train (Optional[bool]): Whether to refine and calibrate a trained model.
        refine_and_calibrate (Optional[bool]): Whether to skip training.
    """
    ckpt_file = Path(ckpt_file)

    # Split the dataset into train/validation
    if spatial_partitions is not None:
        # TODO: We removed `dataset.split_train_val_by_partition` but
        # could make it an option in future versions.
        train_ds, val_ds = dataset.split_train_val(
            val_frac=val_frac, spatial_overlap_allowed=False
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
    temperature_data_module = EdgeDataModule(
        train_ds=val_ds,
        batch_size=batch_size,
        num_workers=load_batch_workers,
        shuffle=True,
    )

    # Setup the Lightning model
    lit_model = CultioLitModel(
        num_features=train_ds.num_features,
        num_time_features=train_ds.num_time_features,
        num_classes=num_classes,
        filters=filters,
        model_type=model_type,
        activation_type=activation_type,
        dilations=dilations,
        res_block_type=res_block_type,
        attention_weights=attention_weights,
        deep_sup_dist=deep_sup_dist,
        deep_sup_edge=deep_sup_edge,
        deep_sup_mask=deep_sup_mask,
        optimizer=optimizer,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        steplr_step_size=steplr_step_size,
        weight_decay=weight_decay,
        class_counts=class_counts,
        edge_class=edge_class,
        scale_pos_weight=scale_pos_weight,
        save_batch_val_metrics=save_batch_val_metrics,
    )

    if reset_model:
        if ckpt_file.is_file():
            ckpt_file.unlink()
        model_file = ckpt_file.parent / "cultionet.pt"
        if model_file.is_file():
            model_file.unlink()

    # Checkpoint
    cb_train_loss = ModelCheckpoint(monitor="loss")
    # Validation and test loss
    cb_val_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode="min",
        monitor="val_score",
        every_n_train_steps=0,
        every_n_epochs=1,
    )
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_score",
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode="min",
        check_on_train_epoch_end=False,
    )
    # Learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
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

    trainer = pl.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=precision,
        devices=None if device == "cpu" else devices,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=50,
        profiler=profiler,
        deterministic=False,
        benchmark=False,
    )
    temperature_ckpt_file = ckpt_file.parent / "temperature" / ckpt_file.name
    temperature_ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    # Temperature checkpoints
    temperature_cb_train_loss = ModelCheckpoint(
        dirpath=temperature_ckpt_file.parent,
        filename=temperature_ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode="min",
        monitor="loss",
        every_n_train_steps=0,
        every_n_epochs=1,
    )
    # Early stopping
    temperature_early_stop_callback = EarlyStopping(
        monitor="loss",
        min_delta=early_stopping_min_delta,
        patience=5,
        mode="min",
        check_on_train_epoch_end=False,
    )
    temperature_callbacks = [
        lr_monitor,
        temperature_cb_train_loss,
        temperature_early_stop_callback,
    ]
    temperature_trainer = pl.Trainer(
        default_root_dir=str(temperature_ckpt_file.parent),
        callbacks=temperature_callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="value",
        check_val_every_n_epoch=1,
        min_epochs=1 if epochs >= 1 else epochs,
        max_epochs=10,
        precision=32,
        devices=None if device == "cpu" else devices,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=50,
        profiler=profiler,
        deterministic=False,
        benchmark=False,
    )

    if auto_lr_find:
        trainer.tune(model=lit_model, datamodule=data_module)
    else:
        if not skip_train:
            trainer.fit(
                model=lit_model,
                datamodule=data_module,
                ckpt_path=ckpt_file if ckpt_file.is_file() else None,
            )
        if refine_and_calibrate:
            # Calibrate the logits
            temperature_model = TemperatureScaling(
                num_classes=num_classes,
                edge_class=edge_class,
                class_counts=class_counts,
                cultionet_ckpt=ckpt_file,
            )
            temperature_trainer.fit(
                model=temperature_model,
                datamodule=temperature_data_module,
                ckpt_path=temperature_ckpt_file
                if temperature_ckpt_file.is_file()
                else None,
            )
        if test_dataset is not None:
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
    lit_model: T.Optional[CultioLitModel] = None,
    enable_progress_bar: T.Optional[bool] = True,
    return_trainer: T.Optional[bool] = False,
) -> T.Tuple[T.Union[None, pl.Trainer], CultioLitModel]:
    """Loads a model from file.

    Args:
        ckpt_file (str | Path): The model checkpoint file.
        model_file (str | Path): The model file.
        device (str): The device to apply inference on.
        lit_model (CultioLitModel): A model to predict with. If `None`, the model
            is loaded from file.
        enable_progress_bar (Optional[bool]): Whether to use the progress bar.
        return_trainer (Optional[bool]): Whether to return the `pytorch_lightning` `Trainer`.
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
            devices=None if device == "cpu" else devices,
            gpus=1 if device == "gpu" else None,
            accelerator=device,
            num_processes=0,
            log_every_n_steps=0,
            logger=False,
            enable_progress_bar=enable_progress_bar,
        )
        if trainer_kwargs["accelerator"] == "cpu":
            del trainer_kwargs["devices"]
            del trainer_kwargs["gpus"]

        trainer = pl.Trainer(**trainer_kwargs)

    if lit_model is None:
        if model_file is not None:
            assert model_file.is_file(), "The model file does not exist."
            if not isinstance(num_features, int) or not isinstance(
                num_time_features, int
            ):
                raise TypeError(
                    "The features must be given to load the model file."
                )
            lit_model = CultioLitModel(
                num_features=num_features,
                num_time_features=num_time_features,
                filters=filters,
                num_classes=num_classes,
            )
            lit_model.load_state_dict(state_dict=torch.load(model_file))
        else:
            assert ckpt_file.is_file(), "The checkpoint file does not exist."
            lit_model = CultioLitModel.load_from_checkpoint(
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
    batch_size: int,
    load_batch_workers: int,
    device: str,
    devices: int,
    precision: int,
    num_classes: int,
    resampling: str,
    ref_res: float,
    compression: str,
    crop_temperature: T.Optional[torch.Tensor] = None,
    temperature_ckpt: T.Optional[Path] = None,
):
    reference_image = Path(reference_image)
    out_path = Path(out_path)
    ckpt_file = Path(ckpt)
    assert ckpt_file.is_file(), "The checkpoint file does not exist."

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
        ref_res=ref_res,
        resampling=resampling,
        compression=compression,
    )
    trainer_kwargs = dict(
        default_root_dir=str(ckpt_file.parent),
        callbacks=[pred_writer],
        precision=precision,
        devices=None if device == "cpu" else devices,
        gpus=1 if device == "gpu" else None,
        accelerator=device,
        num_processes=0,
        log_every_n_steps=0,
        logger=False,
    )

    trainer = pl.Trainer(**trainer_kwargs)
    cultionet_lit_model = CultioLitModel.load_from_checkpoint(
        checkpoint_path=str(ckpt_file)
    )

    geo_refine_model = None
    if crop_temperature is not None:
        geo_refine_model = GeoRefinement(out_channels=num_classes)
        geo_refine_model.load_state_dict(
            torch.load(temperature_ckpt.parent / "temperature.pt")
        )
        geo_refine_model.eval()
    setattr(cultionet_lit_model, "crop_temperature", crop_temperature)
    setattr(cultionet_lit_model, "temperature_lit_model", geo_refine_model)

    # Make predictions
    trainer.predict(
        model=cultionet_lit_model,
        datamodule=data_module,
        return_predictions=False,
    )


def predict(
    lit_model: CultioLitModel,
    data: Data,
    written: np.ndarray,
    data_values: torch.Tensor,
    w: Window = None,
    w_pad: Window = None,
    device: str = "cpu",
    include_maskrcnn: bool = False,
) -> np.ndarray:
    """Applies a model to predict image labels|values.

    Args:
        lit_model (CultioLitModel): A model to predict with.
        data (Data): The data to predict on.
        written (ndarray)
        data_values (Tensor)
        w (Optional[int]): The ``rasterio.windows.Window`` to write to.
        w_pad (Optional[int]): The ``rasterio.windows.Window`` to predict on.
        device (Optional[str])
    """
    norm_batch = zscores(data, data_values.mean, data_values.std)
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
