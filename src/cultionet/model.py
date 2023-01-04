import typing as T
from pathlib import Path
import logging
import json
import filelock

import numpy as np

from .data.const import SCALE_FACTOR
from .data.datasets import EdgeDataset, zscores
from .data.modules import EdgeDataModule
from .models.lightning import CultioLitModel, MaskRCNNLitModel, TemperatureScaling
from .utils.reshape import ModelOutputs
from .utils.logging import set_color_logger

import geowombat as gw
from scipy.stats import mode as sci_mode
import rasterio as rio
from rasterio.windows import Window
import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    StochasticWeightAveraging,
    ModelPruning,
    BasePredictionWriter
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

logging.getLogger('lightning').addHandler(logging.NullHandler())
logging.getLogger('lightning').propagate = False

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
    device: T.Optional[str] = 'gpu',
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
    trainable_backbone_layers: T.Optional[int] = 3
):
    """Fits a Mask R-CNN instance model

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
        shuffle=True
    )
    lit_model = MaskRCNNLitModel(
        cultionet_model_file=ckpt_file.parent / 'cultionet.pt',
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
        trainable_backbone_layers=trainable_backbone_layers
    )

    if reset_model:
        if ckpt_file.is_file():
            ckpt_file.unlink()
        model_file = ckpt_file.parent / 'maskrcnn.pt'
        if model_file.is_file():
            model_file.unlink()

    # Checkpoint
    cb_train_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode='min',
        monitor='loss',
        every_n_train_steps=0,
        every_n_epochs=1
    )
    # Validation and test loss
    cb_val_loss = ModelCheckpoint(monitor='val_loss')
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode='min',
        check_on_train_epoch_end=False
    )
    # Learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [
        lr_monitor,
        cb_train_loss,
        cb_val_loss,
        early_stop_callback
    ]
    if stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(
            swa_lrs=stochastic_weight_averaging_lr,
            swa_epoch_start=stochastic_weight_averaging_start
        ))
    if 0 < model_pruning <= 1:
        callbacks.append(
            ModelPruning('l1_unstructured', amount=model_pruning)
        )

    trainer = pl.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm='value',
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=precision,
        devices=1 if device == 'gpu' else None,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=10,
        profiler=None,
        deterministic=False,
        benchmark=False
    )

    if auto_lr_find:
        trainer.tune(model=lit_model, datamodule=data_module)
    else:
        trainer.fit(
            model=lit_model,
            datamodule=data_module,
            ckpt_path=ckpt_file if ckpt_file.is_file() else None
        )
        if test_dataset is not None:
            trainer.test(
                model=lit_model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path='last'
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
    filters: T.Optional[int] = 64,
    num_classes: T.Optional[int] = 2,
    edge_class: T.Optional[int] = None,
    class_weights: T.Sequence[float] = None,
    edge_weights: T.Sequence[float] = None,
    learning_rate: T.Optional[float] = 0.001,
    epochs: T.Optional[int] = 30,
    save_top_k: T.Optional[int] = 1,
    early_stopping_patience: T.Optional[int] = 7,
    early_stopping_min_delta: T.Optional[float] = 0.01,
    gradient_clip_val: T.Optional[float] = 1.0,
    reset_model: T.Optional[bool] = False,
    auto_lr_find: T.Optional[bool] = False,
    device: T.Optional[str] = 'gpu',
    profiler: T.Optional[str] = None,
    weight_decay: T.Optional[float] = 1e-5,
    precision: T.Optional[int] = 32,
    stochastic_weight_averaging: T.Optional[bool] = False,
    stochastic_weight_averaging_lr: T.Optional[float] = 0.05,
    stochastic_weight_averaging_start: T.Optional[float] = 0.8,
    model_pruning: T.Optional[bool] = False
):
    """Fits a model

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
    """
    ckpt_file = Path(ckpt_file)

    # Split the dataset into train/validation
    if spatial_partitions is not None:
        train_ds, val_ds = dataset.split_train_val_by_partition(
            spatial_partitions=spatial_partitions,
            partition_column=partition_column,
            val_frac=val_frac,
            partition_name=partition_name
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
        shuffle=True
    )
    temperature_data_module = EdgeDataModule(
        train_ds=val_ds,
        batch_size=batch_size,
        num_workers=load_batch_workers,
        shuffle=True
    )

    # Setup the Lightning model
    lit_model = CultioLitModel(
        num_features=train_ds.num_features,
        num_time_features=train_ds.num_time_features,
        num_classes=num_classes,
        filters=filters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        edge_weights=edge_weights,
        edge_class=edge_class
    )

    if reset_model:
        if ckpt_file.is_file():
            ckpt_file.unlink()
        model_file = ckpt_file.parent / 'cultionet.pt'
        if model_file.is_file():
            model_file.unlink()

    # Checkpoint
    cb_train_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode='min',
        monitor='loss',
        every_n_train_steps=0,
        every_n_epochs=1
    )
    # Validation and test loss
    cb_val_loss = ModelCheckpoint(monitor='val_loss')
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode='min',
        check_on_train_epoch_end=False
    )
    # Learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [
        lr_monitor,
        cb_train_loss,
        cb_val_loss,
        early_stop_callback
    ]
    if stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=stochastic_weight_averaging_lr,
                swa_epoch_start=stochastic_weight_averaging_start
            )
        )
    if 0 < model_pruning <= 1:
        callbacks.append(
            ModelPruning('l1_unstructured', amount=model_pruning)
        )

    trainer = pl.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm='value',
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=precision,
        devices=1 if device == 'gpu' else None,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=10,
        profiler=profiler,
        deterministic=False,
        benchmark=False
    )
    temperature_ckpt_file = ckpt_file.parent / 'temperature' / ckpt_file.name
    temperature_ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    # Temperature checkpoints
    temperature_cb_train_loss = ModelCheckpoint(
        dirpath=temperature_ckpt_file.parent,
        filename=temperature_ckpt_file.stem,
        save_last=True,
        save_top_k=save_top_k,
        mode='min',
        monitor='loss',
        every_n_train_steps=0,
        every_n_epochs=1
    )
    # Early stopping
    temperature_early_stop_callback = EarlyStopping(
        monitor='loss',
        min_delta=early_stopping_min_delta,
        patience=5,
        mode='min',
        check_on_train_epoch_end=False
    )
    temperature_callbacks = [
        lr_monitor,
        temperature_cb_train_loss,
        temperature_early_stop_callback
    ]
    temperature_trainer = pl.Trainer(
        default_root_dir=str(temperature_ckpt_file.parent),
        callbacks=temperature_callbacks,
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm='value',
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=10,
        precision=32,
        devices=1 if device == 'gpu' else None,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=10,
        profiler=profiler,
        deterministic=False,
        benchmark=False
    )

    if auto_lr_find:
        trainer.tune(model=lit_model, datamodule=data_module)
    else:
        trainer.fit(
            model=lit_model,
            datamodule=data_module,
            ckpt_path=ckpt_file if ckpt_file.is_file() else None
        )
        # Calibrate the logits
        temperature_model = TemperatureScaling(
            model=CultioLitModel.load_from_checkpoint(
                checkpoint_path=str(ckpt_file)
            ),
            learning_rate=0.1,
            max_iter=20,
            class_weights=class_weights,
            edge_class=edge_class
        )
        temperature_trainer.fit(
            model=temperature_model,
            datamodule=temperature_data_module,
            ckpt_path=temperature_ckpt_file if temperature_ckpt_file.is_file() else None
        )
        if test_dataset is not None:
            trainer.test(
                model=lit_model,
                dataloaders=data_module.test_dataloader(),
                ckpt_path='last'
            )
            logged_metrics = trainer.logged_metrics
            for k, v in logged_metrics.items():
                logged_metrics[k] = float(v)
            with open(Path(trainer.logger.save_dir) / 'test.metrics', mode='w') as f:
                f.write(json.dumps(logged_metrics))


def load_model(
    ckpt_file: T.Union[str, Path] = None,
    model_file: T.Union[str, Path] = None,
    num_features: T.Optional[int] = None,
    num_time_features: T.Optional[int] = None,
    num_classes: T.Optional[int] = None,
    filters: T.Optional[int] = None,
    device: T.Union[str, bytes] = 'gpu',
    lit_model: T.Optional[CultioLitModel] = None,
    enable_progress_bar: T.Optional[bool] = True,
    return_trainer: T.Optional[bool] = False
) -> T.Tuple[T.Union[None, pl.Trainer], CultioLitModel]:
    """Loads a model from file

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
            devices=1 if device == 'gpu' else None,
            gpus=1 if device == 'gpu' else None,
            accelerator=device,
            num_processes=0,
            log_every_n_steps=0,
            logger=False,
            enable_progress_bar=enable_progress_bar
        )
        if trainer_kwargs['accelerator'] == 'cpu':
            del trainer_kwargs['devices']
            del trainer_kwargs['gpus']

        trainer = pl.Trainer(**trainer_kwargs)

    if lit_model is None:
        if model_file is not None:
            assert model_file.is_file(), 'The model file does not exist.'
            if not isinstance(num_features, int) or not isinstance(num_time_features, int):
                raise TypeError('The features must be given to load the model file.')
            lit_model = CultioLitModel(
                num_features=num_features,
                num_time_features=num_time_features,
                filters=filters,
                num_classes=num_classes
            )
            lit_model.load_state_dict(state_dict=torch.load(model_file))
        else:
            assert ckpt_file.is_file(), 'The checkpoint file does not exist.'
            lit_model = CultioLitModel.load_from_checkpoint(
                checkpoint_path=str(ckpt_file)
            )
        lit_model.eval()
        lit_model.freeze()

    return trainer, lit_model


class LightningGTiffWriter(BasePredictionWriter):
    def __init__(
        self,
        reference_image: Path,
        out_path: Path,
        num_classes: int,
        ref_res: float,
        resampling,
        compression: str,
        write_interval: str = 'batch'
    ):
        super().__init__(write_interval)
        self.reference_image = reference_image
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with gw.config.update(ref_res=ref_res):
            with gw.open(reference_image, resampling=resampling) as src:
                profile = {
                    'crs': src.crs,
                    'transform': src.gw.transform,
                    'height': src.gw.nrows,
                    'width': src.gw.ncols,
                    # distance (+1) + edge (+1) + crop (+1) crop types (+N)
                    # `num_classes` includes background
                    'count': 3 + num_classes - 1,
                    'dtype': 'uint16',
                    'blockxsize': 64 if 64 < src.gw.ncols else src.gw.ncols,
                    'blockysize': 64 if 64 < src.gw.nrows else src.gw.nrows,
                    'driver': 'GTiff',
                    'sharing': False,
                    'compress': compression
                }
        profile['tiled'] = True if max(profile['blockxsize'], profile['blockysize']) >= 16 else False
        with rio.open(out_path, mode='w', **profile):
            pass
        self.dst = rio.open(out_path, mode='r+')

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        self.dst.close()

    def reshape_predictions(
        self,
        batch: Data,
        distance_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        crop_batch: torch.Tensor,
        crop_type_batch: T.Union[torch.Tensor, None],
        batch_index: int
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, T.Union[torch.Tensor, None]
    ]:
        pad_slice2d = (
            slice(
                int(batch.row_pad_before[batch_index]),
                int(batch.height[batch_index])-int(batch.row_pad_after[batch_index])
            ),
            slice(
                int(batch.col_pad_before[batch_index]),
                int(batch.width[batch_index])-int(batch.col_pad_after[batch_index])
            )
        )
        pad_slice3d = (
            slice(0, None),
            slice(
                int(batch.row_pad_before[batch_index]),
                int(batch.height[batch_index])-int(batch.row_pad_after[batch_index])
            ),
            slice(
                int(batch.col_pad_before[batch_index]),
                int(batch.width[batch_index])-int(batch.col_pad_after[batch_index])
            )
        )
        rheight = pad_slice2d[0].stop - pad_slice2d[0].start
        rwidth = pad_slice2d[1].stop - pad_slice2d[1].start
        def reshaper(x: torch.Tensor, channel_dims: int) -> torch.Tensor:
            if channel_dims == 1:
                return x.reshape(
                    int(batch.height[batch_index]), int(batch.width[batch_index])
                )[pad_slice2d].contiguous().view(-1)[:, None]
            else:
                return x.t().reshape(
                    channel_dims,
                    int(batch.height[batch_index]),
                    int(batch.width[batch_index])
                )[pad_slice3d].permute(1, 2, 0).reshape(rheight * rwidth, channel_dims)

        distance_batch = reshaper(distance_batch, channel_dims=1)
        edge_batch = reshaper(edge_batch, channel_dims=2)
        crop_batch = reshaper(crop_batch, channel_dims=2)
        if crop_type_batch is not None:
            num_classes = crop_type_batch.size(1)
            crop_type_batch = reshaper(crop_type_batch, channel_dims=num_classes)

        return distance_batch, edge_batch, crop_batch, crop_type_batch

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        distance = prediction['dist']
        edge = prediction['edge']
        crop = prediction['crop']
        crop_type = prediction['crop_type']
        for batch_index in batch.batch.unique():
            mask = batch.batch == batch_index
            w = Window(
                row_off=int(batch.window_row_off[batch_index]),
                col_off=int(batch.window_col_off[batch_index]),
                height=int(batch.window_height[batch_index]),
                width=int(batch.window_width[batch_index])
            )
            w_pad = Window(
                row_off=int(batch.window_pad_row_off[batch_index]),
                col_off=int(batch.window_pad_col_off[batch_index]),
                height=int(batch.window_pad_height[batch_index]),
                width=int(batch.window_pad_width[batch_index])
            )
            distance_batch, edge_batch, crop_batch, crop_type_batch = self.reshape_predictions(
                batch=batch,
                distance_batch=distance[mask],
                edge_batch=edge[mask],
                crop_batch=crop[mask],
                crop_type_batch=crop_type[mask] if crop_type is not None else None,
                batch_index=batch_index
            )
            if crop_type_batch is None:
                crop_type_batch = torch.zeros((crop_batch.size(0), 2), dtype=crop_batch.dtype)
            mo = ModelOutputs(
                distance=distance_batch,
                edge=edge_batch,
                crop=crop_batch,
                crop_type=crop_type_batch,
                instances=None,
                apply_softmax=False
            )
            stack = mo.stack_outputs(w, w_pad)
            stack = (stack * SCALE_FACTOR).clip(0, SCALE_FACTOR)

            with filelock.FileLock('./dst.lock'):
                self.dst.write(
                    stack,
                    indexes=range(1, self.dst.profile['count']+1),
                    window=w
                )


def predict_lightning(
    reference_image: T.Union[str, Path],
    out_path: T.Union[str, Path],
    ckpt: Path,
    dataset: EdgeDataset,
    batch_size: int,
    load_batch_workers: int,
    device: str,
    precision: int,
    num_classes: int,
    resampling: str,
    ref_res: float,
    compression: str,
    edge_temperature: T.Optional[torch.Tensor] = None,
    crop_temperature: T.Optional[torch.Tensor] = None
):
    reference_image = Path(reference_image)
    out_path = Path(out_path)
    ckpt_file = Path(ckpt)
    assert ckpt_file.is_file(), 'The checkpoint file does not exist.'

    data_module = EdgeDataModule(
        predict_ds=dataset,
        batch_size=batch_size,
        num_workers=load_batch_workers,
        shuffle=False
    )
    pred_writer = LightningGTiffWriter(
        reference_image=reference_image,
        out_path=out_path,
        num_classes=num_classes,
        ref_res=ref_res,
        resampling=resampling,
        compression=compression
    )
    trainer_kwargs = dict(
        default_root_dir=str(ckpt_file.parent),
        callbacks=[pred_writer],
        precision=precision,
        devices=1 if device == 'gpu' else None,
        gpus=1 if device == 'gpu' else None,
        accelerator=device,
        num_processes=0,
        log_every_n_steps=0,
        logger=False
    )

    trainer = pl.Trainer(**trainer_kwargs)
    lit_model = CultioLitModel.load_from_checkpoint(checkpoint_path=str(ckpt_file))
    setattr(lit_model, 'edge_temperature', edge_temperature)
    setattr(lit_model, 'crop_temperature', crop_temperature)

    # Make predictions
    trainer.predict(
        model=lit_model,
        datamodule=data_module,
        return_predictions=False
    )


def predict(
    lit_model: CultioLitModel,
    data: Data,
    written: np.ndarray,
    data_values: torch.Tensor,
    w: Window = None,
    w_pad: Window = None,
    device: str = 'cpu',
    include_maskrcnn: bool = False
) -> np.ndarray:
    """Applies a model to predict image labels|values

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
    if device == 'gpu':
        norm_batch = norm_batch.to('cuda')
        lit_model = lit_model.to('cuda')
    with torch.no_grad():
        distance, dist_1, dist_2, dist_3, dist_4, edge, crop = lit_model(norm_batch)
        crop_type = torch.zeros((crop.size(0), 2), dtype=crop.dtype)

        if include_maskrcnn:
            # TODO: fix this -- separate Mask R-CNN model
            predictions = lit_model.mask_forward(
                distance=distance,
                edge=edge,
                height=norm_batch.height,
                width=norm_batch.width,
                batch=None
            )
    instances = None
    if include_maskrcnn:
        instances = np.zeros((norm_batch.height, norm_batch.width), dtype='float64')
        if include_maskrcnn:
            scores = predictions[0]['scores'].squeeze()
            masks = predictions[0]['masks'].squeeze()
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
                    distance
                    .detach()
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
                instances = np.zeros((norm_batch.height, norm_batch.width), dtype='float64')

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
                                lyr = lyr_ref if scores[lyr_idx_ref] > scores[lyr_idx_targ] else lyr_targ
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
                        ((instances == 0) & conditional),
                        uid,
                        instances
                    )
                    uid = instances.max() + 1
                instances /= SCALE_FACTOR
            else:
                logger.warning('No fields were identified.')

    mo = ModelOutputs(
        distance=distance,
        edge=edge,
        crop=crop,
        crop_type=crop_type,
        instances=instances,
        apply_softmax=False
    )
    stack = mo.stack_outputs(w, w_pad)
    if include_maskrcnn:
        stack[:-1] = (stack[:-1] * SCALE_FACTOR).clip(0, SCALE_FACTOR)
        stack[-1] *= SCALE_FACTOR
    else:
        stack = (stack * SCALE_FACTOR).clip(0, SCALE_FACTOR)

    return stack
