import typing as T
from pathlib import Path
import logging

import numpy as np

from .data.datasets import EdgeDataset
from .data.modules import EdgeDataModule
from .models.lightning import CultioLitModel
from .utils.reshape import ModelOutputs

from rasterio.windows import Window
from torch_geometric import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

logging.getLogger('lightning').addHandler(logging.NullHandler())
logging.getLogger('lightning').propagate = False


def fit(
    dataset: EdgeDataset,
    ckpt_file: T.Union[str, Path],
    val_frac: T.Optional[float] = 0.2,
    batch_size: T.Optional[int] = 4,
    accumulate_grad_batches: T.Optional[int] = 1,
    filters: T.Optional[int] = 32,
    learning_rate: T.Optional[float] = 0.001,
    epochs: T.Optional[int] = 30,
    save_top_k: T.Optional[int] = 1,
    early_stopping_patience: T.Optional[int] = 7,
    early_stopping_min_delta: T.Optional[float] = 0.01,
    gradient_clip_val: T.Optional[float] = 1.0,
    random_seed: T.Optional[int] = 0,
    reset_model: T.Optional[bool] = False,
    auto_lr_find: T.Optional[bool] = False,
    device: T.Optional[str] = 'gpu',
    stochastic_weight_avg: T.Optional[bool] = False,
    weight_decay: T.Optional[float] = 1e-5,
):
    """Fits a model

    Args:
        dataset (EdgeDataset): The dataset to fit on.
        ckpt_file (str | Path): The checkpoint file path.
        val_frac (Optional[float]): The fraction of data to use for model validation.
        batch_size (Optional[int]): The data batch size.
        filters (Optional[int]): The number of initial model filters.
        learning_rate (Optional[float]): The model learning rate.
        epochs (Optional[int]): The number of epochs.
        save_top_k (Optional[int]): The number of top-k model checkpoints to save.
        early_stopping_patience (Optional[int]): The patience (epochs) before early stopping.
        early_stopping_min_delta (Optional[float]): The minimum change threshold before early stopping.
        gradient_clip_val (Optional[float]): A gradient clip limit.
        random_seed (Optional[int]): A random seed.
        reset_model (Optional[bool]): Whether to reset an existing model. Otherwise, pick up from last epoch of
            an existing model.
        auto_lr_find (Optional[bool]): Whether to search for an optimized learning rate.
        device (Optional[str]): The device to train on. Choices are ['cpu', 'gpu'].
    """
    ckpt_file = Path(ckpt_file)

    # Split the dataset into train/validation
    seed_everything(random_seed)
    train_ds, val_ds = dataset.split_train_val(val_frac=val_frac)

    # Setup the data module
    data_module = EdgeDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    # Setup the Lightning model
    lit_model = CultioLitModel(
        num_features=train_ds.num_features,
        num_time_features=train_ds.num_time_features,
        filters=filters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    if reset_model:
        if ckpt_file.is_file():
            ckpt_file.unlink()

    if ckpt_file.is_file():
        ckpt_path = ckpt_file
    else:
        ckpt_path = None

    # Callbacks
    cb_train_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.name,
        save_last=True,
        save_top_k=save_top_k,
        mode='min',
        monitor='loss',
        every_n_train_steps=0,
        every_n_epochs=1
    )

    cb_val_loss = ModelCheckpoint(monitor='val_loss')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode='min',
        check_on_train_epoch_end=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        default_root_dir=str(ckpt_file.parent),
        callbacks=[
            lr_monitor,
            cb_train_loss,
            cb_val_loss,
            early_stop_callback
        ],
        enable_checkpointing=True,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=False,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm='value',
        check_val_every_n_epoch=1,
        min_epochs=5 if epochs >= 5 else epochs,
        max_epochs=epochs,
        precision=32,
        devices=1 if device == 'gpu' else 0,
        gpus=1 if device == 'gpu' else 0,
        num_processes=0,
        accelerator=device,
        log_every_n_steps=10
    )

    if auto_lr_find:
        trainer.tune(model=lit_model, datamodule=data_module)
    else:
        trainer.fit(model=lit_model, datamodule=data_module, ckpt_path=ckpt_path)


def predict(
    dataset: EdgeDataset,
    ckpt_file: T.Union[str, Path],
    filters: T.Optional[int] = 32,
    device: T.Union[str, bytes] = 'gpu',
    w: Window = None,
    w_pad: Window = None,
    lit_model: T.Optional[CultioLitModel] = None
) -> T.Tuple[np.ndarray, CultioLitModel]:
    """Applies a model to predict image labels|values

    Args:
        dataset (EdgeDataset): The dataset to predict on.
        ckpt_file (str | Path): The checkpoint file path.
        filters (Optional[int]): The number of initial model filters.
        device (Optional[str]): The device to predict on. Choices are ['cpu', 'gpu'].
        w (Optional[int]): The ``rasterio.windows.Window`` to write to.
        w_pad (Optional[int]): The ``rasterio.windows.Window`` to predict on.
        lit_model (Optional[CultioLitModel]): A model to predict with. If not given, the model is loaded
            from the checkpoint file.
    """
    ckpt_file = Path(ckpt_file)

    data_module = EdgeDataModule(
        predict_ds=dataset, batch_size=1, num_workers=0
    )

    trainer_kwargs = dict(
        default_root_dir=str(ckpt_file.parent),
        precision=32,
        devices=1 if device == 'gpu' else 0,
        gpus=1 if device == 'gpu' else 0,
        accelerator=device,
        num_processes=0,
        progress_bar_refresh_rate=0,
        log_every_n_steps=0,
        logger=False
    )

    lit_kwargs = dict(
        num_features=dataset.num_features,
        num_time_features=dataset.num_time_features,
        filters=filters
    )

    trainer = pl.Trainer(**trainer_kwargs)
    if lit_model is None:
        assert ckpt_file.is_file(), 'The checkpoint file does not exist.'
        lit_model = CultioLitModel(**lit_kwargs).load_from_checkpoint(checkpoint_path=str(ckpt_file))
        lit_model.eval()
        lit_model.freeze()

    # Make predictions
    distance, edge, crop, crop_r = trainer.predict(model=lit_model, datamodule=data_module)[0]

    mo = ModelOutputs(
        distance=distance,
        edge=edge,
        crop=crop,
        crop_r=crop_r,
        apply_softmax=False
    )
    stack = mo.stack_outputs(w, w_pad)

    return stack, lit_model
