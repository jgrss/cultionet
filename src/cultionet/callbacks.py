import hashlib
import typing as T
from pathlib import Path

import filelock
import geowombat as gw
import rasterio as rio
import torch
from lightning.pytorch.callbacks import (
    BasePredictionWriter,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from rasterio.windows import Window

from .data.constant import SCALE_FACTOR
from .enums import InferenceNames

PROGRESS_BAR_CALLBACK = RichProgressBar(
    refresh_rate=1,
    theme=RichProgressBarTheme(
        description="#cacaca",
        progress_bar="#ACFCD6",
        progress_bar_finished="#ACFCD6",
        progress_bar_pulse="#FCADED",
        batch_progress="#AA9439",
        time="grey54",
        processing_speed="grey70",
        metrics="#cacaca",
        metrics_text_delimiter=" • ",
        metrics_format=".3f",
    ),
)


def tile_size_is_correct(
    blockxsize: int, blockysize: int, tile_limit: int = 16
) -> bool:
    return max(blockxsize, blockysize) >= tile_limit


class LightningGTiffWriter(BasePredictionWriter):
    def __init__(
        self,
        reference_image: Path,
        out_path: Path,
        resampling,
        compression: str,
        write_interval: str = "batch",
    ):
        super().__init__(write_interval)

        self.reference_image = reference_image
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with gw.open(reference_image, resampling=resampling) as src:
            self.crs = src.crs
            rechunk = False
            new_row_chunks = src.gw.check_chunksize(
                src.gw.row_chunks, src.gw.nrows
            )
            if new_row_chunks != src.gw.row_chunks:
                rechunk = True
            new_col_chunks = src.gw.check_chunksize(
                src.gw.col_chunks, src.gw.ncols
            )
            if new_col_chunks != src.gw.col_chunks:
                rechunk = True
            if rechunk:
                src = src.chunk(
                    chunks={
                        'band': -1,
                        'y': new_row_chunks,
                        'x': new_col_chunks,
                    }
                )

            self.profile = {
                "crs": self.crs,
                "transform": src.gw.transform,
                "height": src.gw.nrows,
                "width": src.gw.ncols,
                # distance (+1) + edge (+1) + crop (+1)
                "count": 3,
                "dtype": "uint16",
                "blockxsize": src.gw.col_chunks,
                "blockysize": src.gw.row_chunks,
                "driver": "GTiff",
                "sharing": False,
                "compress": compression,
            }

        self.profile["tiled"] = tile_size_is_correct(
            self.profile["blockxsize"], self.profile["blockysize"]
        )

        with rio.open(self.out_path, mode="w", **self.profile):
            pass

        self.dst = rio.open(self.out_path, mode="r+")

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ):
        self.dst.close()

    def slice_predictions(
        self,
        batch_slice: tuple,
        distance_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        crop_batch: torch.Tensor,
    ) -> T.Dict[str, torch.Tensor]:

        distance_batch = distance_batch[batch_slice]
        edge_batch = edge_batch[batch_slice]
        crop_batch = crop_batch[batch_slice]

        if crop_batch.shape[0] > 1:
            crop_batch = crop_batch[[1]]

        return {
            InferenceNames.DISTANCE: distance_batch,
            InferenceNames.EDGE: edge_batch,
            InferenceNames.CROP: crop_batch,
        }

    def get_batch_slice(self, padding: int, window: Window) -> tuple:
        return (
            slice(0, None),
            slice(
                padding,
                padding + window.height,
            ),
            slice(
                padding,
                padding + window.width,
            ),
        )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        pred_df = prediction.get("pred_df")
        if pred_df is not None:
            if not pred_df.empty:
                pred_df = pred_df.set_crs(crs=self.crs, allow_override=True)
                # Create a hash to avoid long file names
                batch_hash = hashlib.shake_256(
                    '-'.join(batch.batch_id).encode()
                )
                pred_df.to_file(
                    self.out_path.parent
                    / f"{self.out_path.stem}_{batch_hash.hexdigest(16)}.gpkg",
                    driver="GPKG",
                )

        distance = prediction[InferenceNames.DISTANCE]
        edge = prediction[InferenceNames.EDGE]
        crop = prediction[InferenceNames.CROP]

        for batch_index in range(batch.x.shape[0]):
            window_row_off = int(batch.window_row_off[batch_index])
            window_height = int(batch.window_height[batch_index])
            window_col_off = int(batch.window_col_off[batch_index])
            window_width = int(batch.window_width[batch_index])

            if window_row_off + window_height > self.profile["height"]:
                window_height = self.profile["height"] - window_row_off
            if window_col_off + window_width > self.profile["width"]:
                window_width = self.profile["width"] - window_col_off

            write_window = Window(
                row_off=window_row_off,
                col_off=window_col_off,
                height=window_height,
                width=window_width,
            )

            batch_slice = self.get_batch_slice(
                padding=batch.padding[batch_index],
                window=write_window,
            )

            batch_dict = self.slice_predictions(
                batch_slice=batch_slice,
                distance_batch=distance[batch_index],
                edge_batch=edge[batch_index],
                crop_batch=crop[batch_index],
            )

            stack = (
                torch.cat(
                    (
                        batch_dict[InferenceNames.DISTANCE],
                        batch_dict[InferenceNames.EDGE],
                        batch_dict[InferenceNames.CROP],
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            stack = (stack * SCALE_FACTOR).clip(0, SCALE_FACTOR)

            with filelock.FileLock("./dst.lock"):
                self.dst.write(
                    stack,
                    indexes=range(1, self.dst.profile["count"] + 1),
                    window=write_window,
                )


def setup_callbacks(
    ckpt_file: T.Union[str, Path],
    stochastic_weight_averaging: bool = False,
    stochastic_weight_averaging_lr: float = 0.05,
    stochastic_weight_averaging_start: float = 0.8,
    model_pruning: bool = False,
) -> T.Tuple[LearningRateMonitor, T.Sequence[T.Any]]:
    # Checkpoint
    cb_train_loss = ModelCheckpoint(monitor="loss")
    # Validation and test loss
    cb_val_loss = ModelCheckpoint(
        dirpath=ckpt_file.parent,
        filename=ckpt_file.stem,
        save_last=False,
        save_top_k=1,
        mode="min",
        monitor="val_score",
        every_n_train_steps=0,
        every_n_epochs=1,
    )
    # Early stopping
    # early_stop_callback = EarlyStopping(
    #     monitor="val_score",
    #     min_delta=early_stopping_min_delta,
    #     patience=early_stopping_patience,
    #     mode="min",
    #     check_on_train_epoch_end=False,
    # )
    # Learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [lr_monitor, cb_train_loss, cb_val_loss]
    if stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=stochastic_weight_averaging_lr,
                swa_epoch_start=stochastic_weight_averaging_start,
            )
        )
    if 0 < model_pruning <= 1:
        callbacks.append(ModelPruning("l1_unstructured", amount=model_pruning))

    return lr_monitor, callbacks
