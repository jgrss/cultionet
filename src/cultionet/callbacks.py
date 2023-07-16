import typing as T
import filelock
from pathlib import Path

import geowombat as gw
import rasterio as rio
from rasterio.windows import Window
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from torch_geometric.data import Data

from .data.const import SCALE_FACTOR
from .utils.reshape import ModelOutputs


def tile_size_is_correct(
    blockxsize: int, blockysize: int, tile_limit: int = 16
) -> bool:
    return max(blockxsize, blockysize) >= tile_limit


class LightningGTiffWriter(BasePredictionWriter):
    def __init__(
        self,
        reference_image: Path,
        out_path: Path,
        num_classes: int,
        ref_res: float,
        resampling,
        compression: str,
        write_interval: str = "batch",
    ):
        super().__init__(write_interval)

        self.reference_image = reference_image
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with gw.config.update(ref_res=ref_res):
            with gw.open(reference_image, resampling=resampling) as src:
                chunksize = src.gw.check_chunksize(
                    256, min(src.gw.nrows, src.gw.ncols)
                )
                src = src.chunk({"band": -1, "y": chunksize, "x": chunksize})
                profile = {
                    "crs": src.crs,
                    "transform": src.gw.transform,
                    "height": src.gw.nrows,
                    "width": src.gw.ncols,
                    # distance (+1) + edge (+1) + crop (+1) crop types (+N)
                    # `num_classes` includes background
                    "count": 3 + num_classes - 1,
                    "dtype": "uint16",
                    "blockxsize": max(64, src.gw.col_chunks),
                    "blockysize": max(64, src.gw.row_chunks),
                    "driver": "GTiff",
                    "sharing": False,
                    "compress": compression,
                }
        profile["tiled"] = tile_size_is_correct(
            profile["blockxsize"], profile["blockysize"]
        )
        with rio.open(out_path, mode="w", **profile):
            pass
        self.dst = rio.open(out_path, mode="r+")

    def write_on_epoch_end(
        self, trainer, pl_module, predictions, batch_indices
    ):
        self.dst.close()

    def reshape_predictions(
        self,
        batch: Data,
        distance_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        crop_batch: torch.Tensor,
        crop_type_batch: T.Union[torch.Tensor, None],
        batch_index: int,
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, T.Union[torch.Tensor, None]
    ]:
        pad_slice2d = (
            slice(
                int(batch.row_pad_before[batch_index]),
                int(batch.height[batch_index])
                - int(batch.row_pad_after[batch_index]),
            ),
            slice(
                int(batch.col_pad_before[batch_index]),
                int(batch.width[batch_index])
                - int(batch.col_pad_after[batch_index]),
            ),
        )
        pad_slice3d = (
            slice(0, None),
            slice(
                int(batch.row_pad_before[batch_index]),
                int(batch.height[batch_index])
                - int(batch.row_pad_after[batch_index]),
            ),
            slice(
                int(batch.col_pad_before[batch_index]),
                int(batch.width[batch_index])
                - int(batch.col_pad_after[batch_index]),
            ),
        )
        rheight = pad_slice2d[0].stop - pad_slice2d[0].start
        rwidth = pad_slice2d[1].stop - pad_slice2d[1].start

        def reshaper(x: torch.Tensor, channel_dims: int) -> torch.Tensor:
            if channel_dims == 1:
                return (
                    x.reshape(
                        int(batch.height[batch_index]),
                        int(batch.width[batch_index]),
                    )[pad_slice2d]
                    .contiguous()
                    .view(-1)[:, None]
                )
            else:
                return (
                    x.t()
                    .reshape(
                        channel_dims,
                        int(batch.height[batch_index]),
                        int(batch.width[batch_index]),
                    )[pad_slice3d]
                    .permute(1, 2, 0)
                    .reshape(rheight * rwidth, channel_dims)
                )

        distance_batch = reshaper(distance_batch, channel_dims=1)
        edge_batch = reshaper(edge_batch, channel_dims=1)
        crop_batch = reshaper(crop_batch, channel_dims=2)
        if crop_type_batch is not None:
            num_classes = crop_type_batch.size(1)
            crop_type_batch = reshaper(
                crop_type_batch, channel_dims=num_classes
            )

        return distance_batch, edge_batch, crop_batch, crop_type_batch

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
        distance = prediction["dist"]
        edge = prediction["edge"]
        crop = prediction["crop"]
        crop_type = prediction["crop_type"]
        for batch_index in batch.batch.unique():
            mask = batch.batch == batch_index
            w = Window(
                row_off=int(batch.window_row_off[batch_index]),
                col_off=int(batch.window_col_off[batch_index]),
                height=int(batch.window_height[batch_index]),
                width=int(batch.window_width[batch_index]),
            )
            w_pad = Window(
                row_off=int(batch.window_pad_row_off[batch_index]),
                col_off=int(batch.window_pad_col_off[batch_index]),
                height=int(batch.window_pad_height[batch_index]),
                width=int(batch.window_pad_width[batch_index]),
            )
            (
                distance_batch,
                edge_batch,
                crop_batch,
                crop_type_batch,
            ) = self.reshape_predictions(
                batch=batch,
                distance_batch=distance[mask],
                edge_batch=edge[mask],
                crop_batch=crop[mask],
                crop_type_batch=crop_type[mask]
                if crop_type is not None
                else None,
                batch_index=batch_index,
            )
            if crop_type_batch is None:
                crop_type_batch = torch.zeros(
                    (crop_batch.size(0), 2), dtype=crop_batch.dtype
                )
            mo = ModelOutputs(
                distance=distance_batch,
                edge=edge_batch,
                crop=crop_batch,
                crop_type=crop_type_batch,
                instances=None,
                apply_softmax=False,
            )
            stack = mo.stack_outputs(w, w_pad)
            stack = (stack * SCALE_FACTOR).clip(0, SCALE_FACTOR)

            with filelock.FileLock("./dst.lock"):
                self.dst.write(
                    stack,
                    indexes=range(1, self.dst.profile["count"] + 1),
                    window=w,
                )
