import typing as T
from pathlib import Path

import filelock
import geowombat as gw
import rasterio as rio
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from rasterio.windows import Window

from .data.constant import SCALE_FACTOR
from .data.data import Data


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
        resampling,
        compression: str,
        write_interval: str = "batch",
    ):
        super().__init__(write_interval)

        self.reference_image = reference_image
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with gw.open(reference_image, resampling=resampling) as src:
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

            profile = {
                "crs": src.crs,
                "transform": src.gw.transform,
                "height": src.gw.nrows,
                "width": src.gw.ncols,
                # distance (+1) + edge (+1) + crop (+1) crop types (+N)
                # `num_classes` includes background
                "count": 3 + num_classes - 1,
                "dtype": "uint16",
                "blockxsize": src.gw.col_chunks,
                "blockysize": src.gw.row_chunks,
                "driver": "GTiff",
                "sharing": False,
                "compress": compression,
            }
        profile["tiled"] = tile_size_is_correct(
            profile["blockxsize"], profile["blockysize"]
        )

        with rio.open(self.out_path, mode="w", **profile):
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
        crop_type_batch: T.Union[torch.Tensor, None],
    ) -> T.Dict[str, torch.Tensor]:

        distance_batch = distance_batch[batch_slice]
        edge_batch = edge_batch[batch_slice]
        crop_batch = crop_batch[batch_slice][1].unsqueeze(0)
        crop_type_batch = torch.zeros_like(edge_batch)

        return {
            "dist": distance_batch,
            "edge": edge_batch,
            "mask": crop_batch,
            "crop_type": crop_type_batch,
        }

    def get_batch_slice(self, batch: Data, batch_index: int) -> tuple:
        return (
            slice(0, None),
            slice(
                batch.padding[batch_index],
                batch.padding[batch_index] + batch.window_height[batch_index],
            ),
            slice(
                batch.padding[batch_index],
                batch.padding[batch_index] + batch.window_width[batch_index],
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
        distance = prediction["dist"]
        edge = prediction["edge"]
        crop = prediction["mask"]
        crop_type = prediction.get("crop_type")
        for batch_index in range(batch.x.shape[0]):
            write_window = Window(
                row_off=int(batch.window_row_off[batch_index]),
                col_off=int(batch.window_col_off[batch_index]),
                height=int(batch.window_height[batch_index]),
                width=int(batch.window_width[batch_index]),
            )

            batch_slice = self.get_batch_slice(batch, batch_index=batch_index)

            batch_dict = self.slice_predictions(
                batch_slice=batch_slice,
                distance_batch=distance[batch_index],
                edge_batch=edge[batch_index],
                crop_batch=crop[batch_index],
                crop_type_batch=crop_type[batch_index]
                if crop_type is not None
                else None,
            )

            stack = (
                torch.cat(
                    (
                        batch_dict["dist"],
                        batch_dict["edge"],
                        batch_dict["mask"],
                        batch_dict["crop_type"],
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
