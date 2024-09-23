from pathlib import Path
from typing import Union

import dask.array as da
import einops
import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask.delayed import Delayed
from dask.utils import SerializableLock
from rasterio.windows import Window
from retry import retry

from ..utils.logging import set_color_logger
from .data import Data

logger = set_color_logger(__name__)


class BatchStore:
    """``dask.array.store`` for data batches."""

    lock_ = SerializableLock()

    def __init__(
        self,
        data: xr.DataArray,
        write_path: Path,
        res: float,
        resampling: str,
        region: str,
        start_date: str,
        end_date: str,
        window_size: int,
        padding: int,
        compress_method: Union[int, str],
    ):
        self.data = data
        self.res = res
        self.resampling = resampling
        self.region = region
        self.start_date = start_date
        self.end_date = end_date
        self.write_path = write_path
        self.window_size = window_size
        self.padding = padding
        self.compress_method = compress_method

    def __setitem__(self, key: tuple, item: np.ndarray) -> None:
        time_range, index_range, y, x = key

        item_window = Window(
            col_off=x.start,
            row_off=y.start,
            width=x.stop - x.start,
            height=y.stop - y.start,
        )
        pad_window = Window(
            col_off=x.start,
            row_off=y.start,
            width=item.shape[-1],
            height=item.shape[-2],
        )

        self.write_batch(item, w=item_window, w_pad=pad_window)

    @retry(IOError, tries=5, delay=1)
    def write_batch(self, x: np.ndarray, w: Window, w_pad: Window):
        image_height = self.window_size + self.padding * 2
        image_width = self.window_size + self.padding * 2

        # Get row adjustments
        row_after_to_pad = image_height - w_pad.height

        # Get column adjustments
        col_after_to_pad = image_width - w_pad.width

        if any([row_after_to_pad > 0, col_after_to_pad > 0]):
            x = np.pad(
                x,
                pad_width=(
                    (0, 0),
                    (0, 0),
                    (0, row_after_to_pad),
                    (0, col_after_to_pad),
                ),
                mode="constant",
                constant_values=0,
            )

        x = einops.rearrange(
            torch.from_numpy(x.astype('int32')).to(dtype=torch.int32),
            't c h w -> 1 c t h w',
        )

        assert x.shape[-2:] == (
            image_height,
            image_width,
        ), "The padded array does not have the correct height/width dimensions."

        batch_id = f"{self.region}_{self.start_date}_{self.end_date}_{w.row_off}_{w.col_off}"

        # Get the upper left lat/lon
        (
            lat_left,
            lat_bottom,
            lat_right,
            lat_top,
        ) = self.data.gw.geodataframe.to_crs("epsg:4326").total_bounds.tolist()

        batch = Data(
            x=x,
            start_year=torch.tensor(
                [pd.Timestamp(self.start_date).year],
                dtype=torch.int32,
            ),
            end_year=torch.tensor(
                [pd.Timestamp(self.end_date).year],
                dtype=torch.int32,
            ),
            padding=[self.padding],
            window_row_off=[w.row_off],
            window_col_off=[w.col_off],
            window_height=[w.height],
            window_width=[w.width],
            res=[self.res],
            resampling=[self.resampling],
            left=torch.tensor([lat_left], dtype=torch.float32),
            bottom=torch.tensor([lat_bottom], dtype=torch.float32),
            right=torch.tensor([lat_right], dtype=torch.float32),
            top=torch.tensor([lat_top], dtype=torch.float32),
            batch_id=[batch_id],
        )

        batch.to_file(
            self.write_path / f"{batch_id}.pt",
            compress=self.compress_method,
        )

        try:
            _ = batch.from_file(self.write_path / f"{batch_id}.pt")
        except EOFError:
            raise IOError

    def __enter__(self) -> "BatchStore":
        self.closed = False

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def _open(self) -> "BatchStore":
        return self

    def save(self, data: da.Array, **kwargs) -> Delayed:
        da.store(data, self, lock=self.lock_, compute=True, **kwargs)
