import typing as T
from functools import singledispatch
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr

from .data import Data


@singledispatch
def get_empty(template: torch.Tensor) -> torch.Tensor:
    return torch.tensor([])


@get_empty.register
def _(template: np.ndarray) -> np.ndarray:
    return np.array([])


@get_empty.register
def _(template: list) -> list:
    return []


@get_empty.register
def _(template: None) -> None:
    return None


@singledispatch
def concat(value: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.cat((value, other))


@concat.register
def _(value: np.ndarray, other: np.ndarray) -> np.ndarray:
    return np.concatenate((value, other))


@concat.register
def _(value: list, other: list) -> list:
    return value + other


def collate_fn(data_list: T.List[Data]) -> Data:
    kwargs = {}
    # Iterate over data keys
    for key in data_list[0].to_dict().keys():
        # Get an empty container
        key_value = get_empty(getattr(data_list[0], key))
        if key_value is not None:
            # Fill the container
            for sample in data_list:
                key_value = concat(key_value, getattr(sample, key))

        kwargs[key] = key_value

    return Data(**kwargs)


def get_image_list_dims(
    image_list: T.Sequence[T.Union[Path, str]], src: xr.DataArray
) -> T.Tuple[int, int]:
    """Gets the dimensions of the time series."""
    # Get the band count using the unique set of band/variable names
    nbands = len(list(set([Path(fn).parent.name for fn in image_list])))
    # Get the time count (.gw.nbands = number of total features)
    ntime = int(src.gw.nbands / nbands)

    return ntime, nbands


def create_data_object(
    x: np.ndarray,
    ntime: int,
    nbands: int,
    height: int,
    width: int,
    y: T.Optional[np.ndarray] = None,
    mask_y: T.Optional[np.ndarray] = None,
    bdist: T.Optional[np.ndarray] = None,
    zero_padding: T.Optional[int] = 0,
    other: T.Optional[np.ndarray] = None,
    **kwargs,
) -> Data:
    """Creates a training data object."""

    x = torch.from_numpy(x).float()

    boxes = None
    box_labels = None
    box_masks = None
    if mask_y is not None:
        boxes = mask_y["boxes"]
        box_labels = mask_y["labels"]
        box_masks = mask_y["masks"]

    if y is None:
        train_data = Data(
            x=x,
            height=height,
            width=width,
            ntime=ntime,
            nbands=nbands,
            boxes=boxes,
            box_labels=box_labels,
            box_masks=box_masks,
            zero_padding=zero_padding,
            **kwargs,
        )
    else:
        y = torch.from_numpy(y.flatten())
        if "float" in y.dtype.name:
            y = y.float()
        else:
            y = y.long()

        bdist_ = torch.from_numpy(bdist.flatten()).float()

        if other is None:
            train_data = Data(
                x=x,
                y=y,
                bdist=bdist_,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                boxes=boxes,
                box_labels=box_labels,
                box_masks=box_masks,
                zero_padding=zero_padding,
                **kwargs,
            )
        else:
            other_ = torch.from_numpy(other.flatten()).float()

            train_data = Data(
                x=x,
                y=y,
                bdist=bdist_,
                other=other_,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                boxes=boxes,
                box_labels=box_labels,
                box_masks=box_masks,
                zero_padding=zero_padding,
                **kwargs,
            )

    # Ensure the correct node count
    train_data.num_nodes = x.shape[0]

    return train_data


def split_multipolygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Splits a MultiPolygon into a Polygon."""

    # Check for multi-polygons
    multi_polygon_mask = df.geom_type == "MultiPolygon"

    if multi_polygon_mask.any():
        new_polygons = []
        for _, multi_polygon_df in df.loc[multi_polygon_mask].iterrows():
            # Split the multi-polygon into a list of polygons
            polygon_list = list(multi_polygon_df.geometry.geoms)
            # Duplicate the row, replacing the geometry
            for split_polygon in polygon_list:
                new_polygons.append(
                    multi_polygon_df.to_frame().T.assign(
                        geometry=[split_polygon]
                    )
                )

        # Stack and replace
        df = pd.concat(
            (
                df.loc[~multi_polygon_mask],
                pd.concat(new_polygons),
            )
        )

    return df
