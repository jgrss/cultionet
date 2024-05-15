import typing as T
from functools import singledispatch
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from affine import Affine
from rasterio.features import rasterize as rio_rasterize
from scipy.ndimage import label as nd_label
from scipy.ndimage import uniform_filter
from skimage.measure import regionprops

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


def roll(
    arr_pad: np.ndarray,
    shift: T.Union[int, T.Tuple[int, int]],
    axis: T.Union[int, T.Tuple[int, int]],
) -> np.ndarray:
    """Rolls array elements along a given axis and slices off padded edges."""
    return np.roll(arr_pad, shift, axis=axis)[1:-1, 1:-1]


def get_crop_count(array: np.ndarray, edge_class: int) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode="edge")

    rarray = roll(array_pad, 1, axis=0)
    crop_count = np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, -1, axis=0)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, 1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))
    rarray = roll(array_pad, -1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != edge_class))

    return crop_count


def get_edge_count(array: np.ndarray, edge_class: int) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode="edge")

    edge_count = np.uint8(roll(array_pad, 1, axis=0) == edge_class)
    edge_count += np.uint8(roll(array_pad, -1, axis=0) == edge_class)
    edge_count += np.uint8(roll(array_pad, 1, axis=1) == edge_class)
    edge_count += np.uint8(roll(array_pad, -1, axis=1) == edge_class)

    return edge_count


def get_non_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode="edge")

    non_count = np.uint8(roll(array_pad, 1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, 1, axis=1) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=1) == 0)

    return non_count


def cleanup_edges(
    array: np.ndarray,
    original: np.ndarray,
    edge_class: int,
) -> np.ndarray:
    """Removes crop pixels that border non-crop pixels."""
    array_pad = np.pad(original, pad_width=((1, 1), (1, 1)), mode="edge")
    original_zero = np.uint8(roll(array_pad, 1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, 1, axis=1) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=1) == 0)

    # Fill edges
    array = np.where(
        (array == 0)
        & (get_crop_count(array, edge_class) > 0)
        & (get_edge_count(array, edge_class) > 0),
        edge_class,
        array,
    )
    # Remove crops next to non-crop
    array = np.where(
        (array > 0)
        & (array != edge_class)
        & (get_non_count(array) > 0)
        & (get_edge_count(array, edge_class) > 0),
        0,
        array,
    )
    # Fill in non-cropland
    array = np.where(original_zero == 4, 0, array)
    # Remove isolated crop pixels (i.e., crop clumps with 2 or fewer pixels)
    array = np.where(
        (array > 0)
        & (array != edge_class)
        & (get_crop_count(array, edge_class) <= 1),
        0,
        array,
    )

    return array


def create_boundary_distances(
    labels_array: np.ndarray, train_type: str, cell_res: float
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates distances from boundaries."""
    if train_type.lower() == "polygon":
        mask = np.uint8(labels_array)
    else:
        mask = np.uint8(1 - labels_array)
    # Get unique segments
    segments = nd_label(mask)[0]
    # Get the distance from edges
    bdist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    bdist *= cell_res

    grad_x = cv2.Sobel(
        np.pad(bdist, 5, mode="edge"), cv2.CV_32F, dx=1, dy=0, ksize=5
    )
    grad_y = cv2.Sobel(
        np.pad(bdist, 5, mode="edge"), cv2.CV_32F, dx=0, dy=1, ksize=5
    )
    ori = cv2.phase(grad_x, grad_y, angleInDegrees=False)
    ori = ori[5:-5, 5:-5] / np.deg2rad(360)
    ori[labels_array == 0] = 0

    return mask, segments, bdist, ori


def normalize_boundary_distances(
    labels_array: np.ndarray,
    train_type: str,
    cell_res: float,
    normalize: bool = True,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Normalizes boundary distances."""

    # Create the boundary distances
    __, segments, bdist, ori = create_boundary_distances(
        labels_array, train_type, cell_res
    )
    dist_max = 1e9
    if normalize:
        dist_max = 1.0
        # Normalize each segment by the local max distance
        props = regionprops(segments, intensity_image=bdist)
        for p in props:
            if p.label > 0:
                bdist = np.where(
                    segments == p.label, bdist / p.max_intensity, bdist
                )
    bdist = np.nan_to_num(
        bdist.clip(0, dist_max), nan=1.0, neginf=1.0, posinf=1.0
    )
    ori = np.nan_to_num(ori.clip(0, 1), nan=1.0, neginf=1.0, posinf=1.0)

    return bdist, ori


def edge_gradient(array: np.ndarray) -> np.ndarray:
    """Calculates the morphological gradient of crop fields."""
    se = np.array([[1, 1], [1, 1]], dtype="uint8")
    array = np.uint8(
        cv2.morphologyEx(np.uint8(array), cv2.MORPH_GRADIENT, se) > 0
    )

    return array


def polygon_to_array(
    df: gpd.GeoDataFrame,
    reference_data: xr.DataArray,
    column: str,
    fill_value: int = 0,
    default_value: int = 1,
    all_touched: bool = False,
    dtype: str = "uint8",
) -> np.ndarray:
    """Converts a polygon, or polygons, to an array."""

    df = df.copy()

    if df.crs != reference_data.crs:
        # Transform the geometry
        df = df.to_crs(reference_data.crs)

    # Get the reference bounds
    left, bottom, right, top = reference_data.gw.bounds
    # Get intersecting polygons
    df = df.cx[left:right, bottom:top]
    # Clip the polygons to the reference bounds
    df = gpd.clip(df, reference_data.gw.geodataframe)

    # Get the output dimensions
    dst_transform = Affine(
        reference_data.gw.cellx, 0.0, left, 0.0, -reference_data.gw.celly, top
    )

    # Get the shape geometry and encoding value
    shapes = list(zip(df.geometry, df[column]))

    # Override dtype
    if (dtype == "uint8") and (df[column].max() > 255):
        dtype = "int32"

    # Convert the polygon(s) to an array
    polygon_array = rio_rasterize(
        shapes,
        out_shape=(reference_data.gw.nrows, reference_data.gw.ncols),
        fill=fill_value,
        transform=dst_transform,
        all_touched=all_touched,
        default_value=default_value,
        dtype=dtype,
    )

    return polygon_array


def fillz(x: np.ndarray) -> np.ndarray:
    """Fills zeros with the focal mean value."""

    focal_mean = uniform_filter(x, size=(0, 0, 3, 3), mode='reflect')

    return np.where(x == 0, focal_mean, x)
