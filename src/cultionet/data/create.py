import typing as T
from pathlib import Path

from .const import (
    POLYGON_CROP_CLASS,
    POLYGON_FALLOW_CLASS,
    EDGE_CLASS,
    CROP_CLASS,
    FALLOW_CLASS
)
from .lookup import CDL_CROP_LABELS_r
from .utils import LabeledData, get_image_list_dims
from ..augment.augmentation import augment
from ..errors import TopologyClipError
from ..utils.geometry import bounds_to_frame, warp_by_image

import geowombat as gw
from geowombat.core import polygon_to_array
import numpy as np
from scipy.ndimage.measurements import label as nd_label, sum as nd_sum
from scipy import stats as sci_stats
import cv2
import geopandas as gpd
from skimage.measure import regionprops
from skimage.morphology import thin as sk_thin
import xarray as xr
from tqdm.auto import tqdm
import torch
from torch_geometric.data import Data


def remove_noncrop(xvars: np.ndarray, labels_array: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """Removes non-crop distances and edges
    """
    # This masking aligns with the reference studies
    # Remove non-crop distances
    bdist = xvars[-1]
    bdist = bdist * np.uint8(labels_array > 0)
    xvars[-1] = bdist
    # Remove non-crop edges
    bdist = np.pad(bdist, pad_width=((5, 5), (5, 5)), mode='edge')
    dist_mean = np.zeros_like(bdist)
    dist_mean[1:-1, 1:-1] = dist_mean[1:-1, 1:-1] + np.roll(bdist, 1, axis=1)[1:-1, 1:-1]
    dist_mean[1:-1, 1:-1] = dist_mean[1:-1, 1:-1] + np.roll(bdist, -1, axis=1)[1:-1, 1:-1]
    dist_mean[1:-1, 1:-1] = dist_mean[1:-1, 1:-1] + np.roll(bdist, 1, axis=0)[1:-1, 1:-1]
    dist_mean[1:-1, 1:-1] = dist_mean[1:-1, 1:-1] + np.roll(bdist, -1, axis=0)[1:-1, 1:-1]
    dist_mean = (dist_mean / 4.0)[5:-5, 5:-5]
    labels_array = labels_array * np.uint8(dist_mean > 0)

    return xvars, labels_array


def roll(
    arr_pad: np.ndarray,
    shift: T.Union[int, T.Tuple[int, int]],
    axis: T.Union[int, T.Tuple[int, int]]
) -> np.ndarray:
    """Rolls array elements along a given axis and slices off padded edges"""
    return np.roll(arr_pad, shift, axis=axis)[1:-1, 1:-1]


def edge_gradient(array: np.ndarray, train_type: str, cell_res: float) -> np.ndarray:
    """Calculates the morphological gradient of crop fields
    """
    se = np.array(
        [
            [1, 1],
            [1, 1]
        ], dtype='uint8'
    )
    array_shape = array.shape
    dim = (array.shape[0] * 2, array.shape[1] * 2)
    array = cv2.resize(array, dim, interpolation=cv2.INTER_NEAREST)
    bdist = normalize_boundary_distances(
        np.uint8(array > 0),
        train_type,
        cell_res * 0.5,
        normalize=False
    )[0]
    bdist = cv2.resize(np.uint8(bdist > 0), array_shape, interpolation=cv2.INTER_CUBIC)
    array = np.uint8(cv2.morphologyEx(np.uint8(bdist > 0), cv2.MORPH_GRADIENT, se) > 0)

    return array


def focal_compare(arr: np.ndarray) -> np.ndarray:
    """Calculates a focal comparison of neighbors and the central pixel
    """
    # crop mask
    out = [np.uint8(arr > 0)]
    arr_pad = np.pad(arr, pad_width=((1, 1), (1, 1)), mode='edge')
    # Direct neighbors
    out.append(np.where(roll(arr_pad, 1, axis=0) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, -1, axis=0) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, 1, axis=1) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, -1, axis=1) == arr, 1, 0))
    # Corner neighbors
    out.append(np.where(roll(arr_pad, (1, 1), axis=(0, 1)) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, (1, -1), axis=(0, 1)) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, (-1, 1), axis=(0, 1)) == arr, 1, 0))
    out.append(np.where(roll(arr_pad, (-1, -1), axis=(0, 1)) == arr, 1, 0))

    return np.array(out).sum(axis=0) * out[0]


def focal_stat(arr: np.ndarray, stat: str = 'var') -> np.ndarray:
    """Calculates the focal variance
    """
    out = [arr.squeeze().copy()]
    arr_pad = np.pad(arr, pad_width=((1, 1), (1, 1)), mode='edge')
    # Direct neighbors
    out.append(roll(arr_pad, 1, axis=0))
    out.append(roll(arr_pad, -1, axis=0))
    out.append(roll(arr_pad, 1, axis=1))
    out.append(roll(arr_pad, -1, axis=1))
    # Corner neighbors
    out.append(roll(arr_pad, (1, 1), axis=(0, 1)))
    out.append(roll(arr_pad, (1, -1), axis=(0, 1)))
    out.append(roll(arr_pad, (-1, 1), axis=(0, 1)))
    out.append(roll(arr_pad, (-1, -1), axis=(0, 1)))

    return getattr(np, stat)(np.array(out), axis=0)


def fill_field_gaps(arr: np.ndarray, reset_edges: T.Optional[bool] = False) -> np.ndarray:
    """Fills gaps between fields and edges
    """
    arr = arr.squeeze().copy().astype('float64')
    arr[arr == EDGE_CLASS] = np.nan
    out = [arr]
    arr_pad = np.pad(arr, pad_width=((1, 1), (1, 1)), mode='edge')
    # Direct neighbors
    out.append(roll(arr_pad, 1, axis=0))
    out.append(roll(arr_pad, -1, axis=0))
    out.append(roll(arr_pad, 1, axis=1))
    out.append(roll(arr_pad, -1, axis=1))
    # Corner neighbors
    # out.append(np.roll(arr_pad, (1, 1), axis=(0, 1))[1:-1, 1:-1])
    # out.append(np.roll(arr_pad, (1, -1), axis=(0, 1))[1:-1, 1:-1])
    # out.append(np.roll(arr_pad, (-1, 1), axis=(0, 1))[1:-1, 1:-1])
    # out.append(np.roll(arr_pad, (-1, -1), axis=(0, 1))[1:-1, 1:-1])

    nsum = np.nansum(np.array(out), axis=0)
    if reset_edges:
        arr[np.isnan(arr)] = EDGE_CLASS
    out = np.where((arr == 0) & (nsum > 0), CROP_CLASS, arr.astype('int64'))
    if not reset_edges:
        out[np.isnan(out)] = 0

    return out


def check_slivers(
    labels_array: np.ndarray, edges: np.ndarray, segments: np.ndarray
) -> T.Tuple[np.ndarray, list]:
    """Checks for small segment slivers
    """
    props = regionprops(segments)
    for p in props:
        min_row, min_col, max_row, max_col = p.bbox
        if (max_row - min_row <= 1) or (max_col - min_col <= 1):
            labels_array[segments == p.label] = 0
    labels_array[edges == 1] = EDGE_CLASS

    return labels_array, props


def make_crops_uniform(
    labels_array: np.ndarray, edges: np.ndarray, data_type: str
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Makes each segment uniform in its class value
    """
    segments, num_objects = nd_label(np.uint8(edges == 0))
    index = np.unique(segments)
    crop_totals = nd_sum(np.uint8(labels_array > 0), labels=segments, index=index)
    for lidx in range(0, len(index)):
        lab = index[lidx]
        if crop_totals[lidx] == 0:
            # No cropland
            labels_array[segments == lab] = 0
        else:
            seg_area = (segments == lab).sum()
            crop_prop = crop_totals[lidx] / seg_area
            if crop_prop > 0.5:
                # Majority cropland
                if data_type == 'boundaries':
                    labels_array[segments == lab] = CROP_CLASS
                else:
                    idx = np.where(segments == lab)
                    crop_pixels = labels_array[idx].flatten()
                    labels_array[segments == lab] = sci_stats.mode(crop_pixels).mode[0]
            else:
                labels_array[segments == lab] = 0

    return labels_array, segments


def recode_crop_labels(
    labels_array: np.ndarray, lc_array: np.ndarray, lc_is_cdl: bool, data_type: str
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Recodes crop labels for non-edge values
    """
    edges = labels_array.copy()
    # Recode the crop labels
    if lc_is_cdl:
        # Ensure the land cover array is the same shape
        if lc_array.shape != labels_array.shape:
            row_diff = labels_array.shape[0] - lc_array.shape[0]
            col_diff = labels_array.shape[1] - lc_array.shape[1]
            lc_array = np.pad(lc_array, pad_width=((0, row_diff), (0, col_diff)), mode='edge')
        recoded_labels = np.zeros_like(labels_array)
        crop_counter = 1
        for lc_value, lc_name in CDL_CROP_LABELS_r.items():
            # TODO: testing two crops
            # if lc_value not in [CDL_CROP_LABELS['maize'], CDL_CROP_LABELS['soybeans']]:
            #     continue
            # Crop = 1
            if data_type == 'boundaries':
                recoded_labels[(lc_array == lc_value) & (edges == 0)] = 1
            else:
                recoded_labels[(lc_array == lc_value) & (edges == 0)] = crop_counter
                # Check if the code was added
                if (recoded_labels == crop_counter).sum() > 0:
                    crop_counter += 1
    else:
        recoded_labels = np.where(lc_array == 1, 1, 0)

    return recoded_labels, edges


def close_edge_ends(array: np.ndarray) -> np.ndarray:
    """Closes 1 pixel gaps at image edges
    """
    # Top
    idx = np.where(array[1] == 1)
    z = np.zeros(array.shape[1], dtype='uint8')
    z[idx] = 1
    array[0] = z
    # Bottom
    idx = np.where(array[-2] == 1)
    z = np.zeros(array.shape[1], dtype='uint8')
    z[idx] = 1
    array[-1] = z
    # Left
    idx = np.where(array[:, 1] == 1)
    z = np.zeros(array.shape[0], dtype='uint8')
    z[idx] = 1
    array[:, 0] = z
    # Right
    idx = np.where(array[:, -2] == 1)
    z = np.zeros(array.shape[0], dtype='uint8')
    z[idx] = 1
    array[:, -1] = z

    return array


def get_other_crop_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    rarray = roll(array_pad, 1, axis=0)
    crop_count = np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, -1, axis=0)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, 1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))
    rarray = roll(array_pad, -1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != array) & (array > 0))

    return crop_count


def fill_edge_gaps(labels: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fills neighboring 1-pixel edge gaps
    """
    # array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')
    # hsum = roll(array_pad, 1, axis=0) + roll(array_pad, -1, axis=0)
    # vsum = roll(array_pad, 1, axis=1) + roll(array_pad, -1, axis=1)
    # array = np.where(
    #     (hsum == 2) & (vsum == 0), 1, array
    # )
    # array = np.where(
    #     (hsum == 0) & (vsum == 2), 1, array
    # )
    other_count = get_other_crop_count(np.where(array == 1, 0, labels))
    array = np.where(
        other_count > 0, 1, array
    )

    return array


def get_crop_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    rarray = roll(array_pad, 1, axis=0)
    crop_count = np.uint8((rarray > 0) & (rarray != EDGE_CLASS))
    rarray = roll(array_pad, -1, axis=0)
    crop_count += np.uint8((rarray > 0) & (rarray != EDGE_CLASS))
    rarray = roll(array_pad, 1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != EDGE_CLASS))
    rarray = roll(array_pad, -1, axis=1)
    crop_count += np.uint8((rarray > 0) & (rarray != EDGE_CLASS))

    return crop_count


def get_edge_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    edge_count = np.uint8(roll(array_pad, 1, axis=0) == EDGE_CLASS)
    edge_count += np.uint8(roll(array_pad, -1, axis=0) == EDGE_CLASS)
    edge_count += np.uint8(roll(array_pad, 1, axis=1) == EDGE_CLASS)
    edge_count += np.uint8(roll(array_pad, -1, axis=1) == EDGE_CLASS)

    return edge_count


def get_non_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode='edge')

    non_count = np.uint8(roll(array_pad, 1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=0) == 0)
    non_count += np.uint8(roll(array_pad, 1, axis=1) == 0)
    non_count += np.uint8(roll(array_pad, -1, axis=1) == 0)

    return non_count


def cleanup_edges(array: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Removes crop pixels that border non-crop pixels
    """
    array_pad = np.pad(original, pad_width=((1, 1), (1, 1)), mode='edge')
    original_zero = np.uint8(roll(array_pad, 1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=0) == 0)
    original_zero += np.uint8(roll(array_pad, 1, axis=1) == 0)
    original_zero += np.uint8(roll(array_pad, -1, axis=1) == 0)

    # Fill edges
    array = np.where(
        (array == 0) & (get_crop_count(array) > 0) & (get_edge_count(array) > 0),
        EDGE_CLASS, array
    )
    # Remove crops next to non-crop
    array = np.where(
        (array > 0) & (array != EDGE_CLASS) & (get_non_count(array) > 0) & (get_edge_count(array) > 0),
        0, array
    )
    # Fill in non-cropland
    array = np.where(
        original_zero == 4, 0, array
    )
    # Remove isolated crop pixels (i.e., crop clumps with 2 or fewer pixels)
    array = np.where(
        (array > 0) & (array != EDGE_CLASS) & (get_crop_count(array) <= 1), 0, array
    )

    return array


def is_grid_processed(
    process_path: Path,
    transforms: T.List[str],
    group_id: str,
    grid: T.Union[str, int],
    n_ts: int
) -> bool:
    """Checks if a grid is already processed
    """
    batch_stored = False
    for aug in transforms:
        if aug.startswith('ts-'):
            for i in range(0, n_ts):
                train_id = f'{group_id}_{grid}_{aug}_{i:03d}'
                train_path = process_path / f'data_{train_id}.pt'
                if train_path.is_file():
                    train_data = torch.load(train_path)
                    if train_data.train_id == train_id:
                        batch_stored = True

        else:
            train_id = f'{group_id}_{grid}_{aug}'
            train_path = process_path / f'data_{train_id}.pt'
            if train_path.is_file():
                train_data = torch.load(train_path)
                if train_data.train_id == train_id:
                    batch_stored = True

    return batch_stored


def create_boundary_distances(
    labels_array: np.ndarray, train_type: str, cell_res: float
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates distances from boundaries
    """
    if train_type.lower() == 'polygon':
        mask = np.uint8(labels_array)
    else:
        mask = np.uint8(1 - labels_array)
    # Get unique segments
    segments = nd_label(mask)[0]
    # Get the distance from edges
    bdist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        3
    )
    bdist *= cell_res

    grad_x = cv2.Sobel(
        np.pad(bdist, 5, mode='edge'),
        cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=5
    )
    grad_y = cv2.Sobel(
        np.pad(bdist, 5, mode='edge'),
        cv2.CV_32F,
        dx=0,
        dy=1,
        ksize=5
    )
    ori = cv2.phase(grad_x, grad_y, angleInDegrees=False)
    ori = ori[5:-5, 5:-5] / np.deg2rad(360)
    ori[labels_array == 0] = 0

    return mask, segments, bdist, ori


def normalize_boundary_distances(
    labels_array: np.ndarray,
    train_type: str,
    cell_res: float,
    normalize: bool = True
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Normalizes boundary distances
    """
    # Create the boundary distances
    __, segments, bdist, ori = create_boundary_distances(labels_array, train_type, cell_res)
    dist_max = 1e9
    if normalize:
        dist_max = 1.0
        # Normalize each segment by the local max distance
        props = regionprops(segments, intensity_image=bdist)
        for p in props:
            if p.label > 0:
                bdist = np.where(
                    segments == p.label,
                    bdist / p.max_intensity,
                    bdist
                )
    bdist = np.nan_to_num(
        bdist.clip(0, dist_max),
        nan=1.0,
        neginf=1.0,
        posinf=1.0
    )
    ori = np.nan_to_num(
        ori.clip(0, 1),
        nan=1.0,
        neginf=1.0,
        posinf=1.0
    )

    return bdist, ori


def create_image_vars(
    image: T.Union[str, Path, list],
    bounds: tuple,
    num_workers: int,
    gain: float = 1e-4,
    offset: float = 0.0,
    grid_edges: T.Optional[gpd.GeoDataFrame] = None,
    ref_res: T.Optional[float] = 10.0,
    resampling: T.Optional[str] = 'nearest',
    crop_column: T.Optional[str] = 'class',
    keep_crop_classes: T.Optional[bool] = False
) -> T.Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Creates the initial image training data
    """
    if isinstance(image, list):
        image = [str(fn) for fn in image]

    # Open the image variables
    with gw.config.update(ref_bounds=bounds, ref_res=ref_res):
        with gw.open(
            image,
            stack_dim='band',
            band_names=list(range(1, len(image) + 1)),
            resampling=resampling
        ) as src_ts:
            # X variables
            time_series = (
                (src_ts.astype('float64') * gain + offset)
                .gw.compute(num_workers=num_workers)
                .clip(0, 1)
            )
            # Get the time and band count
            ntime, nbands = get_image_list_dims(image, src_ts)
            if grid_edges is not None:
                # Get the field polygons
                labels_array = polygon_to_array(
                    grid_edges,
                    col=crop_column,
                    data=src_ts,
                    all_touched=False
                ).squeeze().gw.compute(num_workers=num_workers)
                labels_array_copy = labels_array.copy()
                # Get the field edges
                edges = polygon_to_array(
                    (
                        grid_edges
                        .boundary
                        .to_frame(name='geometry')
                        .reset_index()
                        .rename(columns={'index': crop_column})
                    ),
                    col=crop_column,
                    data=src_ts,
                    all_touched=False
                ).squeeze().gw.compute(num_workers=num_workers)
                edges[edges > 0] = 1
                # def save_labels(out_fig: Path):
                #     import matplotlib.pyplot as plt
                #     fig, axes = plt.subplots(1, 3, figsize=(6, 5), sharey=True, sharex=True, dpi=300)
                #     axes = axes.flatten()
                #     axes[0].imshow(labels_array, interpolation='nearest')
                #     axes[1].imshow(edge_array, interpolation='nearest')
                #     axes[2].imshow(test, interpolation='nearest')
                #     for ax in axes:
                #         ax.axis('off')

                #     plt.tight_layout()
                #     plt.savefig(out_fig, dpi=300)
                # import uuid
                # fig_dir = Path('figures')
                # fig_dir.mkdir(exist_ok=True, parents=True)
                # save_labels(
                #     out_fig=fig_dir / f'{uuid.uuid4().hex}.png'
                # )
                # import ipdb; ipdb.set_trace()
                # Recode
                if not keep_crop_classes:
                    if FALLOW_CLASS != -999:
                        labels_array = np.where(
                            labels_array == POLYGON_FALLOW_CLASS, FALLOW_CLASS, np.where(
                                labels_array == 0, 0, CROP_CLASS
                            )
                        )
                    else:
                        labels_array = np.where(
                            labels_array > 0, CROP_CLASS, 0
                        )
                # Set edges
                labels_array[edges == 1] = EDGE_CLASS
                # No crop pixel should border non-crop
                labels_array = cleanup_edges(labels_array, labels_array_copy)
                # Normalize the boundary distances for each segment
                bdist, ori = normalize_boundary_distances(
                    np.uint8((labels_array > 0) & (labels_array != EDGE_CLASS)),
                    grid_edges.geom_type.values[0],
                    src_ts.gw.celly
                )
                # def save_labels(out_fig: Path):
                #     import matplotlib.pyplot as plt
                #     fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True, dpi=300)
                #     axes = axes.flatten()
                #     for ax, im, title in zip(
                #         axes,
                #         (labels_array_copy, labels_array, bdist, ori),
                #         ('Fields', 'Edges', 'Distance', 'Orientation')
                #     ):
                #         ax.imshow(im, interpolation='nearest')
                #         ax.set_title(title)
                #         ax.axis('off')

                #     plt.tight_layout()
                #     plt.savefig(out_fig, dpi=300)
                # import uuid
                # fig_dir = Path('figures')
                # fig_dir.mkdir(exist_ok=True, parents=True)
                # save_labels(
                #     out_fig=fig_dir / f'{uuid.uuid4().hex}.png'
                # )
            else:
                labels_array = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')
                bdist = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                ori = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='float64')
                edges = np.zeros((src_ts.gw.nrows, src_ts.gw.ncols), dtype='uint8')

    return time_series, labels_array, edges, bdist, ori, ntime, nbands


def create_dataset(
    image_list: T.List[T.List[T.Union[str, Path]]],
    df_grids: gpd.GeoDataFrame,
    df_edges: gpd.GeoDataFrame,
    group_id: str = None,
    process_path: Path = None,
    transforms: T.List[str] = None,
    gain: float = 0.0001,
    offset: float = 0.0,
    ref_res: float = 10.0,
    resampling: str = 'nearest',
    num_workers: int = 1,
    grid_size: T.Optional[T.Union[T.Tuple[int, int], T.List[int], None]] = None,
    lc_path: T.Optional[T.Union[str, None]] = None,
    n_ts: T.Optional[int] = 2,
    data_type: T.Optional[str] = 'boundaries',
    instance_seg: T.Optional[bool] = False,
    zero_padding: T.Optional[int] = 0,
    crop_column: T.Optional[str] = 'class',
    keep_crop_classes: T.Optional[bool] = False
) -> None:
    """Creates a dataset for training

    Args:
        image_list: A list of images.
        df_grids: The training grids.
        df_edges: The training edges.
        group_id: A group identifier, used for logging.
        process_path: The main processing path.
        transforms: A list of augmentation transforms to apply.
        gain: A gain factor to apply to the images.
        offset: An offset factor to apply to the images.
        ref_res: The reference cell resolution to resample the images to.
        resampling: The image resampling method.
        num_workers: The number of dask workers.
        grid_size: The requested grid size, in (rows, columns) or (height, width).
        lc_path: The land cover image path.
        n_ts: The number of temporal augmentations.
        data_type: The target data type.
        instance_seg: Whether to get instance segmentation mask targets.
        zero_padding: Zero padding to apply.
        crop_column: The crop column name in the polygon vector files.
        keep_crop_classes: Whether to keep the crop classes as they are (True) or recode all
            non-zero classes to crop (False).
    """
    if transforms is None:
        transforms = ['none']

    merged_grids = []
    sindex = df_grids.sindex

    # Get the image CRS
    with gw.open(image_list[0]) as src:
        image_crs = src.crs

    with tqdm(total=df_grids.shape[0], desc='Check') as pbar:
        for row in df_grids.itertuples():
            # Clip the edges to the current grid
            try:
                grid_edges = gpd.clip(df_edges, row.geometry)
            except:
                print(TopologyClipError('The input GeoDataFrame contains topology errors.'))
                df_edges = gpd.GeoDataFrame(
                    data=df_edges['class'].values, columns=['class'], geometry=df_edges.buffer(0).geometry
                )
                grid_edges = gpd.clip(df_edges, row.geometry)

            # These are grids with no crop fields. They should still
            # be used for training.
            if grid_edges.loc[~grid_edges.is_empty].empty:
                grid_edges = df_grids.copy()
                grid_edges.loc[:, 'class'] = 0
            # Remove empty geometry
            grid_edges = grid_edges.loc[~grid_edges.is_empty]

            if not grid_edges.empty:
                # Check if the edges overlap multiple grids
                int_idx = sorted(list(sindex.intersection(tuple(grid_edges.total_bounds.flatten()))))

                if len(int_idx) > 1:
                    # Check if any of the grids have already been stored
                    if any([rowg in merged_grids for rowg in df_grids.iloc[int_idx].grid.values.tolist()]):
                        pbar.update(1)
                        pbar.set_description(f'No edges for {group_id}')
                        continue

                    grid_edges = gpd.clip(df_edges, df_grids.iloc[int_idx].geometry)
                    merged_grids.append(row.grid)

                # Make polygons unique
                nonzero_mask = grid_edges['class'] != 0
                if nonzero_mask.any():
                    grid_edges.loc[nonzero_mask, 'class'] = range(1, nonzero_mask.sum()+1)

                # left, bottom, right, top
                ref_bounds = df_grids.to_crs(image_crs).iloc[int_idx].total_bounds.tolist()
                if grid_size is not None:
                    height, width = grid_size
                    left, bottom, right, top = ref_bounds
                    ref_bounds = [left, top-ref_res*height, left+ref_res*width, top]

                # Data for graph network
                xvars, labels_array, edges, bdist, ori, ntime, nbands = create_image_vars(
                    image_list,
                    bounds=ref_bounds,
                    num_workers=num_workers,
                    gain=gain,
                    offset=offset,
                    grid_edges=grid_edges if nonzero_mask.any() else None,
                    ref_res=ref_res,
                    resampling=resampling,
                    crop_column=crop_column,
                    keep_crop_classes=keep_crop_classes
                )

                if (xvars.shape[1] < 5) or (xvars.shape[2] < 5):
                    pbar.update(1)
                    pbar.set_description(f'{group_id} is too small')
                    continue

                # Check if the grid has already been saved
                if hasattr(row, 'grid'):
                    row_grid_id = row.grid
                elif hasattr(row, 'region'):
                    row_grid_id = row.region
                else:
                    raise AttributeError("The grid id should be given as 'grid' or 'region'.")

                batch_stored = is_grid_processed(
                    process_path,
                    transforms,
                    group_id,
                    row_grid_id,
                    n_ts
                )
                if batch_stored:
                    pbar.update(1)
                    pbar.set_description(f'{group_id} is already stored.')
                    continue

                # Get the upper left lat/lon
                left, bottom, right, top = (
                    df_grids.iloc[int_idx]
                    .to_crs('epsg:4326')
                    .total_bounds
                    .tolist()
                )

                if isinstance(group_id, str):
                    end_year = int(group_id.split('_')[-1])
                    start_year = end_year - 1
                else:
                    start_year, end_year = None, None

                # Get land cover over the block sample
                if isinstance(lc_path, (str, Path)):
                    # Convert the grid bounding box from lat/lon to projected coordinates
                    df_latlon = bounds_to_frame(left, bottom, right, top, crs='epsg:4326')
                    df_latlon, ref_crs = warp_by_image(df_latlon, lc_path)

                    # Open the projected land cover
                    with gw.config.update(
                        ref_bounds=df_latlon.total_bounds.tolist(),
                        ref_crs=ref_crs,
                        ref_res=ref_res
                    ):
                        with gw.open(lc_path, chunks=2048) as src:
                            lc_labels = src.squeeze()[:labels_array.shape[0], :labels_array.shape[1]].data.compute()

                    # Close ends
                    labels_array = close_edge_ends(labels_array)
                    # Recode crop labels
                    labels_array, edges = recode_crop_labels(
                        labels_array, lc_labels, lc_is_cdl=True, data_type=data_type
                    )
                    # Make each segment uniform in its class value
                    labels_array, segments = make_crops_uniform(labels_array, edges, data_type=data_type)
                    # Check for slivers
                    labels_array, props = check_slivers(labels_array, edges, segments)
                    # Remove non-crop edges
                    xvars, labels_array = remove_noncrop(xvars, labels_array)
                else:
                    segments = nd_label(labels_array)[0]
                    props = regionprops(segments)

                ldata = LabeledData(
                    x=xvars,
                    y=labels_array,
                    bdist=bdist,
                    ori=ori,
                    segments=segments,
                    props=props
                )
                def save_and_update(train_data: Data) -> None:
                    train_path = process_path / f'data_{train_data.train_id}.pt'
                    torch.save(train_data, train_path)

                for aug in transforms:
                    if aug.startswith('ts-'):
                        for i in range(0, n_ts):
                            train_id = f'{group_id}_{row_grid_id}_{aug}_{i:03d}'
                            train_data = augment(
                                ldata,
                                aug=aug,
                                ntime=ntime,
                                nbands=nbands,
                                k=3,
                                instance_seg=instance_seg,
                                zero_padding=zero_padding,
                                start_year=start_year,
                                end_year=end_year,
                                left=left,
                                bottom=bottom,
                                right=right,
                                top=top,
                                res=ref_res,
                                train_id=train_id
                            )
                            if instance_seg:
                                if hasattr(train_data, 'boxes') and train_data.boxes is not None:
                                    save_and_update(train_data)
                            else:
                                save_and_update(train_data)
                    else:
                        train_id = f'{group_id}_{row_grid_id}_{aug}'
                        train_data = augment(
                            ldata,
                            aug=aug,
                            ntime=ntime,
                            nbands=nbands,
                            k=3,
                            instance_seg=instance_seg,
                            zero_padding=zero_padding,
                            start_year=start_year,
                            end_year=end_year,
                            left=left,
                            bottom=bottom,
                            right=right,
                            top=top,
                            res=ref_res,
                            train_id=train_id
                        )
                        if instance_seg:
                            # Grids without boxes are set as None, and Data() does not
                            # keep None types.
                            if hasattr(train_data, 'boxes') and train_data.boxes is not None:
                                save_and_update(train_data)
                        else:
                            save_and_update(train_data)

            pbar.update(1)
            pbar.set_description(group_id)
