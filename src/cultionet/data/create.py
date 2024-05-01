import typing as T
from functools import partial
from pathlib import Path

import cv2
import einops
import geopandas as gpd
import geowombat as gw
import joblib
import numpy as np
import pandas as pd
import torch
import xarray as xr
from affine import Affine
from geowombat.core import polygon_to_array
from geowombat.core.windows import get_window_offsets
from joblib import Parallel, delayed, parallel_backend
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, from_bounds
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops
from threadpoolctl import threadpool_limits

from ..augment.augmenters import AugmenterMapping
from ..utils.logging import set_color_logger
from .data import Data, LabeledData
from .utils import get_image_list_dims

logger = set_color_logger(__name__)


def roll(
    arr_pad: np.ndarray,
    shift: T.Union[int, T.Tuple[int, int]],
    axis: T.Union[int, T.Tuple[int, int]],
) -> np.ndarray:
    """Rolls array elements along a given axis and slices off padded edges."""
    return np.roll(arr_pad, shift, axis=axis)[1:-1, 1:-1]


def close_edge_ends(array: np.ndarray) -> np.ndarray:
    """Closes 1 pixel gaps at image edges."""
    # Top
    idx = np.where(array[1] == 1)
    z = np.zeros(array.shape[1], dtype="uint8")
    z[idx] = 1
    array[0] = z
    # Bottom
    idx = np.where(array[-2] == 1)
    z = np.zeros(array.shape[1], dtype="uint8")
    z[idx] = 1
    array[-1] = z
    # Left
    idx = np.where(array[:, 1] == 1)
    z = np.zeros(array.shape[0], dtype="uint8")
    z[idx] = 1
    array[:, 0] = z
    # Right
    idx = np.where(array[:, -2] == 1)
    z = np.zeros(array.shape[0], dtype="uint8")
    z[idx] = 1
    array[:, -1] = z

    return array


def get_other_crop_count(array: np.ndarray) -> np.ndarray:
    array_pad = np.pad(array, pad_width=((1, 1), (1, 1)), mode="edge")

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
    """Fills neighboring 1-pixel edge gaps."""
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
    array = np.where(other_count > 0, 1, array)

    return array


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
    array: np.ndarray, original: np.ndarray, edge_class: int
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


def is_grid_processed(
    process_path: Path,
    transforms: T.List[str],
    region: str,
    start_date: str,
    end_date: str,
    uid_format: str,
) -> bool:
    """Checks if a grid is already processed."""

    batches_stored = []
    for aug in transforms:
        aug_method = AugmenterMapping[aug].value
        train_id = uid_format.format(
            REGION_ID=region,
            START_DATE=start_date,
            END_DATE=end_date,
            AUGMENTER=aug_method.name_,
        )
        train_path = process_path / aug_method.file_name(train_id)

        if train_path.exists():
            batch_stored = True
        else:
            batch_stored = False

        batches_stored.append(batch_stored)

    return all(batches_stored)


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


class ReferenceArrays:
    def __init__(
        self,
        labels_array: np.ndarray = None,
        boundary_distance: np.ndarray = None,
        orientation: np.ndarray = None,
        edge_array: np.ndarray = None,
    ):
        self.labels_array = labels_array
        self.boundary_distance = boundary_distance
        self.orientation = orientation
        self.edge_array = edge_array

    @classmethod
    def from_polygons(
        cls,
        df_polygons_grid: gpd.GeoDataFrame,
        max_crop_class: int,
        edge_class: int,
        crop_column: str,
        keep_crop_classes: bool,
        data_array: xr.DataArray,
        num_workers: int,
    ) -> "ReferenceArrays":
        # Polygon label array, where each polygon has a
        # unique raster value.
        labels_array_unique = (
            polygon_to_array(
                df_polygons_grid.copy().assign(
                    **{crop_column: range(1, len(df_polygons_grid.index) + 1)}
                ),
                col=crop_column,
                data=data_array,
                all_touched=False,
            )
            .squeeze()
            .gw.compute(num_workers=num_workers)
        )

        # Polygon label array, where each polygon has a value
        # equal to the GeoDataFrame `crop_column`.
        labels_array = (
            polygon_to_array(
                df_polygons_grid.copy(),
                col=crop_column,
                data=data_array,
                all_touched=False,
            )
            .squeeze()
            .gw.compute(num_workers=num_workers)
        )

        # Get the polygon edges as an array
        edge_array = (
            polygon_to_array(
                (
                    df_polygons_grid.copy()
                    .boundary.to_frame(name="geometry")
                    .reset_index()
                    .rename(columns={"index": crop_column})
                    .assign(
                        **{
                            crop_column: range(
                                1, len(df_polygons_grid.index) + 1
                            )
                        }
                    )
                ),
                col=crop_column,
                data=data_array,
                all_touched=False,
            )
            .squeeze()
            .gw.compute(num_workers=num_workers)
        )
        if not edge_array.flags["WRITEABLE"]:
            edge_array = edge_array.copy()

        edge_array[edge_array > 0] = 1
        assert edge_array.max() <= 1, "Edges were not created."

        # Get the edges from the unique polygon array
        image_grad = edge_gradient(labels_array_unique)
        # Fill in edges that may have been missed by the polygon boundary
        image_grad_count = get_crop_count(image_grad, edge_class)
        edge_array = np.where(image_grad_count > 0, edge_array, 0)

        if not keep_crop_classes:
            # Recode all crop polygons to a single class
            labels_array = np.where(labels_array > 0, max_crop_class, 0)

        # Set edges within the labels array
        # E.g.,
        # 0 = background
        # 1 = crop
        # 2 = crop edge
        labels_array[edge_array == 1] = edge_class
        # No crop pixel should border non-crop
        labels_array = cleanup_edges(
            labels_array, labels_array_unique, edge_class
        )

        assert (
            labels_array.max() <= edge_class
        ), "The labels array have larger than expected values."

        # Normalize the boundary distances for each segment
        boundary_distance, orientation = normalize_boundary_distances(
            np.uint8((labels_array > 0) & (labels_array != edge_class)),
            df_polygons_grid.geom_type.values[0],
            data_array.gw.celly,
        )

        return cls(
            labels_array=labels_array,
            boundary_distance=boundary_distance,
            orientation=orientation,
            edge_array=edge_array,
        )


def reshape_and_mask_array(
    data: xr.DataArray,
    num_time: int,
    num_bands: int,
    gain: float,
    offset: int,
) -> xr.DataArray:
    """Reshapes an array and masks no-data values."""

    src_ts_stack = xr.DataArray(
        # Date are stored [(band x time) x height x width]
        (
            data.data.reshape(
                num_bands,
                num_time,
                data.gw.nrows,
                data.gw.ncols,
            ).transpose(1, 0, 2, 3)
        ).astype('float32'),
        dims=('time', 'band', 'y', 'x'),
        coords={
            'time': range(num_time),
            'band': range(num_bands),
            'y': data.y,
            'x': data.x,
        },
        attrs=data.attrs.copy(),
    )

    with xr.set_options(keep_attrs=True):
        time_series = (src_ts_stack.gw.mask_nodata() * gain + offset).fillna(0)

    return time_series


class ImageVariables:
    def __init__(
        self,
        time_series: np.ndarray,
        labels_array: np.ndarray,
        boundary_distance: np.ndarray,
        orientation: np.ndarray,
        edge_array: np.ndarray,
        num_time: int,
        num_bands: int,
    ):
        self.time_series = time_series
        self.labels_array = labels_array
        self.boundary_distance = boundary_distance
        self.orientation = orientation
        self.edge_array = edge_array
        self.num_time = num_time
        self.num_bands = num_bands

    @staticmethod
    def recode_polygons(
        df_polygons_grid: gpd.GeoDataFrame,
        crop_column: str,
        replace_dict: dict,
    ) -> gpd.GeoDataFrame:
        # Recode labels
        for crop_class in df_polygons_grid[crop_column].unique():
            if crop_class not in list(replace_dict.keys()):
                df_polygons_grid[crop_column] = df_polygons_grid[
                    crop_column
                ].replace({crop_class: -999})

        replace_dict[-999] = 1
        df_polygons_grid[crop_column] = df_polygons_grid[crop_column].replace(
            replace_dict
        )

        # Remove any non-crop polygons
        df_polygons_grid = df_polygons_grid.query(f"{crop_column} != 0")

        return df_polygons_grid

    @staticmethod
    def get_default_arrays(num_rows: int, num_cols: int) -> tuple:
        labels_array = np.zeros((num_rows, num_cols), dtype="uint8")
        boundary_distance = np.zeros((num_rows, num_cols), dtype="float64")
        orientation = np.zeros_like(boundary_distance)
        edge_array = np.zeros_like(labels_array)

        return labels_array, boundary_distance, orientation, edge_array

    @classmethod
    def create_image_vars(
        cls,
        image: T.Union[str, Path, list],
        reference_grid: gpd.GeoDataFrame,
        max_crop_class: int,
        num_workers: int,
        grid_size: T.Optional[
            T.Union[T.Tuple[int, int], T.List[int], None]
        ] = None,
        gain: float = 1e-4,
        offset: float = 0.0,
        df_polygons_grid: T.Optional[gpd.GeoDataFrame] = None,
        ref_res: float = 10.0,
        resampling: str = "nearest",
        crop_column: str = "class",
        keep_crop_classes: bool = False,
        replace_dict: T.Optional[T.Dict[int, int]] = None,
    ) -> "ImageVariables":
        """Creates the initial image training data."""

        ref_bounds = reference_grid.total_bounds.tolist()

        if grid_size is not None:
            ref_window = from_bounds(
                *ref_bounds,
                Affine(
                    ref_res, 0.0, ref_bounds[0], 0.0, -ref_res, ref_bounds[3]
                ),
            )
            assert (ref_window.height == grid_size[0]) and (
                ref_window.width == grid_size[1]
            ), (
                f"The reference grid size is {ref_window.height} rows x {ref_window.width} columns, but the expected "
                f"dimensions are {grid_size[0]} rows x {grid_size[1]} columns"
            )

        # Open the image variables
        with gw.config.update(
            ref_bounds=ref_bounds,
            ref_crs=reference_grid.crs,
            ref_res=ref_res,
        ):
            with gw.open(
                image,
                stack_dim="band",
                band_names=list(range(1, len(image) + 1)),
                resampling=resampling,
            ) as src_ts:

                # Get the time and band count
                num_time, num_bands = get_image_list_dims(image, src_ts)

                time_series = reshape_and_mask_array(
                    data=src_ts,
                    num_time=num_time,
                    num_bands=num_bands,
                    gain=gain,
                    offset=offset,
                ).data.compute(num_workers=num_workers)

                # Default outputs
                (
                    labels_array,
                    boundary_distance,
                    orientation,
                    edge_array,
                ) = cls.get_default_arrays(
                    num_rows=src_ts.gw.nrows, num_cols=src_ts.gw.ncols
                )

                if df_polygons_grid is not None:
                    if replace_dict is not None:
                        # Recode polygons
                        df_polygons_grid = cls.recode_polygons(
                            df_polygons_grid=df_polygons_grid,
                            crop_column=crop_column,
                            replace_dict=replace_dict,
                        )

                    if not df_polygons_grid.empty:
                        reference_arrays: ReferenceArrays = (
                            ReferenceArrays.from_polygons(
                                df_polygons_grid=df_polygons_grid,
                                max_crop_class=max_crop_class,
                                edge_class=max_crop_class + 1,
                                crop_column=crop_column,
                                keep_crop_classes=keep_crop_classes,
                                data_array=src_ts,
                                num_workers=num_workers,
                            )
                        )

                        if reference_arrays.labels_array is not None:
                            labels_array = reference_arrays.labels_array
                            boundary_distance = (
                                reference_arrays.boundary_distance
                            )
                            orientation = reference_arrays.orientation
                            edge_array = reference_arrays.edge_array

                    # import matplotlib.pyplot as plt
                    # def save_labels(out_fig: Path):
                    #     fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True, dpi=300)
                    #     axes = axes.flatten()
                    #     for ax, im, title in zip(
                    #         axes,
                    #         (labels_array_unique, labels_array, boundary_distance, orientation),
                    #         ('Fields', 'Edges', 'Distance', 'orientationentation')
                    #     ):
                    #         ax.imshow(im, interpolation='nearest')
                    #         ax.set_title(title)
                    #         ax.axis('off')

                    #     plt.tight_layout()
                    #     plt.savefig(out_fig, dpi=300)
                    # import uuid
                    # fig_dir = Path('figures')
                    # fig_dir.mkdir(exist_ok=True, parents=True)
                    # hash_id = uuid.uuid4().hex
                    # save_labels(
                    #     out_fig=fig_dir / f'{hash_id}.png'
                    # )

        return cls(
            time_series=time_series,
            labels_array=labels_array,
            boundary_distance=boundary_distance,
            orientation=orientation,
            edge_array=edge_array,
            num_time=num_time,
            num_bands=num_bands,
        )


def save_and_update(
    write_path: Path, predict_data: Data, name: str, compress: int = 5
) -> None:
    predict_path = write_path / f"data_{name}.pt"
    joblib.dump(predict_data, predict_path, compress=compress)


def read_slice(darray: xr.DataArray, w_pad: Window) -> xr.DataArray:
    slicer = (
        slice(0, None),
        slice(0, None),
        slice(w_pad.row_off, w_pad.row_off + w_pad.height),
        slice(w_pad.col_off, w_pad.col_off + w_pad.width),
    )

    return darray[slicer]


def get_window_chunk(windows: T.List[T.Tuple[Window, Window]], chunksize: int):
    for i in range(0, len(windows), chunksize):
        yield windows[i : i + chunksize]


def create_and_save_window(
    write_path: Path,
    res: float,
    resampling: str,
    region: str,
    start_date: str,
    end_date: str,
    window_size: int,
    padding: int,
    compress_method: T.Union[int, str],
    darray: xr.DataArray,
    gain: float,
    w: Window,
    w_pad: Window,
) -> None:
    x = darray.data.compute(num_workers=1)

    image_height = window_size + padding * 2
    image_width = window_size + padding * 2

    # Get row adjustments
    row_before_padded = abs(w_pad.row_off - w.row_off)
    row_before_to_pad = padding - row_before_padded
    row_after_to_pad = image_height - w_pad.height - row_before_to_pad

    # Get column adjustments
    col_before_padded = abs(w_pad.col_off - w.col_off)
    col_before_to_pad = padding - col_before_padded
    col_after_to_pad = image_width - w_pad.width - col_before_to_pad

    x = np.pad(
        x,
        pad_width=(
            (0, 0),
            (0, 0),
            (row_before_to_pad, row_after_to_pad),
            (col_before_to_pad, col_after_to_pad),
        ),
        mode="constant",
        constant_values=0,
    )

    x = einops.rearrange(
        torch.from_numpy(x / gain).to(dtype=torch.int32),
        't c h w -> 1 c t h w',
    )

    assert x.shape[-2:] == (
        image_height,
        image_width,
    ), "The padded array does not have the correct height/width dimensions."

    batch_id = f"{region}_{start_date}_{end_date}_{w.row_off}_{w.col_off}"

    batch = Data(
        x=x,
        start_year=[start_date],
        end_year=[end_date],
        padding=[padding],
        window_row_off=[w.row_off],
        window_col_off=[w.col_off],
        window_height=[w.height],
        window_width=[w.width],
        window_pad_row_off=[w_pad.row_off],
        window_pad_col_off=[w_pad.col_off],
        window_pad_height=[w_pad.height],
        window_pad_width=[w_pad.width],
        row_before_to_pad=[row_before_to_pad],
        row_after_to_pad=[row_after_to_pad],
        col_before_to_pad=[col_before_to_pad],
        col_after_to_pad=[col_after_to_pad],
        res=[res],
        resampling=[resampling],
        left=[darray.gw.left],
        bottom=[darray.gw.bottom],
        right=[darray.gw.right],
        top=[darray.gw.top],
        batch_id=[batch_id],
    )

    batch.to_file(
        write_path / f"{batch_id}.pt",
        compress=compress_method,
    )


@threadpool_limits.wrap(limits=1, user_api="blas")
def create_predict_dataset(
    image_list: T.List[T.List[T.Union[str, Path]]],
    region: str,
    process_path: Path = None,
    date_format: str = "%Y%j",
    gain: float = 1e-4,
    offset: float = 0.0,
    ref_res: T.Union[float, T.Tuple[float, float]] = 10.0,
    resampling: str = "nearest",
    window_size: int = 100,
    padding: int = 101,
    num_workers: int = 1,
    chunksize: int = 100,
    compress_method: T.Union[int, str] = 'zlib',
):
    with gw.config.update(ref_res=ref_res):
        with gw.open(
            image_list,
            stack_dim="band",
            band_names=list(range(1, len(image_list) + 1)),
            resampling=resampling,
            chunks=512,
        ) as src_ts:

            windows = get_window_offsets(
                src_ts.gw.nrows,
                src_ts.gw.ncols,
                window_size,
                window_size,
                padding=(padding, padding, padding, padding),
            )

            num_time, num_bands = get_image_list_dims(image_list, src_ts)

            time_series: xr.DataArray = reshape_and_mask_array(
                data=src_ts,
                num_time=num_time,
                num_bands=num_bands,
                gain=gain,
                offset=offset,
            )

            partial_create_and_save_window = partial(
                create_and_save_window,
                write_path=process_path,
                res=ref_res,
                resampling=resampling,
                region=region,
                start_date=pd.to_datetime(
                    Path(image_list[0]).stem, format=date_format
                ).strftime("%Y%m%d"),
                end_date=pd.to_datetime(
                    Path(image_list[-1]).stem, format=date_format
                ).strftime("%Y%m%d"),
                window_size=window_size,
                padding=padding,
                compress_method=compress_method,
                gain=gain,
            )

            with parallel_backend(backend="threading", n_jobs=num_workers):
                for window_chunk in get_window_chunk(windows, chunksize):
                    with Parallel(temp_folder="/tmp") as pool:
                        __ = pool(
                            delayed(partial_create_and_save_window)(
                                darray=read_slice(time_series, window_pad),
                                w=window,
                                w_pad=window_pad,
                            )
                            for window, window_pad in window_chunk
                        )


def get_reference_bounds(
    df_grid: gpd.GeoDataFrame,
    grid_size: tuple,
    filename: T.Union[Path, str],
    ref_res: tuple,
) -> T.List[float]:
    ref_bounds = df_grid.total_bounds.tolist()

    if grid_size is not None:
        # Enforce bounds given height/width dimensions
        height, width = grid_size
        left, bottom, right, top = ref_bounds

        with gw.open(filename) as src:
            image_crs = src.gw.crs_to_pyproj
            if ref_res is None:
                ref_res = (src.gw.celly, src.gw.cellx)
            else:
                ref_res = (ref_res, ref_res)

        (dst_transform, dst_width, dst_height,) = calculate_default_transform(
            src_crs=image_crs,
            dst_crs=df_grid.crs,
            width=int(abs(round((right - left) / ref_res[1]))),
            height=int(abs(round((top - bottom) / ref_res[0]))),
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            dst_width=width,
            dst_height=height,
        )
        dst_left = dst_transform[2]
        dst_top = dst_transform[5]
        dst_right = dst_left + abs(dst_width * dst_transform[0])
        dst_bottom = dst_top - abs(dst_height * dst_transform[4])
        ref_bounds = [dst_left, dst_bottom, dst_right, dst_top]

    return ref_bounds


def create_train_batch(
    image_list: T.List[T.List[T.Union[str, Path]]],
    df_grid: gpd.GeoDataFrame,
    df_polygons: gpd.GeoDataFrame,
    max_crop_class: int,
    region: str,
    process_path: Path = None,
    date_format: str = "%Y%j",
    transforms: T.List[str] = None,
    gain: float = 1e-4,
    offset: float = 0.0,
    ref_res: float = 10.0,
    resampling: str = "nearest",
    num_workers: int = 1,
    grid_size: T.Optional[
        T.Union[T.Tuple[int, int], T.List[int], None]
    ] = None,
    crop_column: T.Optional[str] = "class",
    keep_crop_classes: T.Optional[bool] = False,
    replace_dict: T.Optional[T.Dict[int, int]] = None,
    compress_method: T.Union[int, str] = 'zlib',
) -> None:
    """Creates a batch file for training.

    Args:
        image_list: A list of images.
        df_grid: The training grid.
        df_polygons: The training polygons.
        max_crop_class: The maximum expected crop class value.
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
        replace_dict: A dictionary of crop class remappings.
    """
    start_date = pd.to_datetime(
        Path(image_list[0]).stem, format=date_format
    ).strftime("%Y%m%d")
    end_date = pd.to_datetime(
        Path(image_list[0]).stem, format=date_format
    ).strftime("%Y%m%d")

    uid_format = "{REGION_ID}_{START_DATE}_{END_DATE}_none"
    group_id = f"{region}_{start_date}_{end_date}_none"

    if transforms is None:
        transforms = ["none"]

    # Check if the grid has already been saved
    batch_stored = is_grid_processed(
        process_path=process_path,
        transforms=transforms,
        region=region,
        start_date=start_date,
        end_date=end_date,
        uid_format=uid_format,
    )

    if batch_stored:
        return

    # # Clip the polygons to the current grid
    # try:
    #     df_polygons_grid = gpd.clip(df_polygons, row.geometry)
    # except ValueError:
    #     logger.warning(
    #         TopologyClipError(
    #             "The input GeoDataFrame contains topology errors."
    #         )
    #     )
    #     df_polygons = gpd.GeoDataFrame(
    #         data=df_polygons[crop_column].values,
    #         columns=[crop_column],
    #         geometry=df_polygons.buffer(0).geometry,
    #     )
    #     df_polygons_grid = gpd.clip(df_polygons, row.geometry)

    # These are grids with no crop fields. They should still
    # be used for training.
    if df_polygons.loc[~df_polygons.is_empty].empty:
        df_polygons = df_grid.copy()
        df_polygons = df_polygons.assign(**{crop_column: 0})

    # Remove empty geometry
    df_polygons = df_polygons.loc[~df_polygons.is_empty]

    if not df_polygons.empty:
        # Get a mask of valid polygons
        nonzero_mask = df_polygons[crop_column] != 0

        # Get the reference bounding box from the grid
        # ref_bounds = get_reference_bounds(
        #     df_grid=df_grid,
        #     grid_size=grid_size,
        #     filename=image_list[0],
        #     ref_res=ref_res,
        # )

        # Data for the model network
        image_variables = ImageVariables.create_image_vars(
            image=image_list,
            reference_grid=df_grid,
            df_polygons_grid=df_polygons if nonzero_mask.any() else None,
            max_crop_class=max_crop_class,
            num_workers=num_workers,
            grid_size=grid_size,
            gain=gain,
            offset=offset,
            ref_res=ref_res,
            resampling=resampling,
            crop_column=crop_column,
            keep_crop_classes=keep_crop_classes,
            replace_dict=replace_dict,
        )

        if image_variables.time_series is None:
            return

        if (image_variables.time_series.shape[1] < 5) or (
            image_variables.time_series.shape[2] < 5
        ):
            return

        # Get the upper left lat/lon
        lat_left, lat_bottom, lat_right, lat_top = df_grid.to_crs(
            "epsg:4326"
        ).total_bounds.tolist()

        segments = nd_label(
            (image_variables.labels_array > 0)
            & (image_variables.labels_array < max_crop_class + 1)
        )[0]
        props = regionprops(segments)

        labeled_data = LabeledData(
            x=image_variables.time_series,
            y=image_variables.labels_array,
            bdist=image_variables.boundary_distance,
            ori=image_variables.orientation,
            segments=segments,
            props=props,
        )

        batch = Data(
            x=einops.rearrange(
                torch.from_numpy(labeled_data.x / gain).to(dtype=torch.int32),
                't c h w -> 1 c t h w',
            ),
            y=einops.rearrange(
                torch.from_numpy(labeled_data.y).to(dtype=torch.uint8),
                'b w -> 1 b w',
            ),
            bdist=einops.rearrange(
                torch.from_numpy(labeled_data.bdist / gain).to(
                    dtype=torch.int32
                ),
                'b w -> 1 b w',
            ),
            start_year=torch.tensor(
                [pd.Timestamp(Path(image_list[0]).stem).year],
                dtype=torch.int32,
            ),
            end_year=torch.tensor(
                [pd.Timestamp(Path(image_list[-1]).stem).year],
                dtype=torch.int32,
            ),
            left=torch.tensor([lat_left], dtype=torch.float32),
            bottom=torch.tensor([lat_bottom], dtype=torch.float32),
            right=torch.tensor([lat_right], dtype=torch.float32),
            top=torch.tensor([lat_top], dtype=torch.float32),
            batch_id=[group_id],
        )

        # FIXME: this doesn't support augmentations
        for aug in transforms:
            aug_method = AugmenterMapping[aug].value
            train_id = uid_format.format(
                REGION_ID=region,
                START_DATE=start_date,
                END_DATE=end_date,
                AUGMENTER=aug_method.name_,
            )
            train_path = process_path / aug_method.file_name(train_id)
            batch.to_file(train_path, compress=compress_method)
