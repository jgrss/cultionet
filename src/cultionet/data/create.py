import typing as T
from pathlib import Path

import dask
import dask.array as da
import einops
import geopandas as gpd
import geowombat as gw
import numpy as np
import pandas as pd
import torch
import xarray as xr
from affine import Affine
from dask.distributed import Client, LocalCluster, progress
from rasterio.windows import Window, from_bounds
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops
from threadpoolctl import threadpool_limits

from ..augment.augmenters import AUGMENTER_METHODS
from ..utils.logging import set_color_logger
from .data import Data, LabeledData
from .store import BatchStore
from .utils import (
    cleanup_edges,
    edge_gradient,
    fillz,
    get_crop_count,
    get_image_list_dims,
    normalize_boundary_distances,
    polygon_to_array,
)

logger = set_color_logger(__name__)


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
        aug_method = AUGMENTER_METHODS[aug]()
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
    compress_method: T.Union[int, str] = 'zlib',
):
    """Creates a prediction dataset for an image."""

    with gw.config.update(ref_res=ref_res):
        with gw.open(
            image_list,
            stack_dim="band",
            band_names=list(range(1, len(image_list) + 1)),
            resampling=resampling,
            chunks=512,
        ) as src_ts:

            num_time, num_bands = get_image_list_dims(image_list, src_ts)

            time_series: xr.DataArray = reshape_and_mask_array(
                data=src_ts,
                num_time=num_time,
                num_bands=num_bands,
                gain=gain,
                offset=offset,
            )

            # Chunk the array into the windows
            time_series_array = time_series.chunk(
                {"time": -1, "band": -1, "y": window_size, "x": window_size}
            ).data

            # Check if the array needs to be padded
            # First, get the end chunk size of rows and columns
            height_end_chunk = time_series_array.chunks[-2][-1]
            width_end_chunk = time_series_array.chunks[-1][-1]

            height_padding = 0
            width_padding = 0
            if padding > height_end_chunk:
                height_padding = padding - height_end_chunk
            if padding > width_end_chunk:
                width_padding = padding - width_end_chunk

            if (height_padding > 0) or (width_padding > 0):
                # Pad the full array if the end chunk is smaller than the padding
                time_series_array = da.pad(
                    time_series_array,
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (0, height_padding),
                        (0, width_padding),
                    ),
                ).rechunk({0: -1, 1: -1, 2: window_size, 3: window_size})

            # Add the padding to each chunk
            time_series_array = time_series_array.map_overlap(
                lambda x: x,
                depth={0: 0, 1: 0, 2: padding, 3: padding},
                boundary=0,
                trim=False,
            )

            with dask.config.set(
                {
                    "distributed.worker.memory.terminate": False,
                    "distributed.comm.retry.count": 10,
                    "distributed.comm.timeouts.connect": 5,
                    "distributed.scheduler.allowed-failures": 20,
                }
            ):
                with LocalCluster(
                    processes=True,
                    n_workers=num_workers,
                    threads_per_worker=1,
                    memory_target_fraction=0.97,
                    memory_limit="4GB",  # per worker limit
                ) as cluster:
                    with Client(cluster) as client:
                        with BatchStore(
                            data=time_series,
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
                        ) as batch_store:
                            save_tasks = batch_store.save(time_series_array)
                            results = client.persist(save_tasks)
                            progress(results)


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
        nonag_is_unknown: bool = False,
        all_touched: bool = True,
    ) -> "ReferenceArrays":
        # Polygon label array, where each polygon has a
        # unique raster value.
        labels_array_unique = polygon_to_array(
            df=df_polygons_grid.assign(
                **{crop_column: range(1, len(df_polygons_grid.index) + 1)}
            ),
            reference_data=data_array,
            column=crop_column,
        )

        # Polygon label array, where each polygon has a value
        # equal to the GeoDataFrame `crop_column`.
        fill_value = 0
        dtype = "uint8"
        if nonag_is_unknown:
            # Background values are unknown, so they need to be
            # filled with -1
            fill_value = -1
            dtype = "int16"

        labels_array = polygon_to_array(
            df=df_polygons_grid,
            reference_data=data_array,
            column=crop_column,
            fill_value=fill_value,
            dtype=dtype,
        )

        # Get the polygon edges as an array
        edge_array = polygon_to_array(
            df=(
                df_polygons_grid.boundary.to_frame(name="geometry")
                .reset_index()
                .rename(columns={"index": crop_column})
                .assign(
                    **{crop_column: range(1, len(df_polygons_grid.index) + 1)}
                )
            ),
            reference_data=data_array,
            column=crop_column,
            all_touched=all_touched,
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
            labels_array = np.where(
                labels_array > 0, max_crop_class, fill_value
            )

        # Set edges within the labels array
        # E.g.,
        # 0 = background
        # 1 = crop
        # 2 = crop edge
        labels_array[edge_array == 1] = edge_class
        # No crop pixel should border non-crop
        labels_array = cleanup_edges(
            np.where(labels_array == fill_value, 0, labels_array),
            labels_array_unique,
            edge_class,
        )
        labels_array = np.where(labels_array == 0, fill_value, labels_array)

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


class ImageVariables:
    def __init__(
        self,
        time_series: np.ndarray = None,
        labels_array: np.ndarray = None,
        boundary_distance: np.ndarray = None,
        orientation: np.ndarray = None,
        edge_array: np.ndarray = None,
        num_time: int = None,
        num_bands: int = None,
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
        """Recodes polygon labels."""

        df_polygons_grid[crop_column] = df_polygons_grid[crop_column].replace(
            to_replace=replace_dict
        )

        # Remove any non-crop polygons
        return df_polygons_grid.query(f"{crop_column} != 0")

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
        region: str,
        image: T.Union[str, Path, list],
        reference_grid: gpd.GeoDataFrame,
        max_crop_class: int,
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
        nonag_is_unknown: bool = False,
        all_touched: bool = True,
    ) -> "ImageVariables":
        """Creates the initial image training data."""

        # Get the reference bounds from the grid geometry
        ref_bounds = reference_grid.total_bounds.tolist()

        # Pre-check before opening files
        if grid_size is not None:
            ref_window = from_bounds(
                *ref_bounds,
                Affine(
                    ref_res, 0.0, ref_bounds[0], 0.0, -ref_res, ref_bounds[3]
                ),
            )

            ref_window = Window(
                row_off=int(ref_window.row_off),
                col_off=int(ref_window.col_off),
                height=int(round(ref_window.height)),
                width=int(round(ref_window.width)),
            )

            assert (int(ref_window.height) == grid_size[0]) and (
                int(ref_window.width) == grid_size[1]
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
                if grid_size is not None:
                    if not (
                        (src_ts.gw.nrows == grid_size[0])
                        and (src_ts.gw.ncols == grid_size[1])
                    ):
                        logger.warning(
                            f"The reference image size is {src_ts.gw.nrows} rows x {src_ts.gw.ncols} columns, but the expected "
                            f"dimensions are {grid_size[0]} rows x {grid_size[1]} columns"
                        )
                        return cls()

                # Get the time and band count
                num_time, num_bands = get_image_list_dims(image, src_ts)

                time_series = reshape_and_mask_array(
                    data=src_ts,
                    num_time=num_time,
                    num_bands=num_bands,
                    gain=gain,
                    offset=offset,
                ).data.compute(num_workers=1)

                # Fill isolated zeros
                time_series = fillz(time_series)

                # NaNs are filled with 0 in reshape_and_mask_array()
                zero_mask = time_series.sum(axis=0) == 0
                if zero_mask.all():
                    logger.warning(
                        f"The {region} time series contains all NaNs."
                    )
                    return cls()

                # Default outputs
                (
                    labels_array,
                    boundary_distance,
                    orientation,
                    edge_array,
                ) = cls.get_default_arrays(
                    num_rows=src_ts.gw.nrows, num_cols=src_ts.gw.ncols
                )

                # Any polygons intersecting the grid?
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
                                nonag_is_unknown=nonag_is_unknown,
                                all_touched=all_touched,
                            )
                        )

                        if reference_arrays.labels_array is not None:
                            labels_array = reference_arrays.labels_array
                            boundary_distance = (
                                reference_arrays.boundary_distance
                            )
                            orientation = reference_arrays.orientation
                            edge_array = reference_arrays.edge_array

        return cls(
            time_series=time_series,
            labels_array=labels_array,
            boundary_distance=boundary_distance,
            orientation=orientation,
            edge_array=edge_array,
            num_time=num_time,
            num_bands=num_bands,
        )


@threadpool_limits.wrap(limits=1, user_api="blas")
def create_train_batch(
    image_list: T.List[T.List[T.Union[str, Path]]],
    df_grid: gpd.GeoDataFrame,
    df_polygons: gpd.GeoDataFrame,
    max_crop_class: int,
    region: str,
    process_path: Path = None,
    date_format: str = "%Y%j",
    gain: float = 1e-4,
    offset: float = 0.0,
    ref_res: float = 10.0,
    resampling: str = "nearest",
    grid_size: T.Optional[
        T.Union[T.Tuple[int, int], T.List[int], None]
    ] = None,
    crop_column: T.Optional[str] = "class",
    keep_crop_classes: T.Optional[bool] = False,
    replace_dict: T.Optional[T.Dict[int, int]] = None,
    nonag_is_unknown: bool = False,
    all_touched: bool = True,
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
        gain: A gain factor to apply to the images.
        offset: An offset factor to apply to the images.
        ref_res: The reference cell resolution to resample the images to.
        resampling: The image resampling method.
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
        nonag_is_unknown: Whether the non-agricultural background is unknown.
        all_touched: Rasterio/Shapely rasterization flag.
    """
    start_date = pd.to_datetime(
        Path(image_list[0]).stem, format=date_format
    ).strftime("%Y%m%d")
    end_date = pd.to_datetime(
        Path(image_list[-1]).stem, format=date_format
    ).strftime("%Y%m%d")

    uid_format = "{REGION_ID}_{START_DATE}_{END_DATE}_none"
    group_id = f"{region}_{start_date}_{end_date}_none"

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

    # These are grids with no crop fields. They should still
    # be used for training.
    if df_polygons.loc[~df_polygons.is_empty].empty:
        df_polygons = df_grid.copy()
        df_polygons = df_polygons.assign(**{crop_column: 0})

    # Remove empty geometries
    df_polygons = df_polygons.loc[~df_polygons.is_empty]

    if not df_polygons.empty:
        type_mask = df_polygons.geom_type == "GeometryCollection"
        if type_mask.any():
            exploded_collections = df_polygons.loc[type_mask].explode(
                column="geometry"
            )
            exploded_collections = exploded_collections.loc[
                (exploded_collections.geom_type == "Polygon")
                | (exploded_collections.geom_type == "MultiPolygon")
            ]
            df_polygons = pd.concat(
                (
                    df_polygons.loc[~type_mask],
                    exploded_collections.droplevel(1),
                )
            )

        df_polygons = df_polygons.reset_index(drop=True)
        df_polygons = df_polygons.loc[df_polygons.geom_type != "Point"]
        type_mask = df_polygons.geom_type == "MultiPolygon"
        if type_mask.any():
            raise TypeError("MultiPolygons should not exist.")

        # Get a mask of valid polygons
        nonzero_mask = df_polygons[crop_column] != 0

        # Data for the model network
        image_variables = ImageVariables.create_image_vars(
            region=region,
            image=image_list,
            reference_grid=df_grid,
            df_polygons_grid=df_polygons if nonzero_mask.any() else None,
            max_crop_class=max_crop_class,
            grid_size=grid_size,
            gain=gain,
            offset=offset,
            ref_res=ref_res,
            resampling=resampling,
            crop_column=crop_column,
            keep_crop_classes=keep_crop_classes,
            replace_dict=replace_dict,
            nonag_is_unknown=nonag_is_unknown,
            all_touched=all_touched,
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
                torch.from_numpy(labeled_data.y).to(
                    dtype=torch.int16 if nonag_is_unknown else torch.uint8
                ),
                'h w -> 1 h w',
            ),
            bdist=einops.rearrange(
                torch.from_numpy(labeled_data.bdist / gain).to(
                    dtype=torch.int32
                ),
                'h w -> 1 h w',
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
            aug_method = AUGMENTER_METHODS[aug]()
            train_id = uid_format.format(
                REGION_ID=region,
                START_DATE=start_date,
                END_DATE=end_date,
                AUGMENTER=aug_method.name_,
            )
            train_path = process_path / aug_method.file_name(train_id)
            batch.to_file(train_path, compress=compress_method)
