import typing as T
from copy import deepcopy
from functools import partial
from pathlib import Path

import attr
import geopandas as gpd
import lightning as L
import numpy as np
import psutil
import pygrts
import torch
from joblib import delayed, parallel_backend
from scipy.ndimage.measurements import label as nd_label
from shapely.geometry import box
from skimage.measure import regionprops
from tqdm.auto import tqdm

from ..augment.augmenters import Augmenters
from ..errors import TensorShapeError
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import ParallelProgress
from ..utils.normalize import NormValues
from .data import Data
from .spatial_dataset import SpatialDataset

ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional

logger = set_color_logger(__name__)


def _check_shape(
    expected_time: int,
    expected_height: int,
    expected_width: int,
    in_time: int,
    in_height: int,
    in_width: int,
    index: int,
    uid: str,
) -> T.Tuple[bool, int, str]:
    if (
        (expected_time != in_time)
        or (expected_height != in_height)
        or (expected_width != in_width)
    ):
        return False, index, uid
    return True, index, uid


class EdgeDataset(SpatialDataset):
    """An edge dataset."""

    data_list_ = None
    grid_id_column = "grid_id"

    def __init__(
        self,
        root: T.Union[str, Path, bytes] = ".",
        norm_values: T.Optional[NormValues] = None,
        pattern: str = "data*.pt",
        processes: int = psutil.cpu_count(),
        threads_per_worker: int = 1,
        random_seed: int = 42,
        augment_prob: float = 0.0,
    ):
        self.root = root
        self.norm_values = norm_values
        self.pattern = pattern
        self.processes = processes
        self.threads_per_worker = threads_per_worker
        self.random_seed = random_seed
        self.augment_prob = augment_prob

        L.seed_everything(self.random_seed)
        self.rng = np.random.default_rng(self.random_seed)

        self.augmentations_ = [
            'tswarp',
            'tsnoise',
            'tsdrift',
            'tspeaks',
            'rot90',
            'rot180',
            'rot270',
            'roll',
            'fliplr',
            'flipud',
            'gaussian',
            'saltpepper',
            'cropresize',
            'perlin',
        ]

        self.data_list_ = None
        self.processed_dir = Path(self.root) / 'processed'
        self.get_data_list()

    def get_data_list(self):
        """Gets the list of data files."""
        data_list_ = sorted(list(Path(self.processed_dir).glob(self.pattern)))

        if not data_list_:
            logger.exception(
                f"No .pt files were found with pattern {self.pattern}."
            )

        self.data_list_ = np.array(data_list_)

    @property
    def data_list(self):
        """Get a list of processed files."""
        return self.data_list_

    def __len__(self):
        """Returns the dataset length."""
        return len(self.data_list)

    def cleanup(self):
        for fn in self.data_list_:
            fn.unlink()

        self.data_list_ = []

    def shuffle(self, data: T.Optional[list] = None):
        """Applies a random in-place shuffle to the data list."""
        if data is not None:
            self.rng.shuffle(data)
        else:
            self.rng.shuffle(self.data_list_)

    @property
    def num_channels(self) -> int:
        return self[0].num_channels

    @property
    def num_time(self) -> int:
        """Get the number of time features."""
        return self[0].num_time

    def get_spatial_partitions(
        self,
        spatial_partitions: T.Union[str, Path, gpd.GeoDataFrame],
        splits: int = 0,
    ) -> None:
        """Gets the spatial partitions."""
        self.create_spatial_index(
            id_column=self.grid_id_column, n_jobs=self.processes
        )
        if isinstance(spatial_partitions, (str, Path)):
            spatial_partitions = gpd.read_file(spatial_partitions)
        else:
            spatial_partitions = self.to_frame(
                id_column=self.grid_id_column,
                n_jobs=self.processes,
            )

        if splits > 0:
            qt = pygrts.QuadTree(spatial_partitions, force_square=False)
            for __ in range(splits):
                qt.split()
            spatial_partitions = qt.to_frame()

        self.spatial_partitions = spatial_partitions.to_crs("epsg:4326")

    def query_partition_by_name(
        self,
        partition_column: str,
        partition_name: str,
        val_frac: float = None,
    ) -> list:
        """Queries grid centroids that are within the partition."""
        # Get the partition
        df = self.spatial_partitions.query(
            f"{partition_column} == '{partition_name}'"
        )
        df_points = self.dataset_df.overlay(df, how="intersection")
        if df_points.empty:
            logger.warning(
                f"Partition {partition_name} does not have any data."
            )
            return None

        # TODO: currently doesn't work with overlapping augmentated points
        # if val_frac is not None:
        #     qt = QuadTree(df_points.to_crs('epsg:8858'), force_square=False)
        #     qt.split_recursive(max_length=1_000)
        #     n_train = int(len(df_points.index) * (1.0 - val_frac))
        #     df_points_train = qt.sample(n=n_train)
        # else:
        grid_names = df_points[self.grid_id_column].values.tolist()
        indices = np.intersect1d(
            grid_names,
            [
                data_path.stem.replace("data_", "")
                for data_path in self.data_list_
            ],
            return_indices=True,
        )[2].tolist()

        return indices

    def split_indices(
        self,
        indices: list,
        return_all: bool = True,
        return_indices: bool = False,
    ) -> T.Tuple["EdgeDataset", "EdgeDataset"]:
        """Splits a list of indices into train and validation datasets."""
        train_idx = list(set(range(len(self))).difference(indices))

        if return_all:
            if return_indices:
                return train_idx, indices
            else:
                return self[train_idx], self[indices]
        else:
            if return_indices:
                return indices
            else:
                return self[indices]

    def spatial_kfoldcv_iter(self, partition_column: str):
        """Yield generator to iterate over spatial partitions."""

        for kfold in self.spatial_partitions.itertuples():
            # Bounding box and indices of the kth fold
            kfold_indices = self.query_partition_by_name(
                partition_column, str(getattr(kfold, partition_column))
            )
            if not kfold_indices:
                continue

            train_ds, test_ds = self.split_indices(kfold_indices)

            yield str(getattr(kfold, partition_column)), train_ds, test_ds

    def check_dims(
        self,
        expected_time: int,
        expected_height: int,
        expected_width: int,
        delete_mismatches: bool = False,
        tqdm_color: str = "ffffff",
    ):
        """Checks if all tensors in the dataset match in shape dimensions."""
        check_partial = partial(
            _check_shape,
            expected_time=expected_time,
            expected_height=expected_height,
            expected_width=expected_width,
        )
        with parallel_backend(
            backend="loky",
            n_jobs=self.processes,
        ):
            with ParallelProgress(
                tqdm_kwargs={
                    "total": len(self),
                    "desc": "Checking dimensions",
                    "colour": tqdm_color,
                }
            ) as pool:
                results = pool(
                    delayed(check_partial)(
                        in_time=self[i].num_time,
                        in_height=self[i].height,
                        in_width=self[i].width,
                        index=i,
                        uid=self[i].batch_id,
                    )
                    for i in range(0, len(self))
                )

        matches, indices, ids = list(map(list, zip(*results)))
        if not all(matches):
            indices = np.array(indices)
            null_indices = indices[~np.array(matches)]
            null_ids = np.array(ids)[null_indices].tolist()
            logger.warning(",".join(null_ids))
            logger.warning(
                f"{null_indices.shape[0]:,d} ids did not match the reference dimensions."
            )

            if delete_mismatches:
                logger.warning(
                    f"Removing {null_indices.shape[0]:,d} .pt files."
                )
                for i in null_indices:
                    self.data_list_[i].unlink()
            else:
                raise TensorShapeError

    def split_train_val_by_partition(
        self,
        spatial_partitions: str,
        partition_column: str,
        val_frac: float,
        partition_name: T.Optional[str] = None,
    ) -> T.Tuple["EdgeDataset", "EdgeDataset"]:
        self.get_spatial_partitions(spatial_partitions=spatial_partitions)
        train_indices = []
        val_indices = []
        self.shuffle()
        # self.spatial_partitions is a GeoDataFrame with Point geometry
        for row in tqdm(
            self.spatial_partitions.itertuples(),
            total=len(self.spatial_partitions.index),
            desc="Sampling partitions",
        ):
            if partition_name is not None:
                if str(getattr(row, partition_column)) != partition_name:
                    continue
            # Query grid centroids within the partition
            indices = self.query_partition_by_name(
                partition_column, str(getattr(row, partition_column))
            )
            if indices is None:
                continue

            n_train = int(len(indices) * (1.0 - val_frac))
            train_indices_split = indices[:n_train]
            val_indices_split = indices[n_train:]
            train_indices.extend(train_indices_split)
            val_indices.extend(val_indices_split)

        train_ds = self[train_indices]
        val_ds = self[val_indices]

        return train_ds, val_ds

    def split_train_val(
        self,
        val_frac: float,
        spatial_overlap_allowed: bool = True,
        spatial_balance: bool = True,
        crs: str = "EPSG:8857",
    ) -> T.Tuple["EdgeDataset", "EdgeDataset"]:
        """Splits the dataset into train and validation.

        Args:
            val_frac (float): The validation fraction.

        Returns:
            train dataset, validation dataset
        """
        # We do not need augmentations when loading batches for
        # sample splits.
        augment_prob = deepcopy(self.augment_prob)
        self.augment_prob = 0.0

        if spatial_overlap_allowed:
            self.shuffle()
            n_train = int(len(self) * (1.0 - val_frac))
            train_ds = self[:n_train]
            val_ds = self[n_train:]
        else:
            self.create_spatial_index(
                id_column=self.grid_id_column,
                n_jobs=self.processes,
            )

            train_ds, val_ds = self.spatial_splits(
                val_frac=val_frac,
                id_column=self.grid_id_column,
                spatial_balance=spatial_balance,
                crs=crs,
                random_state=self.random_seed,
            )

        train_ds.augment_prob = augment_prob
        val_ds.augment_prob = 0.0

        return train_ds, val_ds

    def load_file(self, filename: T.Union[str, Path]) -> Data:
        return Data.from_file(filename)

    def __getitem__(
        self, idx: T.Union[int, np.ndarray]
    ) -> T.Union[dict, "EdgeDataset"]:
        if isinstance(idx, (int, np.integer)):
            return self.get(idx)
        else:
            return self.index_select(idx)

    def index_select(self, idx: np.ndarray) -> "EdgeDataset":
        dataset = deepcopy(self)
        dataset.data_list_ = self.data_list_[idx]

        return dataset

    def get(self, idx: int) -> dict:
        """Gets an individual data object from the dataset.

        Args:
            idx (int): The dataset index position.
        """

        batch = self.load_file(self.data_list_[idx])

        batch.x = (batch.x * 1e-4).clip(1e-9, 1)

        if hasattr(batch, 'bdist'):
            batch.bdist = (batch.bdist * 1e-4).clip(0, 1)

        if batch.y is not None:
            if self.rng.random() > (1 - self.augment_prob):
                # Choose one augmentation to apply
                aug_name = self.rng.choice(self.augmentations_)

                if aug_name in (
                    'roll',
                    'tswarp',
                    'tsnoise',
                    'tsdrift',
                    'tspeaks',
                ):
                    # FIXME: By default, the crop value is 1 (background is 0 and edges are 2).
                    # But, it would be better to get 1 from an argument.
                    # Label properties are only used in 5 augmentations
                    batch.segments = np.uint8(
                        nd_label(batch.y.squeeze().numpy() == 1)[0]
                    )
                    batch.props = regionprops(batch.segments)

                # Create the augmenter object
                aug_modules = Augmenters(
                    # NOTE: apply a single augmenter
                    # TODO: could apply a series of augmenters
                    augmentations=[aug_name],
                    rng=self.rng,
                )

                # Apply the object
                batch = aug_modules(batch)
                batch.segments = None
                batch.props = None

        # batch.x = torch.log(batch.x * 50.0 + 1.0).clip(1e-9, float('inf'))

        # if self.norm_values is not None:
        #     batch = self.norm_values(batch)

        # Get the centroid
        centroid = box(
            float(batch.left),
            float(batch.bottom),
            float(batch.right),
            float(batch.top),
        ).centroid
        batch.lon = torch.tensor([centroid.x])
        batch.lat = torch.tensor([centroid.y])

        return batch
