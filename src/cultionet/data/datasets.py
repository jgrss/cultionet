import typing as T
from copy import deepcopy
from functools import partial
from pathlib import Path

import attr
import geopandas as gpd
import joblib
import numpy as np
import psutil
import pygrts
from joblib import delayed, parallel_backend
from pytorch_lightning import seed_everything
from scipy.ndimage.measurements import label as nd_label
from shapely.geometry import box
from skimage.measure import regionprops
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..augment.augmenters import Augmenters
from ..errors import TensorShapeError
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import TqdmParallel
from ..utils.normalize import NormValues
from .data import Data

ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional

logger = set_color_logger(__name__)


def _check_shape(
    d1: int, h1: int, w1: int, d2: int, h2: int, w2: int, index: int, uid: str
) -> T.Tuple[bool, int, str]:
    if (d1 != d2) or (h1 != h2) or (w1 != w2):
        return False, index, uid
    return True, index, uid


class EdgeDataset(Dataset):
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

        seed_everything(self.random_seed, workers=True)
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
            'speckle',
        ]

        self.data_list_ = None
        self.processed_dir = Path(self.root) / 'processed'
        self.get_data_list()

    def get_data_list(self):
        """Gets the list of data files."""
        data_list_ = list(Path(self.processed_dir).glob(self.pattern))

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

    def shuffle_items(self, data: T.Optional[list] = None):
        """Applies a random in-place shuffle to the data list."""
        if data is not None:
            self.rng.shuffle(data)
        else:
            self.rng.shuffle(self.data_list_)

    @property
    def num_time_features(self):
        """Get the number of time features."""
        data = self[0]
        return int(data.ntime)

    def to_frame(self) -> gpd.GeoDataFrame:
        """Converts the Dataset to a GeoDataFrame."""

        def get_box_id(data_id: str, *bounds):
            return data_id, box(*list(map(float, bounds))).centroid

        with parallel_backend(backend="loky", n_jobs=self.processes):
            with TqdmParallel(
                tqdm_kwargs={
                    "total": len(self),
                    "desc": "Building GeoDataFrame",
                }
            ) as pool:
                results = pool(
                    delayed(get_box_id)(
                        data.batch_id,
                        data.left,
                        data.bottom,
                        data.right,
                        data.top,
                    )
                    for data in self
                )

        ids, geometry = list(map(list, zip(*results)))
        df = gpd.GeoDataFrame(
            data=ids,
            columns=[self.grid_id_column],
            geometry=geometry,
            crs="epsg:4326",
        )

        return df

    def get_spatial_partitions(
        self,
        spatial_partitions: T.Union[str, Path, gpd.GeoDataFrame],
        splits: int = 0,
    ) -> None:
        """Gets the spatial partitions."""
        self.create_spatial_index()
        if isinstance(spatial_partitions, (str, Path)):
            spatial_partitions = gpd.read_file(spatial_partitions)
        else:
            spatial_partitions = self.to_frame()

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

    def create_spatial_index(self):
        """Creates the spatial index."""
        dataset_grid_path = self.root / "dataset_grids.gpkg"

        if dataset_grid_path.is_file():
            self.dataset_df = gpd.read_file(dataset_grid_path)
        else:
            self.dataset_df = self.to_frame()
            self.dataset_df.to_file(dataset_grid_path, driver="GPKG")

    def check_dims(
        self,
        expected_dim: int,
        expected_height: int,
        expected_width: int,
        delete_mismatches: bool = False,
        tqdm_color: str = "ffffff",
    ):
        """Checks if all tensors in the dataset match in shape dimensions."""
        check_partial = partial(
            _check_shape, expected_dim, expected_height, expected_width
        )
        with parallel_backend(
            backend="loky",
            n_jobs=self.processes,
            inner_max_num_threads=self.threads_per_worker,
        ):
            with TqdmParallel(
                tqdm_kwargs={
                    "total": len(self),
                    "desc": "Checking dimensions",
                    "colour": tqdm_color,
                }
            ) as pool:
                results = pool(
                    delayed(check_partial)(
                        self[i].x.shape[1],
                        self[i].height,
                        self[i].width,
                        i,
                        self[i].batch_id,
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
        self.shuffle_items()
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
    ) -> T.Tuple["EdgeDataset", "EdgeDataset"]:
        """Splits the dataset into train and validation.

        Args:
            val_frac (float): The validation fraction.

        Returns:
            train dataset, validation dataset
        """
        id_column = "common_id"
        self.shuffle_items()

        if spatial_overlap_allowed:
            n_train = int(len(self) * (1.0 - val_frac))
            train_ds = self[:n_train]
            val_ds = self[n_train:]
        else:
            # Create a GeoDataFrame of every .pt file in
            # the dataset.
            self.create_spatial_index()

            if spatial_balance:
                # Separate train and validation by spatial location

                # Setup a quad-tree using the GRTS method
                # (see https://github.com/jgrss/pygrts for details)
                qt = pygrts.QuadTree(
                    self.dataset_df.to_crs("EPSG:8858"),
                    force_square=False,
                )

                # Recursively split the quad-tree until each grid has
                # only one sample.
                qt.split_recursive(max_samples=1)

                n_val = int(val_frac * len(self.dataset_df.index))
                # `qt.sample` random samples from the quad-tree in a
                # spatially balanced manner. Thus, `df_val_sample` is
                # a GeoDataFrame with `n_val` sites spatially balanced.
                df_val_sample = qt.sample(n=n_val)

                # Since we only took one sample from each coordinate,
                # we need to find all of the .pt files that share
                # coordinates with the sampled sites.
                val_mask = self.dataset_df[self.grid_id_column].isin(
                    df_val_sample[self.grid_id_column]
                )
            else:
                # Randomly sample a percentage for validation
                df_val_ids = self.dataset_df.sample(
                    frac=val_frac, random_state=self.random_seed
                ).to_frame(name=id_column)
                # Get all ids for validation samples
                val_mask = self.dataset_df[self.grid_id_column].isin(
                    df_val_ids[self.grid_id_column]
                )

            # Get train/val indices
            val_idx = self.dataset_df.loc[val_mask].index.values
            train_idx = self.dataset_df.loc[~val_mask].index.values

            # Slice the dataset
            train_ds = self[train_idx]
            val_ds = self[val_idx]

        val_ds.augment_prob = 0.0

        return train_ds, val_ds

    def load_file(self, filename: T.Union[str, Path]) -> Data:
        return joblib.load(filename)

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

        batch = Data.from_file(self.data_list_[idx])

        if batch.y is not None:
            if self.rng.normal() > 1 - self.augment_prob:
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
                    # Label properties are only used in 4 augmentations
                    batch.segments = np.uint8(
                        nd_label(batch.y.squeeze().numpy() == 1)[0]
                    )
                    batch.props = regionprops(batch.segments)

                # Create the augmenter object
                augmenters = Augmenters(augmentations=[aug_name])

                # Apply the object
                augmenter = augmenters.augmenters_[0]
                batch = augmenter(batch, aug_args=augmenters.aug_args)
                batch.segments = None
                batch.props = None

        if self.norm_values is not None:
            batch = self.norm_values(batch)

        return batch
