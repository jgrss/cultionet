import typing as T
from pathlib import Path
from functools import partial

import numpy as np
import attr
import torch
import psutil
import joblib
from joblib import delayed, parallel_backend
import pandas as pd
import geopandas as gpd
import pygrts
from pytorch_lightning import seed_everything
from shapely.geometry import box
from scipy.ndimage.measurements import label as nd_label
from skimage.measure import regionprops
from torch_geometric.data import Data, Dataset
from tqdm.auto import tqdm

from .utils import LabeledData
from ..augment.augmenters import Augmenters
from ..errors import TensorShapeError
from ..models import model_utils
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import TqdmParallel


ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional

logger = set_color_logger(__name__)


def add_dims(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


def update_data(
    batch: Data,
    idx: T.Optional[int] = None,
    x: T.Optional[torch.Tensor] = None,
) -> Data:
    image_id = None
    if idx is not None:
        if hasattr(batch, "boxes"):
            if batch.boxes is not None:
                image_id = (
                    torch.zeros_like(batch.box_labels, dtype=torch.int64) + idx
                )

    if x is not None:
        exclusion = ("x",)

        return Data(
            x=x,
            image_id=image_id,
            **{
                k: getattr(batch, k)
                for k in batch.keys()
                if k not in exclusion
            },
        )
    else:
        return Data(
            image_id=image_id, **{k: getattr(batch, k) for k in batch.keys()}
        )


def zscores(
    batch: Data,
    data_means: torch.Tensor,
    data_stds: torch.Tensor,
    idx: T.Optional[int] = None,
) -> Data:
    """Normalizes data to z-scores.

    Args:
        batch (Data): A `torch_geometric` data object.
        data_means (Tensor): The data feature-wise means.
        data_stds (Tensor): The data feature-wise standard deviations.

    z = (x - μ) / σ
    """
    x = (batch.x - add_dims(data_means)) / add_dims(data_stds)

    return update_data(batch=batch, idx=idx, x=x)


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
        data_means: T.Optional[torch.Tensor] = None,
        data_stds: T.Optional[torch.Tensor] = None,
        crop_counts: T.Optional[torch.Tensor] = None,
        edge_counts: T.Optional[torch.Tensor] = None,
        pattern: T.Optional[str] = "data*.pt",
        processes: T.Optional[int] = psutil.cpu_count(),
        threads_per_worker: T.Optional[int] = 1,
        random_seed: T.Optional[int] = 42,
        transform: T.Any = None,
        pre_transform: T.Any = None,
        pre_filter: T.Any = None,
        augment_prob: float = 0.0,
    ):
        self.data_means = data_means
        self.data_stds = data_stds
        self.crop_counts = crop_counts
        self.edge_counts = edge_counts
        self.pattern = pattern
        self.processes = processes
        self.threads_per_worker = threads_per_worker
        self.random_seed = random_seed
        seed_everything(self.random_seed, workers=True)
        self.rng = np.random.default_rng(self.random_seed)
        self.augment_prob = augment_prob

        self.ct = model_utils.ConvToTime()
        self.gc = model_utils.GraphToConv()
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

        super().__init__(root, transform, pre_transform, pre_filter)

    def get_data_list(self):
        """Gets the list of data files."""
        self.data_list_ = list(Path(self.processed_dir).glob(self.pattern))

        if not self.data_list_:
            logger.exception(
                f"No .pt files were found with pattern {self.pattern}."
            )

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

    @property
    def raw_file_names(self):
        """Get the raw file names."""
        if not self.data_list_:
            self.get_data_list()

        return self.data_list_

    def to_frame(self) -> gpd.GeoDataFrame:
        """Converts the Dataset to a GeoDataFrame."""

        def get_box_id(data_id: str, *bounds):
            return data_id, box(*bounds).centroid

        with parallel_backend(backend="loky", n_jobs=self.processes):
            with TqdmParallel(
                tqdm_kwargs={
                    "total": len(self),
                    "desc": "Building GeoDataFrame",
                }
            ) as pool:
                results = pool(
                    delayed(get_box_id)(
                        data.train_id,
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

    def spatial_kfoldcv_iter(
        self, partition_column: str
    ) -> T.Tuple[str, "EdgeDataset", "EdgeDataset"]:
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
        dataset_grid_path = (
            Path(self.processed_dir).parent.parent / "dataset_grids.gpkg"
        )
        if dataset_grid_path.is_file():
            self.dataset_df = gpd.read_file(dataset_grid_path)
        else:
            self.dataset_df = self.to_frame()
            self.dataset_df.to_file(dataset_grid_path, driver="GPKG")

    def download(self):
        pass

    def process(self):
        pass

    @property
    def processed_file_names(self):
        """Get a list of processed files."""
        return self.data_list_

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
                        self[i].train_id,
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

    def len(self):
        """Returns the dataset length."""
        return len(self.processed_file_names)

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

            # Create column of each site's common id
            # (i.e., without the year and augmentation).
            self.dataset_df[id_column] = self.dataset_df.grid_id.str.split(
                "_", expand=True
            ).loc[:, 0]

            unique_ids = self.dataset_df.common_id.unique()
            if spatial_balance:
                # Separate train and validation by spatial location

                # Get unique site coordinates
                # NOTE: We do this because augmentations are stacked at
                # the same site, thus creating multiple files with the
                # same centroid.
                df_unique_locations = gpd.GeoDataFrame(
                    pd.Series(unique_ids)
                    .to_frame(name=id_column)
                    .merge(self.dataset_df, on=id_column)
                    .drop_duplicates(id_column)
                    .drop(columns=["grid_id"])
                ).to_crs("EPSG:8858")

                # Setup a quad-tree using the GRTS method
                # (see https://github.com/jgrss/pygrts for details)
                qt = pygrts.QuadTree(df_unique_locations, force_square=False)

                # Recursively split the quad-tree until each grid has
                # only one sample.
                qt.split_recursive(max_samples=1)

                n_val = int(val_frac * len(df_unique_locations.index))
                # `qt.sample` random samples from the quad-tree in a
                # spatially balanced manner. Thus, `df_val_sample` is
                # a GeoDataFrame with `n_val` sites spatially balanced.
                df_val_sample = qt.sample(n=n_val)

                # Since we only took one sample from each coordinate,
                # we need to find all of the .pt files that share
                # coordinates with the sampled sites.
                val_mask = self.dataset_df.common_id.isin(
                    df_val_sample.common_id
                )
            else:
                # Randomly sample a percentage for validation
                df_val_ids = (
                    pd.Series(unique_ids)
                    .sample(frac=val_frac, random_state=self.random_seed)
                    .to_frame(name=id_column)
                )
                # Get all ids for validation samples
                val_mask = self.dataset_df.common_id.isin(df_val_ids.common_id)

            # Get train/val indices
            val_idx = self.dataset_df.loc[val_mask].index.tolist()
            train_idx = self.dataset_df.loc[~val_mask].index.tolist()

            # Slice the dataset
            train_ds = self[train_idx]
            val_ds = self[val_idx]

        return train_ds, val_ds

    def load_file(self, filename: T.Union[str, Path]) -> Data:
        return joblib.load(filename)

    def get(self, idx):
        """Gets an individual data object from the dataset.

        Args:
            idx (int): The dataset index position.

        Returns:
            A `torch_geometric` data object.
        """
        batch = self.load_file(self.data_list_[idx])

        if batch.y is not None:
            if self.rng.normal() > 1 - self.augment_prob:
                # TODO: get segments from crops, not edges
                y = batch.y.reshape(batch.height, batch.width)
                # Reshape from ((H*W) x (C*T)) -> (B x (C * T) x H x W)
                x = self.gc(batch.x, 1, batch.height, batch.width)

                # Choose one augmentation to apply
                aug_name = self.rng.choice(self.augmentations_)
                props = None
                if aug_name in (
                    'tswarp',
                    'tsnoise',
                    'tsdrift',
                    'tspeaks',
                ):
                    # FIXME: By default, the crop value is 1 (background is 0 and edges are 2).
                    # But, it would be better to get 1 from an argument.
                    # Label properties are only used in 4 augmentations
                    props = regionprops(np.uint8(nd_label(y == 1)[0]))

                labeled_data = LabeledData(
                    x=x.squeeze(dim=0),
                    y=y,
                    bdist=batch.bdist.reshape(batch.height, batch.width),
                    ori=None,
                    segments=None,
                    props=props,
                )

                # Create the augmenter object
                augmenters = Augmenters(
                    augmentations=[aug_name],
                    ntime=batch.ntime,
                    nbands=batch.nbands,
                )
                # Apply the object
                augmenter = augmenters.augmenters_[0]
                batch = augmenter(labeled_data, aug_args=augmenters.aug_args)

        if isinstance(self.data_means, torch.Tensor):
            batch = zscores(batch, self.data_means, self.data_stds, idx=idx)
        else:
            batch = update_data(batch=batch, idx=idx)

        return batch
