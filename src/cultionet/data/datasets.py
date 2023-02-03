import typing as T
from pathlib import Path
from functools import partial

from ..errors import TensorShapeError
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import TqdmParallel

import numpy as np
import attr
import torch
from torch_geometric.data import Data, Dataset
import psutil
import joblib
from joblib import delayed, parallel_backend
import geopandas as gpd
from shapely.geometry import box
from pytorch_lightning import seed_everything
from geosample import QuadTree
from tqdm.auto import tqdm

ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional

logger = set_color_logger(__name__)


def add_dims(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


def update_data(
    batch: Data,
    idx: T.Optional[int] = None,
    x: T.Optional[torch.Tensor] = None
) -> Data:
    image_id = None
    if idx is not None:
        if hasattr(batch, 'boxes'):
            if batch.boxes is not None:
                image_id = torch.zeros_like(batch.box_labels, dtype=torch.int64) + idx

    if x is not None:
        exclusion = ('x',)

        return Data(
            x=x,
            image_id=image_id,
            **{k: getattr(batch, k) for k in batch.keys if k not in exclusion}
        )
    else:
        return Data(
            image_id=image_id,
            **{k: getattr(batch, k) for k in batch.keys}
        )


def zscores(
    batch: Data,
    data_means: torch.Tensor,
    data_stds: torch.Tensor,
    idx: T.Optional[int] = None
) -> Data:
    """Normalizes data to z-scores

    Args:
        batch (Data): A `torch_geometric` data object.
        data_means (Tensor)
        data_stds (TEnsor)

    z = (x - μ) / σ
    """
    x = ((batch.x - add_dims(data_means)) / add_dims(data_stds))

    return update_data(
        batch=batch,
        idx=idx,
        x=x
    )


def _check_shape(
    d1: int,
    h1: int,
    w1: int,
    d2: int,
    h2: int,
    w2: int,
    index: int,
    uid: str
) -> T.Tuple[bool, int, str]:
    if (d1 != d2) or (h1 != h2) or (w1 != w2):
        return False, index, uid
    return True, index, uid


@attr.s
class EdgeDataset(Dataset):
    """An edge dataset
    """
    root: T.Union[str, Path, bytes] = attr.ib(default='.')
    transform: T.Any = attr.ib(default=None)
    pre_transform: T.Any = attr.ib(default=None)
    data_means: T.Optional[torch.Tensor] = attr.ib(
        validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None
    )
    data_stds: T.Optional[torch.Tensor] = attr.ib(
        validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None
    )
    crop_counts: T.Optional[torch.Tensor] = attr.ib(
        validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None
    )
    edge_counts: T.Optional[torch.Tensor] = attr.ib(
        validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None
    )
    pattern: T.Optional[str] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(str)), default='data*.pt')
    processes: T.Optional[int] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(int)), default=psutil.cpu_count())
    threads_per_worker: T.Optional[int] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(int)), default=1)
    random_seed: T.Optional[int] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(int)), default=42)

    data_list_ = None
    grid_id_column = 'grid_id'

    def __attrs_post_init__(self):
        super(EdgeDataset, self).__init__(
            str(self.root), transform=self.transform, pre_transform=self.pre_transform
        )
        seed_everything(self.random_seed, workers=True)
        self.rng = np.random.default_rng(self.random_seed)

    def get_data_list(self):
        """Gets the list of data files"""
        self.data_list_ = list(Path(self.processed_dir).glob(self.pattern))

        if not self.data_list_:
            logger.exception(f"No .pt files were found with pattern {self.pattern}.")

    def cleanup(self):
        for fn in self.data_list_:
            fn.unlink()

    def shuffle_items(self, data: T.Optional[list] = None):
        """Applies a random in-place shuffle to the data list"""
        if data is not None:
            self.rng.shuffle(data)
        else:
            self.rng.shuffle(self.data_list_)

    @property
    def num_time_features(self):
        """Get the number of time features
        """
        data = self[0]
        return int(data.ntime)

    @property
    def raw_file_names(self):
        """Get the raw file names
        """
        if not self.data_list_:
            self.get_data_list()

        return self.data_list_

    def to_frame(self) -> gpd.GeoDataFrame:
        """Converts the Dataset to a GeoDataFrame
        """
        def get_box_id(data_id: str, *bounds):
            return data_id, box(*bounds).centroid

        with parallel_backend(
            backend='loky',
            n_jobs=self.processes
        ):
            with TqdmParallel(
                tqdm_kwargs={
                    'total': len(self),
                    'desc': 'Building GeoDataFrame'
                }
            ) as pool:
                results = pool(
                    delayed(get_box_id)(
                        data.train_id,
                        data.left,
                        data.bottom,
                        data.right,
                        data.top
                    ) for data in self
                )

        ids, geometry = list(map(list, zip(*results)))
        df = gpd.GeoDataFrame(
            data=ids,
            columns=[self.grid_id_column],
            geometry=geometry,
            crs='epsg:4326'
        )

        return df

    def get_spatial_partitions(
        self,
        spatial_partitions: T.Union[str, Path, gpd.GeoDataFrame],
        splits: int = 0
    ) -> None:
        """Gets the spatial partitions
        """
        self.create_spatial_index()
        if isinstance(spatial_partitions, (str, Path)):
            spatial_partitions = gpd.read_file(spatial_partitions)
        else:
            spatial_partitions = self.to_frame()

        if splits > 0:
            qt = QuadTree(spatial_partitions, force_square=False)
            for __ in range(splits):
                qt.split()
            spatial_partitions = qt.to_frame()

        self.spatial_partitions = spatial_partitions.to_crs('epsg:4326')

    def query_partition_by_name(
        self,
        partition_column: str,
        partition_name: str,
        val_frac: float = None
    ) -> list:
        """Queries grid centroids that are within the partition
        """
        # Get the partition
        df = self.spatial_partitions.query(f"{partition_column} == '{partition_name}'")
        df_points = self.dataset_df.overlay(df, how='intersection')
        if df_points.empty:
            logger.warning(f"Partition {partition_name} does not have any data.")
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
            [data_path.stem.replace('data_', '') for data_path in self.data_list_],
            return_indices=True
        )[2].tolist()

        return indices

    def split_indices(
        self,
        indices: list,
        return_all: bool = True,
        return_indices: bool = False
    ) -> T.Tuple['EdgeDataset', 'EdgeDataset']:
        """Splits a list of indices into train and validation datasets
        """
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
        self,
        partition_column: str
    ) -> T.Tuple[str, 'EdgeDataset', 'EdgeDataset']:
        """Yield generator to iterate over spatial partitions
        """
        for kfold in self.spatial_partitions.itertuples():
            # Bounding box and indices of the kth fold
            kfold_indices = self.query_partition_by_name(
                partition_column,
                str(getattr(kfold, partition_column))
            )
            if not kfold_indices:
                continue

            train_ds, test_ds = self.split_indices(kfold_indices)

            yield str(getattr(kfold, partition_column)), train_ds, test_ds

    def create_spatial_index(self):
        """Creates the spatial index
        """
        dataset_grid_path = Path(self.processed_dir).parent.parent / 'dataset_grids.gpkg'
        if dataset_grid_path.is_file():
            self.dataset_df = gpd.read_file(dataset_grid_path)
        else:
            self.dataset_df = self.to_frame()
            self.dataset_df.to_file(dataset_grid_path, driver='GPKG')

    def download(self):
        pass

    def process(self):
        pass

    @property
    def processed_file_names(self):
        """Get a list of processed files"""
        return self.data_list_

    def check_dims(
        self,
        expected_dim: int,
        expected_height: int,
        expected_width: int,
        delete_mismatches: bool = False,
        tqdm_color: str = 'ffffff'
    ):
        """Checks if all tensors in the dataset match in shape dimensions
        """
        check_partial = partial(
            _check_shape,
            expected_dim,
            expected_height,
            expected_width
        )

        with parallel_backend(
            backend='loky',
            n_jobs=self.processes,
            inner_max_num_threads=self.threads_per_worker
        ):
            with TqdmParallel(
                tqdm_kwargs={
                    'total': len(self),
                    'desc': 'Checking dimensions',
                    'colour': tqdm_color
                }
            ) as pool:
                results = pool(
                    delayed(check_partial)(
                        self[i].x.shape[1],
                        self[i].height,
                        self[i].width,
                        i,
                        self[i].train_id
                    ) for i in range(0, len(self))
                )
        matches, indices, ids = list(map(list, zip(*results)))
        if not all(matches):
            indices = np.array(indices)
            null_indices = indices[~np.array(matches)]
            null_ids = np.array(ids)[null_indices].tolist()
            logger.warning(','.join(null_ids))
            logger.warning(f'{null_indices.shape[0]:,d} ids did not match the reference dimensions.')

            if delete_mismatches:
                logger.warning(f'Removing {null_indices.shape[0]:,d} .pt files.')
                for i in null_indices:
                    self.data_list_[i].unlink()
            else:
                raise TensorShapeError

    def len(self):
        """Returns the dataset length"""
        return len(self.processed_file_names)

    def split_train_val_by_partition(
        self,
        spatial_partitions: str,
        partition_column: str,
        val_frac: float,
        partition_name: T.Optional[str] = None,
    ) -> T.Tuple['EdgeDataset', 'EdgeDataset']:
        self.get_spatial_partitions(spatial_partitions=spatial_partitions)
        train_indices = []
        val_indices = []
        self.shuffle_items()
        # self.spatial_partitions is a GeoDataFrame with Point geometry
        for row in tqdm(
            self.spatial_partitions.itertuples(),
            total=len(self.spatial_partitions.index),
            desc='Sampling partitions'
        ):
            if partition_name is not None:
                if str(getattr(row, partition_column)) != partition_name:
                    continue
            # Query grid centroids within the partition
            indices = self.query_partition_by_name(
                partition_column,
                str(getattr(row, partition_column))
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
        self, val_frac: float,
        spatial_overlap_allowed: bool = True
    ) -> T.Tuple['EdgeDataset', 'EdgeDataset']:
        """Splits the dataset into train and validation

        Args:
            val_frac (float): The validation fraction.

        Returns:
            train dataset, validation dataset
        """
        self.shuffle_items()
        if spatial_overlap_allowed:
            n_train = int(len(self) * (1.0 - val_frac))
            train_ds = self[:n_train]
            val_ds = self[n_train:]
        else:
            self.create_spatial_index()
            # Keep only one centroid for each site
            df_isolated = self.dataset_df.drop_duplicates('geometry')
            # Randomly sample a percentage for validation
            df_val = df_isolated.sample(
                frac=val_frac, random_state=self.random_seed
            )
            val_geometries = df_val.geometry.tolist()
            train_idx = []
            val_idx = []
            # Iterate over all sample sites
            for i, row in enumerate(self.dataset_df.itertuples()):
                # Find matching geometry
                if row.geometry in val_geometries:
                    val_idx.append(i)
                else:
                    train_idx.append(i)

            train_ds = self[train_idx]
            val_ds = self[val_idx]

        return train_ds, val_ds

    def load_file(self, filename: T.Union[str, Path]) -> Data:
        return joblib.load(filename)

    def get(self, idx):
        """Gets an individual data object from the dataset

        Args:
            idx (int): The dataset index position.

        Returns:
            A `torch_geometric` data object.
        """
        batch = self.load_file(self.data_list_[idx])
        if isinstance(self.data_means, torch.Tensor):
            batch = zscores(batch, self.data_means, self.data_stds, idx=idx)
        else:
            batch = update_data(
                batch=batch,
                idx=idx
            )

        return batch
