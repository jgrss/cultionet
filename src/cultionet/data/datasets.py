import typing as T
from pathlib import Path
import random
from functools import partial

from ..errors import TensorShapeError
from ..utils.logging import set_color_logger
from ..utils.model_preprocessing import TqdmParallel

import numpy as np
import attr
import torch
from torch_geometric.data import Data, Dataset
import psutil
from joblib import delayed, parallel_backend
import rtree
import geopandas as gpd

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
    if x is not None:
        exclusion = ('x',)
    image_id = None
    if idx is not None:
        if hasattr(batch, 'boxes'):
            if batch.boxes is not None:
                image_id = torch.zeros_like(batch.box_labels, dtype=torch.int64) + idx

    if x is not None:
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
    d1: int, d2: int, index: int, uid: str
) -> T.Tuple[bool, int, str]:
    if d1 != d2:
        return False, index, uid
    return True, index, uid


def _check_shape(
    d1: int, d2: int, index: int, uid: str
) -> T.Tuple[bool, int, str]:
    if d1 != d2:
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
    pattern: T.Optional[str] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(str)), default='data*.pt')
    processes: T.Optional[int] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(int)), default=psutil.cpu_count())
    threads_per_worker: T.Optional[int] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(int)), default=1)

    data_list_ = None

    def __attrs_post_init__(self):
        super(EdgeDataset, self).__init__(
            str(self.root), transform=self.transform, pre_transform=self.pre_transform
        )

    def get_data_list(self):
        """Gets the list of data files"""
        self.data_list_ = list(Path(self.processed_dir).glob(self.pattern))

    def shuffle_items(self):
        """Applies a random in-place shuffle to the data list"""
        random.shuffle(self.data_list_)

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

    def to_frame(self, processes: int = 1) -> gpd.GeoDataFrame:
        """Converts the Dataset to a GeoDataFrame
        """
        from shapely.geometry import box

        with parallel_backend(
            backend='loky',
            n_jobs=processes
        ):
            with TqdmParallel(
                tqdm_kwargs={
                    'total': len(self),
                    'desc': 'Building GeoDataFrame'
                }
            ) as pool:
                geometry = pool(
                    delayed(box)(
                        data.left, data.bottom, data.right, data.top
                    ) for data in self
                )

        df = gpd.GeoDataFrame(
            geometry=geometry,
            crs='epsg:4326'
        )

        return df

    def get_skfoldcv_partitions(
        self,
        spatial_partitions: T.Union[str, Path, gpd.GeoDataFrame],
        splits: int = 0,
        processes: int = 0
    ) -> None:
        """Gets the spatial partitions
        """
        self.create_rtree()
        if isinstance(spatial_partitions, (str, Path)):
            spatial_partitions = gpd.read_file(spatial_partitions)
        else:
            spatial_partitions = self.to_frame(processes=processes)

        if splits > 0:
            try:
                from geosample import QuadTree
            except ImportError:
                raise ImportError('geosample must be installed to partition into k-folds.')

            qt = QuadTree(spatial_partitions, force_square=False)
            for __ in range(splits):
                qt.split()
            spatial_partitions = qt.to_frame()

        self.spatial_partitions = spatial_partitions

    def spatial_kfoldcv_iter(self) -> T.Tuple['EdgeDataset', 'EdgeDataset']:
        """Yield generator to iterate over spatial partitions
        """
        for kfold in self.spatial_partitions.itertuples():
            # Bounding box and indices of the kth fold
            bbox = kfold.geometry.bounds
            kfold_indices = list(self.rtree_index.contains(bbox))

            train_idx = []
            test_idx = []
            for i in range(0, len(self)):
                if i in kfold_indices:
                    test_idx.append(i)
                else:
                    train_idx.append(i)

            yield self[train_idx], self[test_idx]

    def create_rtree(self):
        """Creates the Rtree spatial index
        """
        self.rtree_index = rtree.index.Index()
        for idx, fn in enumerate(self.data_list_):
            data = torch.load(fn)
            left = data.left
            bottom = data.bottom
            right = data.right
            top = data.top
            self.rtree_index.insert(idx, (left, bottom, right, top))

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
        delete_mismatches: bool = False,
        tqdm_color: str = 'ffffff'
    ):
        """Checks if all tensors in the dataset match in shape dimensions
        """
        check_partial = partial(_check_shape, expected_dim)

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
                        self[i].x.shape[1], i, self[i].train_id
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

    def split_train_val(self, val_frac: float) -> T.Tuple['EdgeDataset', 'EdgeDataset']:
        """Splits the dataset into train and validation

        Args:
            val_frac (float): The validation fraction.

        Returns:
            train dataset, validation dataset
        """
        n_train = int(len(self) * (1-val_frac))
        self.shuffle_items()
        train_ds = self[:n_train]
        val_ds = self[n_train:]

        return train_ds, val_ds

    def get(self, idx):
        """Gets an individual data object from the dataset

        Args:
            idx (int): The dataset index position.

        Returns:
            A `torch_geometric` data object.
        """
        batch = torch.load(self.data_list_[idx])
        if isinstance(self.data_means, torch.Tensor):
            batch = zscores(batch, self.data_means, self.data_stds, idx=idx)
        else:
            batch = update_data(
                batch=batch,
                idx=idx
            )

        return batch
