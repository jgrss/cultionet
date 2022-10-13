import typing as T
from pathlib import Path
import random
import logging
from functools import partial

from ..errors import TensorShapeError
from ..utils.logging import set_color_logger

import numpy as np
import attr
import torch
from torch_geometric.data import Data, Dataset
import psutil
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend

ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional

logger = set_color_logger(__name__)


def add_dims(d: torch.Tensor) -> torch.Tensor:
    return d.unsqueeze(0)


def zscores(
    batch: Data,
    data_means: torch.Tensor,
    data_stds: torch.Tensor
) -> Data:
    """Normalizes data to z-scores

    Args:
        batch (Data): A `torch_geometric` data object.
        data_means (Tensor)
        data_stds (TEnsor)

    z = (x - μ) / σ
    """
    x = ((batch.x - add_dims(data_means)) / add_dims(data_stds))

    return Data(x=x, **{k: getattr(batch, k) for k in batch.keys if k != 'x'})


def _check_shape(
    d1: int, d2: int, index: int, uid: str
) -> T.Tuple[bool, int, str]:
    if d1 != d2:
        return False, index, uid
    return True, index, uid


class TqdmParallel(Parallel):
    """A tqdm progress bar for joblib Parallel tasks

    Reference:
        https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
    """
    def __init__(self, tqdm_kwargs: dict):
        self.tqdm_kwargs = tqdm_kwargs
        super().__init__()

    def __call__(self, *args, **kwargs):
        with tqdm(**self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


@attr.s
class EdgeDataset(Dataset):
    """An edge dataset
    """
    root: T.Union[str, Path, bytes] = attr.ib(default='.')
    transform: T.Any = attr.ib(default=None)
    pre_transform: T.Any = attr.ib(default=None)
    data_means: T.Optional[torch.Tensor] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None)
    data_stds: T.Optional[torch.Tensor] = attr.ib(validator=ATTRVOPTIONAL(ATTRVINSTANCE(torch.Tensor)), default=None)
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
            return zscores(batch, self.data_means, self.data_stds)
        else:
            return batch
