import typing as T
from pathlib import Path
import random
import logging

from ..errors import TensorShapeError

import attr
import torch
from torch_geometric.data import Data, Dataset

ATTRVINSTANCE = attr.validators.instance_of
ATTRVIN = attr.validators.in_
ATTRVOPTIONAL = attr.validators.optional


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
    x = torch.cat([
        (batch.x[:, :-1] - add_dims(data_means)) / add_dims(data_stds),
        batch.x[:, -1][:, None]
    ], dim=1)

    return Data(x=x, **{k: getattr(batch, k) for k in batch.keys if k != 'x'})


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

    def check_dims(self):
        """Checks if all tensors in the dataset match in shape dimensions
        """
        ref_dim = self[0].x.shape
        for i in range(1, len(self)):
            if self[i].x.shape != ref_dim:
                raise TensorShapeError(f'{Path(self.data_list_[i]).name} does not match the reference.')

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
