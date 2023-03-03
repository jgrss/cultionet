import typing as T

from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from .datasets import EdgeDataset


class EdgeDataModule(LightningDataModule):
    """A Lightning data module
    """
    def __init__(
        self,
        train_ds: T.Optional[EdgeDataset] = None,
        val_ds: T.Optional[EdgeDataset] = None,
        test_ds: T.Optional[EdgeDataset] = None,
        predict_ds: T.Optional[EdgeDataset] = None,
        batch_size: int = 5,
        num_workers: int = 0,
        shuffle: bool = True
    ):
        super().__init__()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.predict_ds = predict_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        """Returns a data loader for train data
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns a data loader for validation data
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns a data loader for test data
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Returns a data loader for predict data
        """
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
