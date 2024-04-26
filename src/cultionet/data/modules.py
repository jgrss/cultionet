import typing as T

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler

from .datasets import EdgeDataset
from .utils import collate_fn


class EdgeDataModule(LightningDataModule):
    """A Lightning data module."""

    def __init__(
        self,
        train_ds: T.Optional[EdgeDataset] = None,
        val_ds: T.Optional[EdgeDataset] = None,
        test_ds: T.Optional[EdgeDataset] = None,
        predict_ds: T.Optional[EdgeDataset] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        shuffle: bool = True,
        sampler: T.Optional[Sampler] = None,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        generator: T.Optional[torch.Generator] = None,
    ):
        super().__init__()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.predict_ds = predict_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sampler = sampler
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.generator = generator

    def train_dataloader(self):
        """Returns a data loader for train data."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=None if self.sampler is not None else self.shuffle,
            num_workers=self.num_workers,
            sampler=self.sampler,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            generator=self.generator,
        )

    def val_dataloader(self):
        """Returns a data loader for validation data."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=self.generator,
        )

    def test_dataloader(self):
        """Returns a data loader for test data."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=self.generator,
        )

    def predict_dataloader(self):
        """Returns a data loader for predict data."""
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            generator=self.generator,
        )
