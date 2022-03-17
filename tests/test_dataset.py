import os
from pathlib import Path

from .data import load_data

from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
import torch
import pytest


project_path = Path(os.path.abspath(os.path.dirname(__file__)))
ppaths = setup_paths(project_path)
ds = EdgeDataset(ppaths.train_path)
data = next(iter(ds))
loaded_data = load_data()


def test_load():
    assert torch.allclose(data.x, loaded_data.x)
    assert torch.allclose(data.y, loaded_data.y)


def test_ds_type():
    assert isinstance(ds, EdgeDataset)


def test_ds_len():
    assert len(ds) == 1


def test_x_type():
    assert isinstance(data.x, torch.Tensor)


def test_x_shape():
    assert data.x.shape == (10000, 39)


def test_y_shape():
    assert data.y.shape == (10000,)


def test_nbands_attr():
    assert data.nbands == 13


def test_image_shape():
    assert data.height == 100
    assert data.width == 100
