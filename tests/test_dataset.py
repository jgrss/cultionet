import os
from pathlib import Path

from .data import batch_file
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
import torch


project_path = Path(os.path.abspath(os.path.dirname(__file__)))
ppaths = setup_paths(project_path)
ds = EdgeDataset(ppaths.train_path)
data = next(iter(ds))
loaded_data = ds.load_file(batch_file)


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


def test_dims_attr():
    assert data.nbands == 3
    assert data.ntime == 13
    assert data.height == 100
    assert data.width == 100
