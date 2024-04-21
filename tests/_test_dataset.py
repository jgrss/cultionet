from pathlib import Path

import torch

from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths

from .data import batch_file

project_path = Path(__file__).parent.absolute()
ppaths = setup_paths(project_path)
ds = EdgeDataset(ppaths.train_path)
DATA = next(iter(ds))
LOADED_DATA = ds.load_file(batch_file)


def test_load():
    assert torch.allclose(DATA.x, LOADED_DATA.x)
    assert torch.allclose(DATA.y, LOADED_DATA.y)


def test_ds_type():
    assert isinstance(ds, EdgeDataset)


def test_ds_len():
    assert len(ds) == 1


def test_x_type():
    assert isinstance(DATA.x, torch.Tensor)


def test_x_shape():
    assert DATA.x.shape == (10000, 39)


def test_y_shape():
    assert DATA.y.shape == (10000,)


def test_dims_attr():
    assert DATA.nbands == 3
    assert DATA.ntime == 13
    assert DATA.height == 100
    assert DATA.width == 100
