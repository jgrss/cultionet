from pathlib import Path

from .data import batch_file
from cultionet.data.datasets import EdgeDataset
from cultionet.utils.project_paths import setup_paths
from cultionet.models import model_utils

import torch


project_path = Path(__file__).parent.absolute()
ppaths = setup_paths(project_path)
ds = EdgeDataset(ppaths.train_path)
DATA = ds.load_file(batch_file)


def test_graph_to_conv():
    """Test reshaping from graph/column order to multi-dimensional/convolution order
    """
    gc = model_utils.GraphToConv()

    x = gc(DATA.x, 1, DATA.height, DATA.width)

    assert x.shape == (1, DATA.x.shape[1], DATA.height, DATA.width)
    assert torch.allclose(x[0, :, 0, 0], DATA.x[0])
    assert torch.allclose(x[0, :, 0, 1], DATA.x[1])
    assert torch.allclose(x[0, :, -1, -2], DATA.x[-2])
    assert torch.allclose(x[0, :, -1, -1], DATA.x[-1])


def test_conv_to_graph():
    """Test reshaping from multi-dimensional/convolution order to graph/column order
    """
    gc = model_utils.GraphToConv()
    cg = model_utils.ConvToGraph()

    x = gc(DATA.x, 1, DATA.height, DATA.width)
    y = cg(x)

    assert torch.allclose(y, DATA.x)


def test_conv_to_time():
    """Test reshaping from multi-dimensional/convolution order to time order
    """
    gc = model_utils.GraphToConv()
    ct = model_utils.ConvToTime()

    x = gc(DATA.x, 1, DATA.height, DATA.width)
    t = ct(x, nbands=DATA.nbands, ntime=DATA.ntime)

    assert torch.allclose(
        x[0, :DATA.ntime, 0, 0], t[0, 0, :, 0, 0]
    )
    assert torch.allclose(
        x[0, DATA.ntime:DATA.ntime*2, 0, 0], t[0, 1, :, 0, 0]
    )
    assert torch.allclose(
        x[0, DATA.ntime*2:, 0, 0], t[0, 2, :, 0, 0]
    )
    assert torch.allclose(
        x[0, :DATA.ntime, 0, 1], t[0, 0, :, 0, 1]
    )
    assert torch.allclose(
        x[0, DATA.ntime:DATA.ntime*2, 0, 1], t[0, 1, :, 0, 1]
    )
    assert torch.allclose(
        x[0, DATA.ntime*2:, 0, 1], t[0, 2, :, 0, 1]
    )
    assert torch.allclose(
        x[0, :DATA.ntime, 50, 50], t[0, 0, :, 50, 50]
    )
    assert torch.allclose(
        x[0, DATA.ntime:DATA.ntime*2, 50, 50], t[0, 1, :, 50, 50]
    )
    assert torch.allclose(
        x[0, DATA.ntime*2:, 50, 50], t[0, 2, :, 50, 50]
    )
    assert torch.allclose(
        x[0, :DATA.ntime, -1, -1], t[0, 0, :, -1, -1]
    )
    assert torch.allclose(
        x[0, DATA.ntime:DATA.ntime*2, -1, -1], t[0, 1, :, -1, -1]
    )
    assert torch.allclose(
        x[0, DATA.ntime*2:, -1, -1], t[0, 2, :, -1, -1]
    )
