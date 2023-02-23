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
    cg = model_utils.GraphToConv()
    x = cg(DATA.x, 1, DATA.height, DATA.width)

    assert x.shape == (1, DATA.x.shape[1], DATA.height, DATA.width)
    assert torch.allclose(x[0, :, 0, 0], DATA.x[0])
    assert torch.allclose(x[0, :, 0, 1], DATA.x[1])
    assert torch.allclose(x[0, :, -1, -2], DATA.x[-2])
    assert torch.allclose(x[0, :, -1, -1], DATA.x[-1])


def test_conv_to_graph():
    """Test reshaping from multi-dimensional/convolution order to graph/column order
    """
    cg = model_utils.GraphToConv()
    gc = model_utils.ConvToGraph()
    x = cg(DATA.x, 1, DATA.height, DATA.width)
    y = gc(x)

    assert torch.allclose(y, DATA.x)
