from .data import batch_file, test_predictions, model_file, norm_file
from cultionet.model import load_model, predict

import geowombat as gw
import numpy as np
from rasterio.windows import Window
import torch


batch = torch.load(batch_file)

lit_model = load_model(
    model_file=model_file,
    num_features=batch.num_features,
    num_time_features=batch.ntime,
    device='cpu'
)[1]

data_values = torch.load(norm_file)
w = Window(row_off=0, col_off=0, height=100, width=100)
w_pad = w


def test_estimates():
    stack = predict(
        lit_model=lit_model,
        data=batch,
        data_values=data_values,
        w=w,
        w_pad=w_pad
    )

    with gw.open(test_predictions) as test_src:
        matches = (test_src.values == np.uint16(stack*10_000.0)).sum()
        assert matches / stack.size >= 0.999, 'The predictions do not match the reference.'
