import os
from pathlib import Path


p = Path(os.path.abspath(os.path.dirname(__file__)))
batch_file = p / 'train' / 'processed' / 'data_000001_2022_0_none.pt'
test_predictions = p / 'estimates.tif'
model_file = p / 'cultionet.pt'
norm_file = p / 'last.norm'
