import os
from pathlib import Path
import torch
from torch_geometric.data import Data


p = Path(os.path.abspath(os.path.dirname(__file__)))

def load_data() -> Data:
    return torch.load(str(p / 'train' / 'processed' / 'data_000001_2022_0_none.pt'))

