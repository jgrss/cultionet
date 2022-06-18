import shutil
import typing as T
from dataclasses import dataclass
from pathlib import Path
import uuid

from .datasets import EdgeDataset
from ..networks import SingleSensorNetwork
from ..utils.reshape import nd_to_columns
from ..utils.normalize import NormValues

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class LabeledData:
    x: np.ndarray
    y: np.ndarray
    bdist: np.ndarray
    segments: np.ndarray
    props: T.List


def create_data_object(
    x: np.ndarray,
    edge_indices: np.ndarray,
    edge_attrs: np.ndarray,
    xy: np.ndarray,
    ntime: int,
    nbands: int,
    height: int,
    width: int,
    y: T.Optional[np.ndarray] = None,
    bdist: T.Optional[np.ndarray] = None,
    other: T.Optional[np.ndarray] = None,
    **kwargs
) -> Data:
    """Creates a training data object
    """
    edge_indices_ = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attrs_ = torch.tensor(edge_attrs, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    xy = torch.tensor(xy, dtype=torch.float)

    if y is None:
        train_data = Data(
            x=x,
            edge_index=edge_indices_,
            edge_attrs=edge_attrs_,
            pos=xy,
            height=height,
            width=width,
            ntime=ntime,
            nbands=nbands,
            **kwargs
        )
    else:
        y_ = torch.tensor(y.flatten(), dtype=torch.float if 'float' in y.dtype.name else torch.long)
        bdist_ = torch.tensor(bdist.flatten(), dtype=torch.float)

        if other is None:
            train_data = Data(
                x=x,
                edge_index=edge_indices_,
                edge_attrs=edge_attrs_,
                y=y_,
                bdist=bdist_,
                pos=xy,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                **kwargs
            )
        else:
            other_ = torch.tensor(other.flatten(), dtype=torch.float)

            train_data = Data(
                x=x,
                edge_index=edge_indices_,
                edge_attrs=edge_attrs_,
                y=y_,
                bdist=bdist_,
                pos=xy,
                other=other_,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                **kwargs
            )

    # Ensure the correct node count
    train_data.num_nodes = x.shape[0]
    
    return train_data


def create_network_data(xvars: np.ndarray, ntime: int, nbands: int) -> Data:

    # Create the network
    nwk = SingleSensorNetwork(np.ascontiguousarray(xvars, dtype='float64'), k=3)

    edge_indices_a, edge_indices_b, edge_attrs_diffs, edge_attrs_dists, xpos, ypos = nwk.create_network()
    edge_indices = np.c_[edge_indices_a, edge_indices_b]
    edge_attrs = np.c_[edge_attrs_diffs, edge_attrs_dists]
    xy = np.c_[xpos, ypos]
    nfeas, nrows, ncols = xvars.shape
    xvars = nd_to_columns(xvars, nfeas, nrows, ncols)

    return create_data_object(
        xvars, edge_indices, edge_attrs, xy, ntime=ntime, nbands=nbands, height=nrows, width=ncols
    )


class NetworkDataset(object):
    def __init__(self, data: Data, data_path: Path, data_values: NormValues):
        self.data_values = data_values
        self.data_path = data_path

        processed_path = self.data_path / 'processed'
        if processed_path.is_dir():
            shutil.rmtree(str(processed_path))
        processed_path.mkdir(parents=True, exist_ok=True)

        # Create a random filename so that the processed
        # directory can be used by other processes
        filename = str(uuid.uuid4()).replace('-', '')
        pt_name = f'{filename}_.pt'
        self.pattern = f'{filename}*.pt'
        self.pt_file = processed_path / pt_name

        self._save(data)

    def _save(self, data: Data) -> None:
        torch.save(data, self.pt_file)

    def unlink(self) -> None:
        self.pt_file.unlink()

    @property
    def ds(self) -> EdgeDataset:
        return EdgeDataset(
            self.data_path,
            data_means=self.data_values.mean,
            data_stds=self.data_values.std,
            pattern=self.pattern
        )
