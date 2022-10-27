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
import xarray as xr
import torch
from torch_geometric.data import Data


@dataclass
class LabeledData:
    x: np.ndarray
    y: T.Union[None, np.ndarray]
    bdist: T.Union[None, np.ndarray]
    ori: T.Union[None, np.ndarray]
    segments: T.Union[None, np.ndarray]
    props: T.Union[None, T.List]


def get_image_list_dims(
    image_list: T.Sequence[T.Union[Path, str]],
    src: xr.DataArray
) -> T.Tuple[int, int]:
    """Gets the dimensions of the time series
    """
    # Get the band count using the unique set of band/variable names
    nbands = len(list(set([Path(fn).parent.name for fn in image_list])))
    # Get the time count (.gw.nbands = number of total features)
    ntime = int(src.gw.nbands / nbands)

    return ntime, nbands


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
    mask_y: T.Optional[np.ndarray] = None,
    bdist: T.Optional[np.ndarray] = None,
    ori: T.Optional[np.ndarray] = None,
    zero_padding: T.Optional[int] = 0,
    other: T.Optional[np.ndarray] = None,
    **kwargs
) -> Data:
    """Creates a training data object
    """
    edge_indices_ = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attrs_ = torch.tensor(edge_attrs, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    xy = torch.tensor(xy, dtype=torch.float)

    boxes = None
    box_labels = None
    box_masks = None
    if mask_y is not None:
        boxes = mask_y['boxes']
        box_labels = mask_y['labels']
        box_masks = mask_y['masks']

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
            boxes=boxes,
            box_labels=box_labels,
            box_masks=box_masks,
            zero_padding=zero_padding,
            **kwargs
        )
    else:
        y = torch.tensor(y.flatten(), dtype=torch.float if 'float' in y.dtype.name else torch.long)
        bdist_ = torch.tensor(bdist.flatten(), dtype=torch.float)
        ori_ = torch.tensor(ori.flatten(), dtype=torch.float)

        if other is None:
            train_data = Data(
                x=x,
                edge_index=edge_indices_,
                edge_attrs=edge_attrs_,
                y=y,
                bdist=bdist_,
                ori=ori_,
                pos=xy,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                boxes=boxes,
                box_labels=box_labels,
                box_masks=box_masks,
                zero_padding=zero_padding,
                **kwargs
            )
        else:
            other_ = torch.tensor(other.flatten(), dtype=torch.float)

            train_data = Data(
                x=x,
                edge_index=edge_indices_,
                edge_attrs=edge_attrs_,
                y=y,
                bdist=bdist_,
                ori=ori_,
                pos=xy,
                other=other_,
                height=height,
                width=width,
                ntime=ntime,
                nbands=nbands,
                boxes=boxes,
                box_labels=box_labels,
                box_masks=box_masks,
                zero_padding=zero_padding,
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

        self.processed_path = self.data_path / 'processed'
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Create a random filename so that the processed
        # directory can be used by other processes
        filename = str(uuid.uuid4()).replace('-', '')
        pt_name = f'{filename}_.pt'
        self.pattern = f'{filename}*.pt'
        self.pt_file = self.processed_path / pt_name

        self._save(data)

    def _save(self, data: Data) -> None:
        torch.save(data, self.pt_file)

    def clear(self) -> None:
        if self.processed_path.is_dir():
            shutil.rmtree(str(self.processed_path))

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
