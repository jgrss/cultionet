import typing as T

from ..data.utils import create_data_object, LabeledData
from ..networks import SingleSensorNetwork
from ..utils.reshape import nd_to_columns

import numpy as np
import cv2
from skimage import util as sk_util
from tsaug import AddNoise, Drift, TimeWarp
from torch_geometric.data import Data


def augment_time(
    ldata: LabeledData,
    p: T.Any,
    xaug: np.ndarray,
    nbands: int,
    add_noise: bool,
    warper: T.Union[AddNoise, Drift, TimeWarp]
) -> np.ndarray:
    """Applies temporal augmentation to a dataset
    """
    # Get the segment
    min_row, min_col, max_row, max_col = p.bbox
    xseg = xaug[:, min_row:max_row, min_col:max_col].copy()
    seg = ldata.segments[min_row:max_row, min_col:max_col].copy()
    mask = np.uint8(seg == p.label)[np.newaxis]
    
    ntime, nrows, ncols = xseg.shape
    # Reshape from (T x H x W) -> (H*W x T X C)
    xseg = (xseg.transpose(1, 2, 0)
            .reshape(nrows*ncols, ntime)
            .reshape(nrows*ncols, int(ntime/nbands), nbands))

    # if xseg.shape[1] < VegetationIndices().n_vis:
    #     raise ValueError('The time series stack should have 3 layers.')
    # Warp the segment
    xseg_warped = warper.augment(xseg)
    if add_noise:
        noise_warper = AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
        xseg_warped = noise_warper.augment(xseg_warped)
    # Reshape back from (H*W x T x C) -> (T x H x W)
    xseg_warped = (xseg_warped.transpose(0, 2, 1)
                   .reshape(nrows*ncols, ntime)
                   .T.reshape(ntime, nrows, ncols)
                   .clip(0, 1))
    xseg = xseg.reshape(nrows*ncols, ntime).T.reshape(ntime, nrows, ncols)

    # Insert back into full array
    xaug[:, min_row:max_row, min_col:max_col] = np.where(mask == 1, xseg_warped, xseg)

    return xaug


def augment(
    ldata: LabeledData,
    aug: str,
    nbands: int,
    k: int = 3,
    **kwargs
) -> Data:
    """Applies augmentation to a dataset
    """
    x = ldata.x
    y = ldata.y
    bdist = ldata.bdist
    ntime_features = x.shape[0]

    xaug = x.copy()

    label_dtype = 'float' if 'float' in y.dtype.name else 'int'

    if 'none' in aug:
        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 'ts-warp' in aug:
        # Warp each segment
        for p in ldata.props:
            xaug = augment_time(
                ldata, p, xaug, nbands, True,
                TimeWarp(
                    n_speed_change=np.random.randint(low=1, high=3),
                    max_speed_ratio=np.random.uniform(low=1.1, high=3.0),
                    static_rand=True
                )
            )
        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 'ts-noise' in aug:
        # Warp each segment
        for p in ldata.props:
            xaug = augment_time(
                ldata, p, xaug, nbands, False,
                AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
            )
        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 'ts-drift' in aug:
        # Warp each segment
        for p in ldata.props:
            xaug = augment_time(
                ldata, p, xaug, nbands, True,
                Drift(
                    max_drift=np.random.uniform(low=0.05, high=0.1),
                    n_drift_points=int(xaug.shape[0] / 2.0),
                    static_rand=True
                )
            )
        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 'rot' in aug:
        deg = int(aug.replace('rot', ''))

        deg_dict = {90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}

        xaug = np.zeros((xaug.shape[0], *cv2.rotate(np.float32(x[0]), deg_dict[deg]).shape), dtype=xaug.dtype)

        for i in range(0, x.shape[0]):
            xaug[i] = cv2.rotate(np.float32(x[i]), deg_dict[deg])

        if label_dtype == 'float':
            yaug = cv2.rotate(np.float32(y), deg_dict[deg])
        else:
            yaug = cv2.rotate(np.uint8(y), deg_dict[deg])
        bdist_aug = cv2.rotate(np.float32(bdist), deg_dict[deg])

    elif 'roll' in aug:
        ntime = int(xaug.shape[0]/nbands)
        if 'p' in aug:
            shift_percent = int(aug.replace('rollp', ''))/100
            shift = int(ntime*shift_percent)
        elif 's' in aug:
            shift = int(aug.replace('rolls', ''))
        else:
            shift = int(aug.replace('roll', ''))
        xaug_copy = np.zeros((int(xaug.shape[0]/nbands), nbands, xaug.shape[1], xaug.shape[2]))
        for n in range(nbands):
            xaug_copy[:, n] = xaug[ntime*n: ntime*(n+1)]
        xaug_copy = np.roll(xaug_copy, shift=shift, axis=0)
        for n in range(nbands):
            xaug[ntime*n: ntime*(n+1)] = xaug_copy[:, n]
        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 'flip' in aug:
        if aug == 'flipfb':
            # Reverse the channels
            for b in range(0, xaug.shape[0], nbands):
                xaug[b:b+nbands] = xaug[b:b+nbands][::-1]
            yaug = y.copy()
            bdist_aug = bdist.copy()
        else:

            for i in range(0, x.shape[0]):
                xaug[i] = getattr(np, aug)(x[i])

            yaug = getattr(np, aug)(y)
            bdist_aug = getattr(np, aug)(bdist)

    elif 'scale' in aug:
        scale = float(aug.replace('scale', ''))

        height = int(xaug.shape[1] * scale)
        width = int(xaug.shape[2] * scale)

        xaug = np.zeros((xaug.shape[0], height, width), dtype=xaug.dtype)

        dim = (width, height)

        for i in range(0, x.shape[0]):
            xaug[i] = cv2.resize(np.float32(x[i]), dim, interpolation=cv2.INTER_LINEAR)
            # xaug[i] = sk_transform.rescale(x[i], order=1, scale=scale, preserve_range=True, mode='reflect')

        if label_dtype == 'float':
            yaug = cv2.resize(np.float32(y), dim, interpolation=cv2.INTER_LINEAR)
        else:
            yaug = cv2.resize(np.uint8(y), dim, interpolation=cv2.INTER_NEAREST)
        bdist_aug = cv2.resize(np.float32(bdist), dim, interpolation=cv2.INTER_LINEAR)

    elif 'gaussian' in aug:
        var = float(aug.replace('gaussian', ''))

        for i in range(0, x.shape[0]):
            xaug[i] = sk_util.random_noise(x[i], mode='gaussian', clip=True, mean=0, var=var)

        yaug = y.copy()
        bdist_aug = bdist.copy()

    elif 's&p' in aug:
        for i in range(0, x.shape[0]):
            xaug[i] = sk_util.random_noise(x[i], mode='s&p', clip=True)

        yaug = y.copy()
        bdist_aug = bdist.copy()

    # Create the network
    nwk = SingleSensorNetwork(np.ascontiguousarray(xaug, dtype='float64'), k=k)

    edge_indices_a, edge_indices_b, edge_attrs_diffs, edge_attrs_dists, __, __ = nwk.create_network()
    edge_indices = np.c_[edge_indices_a, edge_indices_b]
    edge_attrs = np.c_[edge_attrs_diffs, edge_attrs_dists]

    # Create the node position tensor
    dims, height, width = xaug.shape
    pos_x = np.arange(0, width * kwargs['res'], kwargs['res'])
    pos_y = np.arange(height * kwargs['res'], 0, -kwargs['res'])
    grid_x, grid_y = np.meshgrid(pos_x, pos_y, indexing='xy')
    xy = np.c_[grid_x.flatten(), grid_y.flatten()]

    xaug = nd_to_columns(xaug, dims, height, width)

    return create_data_object(
        xaug,
        edge_indices,
        edge_attrs,
        xy,
        nbands=nbands,
        height=height,
        width=width,
        y=yaug,
        bdist=bdist_aug,
        **kwargs
    )
