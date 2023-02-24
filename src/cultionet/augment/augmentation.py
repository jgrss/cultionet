import typing as T

from ..data.utils import create_data_object, LabeledData
from ..networks import SingleSensorNetwork
from ..utils.reshape import nd_to_columns

import numpy as np
import cv2
from scipy.ndimage.measurements import label as nd_label
from skimage import util as sk_util
from tsaug import AddNoise, Drift, TimeWarp
import torch
from torch_geometric.data import Data


def feature_stack_to_tsaug(
    x: np.ndarray, ntime: int, nbands: int, nrows: int, ncols: int
) -> np.ndarray:
    """Reshapes from (T*C x H x W) -> (H*W x T X C)

    where,
        T = time
        C = channels / bands / variables
        H = height
        W = width

    Args:
        x: The array to reshape. The input shape is (T*C x H x W).
        ntime: The number of array time periods (T).
        nbands: The number of array bands/channels (C).
        nrows: The number of array rows (H).
        ncols: The number of array columns (W).
    """
    return (
        x
        .transpose(1, 2, 0)
        .reshape(nrows * ncols, ntime*nbands)
        .reshape(nrows * ncols, ntime, nbands)
    )


def tsaug_to_feature_stack(
    x: np.ndarray, nfeas: int, nrows: int, ncols: int
) -> np.ndarray:
    """Reshapes from (H*W x T X C) -> (T*C x H x W)

        where,
            T = time
            C = channels / bands / variables
            H = height
            W = width

        Args:
            x: The array to reshape. The input shape is (H*W x T X C).
            nfeas: The number of array features (time x channels).
            nrows: The number of array rows (height).
            ncols: The number of array columns (width).
        """
    return (
        x
        .reshape(nrows * ncols, nfeas)
        .T.reshape(nfeas, nrows, ncols)
    )


def get_prop_data(
    ldata: LabeledData,
    p: T.Any,
    x: np.ndarray
) -> T.Tuple[tuple, np.ndarray, np.ndarray]:
    # Get the segment bounding box
    min_row, min_col, max_row, max_col = p.bbox
    bounds_slice = (
        slice(min_row, max_row),
        slice(min_col, max_col)
    )
    # Get the segment features within the bounds
    xseg = x[(slice(0, None),) + bounds_slice].copy()
    # Get the segments within the bounds
    seg = ldata.segments[bounds_slice].copy()
    # Get the segment mask
    mask = np.uint8(seg == p.label)[np.newaxis]

    return bounds_slice, xseg, mask


def reinsert_prop(
    x: np.ndarray,
    bounds_slice: tuple,
    mask: np.ndarray,
    x_update: np.ndarray,
    x_original: np.ndarray
) -> np.ndarray:
    x[(slice(0, None),) + bounds_slice] = np.where(
        mask == 1, x_update, x_original
    )

    return x


def roll_time(
    ldata: LabeledData,
    p: T.Any,
    x: np.ndarray,
    ntime: int
) -> np.ndarray:
    bounds_slice, xseg, mask = get_prop_data(
        ldata=ldata, p=p, x=x
    )
    xseg_original = xseg.copy()

    # Get a temporal shift for the object
    shift = np.random.choice(
        range(
            -int(x.shape[0]*0.25), int(x.shape[0]*0.25)+1
        ), size=1
    )[0]
    # Shift time in each band separately
    for b in range(0, xseg.shape[0], ntime):
        # Get the slice for the current band, n time steps
        xseg[b:b+ntime] = np.roll(xseg[b:b+ntime], shift=shift, axis=0)

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original
    )

    return x


def augment_time(
    ldata: LabeledData,
    p: T.Any,
    x: np.ndarray,
    ntime: int,
    nbands: int,
    add_noise: bool,
    warper: T.Union[AddNoise, Drift, TimeWarp],
    aug: str
) -> np.ndarray:
    """Applies temporal augmentation to a dataset
    """
    bounds_slice, xseg, mask = get_prop_data(
        ldata=ldata, p=p, x=x
    )

    # xseg shape = (ntime*nbands x nrows x ncols)
    xseg_original = xseg.copy()
    nfeas, nrows, ncols = xseg.shape
    assert nfeas == int(ntime*nbands), \
        'The array feature dimensions do not match the expected shape.'

    # (H*W x T X C)
    xseg = feature_stack_to_tsaug(xseg, ntime, nbands, nrows, ncols)

    if aug == 'ts-peaks':
        new_indices = np.sort(
            np.random.choice(
                range(0, ntime*2-8), replace=False, size=ntime
            )
        )
        xseg = np.concatenate((xseg, xseg), axis=1)[:, 4:-4][:, new_indices]
    # Warp the segment
    xseg = warper.augment(xseg)
    if add_noise:
        noise_warper = AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
        xseg = noise_warper.augment(xseg)
    # Reshape back from (H*W x T x C) -> (T*C x H x W)
    xseg = tsaug_to_feature_stack(
        xseg, nfeas, nrows, ncols
    ).clip(0, 1)

    # Ensure reshaping back to original values
    # assert np.allclose(xseg_original, tsaug_to_feature_stack(xseg, nfeas, nrows, ncols))

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original
    )

    return x


def create_parcel_masks(labels_array: np.ndarray, max_crop_class: int) -> T.Union[None, dict]:
    """
    Creates masks for each instance

    Reference:
        https://torchtutorialstaging.z5.web.core.windows.net/intermediate/torchvision_tutorial.html
    """
    # Remove edges
    mask = np.where((labels_array > 0) & (labels_array <= max_crop_class), 1, 0)
    mask = nd_label(mask)[0]
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    small_box_idx = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        # Fields too small
        if (xmax - xmin == 0) or (ymax - ymin == 0):
            small_box_idx.append(i)
            continue
        boxes.append([xmin, ymin, xmax, ymax])

    if small_box_idx:
        good_idx = np.array(
            [idx for idx in range(0, masks.shape[0]) if idx not in small_box_idx]
        )
        masks = masks[good_idx]
    # convert everything into arrays
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.size(0) == 0:
        return None
    # there is only one class
    labels = torch.ones((masks.shape[0],), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    assert boxes.size(0) == labels.size(0) == masks.size(0), \
        'The tensor sizes do not match.'

    target = {
        'boxes': boxes,
        'labels': labels,
        'masks': masks
    }

    return target


def augment(
    ldata: LabeledData,
    aug: str,
    ntime: int,
    nbands: int,
    max_crop_class: int,
    k: int = 3,
    instance_seg: bool = False,
    zero_padding: int = 0,
    **kwargs
) -> Data:
    """Applies augmentation to a dataset
    """
    x = ldata.x
    y = ldata.y
    bdist = ldata.bdist
    # ori = ldata.ori
    if zero_padding > 0:
        zpad = torch.nn.ZeroPad2d(zero_padding)
        x = zpad(torch.tensor(x)).numpy()
        y = zpad(torch.tensor(y)).numpy()
        bdist = zpad(torch.tensor(bdist)).numpy()
        ori = zpad(torch.tensor(ori)).numpy()

    x = x.copy()
    if y is not None:
        y = y.copy()
        label_dtype = 'float' if 'float' in y.dtype.name else 'int'
    if bdist is not None:
        bdist = bdist.copy()
    # ori = None
    # if ori is not None:
    #     ori = ori.copy()

    if aug in ('ts-warp', 'ts-noise', 'ts-drift', 'ts-peaks'):
        add_noise = False if aug == 'ts-noise' else True
        if aug in ('ts-warp', 'ts-peaks'):
            warper = TimeWarp(
                n_speed_change=np.random.randint(
                    low=1, high=3
                ),
                max_speed_ratio=np.random.uniform(
                    low=1.1, high=1.5
                ),
                static_rand=True
            )
        elif aug == 'ts-noise':
            warper = AddNoise(
                scale=np.random.uniform(
                    low=0.01, high=0.05
                )
            )
        elif aug == 'ts-drift':
            warper = Drift(
                max_drift=np.random.uniform(
                    low=0.05, high=0.1
                ),
                n_drift_points=np.random.randint(
                    low=1, high=int(x.shape[0] / 2.0)
                ),
                static_rand=True
            )

        # Warp each segment
        for p in ldata.props:
            x = augment_time(
                ldata,
                p=p,
                x=x,
                ntime=ntime,
                nbands=nbands,
                add_noise=add_noise,
                warper=warper,
                aug=aug
            )

    elif 'rot' in aug:
        deg = int(aug.replace('rot', ''))
        deg_dict = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }

        # Rotate each feature layer
        x = np.zeros(
            (x.shape[0], *cv2.rotate(np.float32(x[0]), deg_dict[deg]).shape),
            dtype=x.dtype
        )
        for i in range(0, x.shape[0]):
            x[i] = cv2.rotate(np.float32(x[i]), deg_dict[deg])

        # Rotate labels
        if label_dtype == 'float':
            y = cv2.rotate(np.float32(y), deg_dict[deg])
        else:
            y = cv2.rotate(np.uint8(y), deg_dict[deg])
        # Rotate the distance transform
        bdist = cv2.rotate(np.float32(bdist), deg_dict[deg])
        # ori_aug = cv2.rotate(np.float32(ori), deg_dict[deg])

    elif 'roll' in aug:
        for p in ldata.props:
            x = roll_time(ldata, p, x, ntime)

    elif 'flip' in aug:
        if aug == 'flipfb':
            # Reverse the channels
            for b in range(0, x.shape[0], ntime):
                # Get the slice for the current band, n time steps
                x[b:b+ntime] = x[b:b+ntime][::-1]
        else:
            for i in range(0, x.shape[0]):
                x[i] = getattr(np, aug)(x[i])

            y = getattr(np, aug)(y)
            bdist = getattr(np, aug)(bdist)
            # ori_aug = getattr(np, aug)(ori)

    elif 'scale' in aug:
        scale = float(aug.replace('scale', ''))

        height = int(x.shape[1] * scale)
        width = int(x.shape[2] * scale)

        x = np.zeros((x.shape[0], height, width), dtype=x.dtype)
        dim = (width, height)

        for i in range(0, x.shape[0]):
            x[i] = cv2.resize(np.float32(x[i]), dim, interpolation=cv2.INTER_LINEAR)

        if label_dtype == 'float':
            y = cv2.resize(np.float32(y), dim, interpolation=cv2.INTER_LINEAR)
        else:
            y = cv2.resize(np.uint8(y), dim, interpolation=cv2.INTER_NEAREST)
        bdist = cv2.resize(np.float32(bdist), dim, interpolation=cv2.INTER_LINEAR)
        # ori_aug = cv2.resize(np.float32(ori), dim, interpolation=cv2.INTER_LINEAR)

    elif 'gaussian' in aug:
        var = float(aug.replace('gaussian', ''))
        for i in range(0, x.shape[0]):
            x[i] = sk_util.random_noise(
                x[i], mode='gaussian', clip=True, mean=0, var=var
            )

    elif 'speckle' in aug:
        var = float(aug.replace('speckle', ''))
        for i in range(0, x.shape[0]):
            x[i] = sk_util.random_noise(
                x[i], mode='speckle', clip=True, mean=0, var=var
            )

    elif 's&p' in aug:
        amount = float(aug.replace('s&p', ''))
        for i in range(0, x.shape[0]):
            x[i] = sk_util.random_noise(
                x[i], mode='s&p', clip=True, amount=amount
            )

    else:
        if aug != 'none':
            raise NameError(f'The augmentation {aug} is not supported.')

    # Create the network
    nwk = SingleSensorNetwork(np.ascontiguousarray(x, dtype='float64'), k=k)

    edge_indices_a, edge_indices_b, edge_attrs_diffs, edge_attrs_dists, __, __ = nwk.create_network()
    edge_indices = np.c_[edge_indices_a, edge_indices_b]
    edge_attrs = np.c_[edge_attrs_diffs, edge_attrs_dists]

    # Create the node position tensor
    dims, height, width = x.shape
    # pos_x = np.arange(0, width * kwargs['res'], kwargs['res'])
    # pos_y = np.arange(height * kwargs['res'], 0, -kwargs['res'])
    # grid_x, grid_y = np.meshgrid(pos_x, pos_y, indexing='xy')
    # xy = np.c_[grid_x.flatten(), grid_y.flatten()]

    x = nd_to_columns(x, dims, height, width)

    mask_y = None
    if instance_seg:
        mask_y = create_parcel_masks(y, max_crop_class)

    return create_data_object(
        x,
        edge_indices,
        edge_attrs,
        ntime=ntime,
        nbands=nbands,
        height=height,
        width=width,
        y=y,
        mask_y=mask_y,
        bdist=bdist,
        # ori=ori_aug,
        zero_padding=zero_padding,
        **kwargs
    )
