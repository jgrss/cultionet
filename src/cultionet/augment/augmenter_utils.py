import typing as T

import numpy as np
from scipy.ndimage.measurements import label as nd_label
from tsaug import AddNoise, Drift, TimeWarp
import torch

from ..data.utils import LabeledData


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
        x.transpose(1, 2, 0)
        .reshape(nrows * ncols, ntime * nbands)
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
    return x.reshape(nrows * ncols, nfeas).T.reshape(nfeas, nrows, ncols)


def get_prop_data(
    ldata: LabeledData, p: T.Any, x: np.ndarray
) -> T.Tuple[tuple, np.ndarray, np.ndarray]:
    # Get the segment bounding box
    min_row, min_col, max_row, max_col = p.bbox
    bounds_slice = (slice(min_row, max_row), slice(min_col, max_col))
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
    x_original: np.ndarray,
) -> np.ndarray:
    x[(slice(0, None),) + bounds_slice] = np.where(
        mask == 1, x_update, x_original
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
    aug: str,
) -> np.ndarray:
    """Applies temporal augmentation to a dataset."""
    bounds_slice, xseg, mask = get_prop_data(ldata=ldata, p=p, x=x)

    # xseg shape = (ntime*nbands x nrows x ncols)
    xseg_original = xseg.copy()
    nfeas, nrows, ncols = xseg.shape
    assert nfeas == int(
        ntime * nbands
    ), 'The array feature dimensions do not match the expected shape.'

    # (H*W x T X C)
    xseg = feature_stack_to_tsaug(xseg, ntime, nbands, nrows, ncols)

    if aug == 'tspeaks':
        new_indices = np.sort(
            np.random.choice(
                range(0, ntime * 2 - 8), replace=False, size=ntime
            )
        )
        xseg = np.concatenate((xseg, xseg), axis=1)[:, 4:-4][:, new_indices]
    # Warp the segment
    xseg = warper.augment(xseg)
    if add_noise:
        noise_warper = AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
        xseg = noise_warper.augment(xseg)
    # Reshape back from (H*W x T x C) -> (T*C x H x W)
    xseg = tsaug_to_feature_stack(xseg, nfeas, nrows, ncols).clip(0, 1)

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original,
    )

    return x


def roll_time(
    ldata: LabeledData, p: T.Any, x: np.ndarray, ntime: int
) -> np.ndarray:
    bounds_slice, xseg, mask = get_prop_data(ldata=ldata, p=p, x=x)
    xseg_original = xseg.copy()

    # Get a temporal shift for the object
    shift = np.random.choice(
        range(-int(x.shape[0] * 0.25), int(x.shape[0] * 0.25) + 1), size=1
    )[0]
    # Shift time in each band separately
    for b in range(0, xseg.shape[0], ntime):
        # Get the slice for the current band, n time steps
        xseg[b : b + ntime] = np.roll(xseg[b : b + ntime], shift=shift, axis=0)

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original,
    )

    return x


def create_parcel_masks(
    labels_array: np.ndarray, max_crop_class: int
) -> T.Union[None, dict]:
    """Creates masks for each instance.

    Reference:
        https://torchtutorialstaging.z5.web.core.windows.net/intermediate/torchvision_tutorial.html
    """
    # Remove edges
    mask = np.where(
        (labels_array > 0) & (labels_array <= max_crop_class), 1, 0
    )
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
            [
                idx
                for idx in range(0, masks.shape[0])
                if idx not in small_box_idx
            ]
        )
        masks = masks[good_idx]
    # convert everything into arrays
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.size(0) == 0:
        return None
    # there is only one class
    labels = torch.ones((masks.shape[0],), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    assert (
        boxes.size(0) == labels.size(0) == masks.size(0)
    ), 'The tensor sizes do not match.'

    target = {'boxes': boxes, 'labels': labels, 'masks': masks}

    return target
