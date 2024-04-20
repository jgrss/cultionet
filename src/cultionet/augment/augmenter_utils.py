import typing as T

import numpy as np
import torch
from einops import rearrange
from scipy.ndimage.measurements import label as nd_label
from tsaug import AddNoise, Drift, TimeWarp

from ..data.data import Data


def feature_stack_to_tsaug(x: torch.Tensor) -> torch.Tensor:
    """Reshapes from (1 x C x T x H x W) -> (H*W x T X C)

    where,
        T = time
        C = channels / bands / variables
        H = height
        W = width

    Args:
        x: The array to reshape. The input shape is (1 x C x T x H x W).
    """
    return rearrange(x, '1 c t h w -> (h w) t c')


def tsaug_to_feature_stack(
    x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Reshapes from (H*W x T X C) -> (1 x C x T x H x W)

    where,
        T = time
        C = channels / bands / variables
        H = height
        W = width

    Args:
        x: The array to reshape. The input shape is (H*W x T X C).
        height: The number of array rows (height).
        width: The number of array columns (width).
    """
    return rearrange(
        x,
        '(h w) t c -> 1 c t h w',
        h=height,
        w=width,
    )


class SegmentParcel:
    def __init__(
        self,
        coords_slices: tuple,
        dims_slice: tuple,
        xseg: torch.Tensor,
    ):
        self.coords_slices = coords_slices
        self.dims_slice = dims_slice
        self.xseg = xseg

    @classmethod
    def from_prop(cls, ldata: Data, p: T.Any) -> "SegmentParcel":
        # Get the segment bounding box
        min_row, min_col, max_row, max_col = p.bbox
        coords_slices = (slice(0, None),) * 3
        dims_slice = (
            slice(min_row, max_row),
            slice(min_col, max_col),
        )

        # Get the segment features within the bounds
        xseg = ldata.x[coords_slices + dims_slice]

        return cls(
            coords_slices=coords_slices,
            dims_slice=dims_slice,
            xseg=xseg,
        )


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
    ldata: Data,
    p: T.Any,
    add_noise: bool,
    warper: T.Union[AddNoise, Drift, TimeWarp],
    aug: str,
) -> np.ndarray:
    """Applies temporal augmentation to a dataset."""
    segment_parcel = SegmentParcel.from_prop(ldata=ldata, p=p)

    (
        num_batch,
        num_channels,
        num_time,
        height,
        width,
    ) = segment_parcel.xseg.shape

    # -> (H*W x T X C)
    xseg = feature_stack_to_tsaug(segment_parcel.xseg)

    if aug == "tspeaks":
        new_indices = np.sort(
            np.random.choice(
                range(0, num_time * 2 - 8), replace=False, size=num_time
            )
        )
        xseg = torch.cat((xseg, xseg), dim=1)[:, 4:-4][:, new_indices]

    # Warp the segment
    xseg = warper.augment(xseg.numpy())

    if add_noise:
        noise_warper = AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
        xseg = noise_warper.augment(xseg)

    # Reshape back from (H*W x T x C) -> (1 x C x T x H x W)
    xseg = tsaug_to_feature_stack(
        torch.from_numpy(xseg), height=height, width=width
    ).clip(0, 1)

    # Insert the parcel
    ldata.x[
        segment_parcel.coords_slices + segment_parcel.dims_slice
    ] = torch.where(
        rearrange(
            torch.from_numpy(ldata.segments)[segment_parcel.dims_slice],
            'h w -> 1 1 1 h w',
        )
        == p.label,
        xseg,
        ldata.x[segment_parcel.coords_slices + segment_parcel.dims_slice],
    )

    return ldata


def roll_time(ldata: Data, p: T.Any) -> Data:
    segment_parcel = SegmentParcel.from_prop(ldata=ldata, p=p)

    # Get a temporal shift for the object
    shift = np.random.choice(
        range(-int(ldata.num_time * 0.25), int(ldata.num_time * 0.25) + 1)
    )
    # Shift time in each band separately
    for band_idx in range(0, ldata.num_channels):
        # Get the slice for the current band, n time steps
        segment_parcel.xseg[0, band_idx] = torch.roll(
            segment_parcel.xseg[0, band_idx], shift, dims=0
        )

    # Insert the parcel
    ldata.x[
        segment_parcel.coords_slices + segment_parcel.dims_slice
    ] = torch.where(
        rearrange(
            torch.from_numpy(ldata.segments)[segment_parcel.dims_slice],
            'h w -> 1 1 1 h w',
        )
        == p.label,
        segment_parcel.xseg,
        ldata.x[segment_parcel.coords_slices + segment_parcel.dims_slice],
    )

    return ldata


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
    ), "The tensor sizes do not match."

    target = {"boxes": boxes, "labels": labels, "masks": masks}

    return target
