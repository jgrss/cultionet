import typing as T

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
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


def insert_parcel(
    parcel_data: Data,
    augmented: torch.Tensor,
    segment_parcel: SegmentParcel,
    prop: object,
) -> Data:
    parcel_data.x[
        segment_parcel.coords_slices + segment_parcel.dims_slice
    ] = torch.where(
        rearrange(
            torch.from_numpy(parcel_data.segments)[segment_parcel.dims_slice],
            'h w -> 1 1 1 h w',
        )
        == prop.label,
        augmented,
        parcel_data.x[
            segment_parcel.coords_slices + segment_parcel.dims_slice
        ],
    )

    return parcel_data


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

    # (1 x C x T x H x W) -> (H*W x T X C)
    xseg = feature_stack_to_tsaug(segment_parcel.xseg)

    if aug == "tspeaks":
        half_a = F.interpolate(
            rearrange(xseg, 'b t c -> b c t'),
            size=num_time // 2,
            mode='linear',
        )
        half_b = F.interpolate(
            rearrange(xseg, 'b t c -> b c t'),
            size=num_time - half_a.shape[-1],
            mode='linear',
        )
        xseg = rearrange(
            torch.cat((half_a, half_b), dim=-1),
            'b c t -> b t c',
        )

    # Warp the segment
    xseg = warper.augment(xseg.numpy())

    if add_noise:
        noise_warper = AddNoise(scale=np.random.uniform(low=0.01, high=0.05))
        xseg = noise_warper.augment(xseg)

    # Reshape back from (H*W x T x C) -> (1 x C x T x H x W)
    xseg = tsaug_to_feature_stack(
        torch.from_numpy(xseg), height=height, width=width
    ).clip(0, 1)

    return insert_parcel(
        parcel_data=ldata,
        augmented=xseg,
        segment_parcel=segment_parcel,
        prop=p,
    )


def roll_time(ldata: Data, p: T.Any) -> Data:
    segment_parcel = SegmentParcel.from_prop(ldata=ldata, p=p)

    # Get a temporal shift for the object
    shift = np.random.choice(
        range(-int(ldata.num_time * 0.25), int(ldata.num_time * 0.25) + 1)
    )

    # Shift time
    # (1 x C x T x H x W)
    xseg = torch.roll(segment_parcel.xseg, shift, dims=2)

    return insert_parcel(
        parcel_data=ldata,
        augmented=xseg,
        segment_parcel=segment_parcel,
        prop=p,
    )
