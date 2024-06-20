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


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def scale_min_max(
    x: torch.Tensor,
    in_range: tuple,
    out_range: tuple,
) -> torch.Tensor:
    min_in, max_in = in_range
    min_out, max_out = out_range

    return (((max_out - min_out) * (x - min_in)) / (max_in - min_in)) + min_out


def generate_perlin_noise_3d(
    shape: T.Tuple[int, int, int],
    res: T.Tuple[int, int, int],
    tileable: T.Tuple[bool, bool, bool] = (
        False,
        False,
        False,
    ),
    out_range: T.Optional[T.Tuple[float, float]] = None,
    interpolant: T.Callable = interpolant,
    rng: T.Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generates a 3D tensor of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple
            of res.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A tensor with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.

    Source:
        https://github.com/pvigier/perlin-numpy/tree/master

        MIT License

        Copyright (c) 2019 Pierre Vigier
    """
    if out_range is None:
        out_range = (-0.1, 0.1)

    if rng is None:
        rng = np.random.default_rng()

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[
        : res[0] : delta[0], : res[1] : delta[1], : res[2] : delta[2]
    ]
    grid = np.mgrid[
        : res[0] : delta[0], : res[1] : delta[1], : res[2] : delta[2]
    ]
    grid = grid.transpose(1, 2, 3, 0) % 1

    grid = torch.from_numpy(grid)

    # Gradients
    torch.manual_seed(rng.integers(low=0, high=2147483647))
    theta = 2 * np.pi * torch.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    torch.manual_seed(rng.integers(low=0, high=2147483647))
    phi = 2 * np.pi * torch.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = torch.stack(
        (
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
        ),
        axis=3,
    )

    if tileable[0]:
        gradients[-1] = gradients[0]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    if tileable[2]:
        gradients[..., -1] = gradients[..., 0]

    gradients = (
        gradients.repeat_interleave(d[0], 0)
        .repeat_interleave(d[1], 1)
        .repeat_interleave(d[2], 2)
    )
    g000 = gradients[: -d[0], : -d[1], : -d[2]]
    g100 = gradients[d[0] :, : -d[1], : -d[2]]
    g010 = gradients[: -d[0], d[1] :, : -d[2]]
    g110 = gradients[d[0] :, d[1] :, : -d[2]]
    g001 = gradients[: -d[0], : -d[1], d[2] :]
    g101 = gradients[d[0] :, : -d[1], d[2] :]
    g011 = gradients[: -d[0], d[1] :, d[2] :]
    g111 = gradients[d[0] :, d[1] :, d[2] :]

    # Ramps
    n000 = torch.sum(
        torch.stack((grid[..., 0], grid[..., 1], grid[..., 2]), dim=3) * g000,
        dim=3,
    )
    n100 = torch.sum(
        torch.stack((grid[..., 0] - 1, grid[..., 1], grid[..., 2]), dim=3)
        * g100,
        dim=3,
    )
    n010 = torch.sum(
        torch.stack((grid[..., 0], grid[..., 1] - 1, grid[..., 2]), dim=3)
        * g010,
        dim=3,
    )
    n110 = torch.sum(
        torch.stack((grid[..., 0] - 1, grid[..., 1] - 1, grid[..., 2]), dim=3)
        * g110,
        dim=3,
    )
    n001 = torch.sum(
        torch.stack((grid[..., 0], grid[..., 1], grid[..., 2] - 1), dim=3)
        * g001,
        dim=3,
    )
    n101 = torch.sum(
        torch.stack((grid[..., 0] - 1, grid[..., 1], grid[..., 2] - 1), dim=3)
        * g101,
        dim=3,
    )
    n011 = torch.sum(
        torch.stack((grid[..., 0], grid[..., 1] - 1, grid[..., 2] - 1), dim=3)
        * g011,
        dim=3,
    )
    n111 = torch.sum(
        torch.stack(
            (grid[..., 0] - 1, grid[..., 1] - 1, grid[..., 2] - 1), dim=3
        )
        * g111,
        dim=3,
    )

    # Interpolation
    t = interpolant(grid)
    n00 = n000 * (1 - t[..., 0]) + t[..., 0] * n100
    n10 = n010 * (1 - t[..., 0]) + t[..., 0] * n110
    n01 = n001 * (1 - t[..., 0]) + t[..., 0] * n101
    n11 = n011 * (1 - t[..., 0]) + t[..., 0] * n111
    n0 = (1 - t[..., 1]) * n00 + t[..., 1] * n10
    n1 = (1 - t[..., 1]) * n01 + t[..., 1] * n11

    x = (1 - t[..., 2]) * n0 + t[..., 2] * n1

    return scale_min_max(x, in_range=(-0.5, 0.5), out_range=out_range)
