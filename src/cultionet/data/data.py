import inspect
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import geowombat as gw
import joblib
import numpy as np
import torch
import xarray as xr
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.crs import CRSError
from pyproj.database import query_utm_crs_info
from rasterio.coords import BoundingBox
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds


@singledispatch
def sanitize_crs(crs: CRS) -> CRS:
    try:
        return crs
    except CRSError:
        return CRS.from_string("epsg:4326")


@sanitize_crs.register
def _(crs: str) -> CRS:
    return CRS.from_string(crs)


@sanitize_crs.register
def _(crs: int) -> CRS:
    return CRS.from_epsg(crs)


@singledispatch
def sanitize_res(res: tuple) -> Tuple[float, float]:
    return tuple(map(float, res))


@sanitize_res.register(int)
@sanitize_res.register(float)
def _(res) -> Tuple[float, float]:
    return sanitize_res((res, res))


class Data:
    def __init__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.x = x
        self.y = y
        if kwargs is not None:
            for k, v in kwargs.items():
                if v is not None:
                    assert isinstance(
                        v, (torch.Tensor, np.ndarray, list)
                    ), "Only tensors, arrays, and lists are supported."

                setattr(self, k, v)

    def _get_attrs(self) -> set:
        members = inspect.getmembers(
            self, predicate=lambda x: not inspect.ismethod(x)
        )
        return set(dict(members).keys()).intersection(
            set(self.__dict__.keys())
        )

    def to_dict(
        self, device: Optional[str] = None, dtype: Optional[str] = None
    ) -> dict:
        kwargs = {}
        for key in self._get_attrs():
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.clone()
                if device is not None:
                    kwargs[key] = kwargs[key].to(device=device, dtype=dtype)
            elif isinstance(value, np.ndarray):
                kwargs[key] = value.copy()
            else:
                if value is None:
                    kwargs[key] = None
                else:
                    try:
                        kwargs[key] = deepcopy(value)
                    except RecursionError:
                        kwargs[key] = value

        return kwargs

    def to(
        self, device: Optional[str] = None, dtype: Optional[str] = None
    ) -> "Data":
        return Data(**self.to_dict(device=device, dtype=dtype))

    def __add__(self, other: "Data") -> "Data":
        out_dict = {}
        for key, value in self.to_dict().items():
            if isinstance(value, torch.Tensor):
                out_dict[key] = value + getattr(other, key)

        return Data(**out_dict)

    def __iadd__(self, other: "Data") -> "Data":
        self = self + other

        return self

    def copy(self) -> "Data":
        return Data(**self.to_dict())

    @property
    def num_samples(self) -> int:
        return self.x.shape[0]

    @property
    def num_channels(self) -> int:
        return self.x.shape[1]

    @property
    def num_time(self) -> int:
        return self.x.shape[2]

    @property
    def height(self) -> int:
        return self.x.shape[3]

    @property
    def width(self) -> int:
        return self.x.shape[4]

    def to_file(
        self, filename: Union[Path, str], compress: Union[int, str] = 'zlib'
    ) -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            self.to_dict(),
            filename,
            compress=compress,
        )

    @classmethod
    def from_file(cls, filename: Union[Path, str]) -> "Data":
        return Data(**joblib.load(filename))

    def __str__(self):
        data_string = f"Data(x={tuple(self.x.shape)}"
        if self.y is not None:
            data_string += f", y={tuple(self.y.shape)}"

        for k, v in self.to_dict().items():
            if k not in (
                'x',
                'y',
            ):
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    if len(v.shape) == 1:
                        data_string += f", {k}={v.numpy().tolist()}"
                    else:
                        data_string += f", {k}={tuple(v.shape)}"
                elif isinstance(v, list):
                    if len(v) == 1:
                        data_string += f", {k}={v}"
                    else:
                        data_string += f", {k}={[len(v)]}"

        data_string += ")"

        return data_string

    def __repr__(self):
        return str(self)

    def plot(
        self,
        channel: Union[int, Sequence[int]],
        res: Union[float, Sequence[float]],
        crs: Optional[Union[int, str]] = None,
    ) -> tuple:

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharey=True, dpi=150)

        ds = self.to_dataset(res=res, crs=crs)

        bands = ds["bands"].assign_attrs(**ds.attrs).sel(channel=channel)
        bands = bands.where(lambda x: x > 0)
        cv = bands.std(dim='time') / bands.mean(dim='time')

        cv.plot.imshow(
            add_colorbar=False,
            robust=True,
            interpolation="nearest",
            ax=axes[0],
        )
        (
            ds["labels"].where(lambda x: x != -1).assign_attrs(**ds.attrs)
        ).plot.imshow(add_colorbar=False, interpolation="nearest", ax=axes[1])
        (ds["distances"].assign_attrs(**ds.attrs)).plot.imshow(
            add_colorbar=False, interpolation="nearest", ax=axes[2]
        )

        for ax in axes:
            ax.set_xlabel('')
            ax.set_ylabel('')

        axes[0].set_title("CV")
        axes[1].set_title("Labels")
        axes[2].set_title("Distances")

        fig.supxlabel("X")
        fig.supylabel("Y")

        return fig, axes

    def utm_bounds(self) -> CRS:
        utm_crs_info = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=self.left[0],
                south_lat_degree=self.bottom[0],
                east_lon_degree=self.right[0],
                north_lat_degree=self.top[0],
            ),
        )[0]

        return CRS.from_epsg(utm_crs_info.code)

    def transform_bounds(self, crs: CRS) -> BoundingBox:
        """Transforms a bounding box to a new CRS."""

        bounds = transform_bounds(
            src_crs=sanitize_crs("epsg:4326"),
            dst_crs=sanitize_crs(crs),
            left=self.left[0],
            bottom=self.bottom[0],
            right=self.right[0],
            top=self.top[0],
        )

        return BoundingBox(*bounds)

    def from_bounds(
        self,
        bounds: BoundingBox,
        res: Union[float, Sequence[float]],
    ) -> tuple:
        """Converts a bounding box to a transform adjusted by the
        resolution."""

        res = sanitize_res(res)

        adjusted_bounds = BoundingBox(
            left=bounds.left,
            bottom=bounds.top - self.height * float(abs(res[1])),
            right=bounds.left + self.width * float(abs(res[0])),
            top=bounds.top,
        )

        adjusted_transform = from_bounds(
            *adjusted_bounds,
            width=self.width,
            height=self.height,
        )

        return adjusted_bounds, adjusted_transform

    def to_dataset(
        self,
        res: Union[float, Sequence[float]],
        crs: Optional[Union[int, str]] = None,
    ) -> xr.Dataset:
        """Converts a PyTorch data batch to an Xarray Dataset."""

        if crs is None:
            crs = self.utm_bounds()

        crs = sanitize_crs(crs)
        dst_bounds = self.transform_bounds(crs)
        dst_bounds, transform = self.from_bounds(dst_bounds, res=res)

        return xr.Dataset(
            data_vars=dict(
                bands=(
                    ["channel", "time", "y", "x"],
                    self.x[0].numpy() * 1e-4,
                ),
                labels=(["y", "x"], self.y[0].numpy()),
                distances=(["y", "x"], self.bdist[0].numpy() * 1e-4),
            ),
            coords={
                "channel": range(1, self.num_channels + 1),
                "time": range(1, self.num_time + 1),
                "y": np.linspace(
                    dst_bounds.top, dst_bounds.bottom, self.height
                ),
                "x": np.linspace(
                    dst_bounds.left, dst_bounds.right, self.width
                ),
            },
            attrs={
                "name": self.batch_id[0],
                "crs": crs.to_epsg(),
                "res": (float(abs(transform[0])), float(abs(transform[4]))),
                "transform": transform,
                "_FillValue": -1,
            },
        )


@dataclass
class LabeledData:
    x: np.ndarray
    y: Union[None, np.ndarray]
    bdist: Union[None, np.ndarray]
    ori: Union[None, np.ndarray]
    segments: Union[None, np.ndarray]
    props: Union[None, List]
