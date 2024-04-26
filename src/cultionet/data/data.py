import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import torch


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


@dataclass
class LabeledData:
    x: np.ndarray
    y: Union[None, np.ndarray]
    bdist: Union[None, np.ndarray]
    ori: Union[None, np.ndarray]
    segments: Union[None, np.ndarray]
    props: Union[None, List]
