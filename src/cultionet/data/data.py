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
                setattr(self, k, v)

    def _get_attrs(self) -> set:
        members = inspect.getmembers(
            self, predicate=lambda x: not inspect.ismethod(x)
        )
        return set(dict(members).keys()).intersection(
            set(self.__dict__.keys())
        )

    def to_dict(self) -> dict:
        kwargs = {}
        for key in self._get_attrs():
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.clone()
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
    def num_rows(self) -> int:
        return self.x.shape[3]

    @property
    def num_cols(self) -> int:
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
        return (
            "\nData(\n"
            f"   num_samples={self.num_samples}, num_channels={self.num_channels}, num_time={self.num_time}, num_rows={self.num_rows:,d}, num_cols={self.num_cols:,d}\n"
            ")"
        )

    def __repr__(self):
        return "Data(...)"


@dataclass
class LabeledData:
    x: np.ndarray
    y: Union[None, np.ndarray]
    bdist: Union[None, np.ndarray]
    ori: Union[None, np.ndarray]
    segments: Union[None, np.ndarray]
    props: Union[None, List]
