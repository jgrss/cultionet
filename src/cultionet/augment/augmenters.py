import enum
import typing as T
from abc import abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import einops
import joblib
import numpy as np
import torch
from skimage import util as sk_util
from tsaug import AddNoise, Drift, TimeWarp

from ..data.data import Data
from .augmenter_utils import augment_time, roll_time


@dataclass
class DataCopies:
    x: torch.Tensor
    y: T.Union[torch.Tensor, None]
    bdist: T.Union[torch.Tensor, None]


@dataclass
class AugmenterArgs:
    kwargs: dict


class AugmenterModule(object):
    """Prepares, augments, and finalizes data."""

    prefix: str = "data_"
    suffix: str = ".pt"

    def __call__(self, ldata: Data, aug_args: AugmenterArgs) -> Data:
        assert hasattr(self, "name_")
        assert isinstance(self.name_, str)

        cdata = self.forward(ldata.copy(), aug_args)
        cdata.x = cdata.x.float()
        if cdata.y is not None:
            cdata.y = cdata.y.long()

        return cdata

    @abstractmethod
    def forward(self, cdata: Data, aug_args: AugmenterArgs) -> Data:
        raise NotImplementedError

    def file_name(self, uid: str) -> str:
        return f"{self.prefix}{uid}{self.suffix}"

    def save(
        self, out_directory: Path, data: Data, compress: T.Union[int, str] = 5
    ) -> None:
        out_path = out_directory / self.file_name(data.train_id)
        joblib.dump(data, out_path, compress=compress)


class AugmentTimeMixin(AugmenterModule):
    def forward(self, cdata: Data, aug_args: AugmenterArgs) -> Data:
        # Warp each segment
        for p in cdata.props:
            cdata = augment_time(
                cdata,
                p=p,
                add_noise=self.add_noise_,
                warper=self.warper,
                aug=self.name_,
            )

        return cdata


class AugmentTimeWarp(AugmentTimeMixin):
    def __init__(
        self,
        name: str,
        n_speed_change_lim: T.Tuple[int, int] = None,
        max_speed_ratio_lim: T.Tuple[float, float] = None,
    ):
        self.n_speed_change_lim = n_speed_change_lim
        self.max_speed_ratio_lim = max_speed_ratio_lim
        self.name_ = name
        self.add_noise_ = True

        if self.n_speed_change_lim is None:
            self.n_speed_change_lim = (1, 3)
        if self.max_speed_ratio_lim is None:
            self.max_speed_ratio_lim = (1.1, 1.5)

        self.warper = TimeWarp(
            n_speed_change=np.random.randint(
                low=self.n_speed_change_lim[0], high=self.n_speed_change_lim[1]
            ),
            max_speed_ratio=np.random.uniform(
                low=self.max_speed_ratio_lim[0],
                high=self.max_speed_ratio_lim[1],
            ),
            static_rand=True,
        )


class AugmentAddTimeNoise(AugmentTimeMixin):
    def __init__(self, scale_lim: T.Tuple[int, int] = None):
        self.scale_lim = scale_lim
        self.name_ = "tsnoise"
        self.add_noise_ = False

        if self.scale_lim is None:
            self.scale_lim = (0.01, 0.05)

        self.warper = AddNoise(
            scale=np.random.uniform(
                low=self.scale_lim[0], high=self.scale_lim[1]
            )
        )


class AugmentTimeDrift(AugmentTimeMixin):
    def __init__(
        self,
        max_drift_lim: T.Tuple[int, int] = None,
        n_drift_points_lim: T.Tuple[int, int] = None,
    ):
        self.max_drift_lim = max_drift_lim
        self.n_drift_points_lim = n_drift_points_lim
        self.name_ = "tsdrift"
        self.add_noise_ = True

        if self.max_drift_lim is None:
            self.max_drift_lim = (0.05, 0.1)
        if self.n_drift_points_lim is None:
            self.n_drift_points_lim = (1, 6)

        self.warper = Drift(
            max_drift=np.random.uniform(
                low=self.max_drift_lim[0], high=self.max_drift_lim[1]
            ),
            n_drift_points=np.random.randint(
                low=self.n_drift_points_lim[0], high=self.n_drift_points_lim[1]
            ),
            static_rand=True,
        )


class Rotate(AugmenterModule):
    def __init__(self, deg: int):
        self.name_ = f"rotate-{deg}"

        deg_dict = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        self.deg_func = deg_dict[deg]

    def forward(
        self,
        cdata: Data,
        aug_args: AugmenterArgs = None,
    ) -> Data:

        stacked_x = einops.rearrange(cdata.x, '1 c t h w -> (c t) h w').numpy()
        # Create the output array for rotated features
        x = np.zeros(
            (
                cdata.num_channels * cdata.num_time,
                *cv2.rotate(np.float32(stacked_x[0]), self.deg_func).shape,
            ),
            dtype='float32',
        )

        for i in range(0, stacked_x.shape[0]):
            x[i] = cv2.rotate(np.float32(stacked_x[i]), self.deg_func)

        cdata.x = einops.rearrange(
            torch.from_numpy(x),
            '(c t) h w -> 1 c t h w',
            c=cdata.num_channels,
            t=cdata.num_time,
        )

        # Rotate labels
        label_dtype = (
            "float" if "float" in cdata.y.numpy().dtype.name else "int"
        )
        if label_dtype == "float":
            y = cv2.rotate(
                np.float32(cdata.y.squeeze(dim=0).numpy()), self.deg_func
            )
        else:
            y = cv2.rotate(
                np.uint8(cdata.y.squeeze(dim=0).numpy()), self.deg_func
            )

        cdata.y = einops.rearrange(torch.from_numpy(y), 'h w -> 1 h w')

        # Rotate the distance transform
        bdist = cv2.rotate(
            np.float32(cdata.bdist.squeeze(dim=0).numpy()), self.deg_func
        )
        cdata.bdist = einops.rearrange(torch.from_numpy(y), 'h w -> 1 h w')

        return cdata


class Roll(AugmenterModule):
    def __init__(self):
        self.name_ = "roll"

    def forward(
        self,
        cdata: Data,
        aug_args: AugmenterArgs = None,
    ) -> Data:
        for p in cdata.props:
            cdata = roll_time(cdata, p)

        return cdata


class Flip(AugmenterModule):
    def __init__(self, direction: str):
        self.direction = direction
        self.name_ = direction

    def forward(
        self,
        cdata: Data,
        aug_args: AugmenterArgs = None,
    ) -> Data:
        x = einops.rearrange(cdata.x, '1 c t h w -> (c t) h w').numpy()

        flip_func = getattr(np, self.direction)
        for band_idx in range(0, x.shape[0]):
            x[band_idx] = flip_func(x[band_idx])

        cdata.x = einops.rearrange(
            torch.from_numpy(x),
            '(c t) h w',
            c=cdata.num_channels,
            t=cdata.num_time,
        )

        return cdata


class SKLearnMixin(AugmenterModule):
    def forward(
        self,
        cdata: Data,
        aug_args: AugmenterArgs = None,
    ) -> DataCopies:
        x = einops.rearrange(cdata.x, '1 c t h w -> (c t) h w').numpy()
        for i in range(0, x.shape[0]):
            x[i] = sk_util.random_noise(
                x[i], mode=self.name_, clip=True, **self.kwargs
            )

        cdata.x = einops.rearrange(
            torch.from_numpy(x),
            '(c t) h w -> 1 c t h w',
            c=cdata.num_channels,
            t=cdata.num_time,
        )

        return cdata


class GaussianNoise(SKLearnMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = "gaussian"


class SaltAndPepperNoise(SKLearnMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = "s&p"


class SpeckleNoise(SKLearnMixin):
    """
    Example:
        >>> augmenter = SpeckleNoise()
        >>> data = augmenter(labeled_data, **kwargs)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = "speckle"


class NoAugmentation(AugmenterModule):
    def __init__(self):
        self.name_ = "none"

    def forward(
        self,
        cdata: DataCopies,
        ldata: Data = None,
        aug_args: AugmenterArgs = None,
    ) -> DataCopies:
        return cdata


class AugmenterMapping(enum.Enum):
    """Key: Augmenter mappings"""

    tswarp = AugmentTimeWarp(name="tswarp")
    tsnoise = AugmentAddTimeNoise()
    tsdrift = AugmentTimeDrift()
    tspeaks = AugmentTimeWarp("tspeaks")
    rot90 = Rotate(deg=90)
    rot180 = Rotate(deg=180)
    rot270 = Rotate(deg=270)
    roll = Roll()
    fliplr = Flip(direction="fliplr")
    flipud = Flip(direction="flipud")
    gaussian = GaussianNoise(mean=0.0, var=0.005)
    saltpepper = SaltAndPepperNoise(amount=0.01)
    speckle = SpeckleNoise(mean=0.0, var=0.05)
    none = NoAugmentation()


class AugmenterBase(object):
    def __init__(
        self,
        augmentations: T.Sequence[str],
        **kwargs,
    ):
        self.augmentations = augmentations
        self.augmenters_ = []
        self.aug_args = AugmenterArgs(kwargs=kwargs)

        self._init_augmenters()

    def _init_augmenters(self):
        for augmentation in self.augmentations:
            self.augmenters_.append(AugmenterMapping[augmentation].value)

    def update_aug_args(self, **kwargs):
        self.aug_args = replace(self.aug_args, **kwargs)


class Augmenters(AugmenterBase):
    """Applies augmentations for a sequence of augmentation methods.

    Inputs to callables:
        augmentation_method(ldata, aug_args=)

        where,
            ldata: `cultionet.data.utils.LabeledData` object, which consists of
                `x`, `y`, `bdist`, `segments` and `props` `numpy.ndarray` attributes.

                Shapes:
                    x: 3d (time, height, width) features, in [0,1] data range.
                    y: 2d (height, width) class labels.
                    bdist: 2d (height, width) boundary distance transforms.
                    segments: 2d (height, width) label instances (i.e., unique ids).
                    props: sequence of `skimage.measure.regionprops` property objects of
                        each labeled parcel in `y`.

            aug_args: Additional keyword arguments passed to the
                `torch_geometric.data.Data` object.

    Example:
        >>> aug = Augmenters(augmentations=['tswarp'])
        >>>
        >>> for method in aug:
        >>>     method(ldata, aug_args=aug.aug_args)
    """

    def __init__(self, **kwargs):
        super(Augmenters, self).__init__(**kwargs)

    def __iter__(self):
        yield from self.augmenters_
