import typing as T
from abc import abstractmethod
from dataclasses import replace
from pathlib import Path

import einops
import joblib
import numpy as np
import torch
from frozendict import frozendict
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as VF
from tsaug import AddNoise, Drift, TimeWarp

from ..data.data import Data
from .augmenter_utils import augment_time, generate_perlin_noise_3d, roll_time


class AugmenterModule:
    """Prepares, augments, and finalizes data."""

    prefix: str = "data_"
    suffix: str = ".pt"

    def __call__(self, ldata: Data) -> Data:
        assert hasattr(self, "name_")
        assert isinstance(self.name_, str)

        cdata = self.forward(ldata.copy())
        cdata.x = cdata.x.float().clip(1e-9, 1)
        cdata.bdist = cdata.bdist.float().clip(0, 1)
        if cdata.y is not None:
            cdata.y = cdata.y.long()

        return cdata

    @abstractmethod
    def forward(self, cdata: Data) -> Data:
        raise NotImplementedError

    def file_name(self, uid: str) -> str:
        return f"{self.prefix}{uid}{self.suffix}"

    def save(
        self, out_directory: Path, data: Data, compress: T.Union[int, str] = 5
    ) -> None:
        out_path = out_directory / self.file_name(data.train_id)
        joblib.dump(data, out_path, compress=compress)


class AugmentTimeMixin(AugmenterModule):
    def forward(self, cdata: Data) -> Data:
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
        rng: T.Optional[np.random.Generator] = None,
    ):
        self.name_ = name
        self.n_speed_change_lim = n_speed_change_lim
        self.max_speed_ratio_lim = max_speed_ratio_lim
        self.rng = rng
        self.add_noise_ = True

        if self.n_speed_change_lim is None:
            self.n_speed_change_lim = (1, 3)
        if self.max_speed_ratio_lim is None:
            self.max_speed_ratio_lim = (1.1, 1.5)

        self.warper = TimeWarp(
            n_speed_change=int(
                self.rng.integers(
                    low=self.n_speed_change_lim[0],
                    high=self.n_speed_change_lim[1],
                )
            ),
            max_speed_ratio=self.rng.uniform(
                low=self.max_speed_ratio_lim[0],
                high=self.max_speed_ratio_lim[1],
            ),
            static_rand=True,
        )


class AugmentAddTimeNoise(AugmentTimeMixin):
    def __init__(
        self,
        scale_lim: T.Tuple[int, int] = None,
        rng: T.Optional[np.random.Generator] = None,
    ):
        self.scale_lim = scale_lim
        self.rng = rng
        self.name_ = "tsnoise"
        self.add_noise_ = False

        if self.scale_lim is None:
            self.scale_lim = (0.01, 0.05)

        self.warper = AddNoise(
            scale=self.rng.uniform(
                low=self.scale_lim[0], high=self.scale_lim[1]
            )
        )


class AugmentTimeDrift(AugmentTimeMixin):
    def __init__(
        self,
        max_drift_lim: T.Tuple[int, int] = None,
        n_drift_points_lim: T.Tuple[int, int] = None,
        rng: T.Optional[np.random.Generator] = None,
    ):
        self.max_drift_lim = max_drift_lim
        self.n_drift_points_lim = n_drift_points_lim
        self.rng = rng
        self.name_ = "tsdrift"
        self.add_noise_ = True

        if self.max_drift_lim is None:
            self.max_drift_lim = (0.05, 0.1)
        if self.n_drift_points_lim is None:
            self.n_drift_points_lim = (1, 6)

        self.warper = Drift(
            max_drift=self.rng.uniform(
                low=self.max_drift_lim[0],
                high=self.max_drift_lim[1],
            ),
            n_drift_points=int(
                self.rng.integers(
                    low=self.n_drift_points_lim[0],
                    high=self.n_drift_points_lim[1],
                )
            ),
            static_rand=True,
        )


class Roll(AugmenterModule):
    def __init__(self, rng: T.Optional[np.random.Generator] = None):
        self.rng = rng
        self.name_ = "roll"

    def forward(self, cdata: Data) -> Data:
        for p in cdata.props:
            cdata = roll_time(cdata, p, rng=self.rng)

        return cdata


class PerlinNoise(AugmenterModule):
    def __init__(self, rng: T.Optional[np.random.Generator] = None):
        self.rng = rng
        self.name_ = "perlin"

    def forward(self, cdata: Data) -> Data:
        res = self.rng.choice([2, 5, 10])
        noise = generate_perlin_noise_3d(
            shape=cdata.x.shape[2:],
            res=(1, res, res),
            tileable=(False, False, False),
            out_range=(-0.03, 0.03),
            rng=self.rng,
        )

        noise = einops.rearrange(noise, 't h w -> 1 1 t h w')
        cdata.x = cdata.x + noise.to(
            dtype=cdata.x.dtype, device=cdata.x.device
        )

        return cdata


class Rotate(AugmenterModule):
    def __init__(self, deg: int, **kwargs):
        self.deg = deg
        self.name_ = f"rotate-{deg}"

    def forward(self, cdata: Data) -> Data:
        x = einops.rearrange(cdata.x, '1 c t h w -> 1 t c h w')

        x_rotation_transform = v2.RandomRotation(
            degrees=[self.deg, self.deg],
            interpolation=InterpolationMode.BILINEAR,
        )
        y_rotation_transform = v2.RandomRotation(
            degrees=[self.deg, self.deg],
            interpolation=InterpolationMode.NEAREST,
        )

        cdata.x = einops.rearrange(
            x_rotation_transform(x),
            '1 t c h w -> 1 c t h w',
        )
        cdata.bdist = x_rotation_transform(cdata.bdist)
        cdata.y = y_rotation_transform(cdata.y)

        return cdata


class Flip(AugmenterModule):
    def __init__(self, direction: str, **kwargs):
        self.direction = direction
        self.name_ = direction

    def forward(self, cdata: Data) -> Data:
        x = einops.rearrange(cdata.x, '1 c t h w -> 1 t c h w')

        if self.direction == 'fliplr':
            flip_transform = VF.hflip
        elif self.direction == 'flipud':
            flip_transform = VF.vflip
        else:
            raise NameError("The direction is not supported.")

        cdata.x = einops.rearrange(
            flip_transform(x),
            '1 t c h w -> 1 c t h w',
        )
        cdata.bdist = flip_transform(cdata.bdist)
        cdata.y = flip_transform(cdata.y)

        return cdata


class RandomCropResize(AugmenterModule):
    def __init__(self, rng: T.Optional[np.random.Generator] = None):
        self.rng = rng
        self.name_ = "cropresize"

    def forward(self, cdata: Data) -> Data:
        div = self.rng.choice([2, 4])
        size = (cdata.y.shape[-2] // div, cdata.y.shape[-1] // div)

        random_seed = self.rng.integers(low=0, high=2147483647)

        x = einops.rearrange(cdata.x, 'b c t h w -> b t c h w')
        x = self.random_crop(
            x,
            interpolation=InterpolationMode.BILINEAR,
            size=size,
            random_seed=random_seed,
        )
        cdata.x = einops.rearrange(x, 'b t c h w -> b c t h w')
        cdata.bdist = self.random_crop(
            cdata.bdist,
            interpolation=InterpolationMode.BILINEAR,
            size=size,
            random_seed=random_seed,
        )
        cdata.y = self.random_crop(
            cdata.y,
            interpolation=InterpolationMode.NEAREST,
            size=size,
            random_seed=random_seed,
        )

        return cdata

    def random_crop(
        self,
        x: torch.Tensor,
        size: tuple,
        interpolation: str,
        random_seed: int,
    ) -> torch.Tensor:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        transform = v2.RandomCrop(
            size=size,
        )
        resize = v2.Resize(
            size=x.shape[-2:],
            interpolation=interpolation,
        )

        return resize(transform(x))


class GaussianBlur(AugmenterModule):
    def __init__(self, rng: T.Optional[np.random.Generator] = None, **kwargs):
        self.kwargs = kwargs
        self.name_ = "gaussian"

    def forward(self, cdata: Data) -> Data:
        transform = v2.GaussianBlur(kernel_size=3, **self.kwargs)
        cdata.x = transform(cdata.x)

        return cdata


class SaltAndPepperNoise(AugmenterModule):
    def __init__(self, rng: T.Optional[np.random.Generator] = None, **kwargs):
        self.rng = rng
        self.kwargs = kwargs
        self.name_ = "s&p"

    def forward(self, cdata: Data) -> Data:
        random_seed = self.rng.integers(low=0, high=2147483647)
        cdata.x = self.gaussian_noise(
            cdata.x,
            random_seed=random_seed,
            **self.kwargs,
        )

        return cdata

    def gaussian_noise(
        self, x: torch.Tensor, random_seed: int, sigma: float = 0.01
    ) -> torch.Tensor:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        return x + sigma * torch.randn_like(x)


class NoAugmentation(AugmenterModule):
    def __init__(self, **kwargs):
        self.name_ = "none"

    def forward(self, cdata: Data) -> Data:
        return cdata


AUGMENTER_METHODS = frozendict(
    tswarp=AugmentTimeWarp,
    tsnoise=AugmentAddTimeNoise,
    tsdrift=AugmentTimeDrift,
    tspeaks=AugmentTimeWarp,
    rot90=Rotate,
    rot180=Rotate,
    rot270=Rotate,
    roll=Roll,
    fliplr=Flip,
    flipud=Flip,
    gaussian=GaussianBlur,
    saltpepper=SaltAndPepperNoise,
    cropresize=RandomCropResize,
    perlin=PerlinNoise,
    none=NoAugmentation,
)

MODULE_DEFAULTS = dict(
    tswarp=dict(name="tswarp"),
    tsnoise={},
    tsdrift={},
    tspeaks=dict(name="tspeaks"),
    rot90=dict(deg=90),
    rot180=dict(deg=180),
    rot270=dict(deg=270),
    roll={},
    fliplr=dict(direction="fliplr"),
    flipud=dict(direction="flipud"),
    gaussian=dict(sigma=(0.2, 0.5)),
    saltpepper=dict(sigma=0.01),
    cropresize={},
    perlin={},
    none={},
)


class Augmenters:
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
                `Data` object.

    Example:
        >>> augmenters = Augmenters(augmentations=['tswarp'])
        >>> ldata = augmenters(ldata)
    """

    def __init__(
        self,
        augmentations: T.Sequence[str],
        rng: T.Optional[np.random.Generator] = None,
        random_seed: T.Optional[int] = None,
        **kwargs,
    ):
        self.augmentations = augmentations
        self.augmenters_ = []
        self.kwargs = kwargs

        if rng is None:
            rng = np.random.default_rng(random_seed)

        self._init_augmenters(rng)

    def _init_augmenters(self, rng: np.random.Generator):
        for aug_name in self.augmentations:
            self.augmenters_.append(
                AUGMENTER_METHODS[aug_name](
                    **{
                        "rng": rng,
                        **MODULE_DEFAULTS[aug_name],
                        **self.kwargs,
                    }
                )
            )

    def update_aug_args(self, **kwargs):
        self.aug_args = replace(self.aug_args, **kwargs)

    def __iter__(self):
        yield from self.augmenters_

    def __call__(self, batch: Data) -> Data:
        return self.forward(batch)

    def forward(self, batch: Data) -> Data:
        for augmenter in self:
            batch = augmenter(batch)

        return batch
