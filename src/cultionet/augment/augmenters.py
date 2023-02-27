from abc import abstractmethod
import typing as T
import enum
from dataclasses import dataclass, replace
from pathlib import Path

from .augmenter_utils import (
    augment_time,
    create_parcel_masks,
    roll_time
)
from ..data.utils import create_data_object, LabeledData
from ..networks import SingleSensorNetwork
from ..utils.reshape import nd_to_columns

from tsaug import AddNoise, Drift, TimeWarp
import numpy as np
import cv2
from skimage import util as sk_util
from torch_geometric.data import Data
import joblib


@dataclass
class DataCopies:
    x: np.ndarray
    y: T.Union[np.ndarray, None]
    bdist: T.Union[np.ndarray, None]


@dataclass
class AugmenterArgs:
    ntime: int
    nbands: int
    max_crop_class: int
    k: int
    instance_seg: bool
    zero_padding: int
    kwargs: dict


class AugmenterModule(object):
    """Prepares, augments, and finalizes data
    """
    def __call__(
        self,
        ldata: LabeledData,
        aug_args: AugmenterArgs
    ) -> Data:
        assert hasattr(self, 'name_')
        assert isinstance(self.name_, str)

        cdata = self.prepare_data(ldata)
        cdata = self.forward(cdata, ldata, aug_args)
        data = self.finalize(
            x=cdata.x,
            y=cdata.y,
            bdist=cdata.bdist,
            aug_args=aug_args
        )

        return data

    @abstractmethod
    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData,
        aug_args: AugmenterArgs
    ) -> DataCopies:
        raise NotImplementedError

    def save(
        self,
        out_directory: Path,
        data: Data,
        prefix: str = 'data_',
        suffix: str = '.pt',
        compress: int = 5
    ) -> None:
        out_path = out_directory / f'{prefix}{data.train_id}{suffix}'
        joblib.dump(
            data,
            out_path,
            compress=compress
        )

    def prepare_data(self, ldata: LabeledData) -> DataCopies:
        x = ldata.x.copy()
        y = ldata.y
        bdist = ldata.bdist
        # TODO: for orientation layer
        # ori = ldata.ori
        # if zero_padding > 0:
        #     zpad = torch.nn.ZeroPad2d(zero_padding)
        #     x = zpad(torch.tensor(x)).numpy()
        #     y = zpad(torch.tensor(y)).numpy()
        #     bdist = zpad(torch.tensor(bdist)).numpy()
        #     ori = zpad(torch.tensor(ori)).numpy()

        if y is not None:
            y = y.copy()
        if bdist is not None:
            bdist = bdist.copy()

        return DataCopies(
            x=x,
            y=y,
            bdist=bdist
        )

    def finalize(
        self,
        x: np.ndarray,
        y: T.Union[np.ndarray, None],
        bdist: T.Union[np.ndarray, None],
        aug_args: AugmenterArgs
    ) -> Data:
        # Create the network
        nwk = SingleSensorNetwork(
            np.ascontiguousarray(x, dtype='float64'), k=aug_args.k
        )

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
        if aug_args.instance_seg:
            mask_y = create_parcel_masks(y, aug_args.max_crop_class)

        return create_data_object(
            x,
            edge_indices,
            edge_attrs,
            ntime=aug_args.ntime,
            nbands=aug_args.nbands,
            height=height,
            width=width,
            y=y,
            mask_y=mask_y,
            bdist=bdist,
            # ori=ori_aug,
            zero_padding=aug_args.zero_padding,
            **aug_args.kwargs
        )


class AugmentTimeMixin(AugmenterModule):
    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData,
        aug_args: AugmenterArgs
    ) -> DataCopies:
        # Warp each segment
        for p in ldata.props:
            x = augment_time(
                ldata,
                p=p,
                x=cdata.x,
                ntime=aug_args.ntime,
                nbands=aug_args.nbands,
                add_noise=self.add_noise_,
                warper=self.warper,
                aug=self.name_
            )
            cdata = replace(cdata, x=x)

        # y and bdist are unaltered
        return cdata


class AugmentTimeWarp(AugmentTimeMixin):
    def __init__(
        self,
        name: str,
        n_speed_change_lim: T.Tuple[int, int] = None,
        max_speed_ratio_lim: T.Tuple[float, float] = None
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
                low=self.n_speed_change_lim[0],
                high=self.n_speed_change_lim[1]
            ),
            max_speed_ratio=np.random.uniform(
                low=self.max_speed_ratio_lim[0],
                high=self.max_speed_ratio_lim[1]
            ),
            static_rand=True
        )


class AugmentAddTimeNoise(AugmentTimeMixin):
    def __init__(self, scale_lim: T.Tuple[int, int] = None):
        self.scale_lim = scale_lim
        self.name_ = 'ts-noise'
        self.add_noise_ = False

        if self.scale_lim is None:
            self.scale_lim = (0.01, 0.05)

        self.warper = AddNoise(
            scale=np.random.uniform(
                low=self.scale_lim[0],
                high=self.scale_lim[1]
            )
        )


class AugmentTimeDrift(AugmentTimeMixin):
    def __init__(
        self,
        max_drift_lim: T.Tuple[int, int] = None,
        n_drift_points_lim: T.Tuple[int, int] = None
    ):
        self.max_drift_lim = max_drift_lim
        self.n_drift_points_lim = n_drift_points_lim
        self.name_ = 'ts-drift'
        self.add_noise_ = True

        if self.max_drift_lim is None:
            self.max_drift_lim = (0.05, 0.1)
        if self.n_drift_points_lim is None:
            self.n_drift_points_lim = (1, 6)

        self.warper = Drift(
            max_drift=np.random.uniform(
                low=self.max_drift_lim[0],
                high=self.max_drift_lim[1]
            ),
            n_drift_points=np.random.randint(
                low=self.n_drift_points_lim[0],
                high=self.n_drift_points_lim[1]
            ),
            static_rand=True
        )


class Rotate(AugmenterModule):
    def __init__(self, deg: int):
        self.name_ = f'rotate-{deg}'

        deg_dict = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        self.deg_func = deg_dict[deg]

    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData = None,
        aug_args: AugmenterArgs = None
    ) -> DataCopies:
        # Create the output array for rotated features
        x = np.zeros(
            (
                cdata.x.shape[0],
                *cv2.rotate(
                    np.float32(cdata.x[0]),
                    self.deg_func
                ).shape
            ),
            dtype=cdata.x.dtype
        )
        for i in range(0, cdata.x.shape[0]):
            x[i] = cv2.rotate(
                np.float32(cdata.x[i]),
                self.deg_func
            )

        # Rotate labels
        label_dtype = 'float' if 'float' in cdata.y.dtype.name else 'int'
        if label_dtype == 'float':
            y = cv2.rotate(np.float32(cdata.y), self.deg_func)
        else:
            y = cv2.rotate(np.uint8(cdata.y), self.deg_func)
        # Rotate the distance transform
        bdist = cv2.rotate(np.float32(cdata.bdist), self.deg_func)
        # ori_aug = cv2.rotate(np.float32(ori), self.deg_func)

        cdata = replace(cdata, x=x, y=y, bdist=bdist)

        return cdata


class Roll(AugmenterModule):
    def __init__(self):
        self.name_ = 'roll'

    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData = None,
        aug_args: AugmenterArgs = None
    ) -> DataCopies:
        for p in ldata.props:
            x = roll_time(
                ldata, p, cdata.x, aug_args.ntime
            )
            cdata = replace(cdata, x=x)

        # y and bdist are unaltered
        return cdata


class Flip(AugmenterModule):
    def __init__(self, direction: str):
        self.direction = direction
        self.name_ = direction

    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData = None,
        aug_args: AugmenterArgs = None
    ) -> DataCopies:
        x = cdata.x.copy()
        if self.direction == 'flipfb':
            # Reverse the channels
            for b in range(0, cdata.x.shape[0], aug_args.ntime):
                # Get the slice for the current band, n time steps
                x[b:b+aug_args.ntime] = x[b:b+aug_args.ntime][::-1]

            # y and bdist are unaltered
            cdata = replace(cdata)
        else:
            flip_func = getattr(np, self.direction)
            for i in range(0, x.shape[0]):
                x[i] = flip_func(x[i])

            y = flip_func(cdata.y)
            bdist = flip_func(cdata.bdist)
            # ori_aug = getattr(np, aug)(ori)
            cdata = replace(cdata, x=x, y=y, bdist=bdist)

        return cdata


class SKLearnMixin(AugmenterModule):
    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData = None,
        aug_args: AugmenterArgs = None
    ) -> DataCopies:
        x = cdata.x.copy()
        for i in range(0, x.shape[0]):
            x[i] = sk_util.random_noise(
                x[i],
                mode=self.name_,
                clip=True,
                **self.kwargs
            )

        # y and bdist are unaltered
        cdata = replace(cdata, x=x)

        return cdata


class GaussianNoise(SKLearnMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = 'gaussian'


class SaltAndPepperNoise(SKLearnMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = 's&p'


class SpeckleNoise(SKLearnMixin):
    """
    Example:
        >>> augmenter = SpeckleNoise()
        >>> data = augmenter(labeled_data, **kwargs)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name_ = 'speckle'


class NoAugmentation(AugmenterModule):
    def __init__(self):
        self.name_ = 'none'

    def forward(
        self,
        cdata: DataCopies,
        ldata: LabeledData = None,
        aug_args: AugmenterArgs = None
    ) -> DataCopies:
        return cdata


class AugmenterMapping(enum.Enum):
    """Key: Augmenter mappings
    """
    ts_warp = AugmentTimeWarp(name='ts-warp')
    ts_noise = AugmentAddTimeNoise()
    ts_drift = AugmentTimeDrift()
    ts_peaks = AugmentTimeWarp(name='ts-peaks')
    rot90 = Rotate(deg=90)
    rot180 = Rotate(deg=180)
    rot270 = Rotate(deg=270)
    roll = Roll()
    fliplr = Flip(direction='fliplr')
    flipud = Flip(direction='flipud')
    gaussian = GaussianNoise(mean=0.0, var=0.005)
    salt_pepper = SaltAndPepperNoise(amount=0.01)
    speckle = SpeckleNoise(mean=0.0, var=0.05)
    none = NoAugmentation()


class AugmenterBase(object):
    def __init__(
        self,
        augmentations: T.Sequence[str],
        ntime: int,
        nbands: int,
        max_crop_class: int,
        k: int = 3,
        instance_seg: bool = False,
        zero_padding: int = 0,
        **kwargs
    ):
        self.augmentations = augmentations
        self.augmenters_ = []
        self.aug_args = AugmenterArgs(
            ntime=ntime,
            nbands=nbands,
            max_crop_class=max_crop_class,
            k=k,
            instance_seg=instance_seg,
            zero_padding=zero_padding,
            kwargs=kwargs
        )

        self._init_augmenters()

    def _init_augmenters(self):
        for augmentation in self.augmentations:
            self.augmenters_.append(
                AugmenterMapping[augmentation.replace('-', '_')].value
            )

    def update_aug_args(self, **kwargs):
        self.aug_args = replace(self.aug_args, **kwargs)


class Augmenters(AugmenterBase):
    """Applies augmentations for a sequence of augmentation methods

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
        >>> aug = Augmenters(
        >>>     augmentations=['ts-warp'],
        >>>     ntime=13,
        >>>     nbands=5,
        >>>     max_crop_class=1
        >>> )
        >>>
        >>> for method in aug:
        >>>     method(ldata, aug_args=aug.aug_args)
    """
    def __init__(self, **kwargs):
        super(Augmenters, self).__init__(**kwargs)

    def __iter__(self):
        yield from self.augmenters_
