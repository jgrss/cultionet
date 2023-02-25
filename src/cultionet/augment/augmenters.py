import typing as T
import enum
from dataclasses import dataclass, replace
from pathlib import Path

from ..data.utils import create_data_object, LabeledData
from ..networks import SingleSensorNetwork
from ..utils.reshape import nd_to_columns

from tsaug import AddNoise, Drift, TimeWarp
import numpy as np
import cv2
from scipy.ndimage.measurements import label as nd_label
from skimage import util as sk_util
import torch
from torch_geometric.data import Data
import joblib


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
        x
        .transpose(1, 2, 0)
        .reshape(nrows * ncols, ntime*nbands)
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
    return (
        x
        .reshape(nrows * ncols, nfeas)
        .T.reshape(nfeas, nrows, ncols)
    )


def get_prop_data(
    ldata: LabeledData,
    p: T.Any,
    x: np.ndarray
) -> T.Tuple[tuple, np.ndarray, np.ndarray]:
    # Get the segment bounding box
    min_row, min_col, max_row, max_col = p.bbox
    bounds_slice = (
        slice(min_row, max_row),
        slice(min_col, max_col)
    )
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
    x_original: np.ndarray
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
    aug: str
) -> np.ndarray:
    """Applies temporal augmentation to a dataset
    """
    bounds_slice, xseg, mask = get_prop_data(
        ldata=ldata, p=p, x=x
    )

    # xseg shape = (ntime*nbands x nrows x ncols)
    xseg_original = xseg.copy()
    nfeas, nrows, ncols = xseg.shape
    assert nfeas == int(ntime*nbands), \
        'The array feature dimensions do not match the expected shape.'

    # (H*W x T X C)
    xseg = feature_stack_to_tsaug(xseg, ntime, nbands, nrows, ncols)

    if aug == 'ts-peaks':
        new_indices = np.sort(
            np.random.choice(
                range(0, ntime*2-8), replace=False, size=ntime
            )
        )
        xseg = np.concatenate((xseg, xseg), axis=1)[:, 4:-4][:, new_indices]
    # Warp the segment
    xseg = warper.augment(xseg)
    if add_noise:
        noise_warper = AddNoise(
            scale=np.random.uniform(low=0.01, high=0.05)
        )
        xseg = noise_warper.augment(xseg)
    # Reshape back from (H*W x T x C) -> (T*C x H x W)
    xseg = tsaug_to_feature_stack(
        xseg, nfeas, nrows, ncols
    ).clip(0, 1)

    # Ensure reshaping back to original values
    # assert np.allclose(xseg_original, tsaug_to_feature_stack(xseg, nfeas, nrows, ncols))

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original
    )

    return x


def roll_time(
    ldata: LabeledData,
    p: T.Any,
    x: np.ndarray,
    ntime: int
) -> np.ndarray:
    bounds_slice, xseg, mask = get_prop_data(
        ldata=ldata, p=p, x=x
    )
    xseg_original = xseg.copy()

    # Get a temporal shift for the object
    shift = np.random.choice(
        range(
            -int(x.shape[0]*0.25), int(x.shape[0]*0.25)+1
        ), size=1
    )[0]
    # Shift time in each band separately
    for b in range(0, xseg.shape[0], ntime):
        # Get the slice for the current band, n time steps
        xseg[b:b+ntime] = np.roll(xseg[b:b+ntime], shift=shift, axis=0)

    # Insert back into full array
    x = reinsert_prop(
        x=x,
        bounds_slice=bounds_slice,
        mask=mask,
        x_update=xseg,
        x_original=xseg_original
    )

    return x


def create_parcel_masks(labels_array: np.ndarray, max_crop_class: int) -> T.Union[None, dict]:
    """
    Creates masks for each instance

    Reference:
        https://torchtutorialstaging.z5.web.core.windows.net/intermediate/torchvision_tutorial.html
    """
    # Remove edges
    mask = np.where((labels_array > 0) & (labels_array <= max_crop_class), 1, 0)
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
            [idx for idx in range(0, masks.shape[0]) if idx not in small_box_idx]
        )
        masks = masks[good_idx]
    # convert everything into arrays
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.size(0) == 0:
        return None
    # there is only one class
    labels = torch.ones((masks.shape[0],), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    assert boxes.size(0) == labels.size(0) == masks.size(0), \
        'The tensor sizes do not match.'

    target = {
        'boxes': boxes,
        'labels': labels,
        'masks': masks
    }

    return target


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
        cdata = self.prepare_data(ldata)
        cdata = self.forward(cdata, ldata, aug_args)
        data = self.finalize(
            x=cdata.x,
            y=cdata.y,
            bdist=cdata.bdist,
            aug_args=aug_args
        )

        return data

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
        x = ldata.x
        y = ldata.y
        bdist = ldata.bdist
        # ori = ldata.ori
        # if zero_padding > 0:
        #     zpad = torch.nn.ZeroPad2d(zero_padding)
        #     x = zpad(torch.tensor(x)).numpy()
        #     y = zpad(torch.tensor(y)).numpy()
        #     bdist = zpad(torch.tensor(bdist)).numpy()
        #     ori = zpad(torch.tensor(ori)).numpy()

        x = x.copy()
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
            cdata = replace(cdata)
        else:
            for i in range(0, x.shape[0]):
                x[i] = getattr(np, self.direction)(x[i])

            y = getattr(np, self.direction)(cdata.y)
            bdist = getattr(np, self.direction)(cdata.bdist)
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
    """
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
