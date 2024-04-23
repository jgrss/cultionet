import numpy as np
import torch
from scipy.ndimage.measurements import label as nd_label
from skimage.measure import regionprops

from cultionet.augment.augmenter_utils import (
    feature_stack_to_tsaug,
    tsaug_to_feature_stack,
)
from cultionet.augment.augmenters import Augmenters
from cultionet.data.data import Data

NTIME = 12
NBANDS = 3
NROWS = 10
NCOLS = 20
RNG = np.random.default_rng(200)


def test_tensor_reshape():
    """Test array reshaping."""

    x = torch.rand(1, NBANDS, NTIME, NROWS, NCOLS)
    # Reshape to -> (H*W x T X C)
    x_t = feature_stack_to_tsaug(x)

    assert x_t.shape == (
        NROWS * NCOLS,
        NTIME,
        NBANDS,
    ), 'The feature stack was incorrectly reshaped.'

    # First sample, first band, all time
    assert torch.allclose(x_t[0, :, 0], x[0, 0, :, 0, 0])
    # First sample, second band, all time
    assert torch.allclose(x_t[0, :, 1], x[0, 1, :, 0, 0])
    # First sample, last band, all time
    assert torch.allclose(x_t[0, :, -1], x[0, -1, :, 0, 0])
    # Last sample, first band, all time
    assert torch.allclose(x_t[-1, :, 0], x[0, 0, :, -1, -1])

    # Reshape from (H*W x T X C) -> (T*C x H x W)
    x_tr = tsaug_to_feature_stack(x_t, NROWS, NCOLS)

    assert torch.allclose(
        x, x_tr
    ), 'The re-transformed data do not match the original.'


def create_full_batch(
    num_channels: int,
    num_time: int,
    height: int,
    width: int,
) -> Data:
    x = torch.rand(1, num_channels, num_time, height, width)
    y = torch.randint(low=0, high=2, size=(1, height, width))
    bdist = torch.rand(1, height, width)

    return Data(x=x, y=y, bdist=bdist)


def test_augmenter_loading():
    augmentations = [
        'tswarp',
        'tsnoise',
        'tsdrift',
        'tspeaks',
    ]
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_full_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        assert method.name_ == augmentations[i]

        batch.segments = np.uint8(nd_label(batch.y.squeeze().numpy() == 1)[0])
        batch.props = regionprops(batch.segments)
        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert not torch.allclose(aug_batch.x, batch.x)
        assert torch.allclose(aug_batch.y, batch.y)

    augmentations = [
        'gaussian',
        'saltpepper',
        'tsdrift',
        'speckle',
    ]
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_full_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        batch.segments = np.uint8(nd_label(batch.y.squeeze().numpy() == 1)[0])
        batch.props = regionprops(batch.segments)
        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert not torch.allclose(aug_batch.x, batch.x)
        assert torch.allclose(aug_batch.y, batch.y)

    augmentations = [
        'rot90',
        'rot180',
        'rot270',
    ]
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_full_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert not torch.allclose(aug_batch.x, batch.x)
        assert not torch.allclose(aug_batch.y, batch.y)

    augmentations = ['none']
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_full_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert torch.allclose(aug_batch.x, batch.x)
        assert torch.allclose(aug_batch.y, batch.y)
