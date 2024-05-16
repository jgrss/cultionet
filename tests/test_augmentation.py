import numpy as np
import torch
from scipy.ndimage import label as nd_label
from skimage.measure import regionprops

from cultionet.augment.augmenter_utils import (
    feature_stack_to_tsaug,
    tsaug_to_feature_stack,
)
from cultionet.augment.augmenters import Augmenters

from .conftest import create_batch

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


def test_augmenter_loading():
    augmentations = [
        'roll',
        'tswarp',
        'tsnoise',
        'tsdrift',
        'tspeaks',
        'tsdrift',
        'gaussian',
        'saltpepper',
    ]
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        assert batch.x.min() >= 0
        assert batch.x.max() <= 1
        assert batch.y.min() == -1

        batch.segments = np.uint8(nd_label(batch.y.squeeze().numpy() == 1)[0])
        batch.props = regionprops(batch.segments)
        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert not torch.allclose(aug_batch.x, batch.x)
        assert torch.allclose(aug_batch.y, batch.y)
        assert torch.allclose(aug_batch.bdist, batch.bdist)

    augmentations = [
        'rot90',
        'rot180',
        'rot270',
        'fliplr',
        'flipud',
        'cropresize',
    ]
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        assert batch.x.min() >= 0
        assert batch.x.max() <= 1
        assert batch.y.min() == -1

        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        if method.name_ == 'rotate-90':
            assert torch.allclose(
                batch.x[0, 0, :, 0, 0],
                aug_batch.x[0, 0, :, -1, 0],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.x[0, 0, :, 0, -1],
                aug_batch.x[0, 0, :, 0, 0],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.y[0, 0, 0],
                aug_batch.y[0, -1, 0],
            )
            assert torch.allclose(
                batch.y[0, 0, -1],
                aug_batch.y[0, 0, 0],
            )
            assert torch.allclose(
                batch.bdist[0, 0, 0],
                aug_batch.bdist[0, -1, 0],
            )
            assert torch.allclose(
                batch.bdist[0, 0, -1],
                aug_batch.bdist[0, 0, 0],
            )
        elif method.name_ == 'fliplr':
            assert torch.allclose(
                batch.x[0, 0, :, 0, 0],
                aug_batch.x[0, 0, :, 0, -1],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.x[0, 0, :, -1, 0],
                aug_batch.x[0, 0, :, -1, -1],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.y[0, 0, 0],
                aug_batch.y[0, 0, -1],
            )
            assert torch.allclose(
                batch.y[0, -1, 0],
                aug_batch.y[0, -1, -1],
            )
            assert torch.allclose(
                batch.bdist[0, 0, 0],
                aug_batch.bdist[0, 0, -1],
            )
            assert torch.allclose(
                batch.bdist[0, -1, 0],
                aug_batch.bdist[0, -1, -1],
            )
        elif method.name_ == 'flipud':
            assert torch.allclose(
                batch.x[0, 0, :, 0, 0],
                aug_batch.x[0, 0, :, -1, 0],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.x[0, 0, :, 0, -1],
                aug_batch.x[0, 0, :, -1, -1],
                rtol=1e-4,
            )
            assert torch.allclose(
                batch.y[0, 0, 0],
                aug_batch.y[0, -1, 0],
            )
            assert torch.allclose(
                batch.y[0, 0, -1],
                aug_batch.y[0, -1, -1],
            )
            assert torch.allclose(
                batch.bdist[0, 0, 0],
                aug_batch.bdist[0, -1, 0],
            )
            assert torch.allclose(
                batch.bdist[0, 0, -1],
                aug_batch.bdist[0, -1, -1],
            )

        assert not torch.allclose(aug_batch.x, batch.x)
        assert not torch.allclose(aug_batch.y, batch.y)
        assert not torch.allclose(aug_batch.bdist, batch.bdist)

    augmentations = ['none']
    aug = Augmenters(augmentations=augmentations, max_crop_class=1)
    for i, method in enumerate(aug):
        batch = create_batch(
            num_channels=3,
            num_time=12,
            height=50,
            width=50,
        )

        aug_batch = method(batch.copy(), aug_args=aug.aug_args)

        assert torch.allclose(aug_batch.x, batch.x)
        assert torch.allclose(aug_batch.y, batch.y)
        assert torch.allclose(aug_batch.bdist, batch.bdist)
