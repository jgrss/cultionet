from cultionet.augment.augmentation import (
    feature_stack_to_tsaug,
    tsaug_to_feature_stack
)

import numpy as np


NTIME = 12
NBANDS = 3
NROWS = 10
NCOLS = 20
RNG = np.random.default_rng(200)


def test_feature_stack_to_tsaug():
    """Test array reshaping
    """
    x = RNG.random((NTIME*NBANDS, NROWS, NCOLS))
    nfeas = x.shape[0]
    # Reshape from (T*C x H x W) -> (H*W x T X C)
    x_t = feature_stack_to_tsaug(
        x, NTIME, NBANDS, NROWS, NCOLS
    )
    assert x_t.shape == (NROWS*NCOLS, NTIME, NBANDS), \
        'The feature stack was not correctly reshaped.'
    # Reshape from (H*W x T X C) -> (T*C x H x W)
    x_tr = tsaug_to_feature_stack(x_t, nfeas, NROWS, NCOLS)
    assert np.allclose(x, x_tr), \
        'The re-transformed data to not match the original.'
