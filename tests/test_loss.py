import numpy as np
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from cultionet.losses import (
    CombinedLoss,
    LossPreprocessing,
    TanimotoComplementLoss,
    TanimotoDistLoss,
)

rng = np.random.default_rng(100)

BATCH_SIZE = 2
HEIGHT = 20
WIDTH = 20

INPUTS_CROP_LOGIT = torch.from_numpy(
    rng.uniform(low=-3, high=3, size=(BATCH_SIZE, 2, HEIGHT, WIDTH))
).float()
INPUTS_CROP_PROB = rearrange(
    torch.from_numpy(
        rng.dirichlet((0.5, 0.5), size=(BATCH_SIZE * HEIGHT * WIDTH))
    ).float(),
    '(b h w) c -> b c h w',
    b=BATCH_SIZE,
    c=2,
    h=HEIGHT,
    w=WIDTH,
)
INPUTS_EDGE_PROB = torch.from_numpy(
    rng.random((BATCH_SIZE, 1, HEIGHT, WIDTH))
).float()
INPUTS_DIST = torch.from_numpy(
    rng.random((BATCH_SIZE, 1, HEIGHT, WIDTH))
).float()
DISCRETE_TARGETS = torch.from_numpy(
    rng.integers(low=0, high=2, size=(BATCH_SIZE, HEIGHT, WIDTH))
).long()
DISCRETE_EDGE_TARGETS = torch.from_numpy(
    rng.integers(low=0, high=1, size=(BATCH_SIZE, HEIGHT, WIDTH))
).long()
DIST_TARGETS = torch.from_numpy(
    rng.random((BATCH_SIZE, HEIGHT, WIDTH))
).float()
MASK = torch.from_numpy(
    rng.integers(low=0, high=2, size=(BATCH_SIZE, 1, HEIGHT, WIDTH))
).long()


def test_loss_preprocessing():
    # Input logits
    preprocessor = LossPreprocessing(
        transform_logits=True, one_hot_targets=True
    )
    inputs, targets = preprocessor(INPUTS_CROP_LOGIT, DISCRETE_TARGETS)

    assert inputs.shape == (BATCH_SIZE, 2, HEIGHT, WIDTH)
    assert targets.shape == (BATCH_SIZE, 2, HEIGHT, WIDTH)
    assert torch.allclose(
        inputs.sum(dim=1), torch.ones(BATCH_SIZE, HEIGHT, WIDTH), rtol=0.1
    )
    assert torch.allclose(
        inputs,
        F.softmax(INPUTS_CROP_LOGIT, dim=1, dtype=INPUTS_CROP_LOGIT.dtype),
    )

    # Input probabilities
    preprocessor = LossPreprocessing(
        transform_logits=False, one_hot_targets=True
    )
    inputs, targets = preprocessor(INPUTS_CROP_PROB, DISCRETE_TARGETS)

    assert inputs.shape == (BATCH_SIZE, 2, HEIGHT, WIDTH)
    assert targets.shape == (BATCH_SIZE, 2, HEIGHT, WIDTH)
    assert torch.allclose(
        inputs.sum(dim=1), torch.ones(BATCH_SIZE, HEIGHT, WIDTH), rtol=0.1
    )
    assert torch.allclose(
        inputs,
        INPUTS_CROP_PROB,
    )

    preprocessor = LossPreprocessing(
        transform_logits=False, one_hot_targets=True
    )
    inputs, targets = preprocessor(INPUTS_EDGE_PROB, DISCRETE_EDGE_TARGETS)

    assert inputs.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)
    assert targets.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)
    assert torch.allclose(
        inputs,
        INPUTS_EDGE_PROB,
    )

    # Regression
    preprocessor = LossPreprocessing(
        transform_logits=False, one_hot_targets=False
    )
    inputs, targets = preprocessor(INPUTS_DIST, DIST_TARGETS)

    # Preprocessing should not change the inputs other than the shape
    assert torch.allclose(inputs, INPUTS_DIST)
    assert torch.allclose(targets, rearrange(DIST_TARGETS, 'b h w -> b 1 h w'))


def test_tanimoto_classification_loss():
    loss_func = TanimotoDistLoss()

    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS)
    assert round(float(loss.item()), 3) == 0.611

    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS, mask=MASK)
    assert round(float(loss.item()), 3) == 0.431

    loss_func = TanimotoComplementLoss()
    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS)
    assert round(float(loss.item()), 3) == 0.824

    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS, mask=MASK)
    assert round(float(loss.item()), 3) == 0.692

    loss_func = CombinedLoss(
        losses=[
            TanimotoDistLoss(),
            TanimotoComplementLoss(),
        ]
    )
    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS)
    assert round(float(loss.item()), 3) == 0.717

    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS, mask=MASK)
    assert round(float(loss.item()), 3) == 0.561


def test_tanimoto_regression_loss():
    loss_func = TanimotoDistLoss(one_hot_targets=False)
    loss = loss_func(INPUTS_DIST, DIST_TARGETS)
    assert round(float(loss.item()), 3) == 0.417

    loss_func = TanimotoComplementLoss(one_hot_targets=False)
    loss = loss_func(INPUTS_DIST, DIST_TARGETS)
    assert round(float(loss.item()), 3) == 0.704
