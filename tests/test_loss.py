import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Dirichlet

from cultionet.losses import (
    LossPreprocessing,
    TanimotoComplementLoss,
    TanimotoDistLoss,
)

torch.manual_seed(100)
BATCH_SIZE = 2
HEIGHT = 20
WIDTH = 20
INPUTS_CROP_LOGIT = torch.randn(BATCH_SIZE, 2, HEIGHT, WIDTH)
INPUTS_CROP_PROB = rearrange(
    Dirichlet(torch.tensor([0.5, 0.5])).rsample(
        (BATCH_SIZE * HEIGHT * WIDTH,)
    ),
    '(b h w) c -> b c h w',
    b=BATCH_SIZE,
    c=2,
    h=HEIGHT,
    w=WIDTH,
)
INPUTS_DIST = torch.rand(BATCH_SIZE, 1, HEIGHT, WIDTH)
DISCRETE_TARGETS = torch.randint(
    low=0, high=2, size=(BATCH_SIZE, HEIGHT, WIDTH)
)
DIST_TARGETS = torch.rand(BATCH_SIZE, HEIGHT, WIDTH)


def test_loss_preprocessing():
    # Input logits
    preprocessor = LossPreprocessing(
        transform_logits=True, one_hot_targets=True
    )
    inputs, targets = preprocessor(INPUTS_CROP_LOGIT, DISCRETE_TARGETS)

    assert inputs.shape == (BATCH_SIZE * HEIGHT * WIDTH, 2)
    assert targets.shape == (BATCH_SIZE * HEIGHT * WIDTH, 2)
    assert torch.allclose(targets.max(dim=0).values, torch.ones(2))
    assert torch.allclose(
        inputs.sum(dim=1), torch.ones(BATCH_SIZE * HEIGHT * WIDTH), rtol=0.1
    )
    assert torch.allclose(
        inputs,
        rearrange(
            F.softmax(INPUTS_CROP_LOGIT, dim=1, dtype=INPUTS_CROP_LOGIT.dtype),
            'b c h w -> (b h w) c',
        ),
    )

    # Input probabilities
    preprocessor = LossPreprocessing(
        transform_logits=False, one_hot_targets=True
    )
    inputs, targets = preprocessor(INPUTS_CROP_PROB, DISCRETE_TARGETS)

    assert inputs.shape == (BATCH_SIZE * HEIGHT * WIDTH, 2)
    assert targets.shape == (BATCH_SIZE * HEIGHT * WIDTH, 2)
    assert torch.allclose(targets.max(dim=0).values, torch.ones(2))
    assert torch.allclose(
        inputs.sum(dim=1), torch.ones(BATCH_SIZE * HEIGHT * WIDTH), rtol=0.1
    )
    assert torch.allclose(
        inputs,
        rearrange(INPUTS_CROP_PROB, 'b c h w -> (b h w) c'),
    )

    # Regression
    preprocessor = LossPreprocessing(
        transform_logits=False, one_hot_targets=False
    )
    inputs, targets = preprocessor(INPUTS_DIST, DIST_TARGETS)

    assert torch.allclose(
        inputs, rearrange(INPUTS_DIST, 'b c h w -> (b h w) c')
    )
    assert torch.allclose(
        targets, rearrange(DIST_TARGETS, 'b h w -> (b h w) 1')
    )


def test_tanimoto_classification_loss():
    loss_func = TanimotoDistLoss(
        scale_pos_weight=False,
        transform_logits=False,
        one_hot_targets=True,
    )
    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS)
    assert round(float(loss.item()), 4) == 0.6062

    loss_func = TanimotoComplementLoss(
        transform_logits=False,
        one_hot_targets=True,
    )
    loss = loss_func(INPUTS_CROP_PROB, DISCRETE_TARGETS)
    assert round(float(loss.item()), 4) == 0.8214
