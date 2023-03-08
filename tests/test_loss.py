from cultionet.losses import TanimotoDistLoss

import torch


torch.manual_seed(100)
n_samples = 100
INPUTS = torch.randn((n_samples, 2))
TARGETS = torch.randint(low=0, high=2, size=(n_samples,))


def test_tanimoto_loss():
    loss_func = TanimotoDistLoss(scale_pos_weight=False, transform_logits=True)
    loss = loss_func(INPUTS, TARGETS)

    assert round(loss.mean().item(), 4) == 0.5903
