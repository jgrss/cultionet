import typing as T

import attr
import torch
import torch.nn.functional as F
from torchmetrics.functional import stat_scores


@attr.s
class F1Score(object):
    """F1-score

    Reference:
        https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    """
    num_classes: int = attr.ib(validator=attr.validators.instance_of(int))
    epsilon: T.Optional[float] = attr.ib(validator=attr.validators.optional(attr.validators.instance_of(float)),
                                         default=1e-7)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            f1-score (float)
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        tp, fp, tn, fn, sup = stat_scores(inputs, targets, num_classes=self.num_classes)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2.0 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1.0-self.epsilon)

        return f1.mean()


@attr.s
class MatthewsCorrcoef(object):
    """Matthews Correlation Coefficient

    Source:
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """
    num_classes: int = attr.ib(validator=attr.validators.instance_of(int))
    inputs_are_logits: T.Optional[bool] = attr.ib(default=True)

    def __attrs_post_init__(self):
        super(MatthewsCorrcoef, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Matthews correlation coefficient (float)
        """
        if self.inputs_are_logits:
            inputs = F.softmax(inputs, dim=1).argmax(dim=1)

        tp, fp, tn, fn, sup = stat_scores(inputs, targets, num_classes=self.num_classes)

        numerator = (tp * tn) - (fp * fn)
        denominator = torch.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        matthews_corr_coef = numerator / denominator
        matthews_corr_coef = torch.nan_to_num(matthews_corr_coef, nan=0.0, neginf=0.0, posinf=0.0)

        return matthews_corr_coef


@attr.s
class TanimotoDistanceLoss(object):
    """Tanimoto distance loss

    Reference:
        https://github.com/sentinel-hub/eo-flow/blob/master/eoflow/models/losses.py

    MIT License

    Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Matic Lubej, Grega Milčinski (Sinergise)
    Copyright (c) 2017-2020 Devis Peressutti, Jernej Puc, Anže Zupanc, Lojze Žust, Jovan Višnjić (Sinergise)
    """
    volume: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    smooth: T.Optional[float] = attr.ib(
        default=1e-5,
        validator=attr.validators.optional(validator=attr.validators.instance_of(float))
    )
    class_weights: T.Optional[torch.Tensor] = attr.ib(
        default=None,
        validator=attr.validators.optional(validator=attr.validators.instance_of(torch.Tensor))
    )
    inputs_are_logits: T.Optional[bool] = attr.ib(
        default=True,
        validator=attr.validators.optional(validator=attr.validators.instance_of(bool))
    )
    apply_transform: T.Optional[bool] = attr.ib(
        default=True,
        validator=attr.validators.optional(validator=attr.validators.instance_of(bool))
    )

    def __attrs_post_init__(self):
        super(TanimotoDistanceLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        if self.inputs_are_logits:
            if self.apply_transform:
                inputs = F.softmax(inputs, dim=1)
            targets = F.one_hot(targets.contiguous().view(-1), inputs.shape[1]).float()
        else:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

        weights = torch.reciprocal(torch.square(self.volume))
        new_weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = torch.where(torch.isinf(weights), torch.ones_like(weights) * new_weights.max(), weights)
        intersection = (targets * inputs).sum(dim=0)
        sum_ = (targets * targets + inputs * inputs).sum(dim=0)
        num_ = (intersection * weights) + self.smooth
        den_ = ((sum_ - intersection) * weights) + self.smooth
        tanimoto = num_ / den_
        loss = (1.0 - tanimoto)

        if self.class_weights is not None:
            loss = loss * self.class_weights

        return loss.sum()


@attr.s
class QuantileLoss(object):
    """Loss function for quantile regression

    Reference:
        https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss

    THE MIT License

    Copyright 2020 Jan Beitner
    """
    quantiles: T.Tuple[float, float, float] = attr.ib(validator=attr.validators.instance_of(tuple))

    def __attrs_post_init__(self):
        super(QuantileLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Quantile loss (float)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - inputs[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.cat(losses, dim=1).sum(dim=1).mean()

        return loss


@attr.s
class MSELoss(object):
    """Mean squared error loss
    """
    def __attrs_post_init__(self):
        super(MSELoss, self).__init__()
        self.loss_func = torch.nn.MSELoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(inputs, targets)


@attr.s
class HuberLoss(object):
    """Huber loss
    """
    def __attrs_post_init__(self):
        super(HuberLoss, self).__init__()
        self.loss_func = torch.nn.HuberLoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(inputs, targets)
