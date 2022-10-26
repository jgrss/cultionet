import typing as T

import attr
import torch
import torch.nn.functional as F


class ClassifierPreprocessing(object):
    def preprocess(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        if self.inputs_are_logits:
            if (len(targets.unique()) > inputs.size(1)) or (targets.unique().max()+1 > inputs.size(1)):
                raise ValueError(
                    'The targets should be ordered values of equal length to the inputs 2nd dimension.'
                )
            if self.apply_transform:
                inputs = F.softmax(inputs, dim=1, dtype=inputs.dtype)
            targets = F.one_hot(
                targets.contiguous().view(-1), inputs.shape[1]
            ).float()
        else:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

        return inputs, targets


@attr.s
class TanimotoDistanceLoss(ClassifierPreprocessing):
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
        inputs, targets = self.preprocess(inputs, targets)

        weights = torch.reciprocal(torch.square(self.volume))
        new_weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = torch.where(
            torch.isinf(weights), torch.ones_like(weights) * new_weights.max(), weights
        )
        intersection = (targets * inputs).sum(dim=0)
        sum_ = (targets * targets + inputs * inputs).sum(dim=0)
        num_ = (intersection * weights) + self.smooth
        den_ = ((sum_ - intersection) * weights) + self.smooth
        tanimoto = num_ / den_
        loss = 1.0 - tanimoto

        if self.class_weights is not None:
            loss = loss * self.class_weights

        return loss.sum()


@attr.s
class CrossEntropyLoss(object):
    """Cross entropy loss
    """
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
    device: T.Optional[str] = attr.ib(
        default='cpu',
        validator=attr.validators.optional(validator=attr.validators.instance_of(str))
    )

    def __attrs_post_init__(self):
        if self.device == 'cpu':
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights).cuda()

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
        return self.loss_func(inputs, targets.contiguous().view(-1))


@attr.s
class QuantileLoss(object):
    """Loss function for quantile regression

    Reference:
        https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss

    THE MIT License

    Copyright 2020 Jan Beitner
    """
    quantiles: T.Tuple[float, float, float] = attr.ib(validator=attr.validators.instance_of(tuple))

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
class HuberLoss(object):
    """Huber loss
    """
    def __attrs_post_init__(self):
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
        return self.loss_func(
            inputs.contiguous().view(-1),
            targets.contiguous().view(-1)
        )
