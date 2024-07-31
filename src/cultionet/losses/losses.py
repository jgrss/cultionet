import typing as T
import warnings

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from kornia.contrib import distance_transform

try:
    import torch_topological.nn as topnn
except ImportError:
    topnn = None

from ..data.data import Data


class FieldOfJunctionsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        patches: torch.Tensor,
        image_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the objective of our model (see Equation 8 of the paper)."""

        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = einops.reduce(
            (
                einops.rearrange(image_patches, 'b c p k h w -> b 1 c p k h w')
                - patches
            )
            ** 2,
            'b n c p k h w -> b n c h w',
            'mean',
        )
        loss_per_patch = einops.reduce(
            loss_per_patch, 'b n c h w -> b n h w', 'sum'
        )
        # Reduce to the batch mean
        loss_per_patch = einops.reduce(
            loss_per_patch, 'b n h w -> n h w', 'mean'
        )

        return loss_per_patch.mean()


class LossPreprocessing(nn.Module):
    def __init__(
        self, transform_logits: bool = False, one_hot_targets: bool = True
    ):
        super().__init__()

        self.transform_logits = transform_logits
        self.one_hot_targets = one_hot_targets

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to transform logits.

        If logits are single-dimension then they are transformed by Sigmoid. If
        logits are multi-dimension then they are transformed by Softmax.
        """

        if self.transform_logits:
            if inputs.shape[1] == 1:
                inputs = F.sigmoid(inputs).to(dtype=inputs.dtype)
            else:
                inputs = F.softmax(inputs, dim=1, dtype=inputs.dtype)

            inputs = inputs.clip(0, 1)

        if self.one_hot_targets and (inputs.shape[1] > 1):
            targets = einops.rearrange(
                F.one_hot(targets, num_classes=inputs.shape[1]),
                'b h w c -> b c h w',
            )
        else:
            targets = einops.rearrange(targets, 'b h w -> b 1 h w')

        return inputs, targets


class TanimotoComplementLoss(nn.Module):
    """Tanimoto distance loss.

    Adapted from publications and source code below:

        CSIRO BSTD/MIT LICENSE

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that
        the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
            following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
            the following disclaimer in the documentation and/or other materials provided with the distribution.
        3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
            promote products derived from this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
        INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
        WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
        USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        References:
            https://www.mdpi.com/2072-4292/14/22/5738
            https://arxiv.org/abs/2009.02062
            https://github.com/waldnerf/decode/blob/main/FracTAL_ResUNet/nn/loss/ftnmt_loss.py
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        depth: int = 5,
        transform_logits: bool = False,
        one_hot_targets: bool = True,
    ):
        super().__init__()

        self.smooth = smooth
        self.depth = depth

        self.preprocessor = LossPreprocessing(
            transform_logits=transform_logits,
            one_hot_targets=one_hot_targets,
        )

    def tanimoto_distance(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        weights: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / self.depth

        if mask is not None:
            y = y * mask
            yhat = yhat * mask

        tpl = y * yhat
        sq_sum = y**2 + yhat**2

        tpl = tpl.sum(dim=(2, 3))
        sq_sum = sq_sum.sum(dim=(2, 3))

        if weights is not None:
            tpl = tpl * weights
            sq_sum = sq_sum * weights

        denominator = 0.0
        for d in range(0, self.depth):
            a = 2.0**d
            b = -(2.0 * a - 1.0)
            denominator = denominator + torch.reciprocal(
                (a * sq_sum) + (b * tpl)
            )
            denominator = torch.nan_to_num(
                denominator, nan=0.0, posinf=0.0, neginf=0.0
            )

        return ((tpl * denominator) * scale).sum(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model (probabilities or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        inputs, targets = self.preprocessor(inputs, targets)

        loss = 1.0 - self.tanimoto_distance(targets, inputs, mask=mask)
        compl_loss = 1.0 - self.tanimoto_distance(
            1.0 - targets, 1.0 - inputs, mask=mask
        )
        loss = (loss + compl_loss) * 0.5

        return loss.mean()


def tanimoto_dist(
    ypred: torch.Tensor,
    ytrue: torch.Tensor,
    scale_pos_weight: bool,
    class_counts: T.Union[None, torch.Tensor],
    beta: float,
    smooth: float,
    mask: T.Optional[torch.Tensor] = None,
    weights: T.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tanimoto distance."""
    ytrue = ytrue.to(dtype=ypred.dtype)

    # Take the batch mean of the channel sums
    volume = ytrue.sum(dim=(2, 3)).mean(dim=0)
    batch_weight = torch.reciprocal(torch.pow(volume, 2))
    new_weights = torch.where(
        torch.isinf(batch_weight),
        torch.zeros_like(batch_weight),
        batch_weight,
    )
    batch_weight = torch.where(
        torch.isinf(batch_weight),
        torch.ones_like(batch_weight) * torch.max(new_weights),
        batch_weight,
    )

    if scale_pos_weight:
        if class_counts is None:
            class_counts = ytrue.sum(dim=0)
        else:
            class_counts = class_counts
        effective_num = 1.0 - beta**class_counts
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * class_counts.shape[0]

    # Apply a mask to zero-out gradients where mask == 0
    if mask is not None:
        ytrue = ytrue * mask
        ypred = ypred * mask

    tpl = ypred * ytrue
    sq_sum = ypred**2 + ytrue**2

    # Sum over rows and columns
    tpl = tpl.sum(dim=(2, 3))
    sq_sum = sq_sum.sum(dim=(2, 3))

    if weights is not None:
        tpl = tpl * weights
        sq_sum = sq_sum * weights

    numerator = (tpl * batch_weight + smooth).sum(dim=1)
    denominator = ((sq_sum - tpl) * batch_weight + smooth).sum(dim=1)
    distance = numerator / denominator

    return distance


class TanimotoDistLoss(nn.Module):
    """Tanimoto distance loss.

    References:
        https://github.com/feevos/resuneta/blob/145be5519ee4bec9a8cce9e887808b8df011f520/nn/loss/loss.py

            CSIRO BSTD/MIT LICENSE

            Redistribution and use in source and binary forms, with or without modification, are permitted provided that
            the following conditions are met:

            1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
                following disclaimer.
            2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
                the following disclaimer in the documentation and/or other materials provided with the distribution.
            3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
                promote products derived from this software without specific prior written permission.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
            INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
            SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
            WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
            USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        https://github.com/sentinel-hub/eo-flow/blob/master/eoflow/models/losses.py

            MIT License

            Copyright (c) 2017-2020 Matej Aleksandrov, Matej Batič, Matic Lubej, Grega Milčinski (Sinergise)
            Copyright (c) 2017-2020 Devis Peressutti, Jernej Puc, Anže Zupanc, Lojze Žust, Jovan Višnjić (Sinergise)

        Class balancing:
            https://github.com/fcakyon/balanced-loss
            https://github.com/vandit15/Class-balanced-loss-pytorch
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        beta: T.Optional[float] = 0.999,
        class_counts: T.Optional[torch.Tensor] = None,
        scale_pos_weight: bool = False,
        transform_logits: bool = False,
        one_hot_targets: bool = True,
    ):
        super().__init__()

        if scale_pos_weight and (class_counts is None):
            warnings.warn(
                "Cannot balance classes without class weights. Weights will be derived for each batch.",
                UserWarning,
            )

        self.smooth = smooth
        self.beta = beta
        self.class_counts = class_counts
        self.scale_pos_weight = scale_pos_weight

        self.preprocessor = LossPreprocessing(
            transform_logits=transform_logits,
            one_hot_targets=one_hot_targets,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """

        inputs, targets = self.preprocessor(inputs, targets)

        loss = 1.0 - tanimoto_dist(
            inputs,
            targets,
            scale_pos_weight=self.scale_pos_weight,
            class_counts=self.class_counts,
            beta=self.beta,
            smooth=self.smooth,
            mask=mask,
        )
        compl_loss = 1.0 - tanimoto_dist(
            1.0 - inputs,
            1.0 - targets,
            scale_pos_weight=self.scale_pos_weight,
            class_counts=self.class_counts,
            beta=self.beta,
            smooth=self.smooth,
            mask=mask,
        )
        loss = (loss + compl_loss) * 0.5

        return loss.mean()


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss."""

    def __init__(
        self,
        weight: T.Optional[torch.Tensor] = None,
        reduction: T.Optional[str] = "mean",
        label_smoothing: T.Optional[float] = 0.1,
    ):
        super().__init__()

        self.loss_func = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, label_smoothing=label_smoothing
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(inputs, targets)


class FocalLoss(nn.Module):
    """Focal loss.

    Reference:
        https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """

    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 2.0,
        weight: T.Optional[torch.Tensor] = None,
        label_smoothing: T.Optional[float] = 0.1,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.preprocessor = LossPreprocessing(
            inputs_are_logits=True, apply_transform=True
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=weight, reduction="none", label_smoothing=label_smoothing
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        inputs, targets = self.preprocessor(inputs, targets)
        ce_loss = self.cross_entropy_loss(inputs, targets.half())
        ce_exp = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - ce_exp) ** self.gamma * ce_loss

        return focal_loss.mean()


class QuantileLoss(nn.Module):
    """Loss function for quantile regression.

    Reference:
        https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss

    THE MIT License

    Copyright 2020 Jan Beitner
    """

    def __init__(self, quantiles: T.Tuple[float, float, float]):
        super().__init__()

        self.quantiles = quantiles

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

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


class WeightedL1Loss(nn.Module):
    """Weighted L1Loss loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        mae = torch.abs(inputs - targets)
        weight = inputs + targets
        loss = (mae * weight).mean()

        return loss


class MSELoss(nn.Module):
    """MSE loss."""

    def __init__(self):
        super().__init__()

        self.loss_func = nn.MSELoss()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(
            inputs.contiguous().view(-1), targets.contiguous().view(-1)
        )


class BoundaryLoss(nn.Module):
    """Boundary (surface) loss.

    Reference:
        https://github.com/LIVIAETS/boundary-loss
    """

    def __init__(self):
        super().__init__()

    def fill_distances(
        self,
        distances: torch.Tensor,
        targets: torch.Tensor,
    ):
        dt = distance_transform(
            F.pad(
                (targets == 2).long().unsqueeze(1).float(),
                pad=(
                    21,
                    21,
                    21,
                    21,
                ),
            ),
            kernel_size=21,
            h=0.1,
        ).squeeze(dim=1)[:, 21:-21, 21:-21]
        dt /= dt.max()

        idist = torch.where(
            targets == 2, 0, torch.where(targets == 1, distances, 0)
        )
        idist = torch.where(targets > 0, idist, dt)

        return idist

    def forward(
        self,
        probs: torch.Tensor,
        distances: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            probs: Predicted probabilities, shaped (B x H x W).
            distances: Ground truth distance transform, shaped (B x H x W).
            targets: Ground truth labels, shaped (B x H x W).

        Returns:
            Loss (float)
        """
        distances = self.fill_distances(distances, targets)

        return torch.einsum("bhw, bhw -> bhw", distances, 1.0 - probs).mean()


class MultiScaleSSIMLoss(nn.Module):
    """Multi-scale Structural Similarity Index Measure loss."""

    def __init__(self):
        super().__init__()

        self.msssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
            gaussian_kernel=False,
            kernel_size=3,
            data_range=1.0,
            k1=1e-4,
            k2=9e-4,
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, data: Data
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predicted probabilities.
            targets: Ground truth inverse distance transform, where distances
                along edges are 1.
            data: Data object used to extract dimensions.

        Returns:
            Loss (float)
        """
        height = (
            int(data.height) if data.batch is None else int(data.height[0])
        )
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        inputs = self.gc(inputs.unsqueeze(1), batch_size, height, width)
        targets = self.gc(targets.unsqueeze(1), batch_size, height, width).to(
            dtype=inputs.dtype
        )

        loss = 1.0 - self.msssim(inputs, targets)

        return loss


class TopologyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        if topnn is not None:
            self.loss_func = topnn.SummaryStatisticLoss(
                "total_persistence", p=2
            )
            self.cubical = topnn.CubicalComplex(dim=3)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions (probabilities) from model.
            targets: Ground truth values.
        """
        if mask is None:
            targets = targets * mask
            inputs = inputs * mask

        persistence_information_target = self.cubical(targets)
        persistence_information_target = [persistence_information_target[0]]

        persistence_information = self.cubical(inputs)
        persistence_information = [persistence_information[0]]

        loss = self.loss_func(
            persistence_information, persistence_information_target
        )

        return loss


class ClassBalancedMSELoss(nn.Module):
    r"""
    References:
        @article{xia_etal_2024,
            title={Crop field extraction from high resolution remote sensing images based on semantic edges and spatial structure map},
            author={Xia, Liegang and Liu, Ruiyan and Su, Yishao and Mi, Shulin and Yang, Dezhi and Chen, Jun and Shen, Zhanfeng},
            journal={Geocarto International},
            volume={39},
            number={1},
            pages={2302176},
            year={2024},
            publisher={Taylor \& Francis}
        }

        https://github.com/Adillwma/ACB_MSE
    """

    def __init__(self):
        super().__init__()

        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities, shaped (B x C x H x W).
            targets: Ground truth values, shaped (B x C x H x W).
            mask: Shaped (B x C x H x W).
        """
        if mask is not None:
            neg_mask = (targets == 0) & (mask != 0)
            pos_mask = (targets == 1) & (mask != 0)
            target_count = mask.sum()
        else:
            neg_mask = targets == 0
            pos_mask = ~neg_mask
            target_count = targets.nelement()

        targets_neg = targets[neg_mask]
        targets_pos = targets[pos_mask]

        inputs_neg = inputs[neg_mask]
        inputs_pos = inputs[pos_mask]

        beta = targets_pos.sum() / target_count

        neg_loss = self.mse_loss(
            inputs_neg, targets_neg.to(dtype=inputs.dtype)
        )
        pos_loss = self.mse_loss(
            inputs_pos, targets_pos.to(dtype=inputs.dtype)
        )

        if torch.isnan(neg_loss):
            neg_loss = 0.0
        if torch.isnan(pos_loss):
            pos_loss = 0.0

        loss = beta * neg_loss + (1.0 - beta) * pos_loss

        return loss
