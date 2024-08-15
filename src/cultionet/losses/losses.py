import typing as T

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossPreprocessing(nn.Module):
    def __init__(
        self, transform_logits: bool = False, one_hot_targets: bool = True
    ):
        super().__init__()

        self.transform_logits = transform_logits
        self.one_hot_targets = one_hot_targets

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
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

        if mask is not None:
            # Apply a mask to zero-out weight
            inputs = inputs * mask
            targets = targets * mask

        return inputs, targets


class CombinedLoss(nn.Module):
    def __init__(self, losses: T.List[T.Callable]):
        super().__init__()

        self.losses = losses

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model (probabilities or labels), shaped (B, C, H, W).
            targets: Ground truth values, shaped (B, C, H, W).
            mask: Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns:
            Average distance loss (float)
        """

        loss = 0.0
        for loss_func in self.losses:
            loss = loss + loss_func(
                inputs=inputs,
                targets=targets,
                mask=mask,
            )

        loss = loss / len(self.losses)

        return loss


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
    ) -> torch.Tensor:
        scale = 1.0 / self.depth

        tpl = y * yhat
        sq_sum = y**2 + yhat**2

        tpl = tpl.sum(dim=(1, 2, 3))
        sq_sum = sq_sum.sum(dim=(1, 2, 3))

        denominator = 0.0
        for d in range(0, self.depth):
            a = 2.0**d
            b = -(2.0 * a - 1.0)
            denominator = denominator + torch.reciprocal(
                ((a * sq_sum) + (b * tpl)) + self.smooth
            )

        numerator = tpl + self.smooth
        distance = (numerator * denominator) * scale

        return 1.0 - distance

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Args:
            inputs: Predictions from model (probabilities or labels), shaped (B, C, H, W).
            targets: Ground truth values, shaped (B, C, H, W).
            mask: Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns:
            Tanimoto distance loss (float)
        """
        inputs, targets = self.preprocessor(
            inputs=inputs, targets=targets, mask=mask
        )

        loss1 = self.tanimoto_distance(targets, inputs)
        loss2 = self.tanimoto_distance(1.0 - targets, 1.0 - inputs)
        loss = (loss1 + loss2) * 0.5

        return loss.mean()


def tanimoto_dist(
    ypred: torch.Tensor,
    ytrue: torch.Tensor,
    smooth: float,
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

    tpl = ypred * ytrue
    sq_sum = ypred**2 + ytrue**2

    # Sum over rows and columns
    tpl = tpl.sum(dim=(2, 3))
    sq_sum = sq_sum.sum(dim=(2, 3))

    numerator = (tpl * batch_weight).sum(dim=-1) + smooth
    denominator = ((sq_sum - tpl) * batch_weight).sum(dim=-1) + smooth
    distance = numerator / denominator

    loss = 1.0 - distance

    return loss


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
        transform_logits: bool = False,
        one_hot_targets: bool = True,
    ):
        super().__init__()

        self.smooth = smooth

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
            inputs: Predictions from model (probabilities or labels), shaped (B, C, H, W).
            targets: Ground truth values, shaped (B, C, H, W).
            mask: Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns:
            Tanimoto distance loss (float)
        """

        inputs, targets = self.preprocessor(
            inputs=inputs, targets=targets, mask=mask
        )

        loss1 = tanimoto_dist(
            inputs,
            targets,
            smooth=self.smooth,
        )
        loss2 = tanimoto_dist(
            1.0 - inputs,
            1.0 - targets,
            smooth=self.smooth,
        )
        loss = (loss1 + loss2) * 0.5

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
