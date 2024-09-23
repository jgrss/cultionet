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

            with torch.no_grad():
                targets = einops.rearrange(
                    F.one_hot(targets, num_classes=inputs.shape[1]),
                    'b h w c -> b c h w',
                )

        else:
            if len(targets.shape) == 3:
                targets = einops.rearrange(targets, 'b h w -> b 1 h w')

        if mask is not None:

            if len(mask.shape) == 3:
                mask = einops.rearrange(mask, 'b h w -> b 1 h w')

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

        Parameters
        ==========
        inputs
            Predictions from model (probabilities or labels), shaped (B, C, H, W).
        targets
            Ground truth values, shaped (B, C, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns
        =======
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
        self.one_hot_targets = one_hot_targets

        self.preprocessor = LossPreprocessing(
            transform_logits=transform_logits,
            one_hot_targets=one_hot_targets,
        )

    def tanimoto_distance(
        self,
        y: torch.Tensor,
        yhat: torch.Tensor,
        dim: T.Optional[T.Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        if dim is None:
            dim = (1, 2, 3)

        scale = 1.0 / self.depth

        tpl = y * yhat
        sq_sum = y**2 + yhat**2

        tpl = tpl.sum(dim=dim)
        sq_sum = sq_sum.sum(dim=dim)

        denominator = 0.0
        for d in range(0, self.depth):
            a = 2.0**d
            b = -(2.0 * a - 1.0)
            denominator = denominator + torch.reciprocal(
                ((a * sq_sum) + (b * tpl)) + self.smooth
            )

        numerator = tpl + self.smooth

        if dim == (2, 3):
            distance = ((numerator * denominator) * scale).sum(dim=1)
        else:
            distance = (numerator * denominator) * scale

        loss = 1.0 - distance

        return loss

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        dim: T.Optional[T.Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (probabilities or labels), shaped (B, C, H, W).
        targets
            Ground truth values, shaped (B, C, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).

        Returns
        =======
        Tanimoto distance loss (float)
        """
        inputs, targets = self.preprocessor(
            inputs=inputs, targets=targets, mask=mask
        )

        loss1 = self.tanimoto_distance(targets, inputs, dim=dim)
        loss2 = self.tanimoto_distance(1.0 - targets, 1.0 - inputs, dim=dim)
        loss = (loss1 + loss2) * 0.5

        return loss.mean()


def tanimoto_dist(
    ypred: torch.Tensor,
    ytrue: torch.Tensor,
    smooth: float,
    dim: T.Optional[T.Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Tanimoto distance."""

    if dim is None:
        dim = (1, 2, 3)

    ytrue = ytrue.to(dtype=ypred.dtype)

    tpl = ypred * ytrue
    sq_sum = ypred**2 + ytrue**2

    tpl = tpl.sum(dim=dim)
    sq_sum = sq_sum.sum(dim=dim)

    numerator = tpl + smooth
    denominator = (sq_sum - tpl) + smooth
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

        Parameters
        ==========
        inputs
            Predictions from model (probabilities or labels), shaped (B, C, H, W).
        targets
            Ground truth values, shaped (B, C, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns
        =======
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


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (real values), shaped (B, H, W) or (B, 1, H, W).
        targets
            Targets (real values), shaped (B, H, W) or (B, 1, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).

        Returns
        =======
        Log Hyperbolic Cosine loss (float)
        """

        if len(inputs.shape) == 3:
            inputs = einops.rearrange(inputs, 'b h w -> b 1 h w')

        if len(targets.shape) == 3:
            targets = einops.rearrange(targets, 'b h w -> b 1 h w')

        loss = torch.log(torch.cosh(inputs - targets))

        if mask is not None:

            if len(mask.shape) == 3:
                mask = einops.rearrange(mask, 'b h w -> b 1 h w')

            loss = loss * mask
            loss = loss.sum() / mask.sum()

        else:
            loss = loss.mean()

        return loss


class ClassBalancedMSELoss(nn.Module):
    r"""Class-balanced mean squared error loss.

    License:
        MIT License
        Copyright (c) 2023 Adill Al-Ashgar

    References:
        @article{xia_etal_2024,
            title={Crop field extraction from high resolution remote sensing images based on semantic edges and spatial structure map},
            author={Xia, Liegang and Liu, Ruiyan and Su, Yishao and Mi, Shulin and Yang, Dezhi and Chen, Jun and Shen, Zhanfeng},
            journal={Geocarto International},
            volume={39},
            number={1},
            pages={2302176},
            year={2024},
            publisher={Taylor \& Francis},
        }

    Source:
        https://github.com/Adillwma/ACB_MSE
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ==========
        inputs
            Predictions (probabilities), shaped (B, H, W) or (B, 1, H, W).
        targets
            Ground truth values, shaped (B, H, W) or (B, 1, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).
        """

        if len(inputs.shape) == 4:
            inputs = einops.rearrange(inputs, 'b 1 h w -> b h w')

        if len(targets.shape) == 4:
            targets = einops.rearrange(targets, 'b 1 h w -> b h w')

        if mask is not None:

            if len(mask.shape) == 4:
                mask = einops.rearrange(mask, 'b 1 h w -> b h w')

            neg_mask = (targets <= 0.5) & (mask != 0)
            pos_mask = (targets > 0.5) & (mask != 0)
            target_count = mask.sum()

        else:

            neg_mask = targets <= 0.5
            pos_mask = ~neg_mask
            target_count = targets.nelement()

        targets_neg = targets[neg_mask]
        targets_pos = targets[pos_mask]

        inputs_neg = inputs[neg_mask]
        inputs_pos = inputs[pos_mask]

        beta = pos_mask.sum() / target_count

        assert 0 <= beta <= 1

        neg_loss = torch.log(
            torch.cosh(
                torch.pow(inputs_neg - targets_neg.to(dtype=inputs.dtype), 2)
            )
        ).mean()

        pos_loss = torch.log(
            torch.cosh(
                torch.pow(inputs_pos - targets_pos.to(dtype=inputs.dtype), 2)
            )
        ).mean()

        if torch.isnan(neg_loss):
            neg_loss = 0.0

        if torch.isnan(pos_loss):
            pos_loss = 0.0

        loss = beta * neg_loss + (1.0 - beta) * pos_loss

        return loss


class BoundaryLoss(nn.Module):
    """Boundary loss.

    License:
        MIT License
        Copyright (c) 2023 Hoel Kervadec

    Reference:
        @inproceedings{kervadec_etal_2019,
            title={Boundary loss for highly unbalanced segmentation},
            author={Kervadec, Hoel and Bouchtiba, Jihene and Desrosiers, Christian and Granger, Eric and Dolz, Jose and Ayed, Ismail Ben},
            booktitle={International conference on medical imaging with deep learning},
            pages={285--296},
            year={2019},
            organization={PMLR},
        }

    Source:
        https://github.com/LIVIAETS/boundary-loss/tree/108bd9892adca476e6cdf424124bc6268707498e
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (probabilities), shaped (B, 1, H, W).
        targets
            Target distance map, shaped (B, H, W) or (B, 1, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).

        Returns
        =======
        Boundary loss (float)
        """

        if len(targets.shape) == 3:
            targets = einops.rearrange(targets, 'b h w -> b 1 h w')

        if mask is not None:
            if len(mask.shape) == 3:
                mask = einops.rearrange(mask, 'b h w -> b 1 h w')

            # Apply a mask to zero-out weight
            inputs = inputs * mask
            targets = targets * mask

        hadamard_product = torch.einsum('bchw, bchw -> bchw', inputs, targets)

        if mask is not None:
            hadamard_mean = hadamard_product.sum() / mask.sum()
        else:
            hadamard_mean = hadamard_product.mean()

        return 1.0 - hadamard_mean


class SoftSkeleton(nn.Module):
    """Soft skeleton.

    License:
        MIT License
        Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit

    Reference:
        @inproceedings{shit_etal_2021,
            title={clDice-a novel topology-preserving loss function for tubular structure segmentation},
            author={Shit, Suprosanna and Paetzold, Johannes C and Sekuboyina, Anjany and Ezhov, Ivan and Unger, Alexander and Zhylka, Andrey and Pluim, Josien PW and Bauer, Ulrich and Menze, Bjoern H},
            booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
            pages={16560--16569},
            year={2021},
        }

    Source:
        https://github.com/jocpae/clDice/tree/master
    """

    def __init__(self, num_iter: int):
        super().__init__()

        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 4:

            p1 = -F.max_pool2d(
                -img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)
            )
            p2 = -F.max_pool2d(
                -img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            )

            eroded = torch.min(p1, p2)

        elif len(img.shape) == 5:

            p1 = -F.max_pool3d(
                -img,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
            )
            p2 = -F.max_pool3d(
                -img,
                kernel_size=(1, 3, 1),
                stride=(1, 1, 1),
                padding=(0, 1, 0),
            )
            p3 = -F.max_pool3d(
                -img,
                kernel_size=(1, 1, 3),
                stride=(1, 1, 1),
                padding=(0, 0, 1),
            )

            eroded = torch.min(torch.min(p1, p2), p3)

        return eroded

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) == 4:
            dilated = F.max_pool2d(
                img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
        elif len(img.shape) == 5:
            dilated = F.max_pool3d(
                img, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
            )

        return dilated

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    def soft_skeleton(self, img: torch.Tensor) -> torch.Tensor:
        img1 = self.soft_open(img)
        skeleton = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.soft_skeleton(x)


class CLDiceLoss(nn.Module):
    """Centerline Dice loss.

    License:
        MIT License
        Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit

    Reference:
        @inproceedings{shit_etal_2021,
            title={clDice-a novel topology-preserving loss function for tubular structure segmentation},
            author={Shit, Suprosanna and Paetzold, Johannes C and Sekuboyina, Anjany and Ezhov, Ivan and Unger, Alexander and Zhylka, Andrey and Pluim, Josien PW and Bauer, Ulrich and Menze, Bjoern H},
            booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
            pages={16560--16569},
            year={2021},
        }

    Source:
        https://github.com/jocpae/clDice/tree/master
    """

    def __init__(self, smooth: float = 1.0, num_iter: int = 10):
        super().__init__()

        self.smooth = smooth

        self.soft_skeleton = SoftSkeleton(num_iter=num_iter)

    def precision_recall(
        self, skeleton: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return ((skeleton * mask).sum() + self.smooth) / (
            skeleton.sum() + self.smooth
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        transform_logits: bool = True,
        mask: T.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (probabilities), shaped (B, 1, H, W).
        targets
            Binary targets, where background is 0 and targets are 1, shaped (B, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, 1, H, W).

        Returns
        =======
        Centerline Dice loss (float)
        """

        targets = einops.rearrange(targets, 'b h w -> b 1 h w')

        if transform_logits:
            inputs = F.softmax(inputs, dim=1)[:, [1]]

        # Get the predicted label
        y_pred = (inputs > 0.5).long()

        # Add background
        # TODO: this could be optional
        pred_background = (1 - y_pred).abs()
        y_pred = torch.cat((pred_background, y_pred), dim=1)

        true_background = (1 - targets).abs()
        y_true = torch.cat((true_background, targets), dim=1)

        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask

        pred_skeleton = self.soft_skeleton(y_pred.to(dtype=inputs.dtype))
        true_skeleton = self.soft_skeleton(y_true.to(dtype=inputs.dtype))

        topo_precision = self.precision_recall(pred_skeleton, y_true)
        topo_recall = self.precision_recall(true_skeleton, y_pred)

        cl_dice = 1.0 - 2.0 * (topo_precision * topo_recall) / (
            topo_precision + topo_recall
        )

        return cl_dice


class TverskyLoss(nn.Module):
    """Tversky loss."""

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.6,
        smooth: float = 1.0,
        transform_logits: bool = False,
        one_hot_targets: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
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
        dim: T.Optional[T.Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (probabilities), shaped (B, H, W) or (B, 1, H, W).
        targets
            Target labels, shaped (B, H, W) or (B, 1, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).

        Returns
        =======
        Tversky loss (float)
        """

        if dim is None:
            dim = (1, 2, 3)

        inputs, targets = self.preprocessor(
            inputs=inputs, targets=targets, mask=mask
        )

        if mask is not None:

            if len(mask.shape) == 3:
                mask = einops.rearrange(mask, 'b h w -> b 1 h w')

            inputs = inputs * mask
            targets = targets * mask

        tp = (inputs * targets).sum(dim=dim)
        fp = ((1 - targets) * inputs).sum(dim=dim)
        fn = (targets * (1 - inputs)).sum(dim=dim)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = 1.0 - tversky

        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss."""

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 0.8,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()

        self.gamma = gamma

        self.tversky_loss = TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        dim: T.Optional[T.Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Performs a single forward pass.

        Parameters
        ==========
        inputs
            Predictions from model (probabilities), shaped (B, H, W) or (B, 1, H, W).
        targets
            Target labels, shaped (B, H, W) or (B, 1, H, W).
        mask
            Values to mask (0) or keep (1), shaped (B, H, W) or (B, 1, H, W).

        Returns
        =======
        Focal Tversky loss (float)
        """

        tversky_loss = self.tversky_loss(
            inputs=inputs,
            targets=targets,
            mask=mask,
            dim=dim,
        )

        loss = torch.pow(tversky_loss, self.gamma)

        return loss.mean()
