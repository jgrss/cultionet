import typing as T

from . import topological
from ..models import model_utils

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torchmetrics


class LossPreprocessing(torch.nn.Module):
    def __init__(self, inputs_are_logits: bool, apply_transform: bool):
        super(LossPreprocessing, self).__init__()

        self.inputs_are_logits = inputs_are_logits
        self.apply_transform = apply_transform
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor = None
    ) -> T.Tuple[torch.Tensor, T.Union[torch.Tensor, None]]:
        """Forward pass to transform logits

        If logits are single-dimension then they are transformed by Sigmoid.
        If logits are multi-dimension then they are transformed by Softmax.
        """
        if self.inputs_are_logits:
            if targets is not None:
                if (len(targets.unique()) > inputs.size(1)) or (targets.unique().max()+1 > inputs.size(1)):
                    raise ValueError(
                        'The targets should be ordered values of equal length to the inputs 2nd dimension.'
                    )
            if self.apply_transform:
                if inputs.shape[1] == 1:
                    inputs = self.sigmoid(inputs)
                else:
                    inputs = F.softmax(inputs, dim=1, dtype=inputs.dtype)

            inputs = inputs.clip(0, 1)
            if targets is not None:
                targets = F.one_hot(
                    targets.contiguous().view(-1), inputs.shape[1]
                ).float()
        else:
            inputs = inputs.unsqueeze(1)
            targets = targets.unsqueeze(1)

        return inputs, targets


class TopologicalLoss(torch.nn.Module):
    """
    Reference:
        https://arxiv.org/abs/1906.05404
        https://arxiv.org/pdf/1906.05404.pdf
        https://github.com/HuXiaoling/TopoLoss/blob/5cb98177de50a3694f5886137ff7c6f55fd51493/topoloss_pytorch.py
    """
    def __init__(self):
        super(TopologicalLoss, self).__init__()

        self.gc = model_utils.GraphToConv()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        data: Data
    ) -> torch.Tensor:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        input_dims = inputs.shape[1]
        # Probabilities are ether Sigmoid or Softmax
        input_index = 0 if input_dims == 1 else 1

        inputs = self.gc(
            inputs, batch_size, height, width
        )
        targets = self.gc(
            targets.unsqueeze(1), batch_size, height, width
        )
        # Clone tensors before detaching from GPU
        inputs_clone = inputs.clone()
        targets_clone = targets.clone()

        topo_cp_weight_map = np.zeros(inputs_clone[:, input_index].shape, dtype='float32')
        topo_cp_ref_map = np.zeros(inputs_clone[:, input_index].shape, dtype='float32')
        topo_mask = np.zeros(inputs_clone[:, input_index].shape, dtype='uint8')

        # Detach from GPU for gudhi libary
        inputs_clone = inputs_clone[:, input_index].float().cpu().detach().numpy()
        targets_clone = targets_clone[:, 0].float().cpu().detach().numpy()

        pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = topological.critical_points(inputs_clone)
        pd_gt, __, __, pairs_lh_gt = topological.critical_points(targets_clone)

        if pairs_lh_pa and pairs_lh_gt:
            for batch in range(0, batch_size):
                if (pd_lh[batch].size > 0) and (pd_gt[batch].size > 0):
                    __, idx_holes_to_fix, idx_holes_to_remove = topological.compute_dgm_force(
                        pd_lh[batch], pd_gt[batch], pers_thresh=0.03
                    )
                    topo_cp_weight_map[batch], topo_cp_ref_map[batch], topo_mask[batch] = topological.set_topology_weights(
                        likelihood=inputs_clone[batch],
                        topo_cp_weight_map=topo_cp_weight_map[batch],
                        topo_cp_ref_map=topo_cp_ref_map[batch],
                        topo_mask=topo_mask[batch],
                        bcp_lh=bcp_lh[batch],
                        dcp_lh=dcp_lh[batch],
                        idx_holes_to_fix=idx_holes_to_fix,
                        idx_holes_to_remove=idx_holes_to_remove,
                        height=inputs.shape[-2],
                        width=inputs.shape[-1]
                    )

        topo_cp_weight_map = torch.tensor(
            topo_cp_weight_map, dtype=inputs.dtype, device=inputs.device
        )
        topo_cp_ref_map = torch.tensor(
            topo_cp_ref_map, dtype=inputs.dtype, device=inputs.device
        )
        topo_mask = torch.tensor(
            topo_mask, dtype=bool, device=inputs.device
        )
        if not topo_mask.any():
            topo_loss = (
                (
                    (
                        inputs[:, input_index] * topo_cp_weight_map
                    ) - topo_cp_ref_map
                ) ** 2
            )
        else:
            topo_loss = (
                (
                    (
                        inputs[:, input_index][topo_mask] * topo_cp_weight_map[topo_mask]
                    ) - topo_cp_ref_map[topo_mask]
                ) ** 2
            )

        return topo_loss.mean()


class TanimotoComplementLoss(torch.nn.Module):
    """Tanimoto distance loss

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
        targets_are_labels: bool = True
    ):
        super(TanimotoComplementLoss, self).__init__()

        self.smooth = smooth
        self.depth = depth
        self.targets_are_labels = targets_are_labels

        self.preprocessor = LossPreprocessing(
            inputs_are_logits=True,
            apply_transform=True
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities or labels).
            targets: Ground truth values.

        Returns:
            Tanimoto distance loss (float)
        """
        if self.targets_are_labels:
            # Discrete targets
            if inputs.shape[1] > 1:
                # Softmax and One-hot encoding
                inputs, targets = self.preprocessor(inputs, targets)

        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        length = inputs.shape[1]

        def tanimoto(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
            scale = 1.0 / self.depth
            tpl = (y * yhat).sum(dim=0)
            numerator = tpl + self.smooth
            sq_sum = (y**2 + yhat**2).sum(dim=0)
            denominator = torch.zeros(length, dtype=inputs.dtype).to(device=inputs.device)
            for d in range(0, self.depth):
                a = 2**d
                b = -(2.0 * a - 1.0)
                denominator = denominator + torch.reciprocal((a * sq_sum) + (b * tpl) + self.smooth)

            return numerator * denominator * scale

        score = tanimoto(targets, inputs)
        if inputs.shape[1] == 1:
            score = (score + tanimoto(1.0 - targets, 1.0 - inputs)) * 0.5

        return (1.0 - score).mean()


class TanimotoDistLoss(torch.nn.Module):
    """Tanimoto distance loss

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
    """
    def __init__(
        self,
        smooth: float = 1e-5,
        scale_pos_weight: T.Optional[bool] = False,
        transform_logits: T.Optional[bool] = True
    ):
        super(TanimotoDistLoss, self).__init__()

        self.smooth = smooth
        self.scale_pos_weight = scale_pos_weight
        self.transform_logits = transform_logits
        self.preprocessor = LossPreprocessing(
            inputs_are_logits=True,
            apply_transform=True
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: T.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model (probabilities, logits or labels).
            targets: Ground truth values.
            sample_weights: Weights for each sample/pixel.

        Returns:
            Tanimoto distance loss (float)
        """
        if self.transform_logits:
            if inputs.shape[1] == 1:
                inputs, __ = self.preprocessor(inputs)
            else:
                inputs, targets = self.preprocessor(inputs, targets)

        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(1)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        if sample_weights is None:
            sample_weights = 1.0

        def tanimoto_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y.to(dtype=yhat.dtype)
            if self.scale_pos_weight:
                counts = y.sum(dim=0)
                weights = counts.sum() / (counts.shape[0] * counts)
                new_weights = torch.where(
                    torch.isinf(weights),
                    torch.zeros_like(weights),
                    weights
                )
                weights = torch.where(
                    torch.isinf(weights),
                    torch.ones_like(weights) * new_weights.max(),
                    weights
                )
            else:
                weights = torch.ones(
                    inputs.shape[1], dtype=inputs.dtype, device=inputs.device
                )
            # Reduce
            tpl = ((yhat * y) * sample_weights).sum(dim=0)
            sq_sum = ((yhat**2 + y**2) * sample_weights).sum(dim=0)
            numerator = tpl * weights + self.smooth
            denominator = (sq_sum - tpl) * weights + self.smooth
            tanimoto = numerator / denominator
            loss = 1.0 - tanimoto

            return loss

        loss = tanimoto_loss(inputs, targets)
        if inputs.shape[1] == 1:
            compl_loss = tanimoto_loss(1.0 - inputs, 1.0 - targets)
            loss = (loss + compl_loss) * 0.5

        return loss.mean()


class CrossEntropyLoss(torch.nn.Module):
    """Cross entropy loss
    """
    def __init__(
        self,
        weight: T.Optional[torch.Tensor] = None,
        reduction: T.Optional[str] = 'mean',
        label_smoothing: T.Optional[float] = 0.1
    ):
        super(CrossEntropyLoss, self).__init__()

        self.loss_func = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
            label_smoothing=label_smoothing
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        return self.loss_func(
            inputs, targets
        )


class FocalLoss(torch.nn.Module):
    """Focal loss

    Reference:
        https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """
    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 2.0,
        weight: T.Optional[torch.Tensor] = None,
        label_smoothing: T.Optional[float] = 0.1
    ):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.preprocessor = LossPreprocessing(
            inputs_are_logits=True,
            apply_transform=True
        )
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            weight=weight,
            reduction='none',
            label_smoothing=label_smoothing
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        inputs, targets = self.preprocessor(inputs, targets)
        ce_loss = self.cross_entropy_loss(
            inputs, targets.half()
        )
        ce_exp = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - ce_exp)**self.gamma * ce_loss

        return focal_loss.mean()


class QuantileLoss(torch.nn.Module):
    """Loss function for quantile regression

    Reference:
        https://pytorch-forecasting.readthedocs.io/en/latest/_modules/pytorch_forecasting/metrics.html#QuantileLoss

    THE MIT License

    Copyright 2020 Jan Beitner
    """
    def __init__(
        self,
        quantiles: T.Tuple[float, float, float]
    ):
        super(QuantileLoss, self).__init__()

        self.quantiles = quantiles

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


class WeightedL1Loss(torch.nn.Module):
    """Weighted L1Loss loss
    """
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predictions from model.
            targets: Ground truth values.

        Returns:
            Loss (float)
        """
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        mae = torch.abs(
            inputs - targets
        )
        weight = inputs + targets
        loss = (mae * weight).mean()

        return loss


class MSELoss(torch.nn.Module):
    """MSE loss
    """
    def __init__(self):
        super(MSELoss, self).__init__()

        self.loss_func = torch.nn.MSELoss()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
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


class BoundaryLoss(torch.nn.Module):
    """Boundary (surface) loss

    Reference:
        https://github.com/LIVIAETS/boundary-loss
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

        self.gc = model_utils.GraphToConv()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, data: Data
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predicted probabilities.
            targets: Ground truth inverse distance transform, where distances
                along edges are 1.
            data: Data object used to extract dimensions.

        Returns:
            Loss (float)
        """
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        inputs = self.gc(
            inputs.unsqueeze(1), batch_size, height, width
        )
        targets = self.gc(
            targets.unsqueeze(1), batch_size, height, width
        )

        return torch.einsum('bchw, bchw -> bchw', inputs, targets).mean()


class MultiScaleSSIMLoss(torch.nn.Module):
    """Multi-scale Structural Similarity Index Measure loss
    """
    def __init__(self):
        super(MultiScaleSSIMLoss, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.msssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
            gaussian_kernel=False,
            kernel_size=3,
            data_range=1.0,
            k1=1e-4,
            k2=9e-4
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, data: Data
    ) -> torch.Tensor:
        """Performs a single forward pass

        Args:
            inputs: Predicted probabilities.
            targets: Ground truth inverse distance transform, where distances
                along edges are 1.
            data: Data object used to extract dimensions.

        Returns:
            Loss (float)
        """
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        inputs = self.gc(
            inputs.unsqueeze(1), batch_size, height, width
        )
        targets = self.gc(
            targets.unsqueeze(1), batch_size, height, width
        ).to(dtype=inputs.dtype)

        loss = 1.0 - self.msssim(inputs, targets)

        return loss
