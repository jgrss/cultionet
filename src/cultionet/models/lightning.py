import logging
import typing as T
import warnings
from pathlib import Path

import einops
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
from torch.optim import lr_scheduler as optim_lr_scheduler

from .. import losses as cnetlosses
from .. import nn as cunn
from ..data.data import Data
from ..enums import (
    AttentionTypes,
    InferenceNames,
    LearningRateSchedulers,
    LossTypes,
    ModelNames,
    ModelTypes,
    ResBlockTypes,
    ValidationNames,
)
from ..layers.weights import init_conv_weights
from .cultionet import CultioNet
from .maskcrnn import (
    BFasterRCNN,
    ReshapeMaskData,
    create_mask_targets,
    mask_2d_to_3d,
    mask_to_polygon,
    nms_masks,
)

warnings.filterwarnings("ignore")
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False
logging.getLogger("lightning").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("high")


LOSS_DICT = {
    LossTypes.BOUNDARY: {
        "classification": cnetlosses.BoundaryLoss(),
    },
    LossTypes.CLASS_BALANCED_MSE: {
        "classification": cnetlosses.ClassBalancedMSELoss(),
    },
    LossTypes.CROSS_ENTROPY: {
        "classification": cnetlosses.CrossEntropyLoss(),
    },
    LossTypes.LOG_COSH: {
        "regression": cnetlosses.LogCoshLoss(),
    },
    LossTypes.TANIMOTO_COMPLEMENT: {
        "classification": cnetlosses.TanimotoComplementLoss(),
        "regression": cnetlosses.TanimotoComplementLoss(
            transform_logits=False,
            one_hot_targets=False,
        ),
    },
    LossTypes.TANIMOTO: {
        "classification": cnetlosses.TanimotoDistLoss(),
        "regression": cnetlosses.TanimotoDistLoss(
            transform_logits=False,
            one_hot_targets=False,
        ),
    },
    LossTypes.TANIMOTO_COMBINED: {
        "classification": cnetlosses.CombinedLoss(
            losses=[
                cnetlosses.TanimotoDistLoss(),
                cnetlosses.TanimotoComplementLoss(),
            ],
        ),
        "regression": cnetlosses.CombinedLoss(
            losses=[
                cnetlosses.TanimotoDistLoss(
                    transform_logits=False,
                    one_hot_targets=False,
                ),
                cnetlosses.TanimotoComplementLoss(
                    transform_logits=False,
                    one_hot_targets=False,
                ),
            ],
        ),
    },
    LossTypes.TVERSKY: {
        "classification": cnetlosses.TverskyLoss(),
    },
    LossTypes.FOCAL_TVERSKY: {
        "classification": cnetlosses.FocalTverskyLoss(),
    },
}


class LightningModuleMixin(LightningModule):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """Performs a single model forward pass.

        Returns:
            distance: Normalized distance transform (from boundaries), [0,1]. Shaped (B, 1, H, W).
            edge: Edge|non-edge predictions, logits or probabilities. Shaped (B, 1, H, W).
            crop: Logits of crop|non-crop. Shaped (B, C, H, W).
        """
        return self.cultionet_model(batch)

    @property
    def cultionet_model(self) -> CultioNet:
        """Get the network model name."""
        return getattr(self, self.model_attr)

    @staticmethod
    def get_cuda_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"{t * 1e-6:.02f}MB", f"{r * 1e-6:.02f}MB", f"{a * 1e-6:.02f}MB")

    def probas_to_labels(
        self, x: torch.Tensor, thresh: float = 0.5
    ) -> torch.Tensor:
        """Converts probabilities to class labels."""

        if x.shape[1] == 1:
            labels = x.gt(thresh).squeeze(dim=1).long()
        else:
            labels = x.argmax(dim=1).long()

        return labels

    def logits_to_probas(self, x: torch.Tensor) -> T.Union[None, torch.Tensor]:
        """Transforms logits to probabilities."""

        if x is not None:
            if x.shape[1] > 1:
                x = F.softmax(x, dim=1, dtype=x.dtype)
            else:
                # Single-dimension inputs are sigmoid probabilities
                x = F.sigmoid(x)

            x = x.clip(0, 1)

        return x

    def predict_step(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """A prediction step for Lightning."""

        predictions = self.forward(batch, batch_idx=batch_idx)

        if self.train_maskrcnn:
            # Apply a forward pass on Mask RCNN
            mask_data = self.mask_rcnn_forward(
                batch=batch,
                predictions=predictions,
                mode='predict',
            )
            predictions.update(pred_df=mask_data['pred_df'])

        return predictions

    @torch.no_grad
    def get_true_labels(
        self, batch: Data, crop_type: torch.Tensor = None
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        """Gets true labels from the data batch."""

        true_edge = torch.where(batch.y == self.edge_class, 1, 0).long()
        # Recode all crop classes to 1, otherwise 0
        true_crop = torch.where(
            (batch.y > 0) & (batch.y < self.edge_class), 1, 0
        ).long()
        # Same as above, with additional edge class
        # Non-crop = 0
        # Crop | Edge = 1
        true_crop_and_edge = torch.where(batch.y > 0, 1, 0).long()
        # Same as above, with additional edge class
        # Non-crop = 0
        # Crop = 1
        # Edge = 2
        true_crop_or_edge = torch.where(
            (batch.y > 0) & (batch.y < self.edge_class),
            1,
            torch.where(batch.y == self.edge_class, 2, 0),
        ).long()
        true_crop_type = None
        if crop_type is not None:
            # Leave all crop classes as they are
            true_crop_type = torch.where(
                batch.y == self.edge_class, 0, batch.y
            ).long()

        # Weak supervision mask
        mask = None
        if batch.y.min() == -1:
            mask = torch.where(batch.y == -1, 0, 1).to(
                dtype=torch.long, device=batch.y.device
            )
            mask = einops.rearrange(mask, 'b h w -> b 1 h w')

        return {
            ValidationNames.TRUE_EDGE: true_edge,
            ValidationNames.TRUE_CROP: true_crop,
            ValidationNames.TRUE_CROP_AND_EDGE: true_crop_and_edge,
            ValidationNames.TRUE_CROP_OR_EDGE: true_crop_or_edge,
            ValidationNames.TRUE_CROP_TYPE: true_crop_type,
            ValidationNames.MASK: mask,
        }

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: T.Dict[str, torch.Tensor],
    ):
        """Calculates the loss.

        Returns:
            Total loss
        """

        weights = {
            InferenceNames.DISTANCE: 1.0,
            InferenceNames.EDGE: 1.0,
            InferenceNames.CROP: 1.0,
        }

        with torch.no_grad():
            true_labels_dict = self.get_true_labels(
                batch, crop_type=predictions.get(InferenceNames.CROP_TYPE)
            )

            true_edge_distance = torch.where(
                true_labels_dict[ValidationNames.TRUE_EDGE] == 1,
                1,
                torch.where(
                    true_labels_dict[ValidationNames.TRUE_CROP] == 1,
                    (1.0 - batch.bdist) ** 20.0,
                    0,
                ),
            )
            true_crop_distance = torch.where(
                true_labels_dict[ValidationNames.TRUE_CROP] != 1,
                0,
                1.0 - true_edge_distance,
            )

            true_edge_distance = einops.rearrange(
                true_edge_distance, 'b h w -> b 1 h w'
            )
            true_crop_distance = einops.rearrange(
                true_crop_distance, 'b h w -> b 1 h w'
            )

        loss = 0.0

        ##########################
        # Temporal encoding losses
        ##########################

        if predictions[InferenceNames.CLASSES_L2] is not None:
            # Temporal encoding level 2 loss (non-crop=0; crop|edge=1)
            classes_l2_loss = F.cross_entropy(
                predictions[InferenceNames.CLASSES_L2],
                true_labels_dict[ValidationNames.TRUE_CROP_AND_EDGE],
                weight=self.crop_and_edge_weights,
                reduction='none'
                if true_labels_dict[ValidationNames.MASK] is not None
                else 'mean',
            )

            if true_labels_dict[ValidationNames.MASK] is not None:
                classes_l2_loss = classes_l2_loss * einops.rearrange(
                    true_labels_dict[ValidationNames.MASK], 'b 1 h w -> b h w'
                )
                masked_weights = self.crop_and_edge_weights[
                    true_labels_dict[ValidationNames.TRUE_CROP_AND_EDGE]
                ] * einops.rearrange(
                    true_labels_dict[ValidationNames.MASK], 'b 1 h w -> b h w'
                )
                classes_l2_loss = classes_l2_loss.sum() / masked_weights.sum()

            weights[InferenceNames.CLASSES_L2] = 0.01
            loss = loss + classes_l2_loss * weights[InferenceNames.CLASSES_L2]

        if predictions[InferenceNames.CLASSES_L3] is not None:
            # Temporal encoding final loss (non-crop=0; crop=1; edge=2)
            classes_last_loss = F.cross_entropy(
                predictions[InferenceNames.CLASSES_L3],
                true_labels_dict[ValidationNames.TRUE_CROP_OR_EDGE],
                weight=self.crop_or_edge_weights,
                reduction='none'
                if true_labels_dict[ValidationNames.MASK] is not None
                else 'mean',
            )

            if true_labels_dict[ValidationNames.MASK] is not None:
                classes_last_loss = classes_last_loss * einops.rearrange(
                    true_labels_dict[ValidationNames.MASK], 'b 1 h w -> b h w'
                )
                masked_weights = self.crop_or_edge_weights[
                    true_labels_dict[ValidationNames.TRUE_CROP_OR_EDGE]
                ] * einops.rearrange(
                    true_labels_dict[ValidationNames.MASK], 'b 1 h w -> b h w'
                )
                classes_last_loss = (
                    classes_last_loss.sum() / masked_weights.sum()
                )

            weights[InferenceNames.CLASSES_L3] = 0.1
            loss = (
                loss + classes_last_loss * weights[InferenceNames.CLASSES_L3]
            )

        #############
        # Main losses
        #############

        # Distance transform loss
        dist_loss = self.reg_loss(
            # Inputs are 0-1 continuous
            inputs=predictions[InferenceNames.DISTANCE],
            # True data are 0-1 continuous
            targets=batch.bdist,
            mask=true_labels_dict[ValidationNames.MASK],
        )
        loss = loss + dist_loss * weights[InferenceNames.DISTANCE]

        # Edge loss
        edge_loss = self.cls_loss(
            # Inputs are single-layer logits or probabilities
            inputs=predictions[InferenceNames.EDGE],
            # True data are 0|1
            targets=true_labels_dict[ValidationNames.TRUE_EDGE],
            mask=true_labels_dict[ValidationNames.MASK],
        )
        loss = loss + edge_loss * weights[InferenceNames.EDGE]

        # Crop mask loss
        crop_loss = self.cls_loss(
            # Inputs are 2-layer logits or probabilities
            inputs=predictions[InferenceNames.CROP],
            # True data are 0|1
            targets=true_labels_dict[ValidationNames.TRUE_CROP],
            mask=true_labels_dict[ValidationNames.MASK],
        )
        loss = loss + crop_loss * weights[InferenceNames.CROP]

        # Boundary loss for edges
        edge_boundary_loss = self.boundary_loss(
            # Inputs are probabilities
            inputs=predictions[InferenceNames.EDGE],
            # True data are 0-1 continuous
            targets=true_edge_distance,
            mask=true_labels_dict[ValidationNames.MASK],
        )
        weights["edge_boundary_loss"] = 0.1
        loss = loss + edge_boundary_loss * weights["edge_boundary_loss"]

        # Boundary loss for crop
        crop_boundary_loss = self.boundary_loss(
            # Inputs are probabilities
            inputs=predictions[InferenceNames.CROP],
            # True data are 0-1 continuous
            targets=true_crop_distance,
            mask=true_labels_dict[ValidationNames.MASK],
        )
        weights["crop_boundary_loss"] = 0.1
        loss = loss + crop_boundary_loss * weights["crop_boundary_loss"]

        loss_report = {
            "dloss": dist_loss,
            "eloss": edge_loss,
            "closs": crop_loss,
            "ebloss": edge_boundary_loss,
            "cbloss": crop_boundary_loss,
        }

        return loss / sum(weights.values()), loss_report

    def mask_rcnn_forward(
        self,
        batch: Data,
        predictions: T.Dict[str, torch.Tensor],
        mode: str,
    ) -> dict:
        """Mask-RCNN forward."""

        assert mode in (
            'eval',
            'predict',
            'train',
        ), "Choose 'eval', 'predict', or 'train' mode."

        if mode in (
            'eval',
            'train',
        ):
            # NOTE: Mask-RCNN does not return loss in eval() mode
            self.mask_rcnn_model.train()
        else:
            self.mask_rcnn_model.eval()

        if mode == 'eval':
            # Turn off layers
            for module in self.mask_rcnn_model.modules():
                if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
                    module.eval()

        # Iterate over the batches and create box masks
        mask_x = []
        mask_y = []
        for bidx in range(batch.x.shape[0]):
            mask_data = ReshapeMaskData.prepare(
                x=torch.cat(
                    (
                        predictions['dist'][bidx].detach(),
                        predictions[InferenceNames.EDGE][bidx].detach(),
                        einops.rearrange(
                            predictions[InferenceNames.CROP][bidx, 1].detach(),
                            'h w -> 1 h w',
                        ),
                    ),
                    dim=0,
                ),
                y=None if mode == 'predict' else batch.y[bidx],
            )

            if mode == 'predict':
                mask_x.append(mask_data.x[0])
            else:
                mask_data.masks = mask_2d_to_3d(mask_data.masks)
                mask_data.x, mask_data.masks = create_mask_targets(
                    mask_data.x, mask_data.masks
                )
                mask_x.append(mask_data.x[0])
                mask_y.append(mask_data.masks)

        # Apply a forward pass on Mask RCNN
        if mode in (
            'eval',
            'predict',
        ):
            with torch.no_grad():
                mask_outputs = self.mask_rcnn_model(
                    x=mask_x,
                    y=None if mode == 'predict' else mask_y,
                )
        else:
            mask_outputs = self.mask_rcnn_model(x=mask_x, y=mask_y)

        mask_loss = None
        pred_df = None
        if mode in (
            'eval',
            'train',
        ):
            mask_loss = sum([loss for loss in mask_outputs.values()]) / len(
                mask_outputs
            )
        else:
            pred_df = []
            for bidx, batch_output in enumerate(mask_outputs):
                batch_pred_masks = nms_masks(
                    boxes=batch_output['boxes'],
                    scores=batch_output['scores'],
                    masks=batch_output['masks'],
                    iou_threshold=0.7,
                    size=batch.x.shape[-2:],
                )
                pred_frame = mask_to_polygon(
                    batch_pred_masks,
                    image_left=batch.left[0],
                    image_top=batch.top[0],
                    row_off=batch.window_row_off[bidx],
                    col_off=batch.window_col_off[bidx],
                    resolution=10.0,
                    # FIXME: res is currently passing None
                    # resolution=batch.res[0],
                    padding=batch.padding[0],
                )
                pred_df.append(pred_frame)

            pred_df = pd.concat(pred_df)

        return {
            'outputs': mask_outputs,
            'loss': mask_loss,
            'pred_df': pred_df,
        }

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step and logs training step metrics."""

        predictions = self(batch)

        loss, _ = self.calc_loss(batch, predictions)

        if self.train_maskrcnn:
            # Apply a forward pass on Mask RCNN
            mask_data = self.mask_rcnn_forward(
                batch=batch,
                predictions=predictions,
                mode='train',
            )

            loss = loss + mask_data['loss']

        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_samples,
        )

        return loss

    def _shared_eval_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Evaluation step shared between validation and testing."""

        # Forward pass to get predictions
        predictions = self(batch)

        # Calculate the loss
        loss, loss_report = self.calc_loss(batch, predictions)

        # Convert probabilities to class labels
        edge_ypred = self.probas_to_labels(predictions[InferenceNames.EDGE])
        crop_ypred = self.probas_to_labels(predictions[InferenceNames.CROP])

        # Get the true edge and crop labels
        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions.get(InferenceNames.CROP_TYPE)
        )

        if self.train_maskrcnn:
            # Apply a forward pass on Mask RCNN
            mask_data = self.mask_rcnn_forward(
                batch=batch,
                predictions=predictions,
                mode='eval',
            )

            loss = loss + mask_data['loss']

        if true_labels_dict[ValidationNames.MASK] is not None:
            # Valid sample = True; Invalid sample = False
            labels_bool_mask = true_labels_dict[ValidationNames.MASK].to(
                dtype=torch.bool
            )
            predictions[InferenceNames.DISTANCE] = torch.masked_select(
                predictions[InferenceNames.DISTANCE], labels_bool_mask
            )
            bdist = torch.masked_select(
                batch.bdist, labels_bool_mask.squeeze(dim=1)
            )

        else:
            predictions[InferenceNames.DISTANCE] = einops.rearrange(
                predictions[InferenceNames.DISTANCE], 'b 1 h w -> (b h w)'
            )
            bdist = einops.rearrange(batch.bdist, 'b h w -> (b h w)')

        dist_score_args = (predictions[InferenceNames.DISTANCE], bdist)

        dist_mae = self.mae_scorer(*dist_score_args)
        dist_mse = self.mse_scorer(*dist_score_args)

        if true_labels_dict[ValidationNames.MASK] is not None:
            edge_ypred = torch.masked_select(
                edge_ypred, labels_bool_mask.squeeze(dim=1)
            )
            crop_ypred = torch.masked_select(
                crop_ypred, labels_bool_mask.squeeze(dim=1)
            )
            true_labels_dict[ValidationNames.TRUE_EDGE] = torch.masked_select(
                true_labels_dict[ValidationNames.TRUE_EDGE],
                labels_bool_mask.squeeze(dim=1),
            )
            true_labels_dict[ValidationNames.TRUE_CROP] = torch.masked_select(
                true_labels_dict[ValidationNames.TRUE_CROP],
                labels_bool_mask.squeeze(dim=1),
            )

        else:
            edge_ypred = einops.rearrange(edge_ypred, 'b h w -> (b h w)')
            crop_ypred = einops.rearrange(crop_ypred, 'b h w -> (b h w)')
            true_labels_dict[ValidationNames.TRUE_EDGE] = einops.rearrange(
                true_labels_dict[ValidationNames.TRUE_EDGE], 'b h w -> (b h w)'
            )
            true_labels_dict[ValidationNames.TRUE_CROP] = einops.rearrange(
                true_labels_dict[ValidationNames.TRUE_CROP], 'b h w -> (b h w)'
            )

        # Scorer input args
        edge_score_args = (
            edge_ypred,
            true_labels_dict[ValidationNames.TRUE_EDGE],
        )
        crop_score_args = (
            crop_ypred,
            true_labels_dict[ValidationNames.TRUE_CROP],
        )

        # Fβ-score
        edge_fscore = self.f_beta_scorer(*edge_score_args)
        crop_fscore = self.f_beta_scorer(*crop_score_args)

        # MCC
        edge_mcc = self.mcc_scorer(*edge_score_args)
        crop_mcc = self.mcc_scorer(*crop_score_args)

        total_score = (
            loss
            + (1.0 - edge_fscore)
            + (1.0 - crop_fscore)
            + dist_mae
            + (1.0 - edge_mcc.clamp_min(0))
            + (1.0 - crop_mcc.clamp_min(0))
        )

        metrics = {
            "loss": loss,
            "dist_mae": dist_mae,
            "dist_mse": dist_mse,
            "edge_f1": edge_fscore,
            "crop_f1": crop_fscore,
            "edge_mcc": edge_mcc,
            "crop_mcc": crop_mcc,
            "score": total_score,
        }

        metrics.update(loss_report)

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step."""

        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            "vef1": eval_metrics["edge_f1"],
            "vcf1": eval_metrics["crop_f1"],
            "vmae": eval_metrics["dist_mae"],
            "val_score": eval_metrics["score"],
            "val_loss": eval_metrics["loss"],
            "val_dloss": eval_metrics["dloss"],
            "val_eloss": eval_metrics["eloss"],
            "val_closs": eval_metrics["closs"],
            "val_ebloss": eval_metrics["ebloss"],
            "val_cbloss": eval_metrics["cbloss"],
        }

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_samples,
        )

        if self.save_batch_val_metrics:
            self._save_batch_metrics(metrics, self.current_epoch, batch)

        return metrics

    def _save_batch_metrics(
        self, metrics: T.Dict[str, torch.Tensor], epoch: int, batch: Data
    ) -> None:
        """Saves batch metrics to a parquet file."""
        if not self.trainer.sanity_checking:
            write_metrics = {
                "epoch": [epoch] * len(batch.train_id),
                "train_ids": batch.train_id,
            }
            for k, v in metrics.items():
                write_metrics[k] = [float(v)] * len(batch.train_id)
            if self.logger.save_dir is not None:
                metrics_file = (
                    Path(self.logger.save_dir) / "batch_metrics.parquet"
                )
                if not metrics_file.is_file():
                    df = pd.DataFrame(write_metrics)
                    df.to_parquet(metrics_file)
                else:
                    df_new = pd.DataFrame(write_metrics)
                    df = pd.read_parquet(metrics_file)
                    df = pd.concat((df, df_new), axis=0)
                    df.to_parquet(metrics_file)

    def test_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one test step."""

        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            "test_loss": eval_metrics["loss"],
            "tmae": eval_metrics["dist_mae"],
            "tmse": eval_metrics["dist_mse"],
            "tef1": eval_metrics["edge_f1"],
            "tcf1": eval_metrics["crop_f1"],
            "temcc": eval_metrics["edge_mcc"],
            "tcmcc": eval_metrics["crop_mcc"],
            "tedice": eval_metrics["edge_dice"],
            "tcdice": eval_metrics["crop_dice"],
            "tejaccard": eval_metrics["edge_jaccard"],
            "tcjaccard": eval_metrics["crop_jaccard"],
            "test_score": eval_metrics["score"],
        }
        if "crop_type_f1" in eval_metrics:
            metrics["tctf1"] = eval_metrics["crop_type_f1"]

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        """The fβ value.

        To put equal weight on precision and recall, set to 1. To emphasize
        minimizing false positives, set to <1. To emphasize minimizing false
        negatives, set to >1.
        """

        self.mae_scorer = torchmetrics.MeanAbsoluteError()
        self.mse_scorer = torchmetrics.MeanSquaredError()
        self.f_beta_scorer = torchmetrics.FBetaScore(
            task="multiclass", num_classes=2, beta=2.0
        )
        self.mcc_scorer = torchmetrics.MatthewsCorrCoef(
            task="multiclass", num_classes=2
        )

    def calc_weights(self, counts: torch.Tensor) -> torch.Tensor:
        """Calculates class weights."""

        num_samples = counts.sum()
        num_classes = len(counts)
        class_weights = num_samples / (num_classes * counts)
        weights = torch.nan_to_num(class_weights, nan=0, neginf=0, posinf=0)

        return weights

    def configure_loss(self):
        """Configures loss methods."""

        # # Weights for crop AND edge
        # crop_and_edge_counts = torch.zeros(2, device=self.class_counts.device)
        # crop_and_edge_counts[0] = self.class_counts[0]
        # crop_and_edge_counts[1] = self.class_counts[1:].sum()
        # self.crop_and_edge_weights = self.calc_weights(crop_and_edge_counts)

        # # Weights for crop OR edge
        # self.crop_or_edge_weights = self.calc_weights(self.class_counts)

        # # Weights for crop
        # crop_counts = torch.zeros(2, device=self.class_counts.device)
        # crop_counts[0] = self.class_counts[0]
        # crop_counts[1] = self.class_counts[1]
        # self.crop_weights = self.calc_weights(crop_counts)

        # Main loss
        self.reg_loss = LOSS_DICT[self.loss_name].get("regression")
        self.cls_loss = LOSS_DICT[self.loss_name].get("classification")

        # Boundary loss
        self.boundary_loss = LOSS_DICT[LossTypes.BOUNDARY].get(
            "classification"
        )

    def configure_optimizers(self):
        """Configures optimizers."""

        params_list = list(self.cultionet_model.parameters())
        interval = 'epoch'
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_list,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.eps,
                betas=(0.9, 0.98),
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                params_list,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise NameError("Choose either 'AdamW' or 'SGD'.")

        if self.lr_scheduler == LearningRateSchedulers.COSINE_ANNEALING_LR:
            model_lr_scheduler = optim_lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-5, last_epoch=-1
            )
        elif self.lr_scheduler == LearningRateSchedulers.EXPONENTIAL_LR:
            model_lr_scheduler = optim_lr_scheduler.ExponentialLR(
                optimizer, gamma=0.5
            )
        elif self.lr_scheduler == LearningRateSchedulers.ONE_CYCLE_LR:
            model_lr_scheduler = optim_lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.trainer.estimated_stepping_batches,
            )
            interval = 'step'
        elif self.lr_scheduler == LearningRateSchedulers.STEP_LR:
            model_lr_scheduler = optim_lr_scheduler.StepLR(
                optimizer, step_size=self.steplr_step_size, gamma=0.5
            )
        else:
            raise NameError(
                "The learning rate scheduler is not implemented in Cultionet."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": model_lr_scheduler,
                "name": "lr_sch",
                "monitor": "val_score",
                "interval": interval,
                "frequency": 1,
            },
        }


class CultionetLitTransferModel(LightningModuleMixin):
    """Transfer learning module for Cultionet."""

    def __init__(
        self,
        pretrained_ckpt_file: T.Union[Path, str],
        in_channels: int,
        in_time: int,
        num_classes: int = 2,
        hidden_channels: int = 64,
        model_type: str = ModelTypes.TOWERUNET,
        dropout: float = 0.2,
        activation_type: str = "SiLU",
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.NATTEN,
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO_COMPLEMENT,
        learning_rate: float = 0.01,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 1e-3,
        eps: float = 1e-4,
        ckpt_name: str = ModelNames.CKPT_TRANSFER_NAME.replace(".ckpt", ""),
        model_name: str = "cultionet_transfer",
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        class_counts: T.Optional[torch.Tensor] = None,
        edge_class: T.Optional[int] = None,
        scale_pos_weight: bool = False,
        save_batch_val_metrics: bool = False,
        finetune: T.Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer = optimizer
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.steplr_step_size = steplr_step_size
        self.weight_decay = weight_decay
        self.eps = eps
        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_time = in_time
        self.class_counts = class_counts
        self.scale_pos_weight = scale_pos_weight
        self.save_batch_val_metrics = save_batch_val_metrics
        self.finetune = finetune
        self.train_maskrcnn = None

        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.cultionet_model = CultionetLitModel.load_from_checkpoint(
            checkpoint_path=str(pretrained_ckpt_file)
        ).cultionet_model

        if self.finetune != "all":

            # Freeze all parameters if not finetuning the full model
            self.freeze(self.cultionet_model)

            if self.finetune == "fc":
                # Unfreeze fully connected layers
                for name, param in self.cultionet_model.named_parameters():
                    if name.startswith("mask_model.final_"):
                        param.requires_grad = True

            else:

                # Update the post-UNet layer with trainable parameters
                mask_model_final_a = cunn.TowerUNetFinal(
                    in_channels=self.cultionet_model.mask_model.final_a.in_channels,
                    num_classes=self.cultionet_model.mask_model.final_a.num_classes,
                    activation_type=activation_type,
                )
                mask_model_final_a.apply(init_conv_weights)
                self.cultionet_model.mask_model.final_a = mask_model_final_a

                mask_model_final_b = cunn.TowerUNetFinal(
                    in_channels=self.cultionet_model.mask_model.final_b.in_channels,
                    num_classes=self.cultionet_model.mask_model.final_b.num_classes,
                    activation_type=activation_type,
                    resample_factor=2,
                )
                mask_model_final_b.apply(init_conv_weights)
                self.cultionet_model.mask_model.final_b = mask_model_final_b

                mask_model_final_c = cunn.TowerUNetFinal(
                    in_channels=self.cultionet_model.mask_model.final_c.in_channels,
                    num_classes=self.cultionet_model.mask_model.final_c.num_classes,
                    activation_type=activation_type,
                    resample_factor=4,
                )
                mask_model_final_c.apply(init_conv_weights)
                self.cultionet_model.mask_model.final_c = mask_model_final_c

                mask_model_final_combine = cunn.TowerUNetFinalCombine(
                    num_classes=self.cultionet_model.mask_model.final_combine.num_classes,
                    edge_activation=self.cultionet_model.mask_model.final_combine.edge_activation,
                    mask_activation=self.cultionet_model.mask_model.final_combine.mask_activation,
                )
                mask_model_final_combine.apply(init_conv_weights)
                self.cultionet_model.mask_model.final_combine = (
                    mask_model_final_combine
                )

        self.model_attr = f"{model_name}_{model_type}"
        setattr(
            self,
            self.model_attr,
            self.cultionet_model,
        )

        self.configure_loss()
        self.configure_scorer()

    @property
    def is_transfer_model(self) -> bool:
        return True

    def freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def unfreeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

        return layer


class CultionetLitModel(LightningModuleMixin):
    def __init__(
        self,
        in_channels: int,
        in_time: int,
        num_classes: int = 2,
        hidden_channels: int = 64,
        model_type: str = ModelTypes.TOWERUNET,
        dropout: float = 0.2,
        activation_type: str = "SiLU",
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.NATTEN,
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO_COMPLEMENT,
        learning_rate: float = 0.01,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 1e-3,
        eps: float = 1e-4,
        ckpt_name: str = "last",
        model_name: str = "cultionet",
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        class_counts: T.Optional[torch.Tensor] = None,
        edge_class: T.Optional[int] = None,
        scale_pos_weight: bool = False,
        save_batch_val_metrics: bool = False,
        train_maskrcnn: bool = False,
    ):
        """Lightning model."""

        super().__init__()

        self.save_hyperparameters()

        self.optimizer = optimizer
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.steplr_step_size = steplr_step_size
        self.weight_decay = weight_decay
        self.eps = eps
        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_time = in_time
        self.class_counts = class_counts
        self.scale_pos_weight = scale_pos_weight
        self.save_batch_val_metrics = save_batch_val_metrics
        self.train_maskrcnn = train_maskrcnn

        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        if self.train_maskrcnn:
            warnings.warn(
                'RCNN in cultionet is experimental and not well tested.'
            )

            self.mask_rcnn_model = BFasterRCNN(
                in_channels=3,
                out_channels=hidden_channels * 2,
                num_classes=2,  # non-cropland and cropland
                # sizes=(16, 32, 64, 128, 256),
                # aspect_ratios=(0.5, 1.0, 3.0,),
                trainable_backbone_layers=1,
                min_image_size=256,
                max_image_size=256,
            )

        self.model_attr = f"{model_name}_{model_type}"
        setattr(
            self,
            self.model_attr,
            CultioNet(
                in_channels=in_channels,
                in_time=in_time,
                hidden_channels=hidden_channels,
                num_classes=self.num_classes,
                model_type=model_type,
                dropout=dropout,
                activation_type=activation_type,
                dilations=dilations,
                res_block_type=res_block_type,
                attention_weights=attention_weights,
                pool_by_max=pool_by_max,
                repeat_resa_kernel=repeat_resa_kernel,
                batchnorm_first=batchnorm_first,
            ),
        )

        self.configure_loss()
        self.configure_scorer()

    @property
    def is_transfer_model(self) -> bool:
        return False

    # def on_train_epoch_start(self):
    #     # Get the current learning rate from the optimizer
    #     weight_decay = self.optimizers().optimizer.param_groups[0]['weight_decay']
    #     if (weight_decay != self.weight_decay) or (eps != self.eps):
    #         self.configure_optimizers()
