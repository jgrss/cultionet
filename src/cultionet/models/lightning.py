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
    LearningRateSchedulers,
    LossTypes,
    ModelNames,
    ModelTypes,
    ResBlockTypes,
)
from ..layers.weights import init_attention_weights, init_conv_weights
from ..models.temporal_transformer import TemporalTransformerFinal
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

torch.set_float32_matmul_precision("medium")


class LightningModuleMixin(LightningModule):
    def __init__(self):
        super(LightningModuleMixin, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, training: bool = True, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """Performs a single model forward pass.

        Returns:
            distance: Normalized distance transform (from boundaries), [0,1].
            edge: Probabilities of edge|non-edge, [0,1].
            crop: Logits of crop|non-crop.
        """
        return self.cultionet_model(batch, training=training)

    @property
    def cultionet_model(self) -> CultioNet:
        return getattr(self, self.model_attr)

    @staticmethod
    def get_cuda_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"{t * 1e-6:.02f}MB", f"{r * 1e-6:.02f}MB", f"{a * 1e-6:.02f}MB")

    def softmax(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return F.softmax(x, dim=dim, dtype=x.dtype)

    def probas_to_labels(
        self, x: torch.Tensor, thresh: float = 0.5
    ) -> torch.Tensor:
        if x.shape[1] == 1:
            labels = x.gt(thresh).squeeze(dim=1).long()
        else:
            labels = x.argmax(dim=1).long()

        return labels

    def logits_to_probas(self, x: torch.Tensor) -> T.Union[None, torch.Tensor]:
        if x is not None:
            # Single-dimension inputs are sigmoid probabilities
            if x.shape[1] > 1:
                # Transform logits to probabilities
                x = self.softmax(x)
            else:
                x = self.sigmoid(x)
            x = x.clip(0, 1)

        return x

    def predict_step(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """A prediction step for Lightning."""
        with torch.no_grad():
            predictions = self.forward(
                batch, training=False, batch_idx=batch_idx
            )

            if self.train_maskrcnn:
                # Apply a forward pass on Mask RCNN
                mask_data = self.mask_rcnn_forward(
                    batch=batch,
                    predictions=predictions,
                    mode='predict',
                )
                predictions.update(pred_df=mask_data['pred_df'])

        return predictions

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
                dtype=torch.uint8, device=batch.y.device
            )
            mask = einops.rearrange(mask, 'b h w -> b 1 h w')

        return {
            "true_edge": true_edge,
            "true_crop": true_crop,
            "true_crop_and_edge": true_crop_and_edge,
            "true_crop_or_edge": true_crop_or_edge,
            "true_crop_type": true_crop_type,
            "mask": mask,
        }

    # def on_validation_epoch_end(self, *args, **kwargs):
    #     """Save the model on validation end."""
    #     if self.logger.save_dir is not None:
    #         model_file = Path(self.logger.save_dir) / f"{self.model_name}.pt"
    #         if model_file.is_file():
    #             model_file.unlink()
    #         torch.save(self.state_dict(), model_file)

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
            "l2": 0.25,
            "l3": 0.5,
            "dist_loss": 1.0,
            "edge_loss": 1.0,
            "crop_loss": 1.0,
        }

        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions.get("crop_type")
        )

        loss = 0.0

        ##########################
        # Temporal encoding losses
        ##########################

        if predictions["classes_l2"] is not None:
            if self.classes_l2_loss is not None:
                # Temporal encoding level 2 loss (non-crop=0; crop|edge=1)
                classes_l2_loss = self.classes_l2_loss(
                    predictions["classes_l2"],
                    true_labels_dict["true_crop_and_edge"],
                    mask=true_labels_dict["mask"],
                )
                loss = loss + classes_l2_loss * weights["l2"]

        if predictions["classes_l3"] is not None:
            if self.classes_last_loss is not None:
                # Temporal encoding final loss (non-crop=0; crop=1; edge=2)
                classes_last_loss = self.classes_last_loss(
                    predictions["classes_l3"],
                    true_labels_dict["true_crop_or_edge"],
                    mask=true_labels_dict["mask"],
                )
                loss = loss + classes_last_loss * weights["l3"]

        #########################
        # Deep supervision losses
        #########################

        if self.deep_supervision:
            dist_loss_deep_b = self.dist_loss_deep_b(
                predictions["dist_b"],
                batch.bdist,
                mask=true_labels_dict["mask"],
            )
            edge_loss_deep_b = self.edge_loss_deep_b(
                predictions["edge_b"],
                true_labels_dict["true_edge"],
                mask=true_labels_dict["mask"],
            )
            crop_loss_deep_b = self.crop_loss_deep_b(
                predictions["mask_b"],
                true_labels_dict["true_crop"],
                mask=true_labels_dict["mask"],
            )
            dist_loss_deep_c = self.dist_loss_deep_c(
                predictions["dist_c"],
                batch.bdist,
                mask=true_labels_dict["mask"],
            )
            edge_loss_deep_c = self.edge_loss_deep_c(
                predictions["edge_c"],
                true_labels_dict["true_edge"],
                mask=true_labels_dict["mask"],
            )
            crop_loss_deep_c = self.crop_loss_deep_c(
                predictions["mask_c"],
                true_labels_dict["true_crop"],
                mask=true_labels_dict["mask"],
            )

            weights["dist_loss_deep_b"] = 0.25
            weights["edge_loss_deep_b"] = 0.25
            weights["crop_loss_deep_b"] = 0.25
            weights["dist_loss_deep_c"] = 0.1
            weights["edge_loss_deep_c"] = 0.1
            weights["crop_loss_deep_c"] = 0.1

            # Main loss
            loss = (
                loss
                + dist_loss_deep_b * weights["dist_loss_deep_b"]
                + edge_loss_deep_b * weights["edge_loss_deep_b"]
                + crop_loss_deep_b * weights["crop_loss_deep_b"]
                + dist_loss_deep_c * weights["dist_loss_deep_c"]
                + edge_loss_deep_c * weights["edge_loss_deep_c"]
                + crop_loss_deep_c * weights["crop_loss_deep_c"]
            )

        #############
        # Main losses
        #############

        # Distance transform loss
        dist_loss = self.dist_loss(
            predictions["dist"],
            batch.bdist,
            mask=true_labels_dict["mask"],
        )
        loss = loss + dist_loss * weights["dist_loss"]

        # Edge loss
        edge_loss = self.edge_loss(
            predictions["edge"],
            true_labels_dict["true_edge"],
            mask=true_labels_dict["mask"],
        )
        loss = loss + edge_loss * weights["edge_loss"]

        # Crop mask loss
        crop_loss = self.crop_loss(
            predictions["mask"],
            true_labels_dict["true_crop"],
            mask=true_labels_dict["mask"],
        )
        loss = loss + crop_loss * weights["crop_loss"]

        if not self.is_transfer_model:
            # Class-balanced MSE loss
            edge_cmse_loss = self.cmse_loss(
                predictions["edge"].squeeze(dim=1),
                true_labels_dict["true_edge"],
                mask=None
                if true_labels_dict["mask"] is None
                else true_labels_dict["mask"].squeeze(dim=1),
            )
            weights["edge_cmse_loss"] = 0.1
            loss = loss + edge_cmse_loss * weights["edge_cmse_loss"]

            crop_cmse_loss = self.cmse_loss(
                predictions["mask"].sum(dim=1),
                true_labels_dict["true_crop"],
                mask=None
                if true_labels_dict["mask"] is None
                else true_labels_dict["mask"].squeeze(dim=1),
            )
            weights["crop_cmse_loss"] = 0.1
            loss = loss + crop_cmse_loss * weights["crop_cmse_loss"]

            # Topology loss
            # topo_loss = self.topo_loss(
            #     predictions["edge"].squeeze(dim=1),
            #     true_labels_dict["true_edge"],
            # )
            # weights["topo_loss"] = 0.1
            # loss = loss + topo_loss * weights["topo_loss"]

        # if predictions["crop_type"] is not None:
        #     # Upstream (deep) loss on crop-type
        #     crop_type_star_loss = self.crop_type_star_loss(
        #         predictions["crop_type_star"],
        #         true_labels_dict["true_crop_type"],
        #     )
        #     loss = loss + crop_type_star_loss
        #     # Loss on crop-type
        #     crop_type_loss = self.crop_type_loss(
        #         predictions["crop_type"], true_labels_dict["true_crop_type"]
        #     )
        #     loss = loss + crop_type_loss

        return loss / sum(weights.values())

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
                        predictions['edge'][bidx].detach(),
                        einops.rearrange(
                            predictions['mask'][bidx, 1].detach(),
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

        loss = self.calc_loss(batch, predictions)

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
        predictions = self(batch)
        loss = self.calc_loss(batch, predictions)

        # Get the true edge and crop labels
        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions["crop_type"]
        )

        if self.train_maskrcnn:
            # Apply a forward pass on Mask RCNN
            mask_data = self.mask_rcnn_forward(
                batch=batch,
                predictions=predictions,
                mode='eval',
            )

            loss = loss + mask_data['loss']

        if true_labels_dict["mask"] is not None:
            # Valid sample = True; Invalid sample = False
            labels_bool_mask = true_labels_dict["mask"].to(dtype=torch.bool)
            predictions["dist"] = torch.masked_select(
                predictions["dist"], labels_bool_mask
            )
            batch.bdist = torch.masked_select(
                batch.bdist, labels_bool_mask.squeeze(dim=1)
            )

        dist_score_args = (
            predictions["dist"].contiguous().view(-1),
            batch.bdist.contiguous().view(-1),
        )

        dist_mae = self.dist_mae(*dist_score_args)
        dist_mse = self.dist_mse(*dist_score_args)

        # Get the class labels
        edge_ypred = self.probas_to_labels(predictions["edge"])
        crop_ypred = self.probas_to_labels(predictions["mask"])

        if true_labels_dict["mask"] is not None:
            edge_ypred = torch.masked_select(
                edge_ypred, labels_bool_mask.squeeze(dim=1)
            )
            crop_ypred = torch.masked_select(
                crop_ypred, labels_bool_mask.squeeze(dim=1)
            )
            true_labels_dict["true_edge"] = torch.masked_select(
                true_labels_dict["true_edge"], labels_bool_mask.squeeze(dim=1)
            )
            true_labels_dict["true_crop"] = torch.masked_select(
                true_labels_dict["true_crop"], labels_bool_mask.squeeze(dim=1)
            )

        edge_score_args = (
            edge_ypred.contiguous().view(-1),
            true_labels_dict["true_edge"].contiguous().view(-1),
        )
        crop_score_args = (
            crop_ypred.contiguous().view(-1),
            true_labels_dict["true_crop"].contiguous().view(-1),
        )

        # F1-score
        edge_score = self.edge_f1(*edge_score_args)
        crop_score = self.crop_f1(*crop_score_args)
        # MCC
        edge_mcc = self.edge_mcc(*edge_score_args)
        crop_mcc = self.crop_mcc(*crop_score_args)
        # Dice
        edge_dice = self.edge_dice(*edge_score_args)
        crop_dice = self.crop_dice(*crop_score_args)
        # Jaccard/IoU
        edge_jaccard = self.edge_jaccard(*edge_score_args)
        crop_jaccard = self.crop_jaccard(*crop_score_args)

        total_score = (
            loss
            + (1.0 - edge_score)
            + (1.0 - crop_score)
            + dist_mae
            + (1.0 - edge_mcc)
            + (1.0 - crop_mcc)
        )

        metrics = {
            "loss": loss,
            "dist_mae": dist_mae,
            "dist_mse": dist_mse,
            "edge_f1": edge_score,
            "crop_f1": crop_score,
            "edge_mcc": edge_mcc,
            "crop_mcc": crop_mcc,
            "edge_dice": edge_dice,
            "crop_dice": crop_dice,
            "edge_jaccard": edge_jaccard,
            "crop_jaccard": crop_jaccard,
            "score": total_score,
        }

        if predictions["crop_type"] is not None:
            crop_type_ypred = self.probas_to_labels(
                self.logits_to_probas(predictions["crop_type"])
            )
            crop_type_score = self.crop_type_f1(
                crop_type_ypred, true_labels_dict["true_crop_type"]
            )
            metrics["crop_type_f1"] = crop_type_score

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
        }
        if "crop_type_f1" in eval_metrics:
            metrics["vctf1"] = eval_metrics["crop_type_f1"]

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
        self.dist_mae = torchmetrics.MeanAbsoluteError()
        self.dist_mse = torchmetrics.MeanSquaredError()
        self.edge_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.crop_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.edge_mcc = torchmetrics.MatthewsCorrCoef(
            task="multiclass", num_classes=2
        )
        self.crop_mcc = torchmetrics.MatthewsCorrCoef(
            task="multiclass", num_classes=2
        )
        self.edge_dice = torchmetrics.Dice(num_classes=2, average="macro")
        self.crop_dice = torchmetrics.Dice(num_classes=2, average="macro")
        self.edge_jaccard = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.crop_jaccard = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=2, average="weighted"
        )
        if self.num_classes > 2:
            self.crop_type_f1 = torchmetrics.F1Score(
                num_classes=self.num_classes,
                task="multiclass",
                average="weighted",
                ignore_index=0,
            )

    def configure_loss(self):
        # Distance loss
        self.dist_loss = self.loss_dict[self.loss_name].get("regression")
        # Edge loss
        self.edge_loss = self.loss_dict[self.loss_name].get("classification")
        # Crop mask loss
        self.crop_loss = self.loss_dict[self.loss_name].get("classification")

        if not self.is_transfer_model:
            self.cmse_loss = self.loss_dict[LossTypes.CLASS_BALANCED_MSE].get(
                "classification"
            )
            # self.topo_loss = self.loss_dict[LossTypes.TOPOLOGY].get(
            #     "classification"
            # )

        if self.deep_supervision:
            self.dist_loss_deep_b = self.loss_dict[self.loss_name].get(
                "regression"
            )
            self.edge_loss_deep_b = self.loss_dict[self.loss_name].get(
                "classification"
            )
            self.crop_loss_deep_b = self.loss_dict[self.loss_name].get(
                "classification"
            )
            self.dist_loss_deep_c = self.loss_dict[self.loss_name].get(
                "regression"
            )
            self.edge_loss_deep_c = self.loss_dict[self.loss_name].get(
                "classification"
            )
            self.crop_loss_deep_c = self.loss_dict[self.loss_name].get(
                "classification"
            )

        self.classes_l2_loss = None
        self.classes_last_loss = None
        if not self.is_transfer_model:
            # Crop Temporal encoding losses
            self.classes_l2_loss = self.loss_dict[self.loss_name].get(
                "classification"
            )
            self.classes_last_loss = self.loss_dict[self.loss_name].get(
                "classification"
            )

        if self.num_classes > 2:
            self.crop_type_star_loss = self.loss_dict[self.loss_name].get(
                "classification"
            )
            self.crop_type_loss = self.loss_dict[self.loss_name].get(
                "classification"
            )

    def configure_optimizers(self):
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
        attention_weights: str = "spatial_channel",
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO_COMPLEMENT,
        learning_rate: float = 0.01,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 1e-3,
        eps: float = 1e-4,
        ckpt_name: str = ModelNames.CKPT_TRANSFER_NAME.replace(".ckpt", ""),
        model_name: str = "cultionet_transfer",
        deep_supervision: bool = False,
        pool_attention: bool = False,
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
        class_counts: T.Optional[torch.Tensor] = None,
        edge_class: T.Optional[int] = None,
        scale_pos_weight: bool = False,
        save_batch_val_metrics: bool = False,
        finetune: T.Optional[str] = None,
    ):
        super(CultionetLitTransferModel, self).__init__()

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
        self.deep_supervision = deep_supervision
        self.train_maskrcnn = None

        self.sigmoid = torch.nn.Sigmoid()
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.loss_dict = {
            LossTypes.BOUNDARY: {
                "classification": cnetlosses.BoundaryLoss(),
            },
            LossTypes.CLASS_BALANCED_MSE: {
                "classification": cnetlosses.ClassBalancedMSELoss(),
            },
            LossTypes.TANIMOTO_COMPLEMENT: {
                "classification": cnetlosses.TanimotoComplementLoss(),
                "regression": cnetlosses.TanimotoComplementLoss(
                    one_hot_targets=False
                ),
            },
            LossTypes.TANIMOTO: {
                "classification": cnetlosses.TanimotoDistLoss(),
                "regression": cnetlosses.TanimotoDistLoss(
                    one_hot_targets=False
                ),
            },
            LossTypes.TOPOLOGY: {
                "classification": cnetlosses.TopologyLoss(),
            },
        }

        self.cultionet_model = CultionetLitModel.load_from_checkpoint(
            checkpoint_path=str(pretrained_ckpt_file)
        ).cultionet_model

        # import torchinfo
        # torchinfo.summary(
        #     model=self.cultionet_model.mask_model,
        #     input_size=[(1, 5, 13, 100, 100), (1, 64, 100, 100)],
        #     device="cuda",
        # )

        # Freeze all parameters if not finetuning the full model
        if self.finetune != "all":
            for name, param in self.cultionet_model.named_parameters():
                param.requires_grad = False

            if self.finetune == "fc":
                # Unfreeze fully connected layers
                for name, param in self.cultionet_model.named_parameters():
                    if name.startswith("temporal_encoder.final_"):
                        param.requires_grad = True
                    if name.startswith("mask_model.final_"):
                        param.requires_grad = True

            else:
                # Set new final layers to learn new weights
                temporal_encoder_final = TemporalTransformerFinal(
                    hidden_channels=hidden_channels,
                    d_model=self.cultionet_model.temporal_encoder.d_model,
                    num_classes_l2=self.cultionet_model.temporal_encoder.num_classes_l2,
                    num_classes_last=self.cultionet_model.temporal_encoder.num_classes_last,
                    activation_type=activation_type,
                    final_activation=nn.Softmax(dim=1),
                )
                temporal_encoder_final.apply(init_attention_weights)
                self.cultionet_model.temporal_encoder.final = (
                    temporal_encoder_final
                )

                # Update the post-UNet layer with trainable parameters
                mask_model_final_a = cunn.TowerUNetFinal(
                    in_channels=self.cultionet_model.mask_model.final_a.in_channels,
                    num_classes=self.cultionet_model.mask_model.final_a.num_classes,
                    mask_activation=self.cultionet_model.mask_model.final_a.mask_activation,
                    activation_type=activation_type,
                )
                mask_model_final_a.apply(init_conv_weights)
                self.cultionet_model.mask_model.final_a = mask_model_final_a

                if self.deep_supervision:
                    mask_model_final_b = cunn.TowerUNetFinal(
                        in_channels=self.cultionet_model.mask_model.final_b.in_channels,
                        num_classes=self.cultionet_model.mask_model.final_b.num_classes,
                        mask_activation=self.cultionet_model.mask_model.final_b.mask_activation,
                        activation_type=activation_type,
                    )
                    mask_model_final_b.apply(init_conv_weights)
                    self.cultionet_model.mask_model.final_b = (
                        mask_model_final_b
                    )

                    mask_model_final_c = cunn.TowerUNetFinal(
                        in_channels=self.cultionet_model.mask_model.final_c.in_channels,
                        num_classes=self.cultionet_model.mask_model.final_c.num_classes,
                        mask_activation=self.cultionet_model.mask_model.final_c.mask_activation,
                        activation_type=activation_type,
                    )
                    mask_model_final_c.apply(init_conv_weights)
                    self.cultionet_model.mask_model.final_c = (
                        mask_model_final_c
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
        attention_weights: str = "spatial_channel",
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO_COMPLEMENT,
        learning_rate: float = 0.01,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 1e-3,
        eps: float = 1e-4,
        ckpt_name: str = "last",
        model_name: str = "cultionet",
        deep_supervision: bool = False,
        pool_attention: bool = False,
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

        super(CultionetLitModel, self).__init__()

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
        self.deep_supervision = deep_supervision
        self.train_maskrcnn = train_maskrcnn

        self.sigmoid = torch.nn.Sigmoid()
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.loss_dict = {
            LossTypes.BOUNDARY: {
                "classification": cnetlosses.BoundaryLoss(),
            },
            LossTypes.CLASS_BALANCED_MSE: {
                "classification": cnetlosses.ClassBalancedMSELoss(),
            },
            LossTypes.TANIMOTO_COMPLEMENT: {
                "classification": cnetlosses.TanimotoComplementLoss(),
                "regression": cnetlosses.TanimotoComplementLoss(
                    one_hot_targets=False
                ),
            },
            LossTypes.TANIMOTO: {
                "classification": cnetlosses.TanimotoDistLoss(),
                "regression": cnetlosses.TanimotoDistLoss(
                    one_hot_targets=False
                ),
            },
            LossTypes.TOPOLOGY: {
                "classification": cnetlosses.TopologyLoss(),
            },
        }

        if self.train_maskrcnn:
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
                deep_supervision=deep_supervision,
                pool_attention=pool_attention,
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
