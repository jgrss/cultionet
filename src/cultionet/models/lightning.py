import logging
import typing as T
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
from torch.optim import lr_scheduler as optim_lr_scheduler
from torchvision import transforms
from torchvision.ops import box_iou

from .. import nn as cunn
from ..data.data import Data
from ..enums import (
    LearningRateSchedulers,
    LossTypes,
    ModelTypes,
    ResBlockTypes,
)
from ..layers.weights import init_attention_weights
from ..losses import TanimotoComplementLoss, TanimotoDistLoss
from .cultionet import CultioNet, GeoRefinement
from .maskcrnn import BFasterRCNN
from .nunet import PostUNet3Psi

warnings.filterwarnings("ignore")
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False
logging.getLogger("lightning").setLevel(logging.ERROR)


class MaskRCNNLitModel(LightningModule):
    def __init__(
        self,
        cultionet_model_file: Path,
        cultionet_in_channels: int,
        cultionet_num_time: int,
        cultionet_hidden_channels: int,
        cultionet_num_classes: int,
        ckpt_name: str = "maskrcnn",
        model_name: str = "maskrcnn",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        resize_height: int = 201,
        resize_width: int = 201,
        min_image_size: int = 100,
        max_image_size: int = 500,
        trainable_backbone_layers: int = 3,
    ):
        """Lightning model.

        Args:
            in_channels
            num_time
            hidden_channels
            learning_rate
            weight_decay
        """
        super(MaskRCNNLitModel, self).__init__()
        self.save_hyperparameters()

        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = 2
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.cultionet_model = CultionetLitModel(
            in_channels=cultionet_in_channels,
            num_time=cultionet_num_time,
            hidden_channels=cultionet_hidden_channels,
            num_classes=cultionet_num_classes,
        )
        self.cultionet_model.load_state_dict(
            state_dict=torch.load(cultionet_model_file)
        )
        self.cultionet_model.eval()
        self.cultionet_model.freeze()
        self.model = BFasterRCNN(
            in_channels=4,
            out_channels=256,
            num_classes=self.num_classes,
            sizes=(16, 32, 64, 128, 256),
            aspect_ratios=(0.5, 1.0, 3.0),
            trainable_backbone_layers=trainable_backbone_layers,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def mask_forward(
        self,
        distance_ori: torch.Tensor,
        distance: torch.Tensor,
        edge: torch.Tensor,
        crop_r: torch.Tensor,
        height: T.Union[None, int, torch.Tensor],
        width: T.Union[None, int, torch.Tensor],
        batch: T.Union[None, int, torch.Tensor],
        y: T.Union[None, torch.Tensor] = None,
    ):
        height = int(height) if batch is None else int(height[0])
        width = int(width) if batch is None else int(width[0])
        batch_size = 1 if batch is None else batch.unique().size(0)
        x = torch.cat(
            (
                distance_ori,
                distance,
                edge[:, 1][:, None],
                crop_r[:, 1][:, None],
            ),
            dim=1,
        )
        resizer = transforms.Resize((self.resize_height, self.resize_width))
        x = [resizer(image) for image in x]
        targets = None
        if y is not None:
            targets = []
            for bidx in y["image_id"].unique():
                batch_dict = {}
                batch_slice = y["image_id"] == bidx
                for k in y.keys():
                    if k == "masks":
                        batch_dict[k] = resizer(y[k][batch_slice])
                    elif k == "boxes":
                        # [xmin, ymin, xmax, ymax]
                        batch_dict[k] = self.scale_boxes(
                            y[k][batch_slice], batch, [height]
                        )
                    else:
                        batch_dict[k] = y[k][batch_slice]
                    targets.append(batch_dict)
        outputs = self.model(x, targets)

        return outputs

    def scale_boxes(
        self,
        boxes: torch.Tensor,
        batch: torch.Tensor,
        height: T.Union[None, int, T.List[int], torch.Tensor],
    ):
        height = int(height) if batch is None else int(height[0])
        scale = self.resize_height / height

        return boxes * scale

    def forward(
        self,
        batch: Data,
        batch_idx: int = None,
        y: T.Optional[torch.Tensor] = None,
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Performs a single model forward pass."""
        with torch.no_grad():
            distance_ori, distance, edge, __, crop_r = self.cultionet_model(
                batch
            )
        estimates = self.mask_forward(
            distance_ori,
            distance,
            edge,
            crop_r,
            height=batch.height,
            width=batch.width,
            batch=batch.batch,
            y=y,
        )

        return estimates

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint."""
        ckpt_file = Path(self.logger.save_dir) / f"{self.ckpt_name}.ckpt"
        if ckpt_file.is_file():
            ckpt_file.unlink()
        torch.save(checkpoint, ckpt_file)

    def on_validation_epoch_end(self, *args, **kwargs):
        """Save the model on validation end."""
        model_file = Path(self.logger.save_dir) / f"{self.model_name}.pt"
        if model_file.is_file():
            model_file.unlink()
        torch.save(self.state_dict(), model_file)

    def calc_loss(
        self, batch: T.Union[Data, T.List], y: T.Optional[torch.Tensor] = None
    ):
        """Calculates the loss for each layer.

        Returns:
            Average loss
        """
        losses = self(batch, y=y)
        loss = sum(loss for loss in losses.values())

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step."""
        y = {
            "boxes": batch.boxes,
            "labels": batch.box_labels,
            "masks": batch.box_masks,
            "image_id": batch.image_id,
        }
        loss = self.calc_loss(batch, y=y)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _shared_eval_step(self, batch: Data) -> dict:
        # Predictions
        instances = self(batch)
        # True boxes
        true_boxes = self.scale_boxes(batch.boxes, batch, batch.height)

        predict_iou_score = torch.tensor(0.0, device=self.device)
        iou_score = torch.tensor(0.0, device=self.device)
        box_score = torch.tensor(0.0, device=self.device)
        for bidx, batch_value in enumerate(batch.image_id.unique()):
            # This should be low (i.e., low overlap of predicted boxes)
            predict_iou_score += box_iou(
                instances[bidx]["boxes"], instances[bidx]["boxes"]
            ).mean()
            # This should be high (i.e., high overlap of predictions and true boxes)
            iou_score += box_iou(
                true_boxes[batch.image_id == batch_value],
                instances[bidx]["boxes"],
            ).mean()
            # This should be high (i.e., masks should be confident)
            box_score += instances[bidx]["scores"].mean()
        predict_iou_score /= batch.image_id.unique().size(0)
        iou_score /= batch.image_id.unique().size(0)
        box_score /= batch.image_id.unique().size(0)

        total_iou_score = (predict_iou_score + (1.0 - iou_score)) * 0.5
        box_score = 1.0 - box_score
        # Minimize intersection-over-union and maximum score
        total_score = (total_iou_score + box_score) * 0.5

        metrics = {
            "predict_iou_score": predict_iou_score,
            "iou_score": iou_score,
            "box_score": box_score,
            "mean_score": total_score,
        }

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step."""
        eval_metrics = self._shared_eval_step(batch)

        metrics = {
            "val_loss": eval_metrics["mean_score"],
            "val_piou": eval_metrics["predict_iou_score"],
            "val_iou": eval_metrics["iou_score"],
            "val_box": eval_metrics["box_score"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def test_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one test step."""
        eval_metrics = self._shared_eval_step(batch)

        metrics = {
            "test_loss": eval_metrics["mean_score"],
            "test_piou": eval_metrics["predict_iou_score"],
            "test_iou": eval_metrics["iou_score"],
            "test_box": eval_metrics["box_score"],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-4,
        )
        lr_scheduler = optim_lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5
        )

        return {
            "optimizer": optimizer,
            "scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


def scale_logits(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x / t


class RefineLitModel(LightningModule):
    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        eps: float = 1e-4,
        edge_class: int = 2,
        class_counts: T.Optional[torch.Tensor] = None,
        cultionet_ckpt: T.Optional[T.Union[Path, str]] = None,
    ):
        super(RefineLitModel, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.edge_class = edge_class
        self.class_counts = class_counts
        self.cultionet_ckpt = cultionet_ckpt

        self.cultionet_model = None
        self.geo_refine_model = GeoRefinement(
            in_features=in_features, out_channels=num_classes
        )

        self.configure_loss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        predictions: T.Dict[str, torch.Tensor],
        batch: Data,
        batch_idx: int = None,
    ) -> T.Dict[str, torch.Tensor]:
        return self.geo_refine_model(predictions, data=batch)

    def set_true_labels(self, batch: Data) -> torch.Tensor:
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(
            (batch.y > 0) & (batch.y != self.edge_class), 1, 0
        ).long()

        return true_crop

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: T.Dict[str, torch.Tensor],
    ):
        true_crop = self.set_true_labels(batch)
        # Predicted crop values are probabilities
        loss = self.crop_loss(predictions["mask"], true_crop)

        return loss

    def training_step(
        self, batch: Data, batch_idx: int = None, optimizer_idx: int = None
    ):
        """Executes one training step."""
        # Apply inference with the main cultionet model
        if (self.cultionet_ckpt is not None) and (
            self.cultionet_model is None
        ):
            self.cultionet_model = CultionetLitModel.load_from_checkpoint(
                checkpoint_path=str(self.cultionet_ckpt)
            )
            self.cultionet_model.to(self.device)
            self.cultionet_model.eval()
            self.cultionet_model.freeze()
        with torch.no_grad():
            predictions = self.cultionet_model(batch)

        predictions = self(predictions, batch)
        loss = self.calc_loss(batch, predictions)

        metrics = {"loss": loss}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def on_train_epoch_end(self, *args, **kwargs):
        """Save the scaling parameters on training end."""
        if self.logger.save_dir is not None:
            model_file = Path(self.logger.save_dir) / "refine.pt"
            if model_file.is_file():
                model_file.unlink()
            torch.save(self.geo_refine_model.state_dict(), model_file)

    def configure_loss(self):
        self.crop_loss = TanimotoDistLoss(scale_pos_weight=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.geo_refine_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
        lr_scheduler = optim_lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-5, last_epoch=-1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class LightningModuleMixin(LightningModule):
    def __init__(self):
        super(LightningModuleMixin, self).__init__()

        torch.set_float32_matmul_precision("high")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """Performs a single model forward pass.

        Returns:
            distance: Normalized distance transform (from boundaries), [0,1].
            edge: Probabilities of edge|non-edge, [0,1].
            crop: Logits of crop|non-crop.
        """
        return self.cultionet_model(batch)

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
        predictions = self.forward(batch, batch_idx)
        if self.temperature_lit_model is not None:
            predictions = self.temperature_lit_model(predictions, batch)

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

        return {
            "true_edge": true_edge,
            "true_crop": true_crop,
            "true_crop_and_edge": true_crop_and_edge,
            "true_crop_or_edge": true_crop_or_edge,
            "true_crop_type": true_crop_type,
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
        if predictions["classes_l2"] is not None:
            # Temporal encoding level 2 loss (non-crop=0; crop|edge=1)
            classes_l2_loss = self.classes_l2_loss(
                predictions["classes_l2"],
                true_labels_dict["true_crop_and_edge"],
            )
            loss = loss + classes_l2_loss * weights["l2"]

        if predictions["classes_l3"] is not None:
            # Temporal encoding final loss (non-crop=0; crop=1; edge=2)
            classes_last_loss = self.classes_last_loss(
                predictions["classes_l3"],
                true_labels_dict["true_crop_or_edge"],
            )
            loss = loss + classes_last_loss * weights["l3"]

        # Edge losses
        if self.deep_supervision:
            dist_loss_deep_b = self.dist_loss_deep_b(
                predictions["dist_b"], batch.bdist
            )
            edge_loss_deep_b = self.edge_loss_deep_b(
                predictions["edge_b"], true_labels_dict["true_edge"]
            )
            crop_loss_deep_b = self.crop_loss_deep_b(
                predictions["mask_b"], true_labels_dict["true_crop"]
            )
            dist_loss_deep_c = self.dist_loss_deep_c(
                predictions["dist_c"], batch.bdist
            )
            edge_loss_deep_c = self.edge_loss_deep_c(
                predictions["edge_c"], true_labels_dict["true_edge"]
            )
            crop_loss_deep_c = self.crop_loss_deep_c(
                predictions["mask_c"], true_labels_dict["true_crop"]
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

        # Distance transform loss
        dist_loss = self.dist_loss(predictions["dist"], batch.bdist)
        loss = loss + dist_loss * weights["dist_loss"]

        # Edge loss
        edge_loss = self.edge_loss(
            predictions["edge"], true_labels_dict["true_edge"]
        )
        loss = loss + edge_loss * weights["edge_loss"]

        # Crop mask loss
        crop_loss = self.crop_loss(
            predictions["mask"], true_labels_dict["true_crop"]
        )
        loss = loss + crop_loss * weights["crop_loss"]

        if predictions.get("foj_image_patches") is not None:
            foj_loss = self.foj_loss(
                patches=predictions.get("foj_patches"),
                image_patches=predictions.get("foj_image_patches"),
            )
            weights["foj"] = 0.1
            loss = loss + foj_loss

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

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step and logs training step metrics."""
        predictions = self(batch)
        loss = self.calc_loss(batch, predictions)
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

        dist_mae = self.dist_mae(
            # B x 1 x H x W
            predictions["dist"].squeeze(dim=1),
            # B x H x W
            batch.bdist,
        )
        dist_mse = self.dist_mse(
            predictions["dist"].squeeze(dim=1),
            batch.bdist,
        )
        # Get the class labels
        edge_ypred = self.probas_to_labels(predictions["edge"])
        crop_ypred = self.probas_to_labels(predictions["mask"])
        # Get the true edge and crop labels
        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions["crop_type"]
        )

        # F1-score
        edge_score = self.edge_f1(edge_ypred, true_labels_dict["true_edge"])
        crop_score = self.crop_f1(crop_ypred, true_labels_dict["true_crop"])
        # MCC
        edge_mcc = self.edge_mcc(edge_ypred, true_labels_dict["true_edge"])
        crop_mcc = self.crop_mcc(crop_ypred, true_labels_dict["true_crop"])
        # Dice
        edge_dice = self.edge_dice(edge_ypred, true_labels_dict["true_edge"])
        crop_dice = self.crop_dice(crop_ypred, true_labels_dict["true_crop"])
        # Jaccard/IoU
        edge_jaccard = self.edge_jaccard(
            edge_ypred, true_labels_dict["true_edge"]
        )
        crop_jaccard = self.crop_jaccard(
            crop_ypred, true_labels_dict["true_crop"]
        )

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
            "val_loss": eval_metrics["loss"],
            "vef1": eval_metrics["edge_f1"],
            "vcf1": eval_metrics["crop_f1"],
            "vmae": eval_metrics["dist_mae"],
            "val_score": eval_metrics["score"],
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
        # Field of junctions loss
        # self.foj_loss = FieldOfJunctionsLoss()

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
        ckpt_file: T.Union[Path, str],
        in_channels: int,
        num_time: int,
        init_filter: int = 32,
        activation_type: str = "SiLU",
        num_classes: int = 2,
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO,
        learning_rate: float = 1e-3,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 0.01,
        eps: float = 1e-4,
        mask_activation: T.Callable = nn.Softmax(dim=1),
        deep_supervision: bool = True,
        scale_pos_weight: bool = True,
        model_name: str = "cultionet_transfer",
        edge_class: T.Optional[int] = None,
        save_batch_val_metrics: bool = False,
        finetune: bool = False,
    ):
        super(CultionetLitTransferModel, self).__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.steplr_step_size = steplr_step_size
        self.weight_decay = weight_decay
        self.eps = eps
        self.model_name = model_name
        self.temperature_lit_model = None
        self.save_batch_val_metrics = save_batch_val_metrics
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.loss_dict = {
            LossTypes.TANIMOTO_COMPLEMENT: {
                "classification": TanimotoComplementLoss(),
                "regression": TanimotoComplementLoss(one_hot_targets=False),
            },
            LossTypes.TANIMOTO: {
                "classification": TanimotoDistLoss(),
                "regression": TanimotoDistLoss(one_hot_targets=False),
            },
        }

        up_channels = int(init_filter * 5)
        self.in_channels = in_channels
        self.num_time = num_time
        self.deep_supervision = deep_supervision
        self.scale_pos_weight = scale_pos_weight

        self.cultionet_model = CultionetLitModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_file)
        )

        if not finetune:
            # Freeze all parameters for feature extraction
            self.cultionet_model.freeze()

        # layers[-2] ->
        #   TemporalAttention()
        layers = list(self.cultionet_model.cultionet_model.children())
        self.cultionet_model.temporal_encoder = layers[-2]

        if not finetune:
            # Unfreeze the temporal encoder
            self.cultionet_model.temporal_encoder = self.unfreeze_layer(
                self.temporal_encoder
            )
            # Set new final layers to learn new weights
            # Level 2 level (non-crop; crop)
            self.cultionet_model.temporal_encoder.l2 = cunn.FinalConv2dDropout(
                hidden_dim=self.temporal_encoder.l2.net[0]
                .seq.seq[0]
                .seq[0]
                .in_channels,
                dim_factor=1,
                activation_type=activation_type,
                final_activation=nn.Softmax(dim=1),
                num_classes=num_classes,
            )
            self.cultionet_model.temporal_encoder.l2.apply(
                init_attention_weights
            )
            # Last level (non-crop; crop; edges)
            self.cultionet_model.temporal_encoder.final_last = (
                cunn.FinalConv2dDropout(
                    hidden_dim=self.temporal_encoder.final_last.net[0]
                    .seq.seq[0]
                    .seq[0]
                    .in_channels,
                    dim_factor=1,
                    activation_type=activation_type,
                    final_activation=nn.Softmax(dim=1),
                    num_classes=num_classes + 1,
                )
            )
            self.cultionet_model.temporal_encoder.final_last.apply(
                init_attention_weights
            )

            # layers[-1] ->
            #   ResUNet3Psi()
            self.cultionet_model.mask_model = layers[-1]
            # Update the post-UNet layer with trainable parameters
            post_unet = PostUNet3Psi(
                up_channels=up_channels,
                num_classes=num_classes,
                mask_activation=mask_activation,
                deep_supervision=deep_supervision,
            )
            self.cultionet_model.mask_model.post_unet = post_unet

        self.model_attr = model_name
        setattr(
            self,
            self.model_attr,
            self.cultionet_model,
        )
        self.configure_loss()
        self.configure_scorer()

    def unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True

        return layer


class CultionetLitModel(LightningModuleMixin):
    def __init__(
        self,
        in_channels: int = None,
        in_time: int = None,
        num_classes: int = 2,
        hidden_channels: int = 32,
        model_type: str = ModelTypes.TOWERUNET,
        dropout: float = 0.1,
        activation_type: str = "SiLU",
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: str = "spatial_channel",
        optimizer: str = "AdamW",
        loss_name: str = LossTypes.TANIMOTO,
        learning_rate: float = 1e-3,
        lr_scheduler: str = LearningRateSchedulers.ONE_CYCLE_LR,
        steplr_step_size: int = 5,
        weight_decay: float = 0.01,
        eps: float = 1e-4,
        ckpt_name: str = "last",
        model_name: str = "cultionet",
        deep_supervision: bool = False,
        pool_first: bool = False,
        class_counts: T.Optional[torch.Tensor] = None,
        edge_class: T.Optional[int] = None,
        temperature_lit_model: T.Optional[GeoRefinement] = None,
        scale_pos_weight: bool = False,
        save_batch_val_metrics: bool = False,
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
        self.temperature_lit_model = temperature_lit_model
        self.scale_pos_weight = scale_pos_weight
        self.save_batch_val_metrics = save_batch_val_metrics
        self.deep_supervision = deep_supervision
        self.pool_first = pool_first
        self.sigmoid = torch.nn.Sigmoid()
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.loss_dict = {
            LossTypes.TANIMOTO_COMPLEMENT: {
                "classification": TanimotoComplementLoss(),
                "regression": TanimotoComplementLoss(one_hot_targets=False),
            },
            LossTypes.TANIMOTO: {
                "classification": TanimotoDistLoss(),
                "regression": TanimotoDistLoss(one_hot_targets=False),
            },
        }

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
                pool_first=pool_first,
            ),
        )

        self.configure_loss()
        self.configure_scorer()

    # def on_train_epoch_start(self):
    #     # Get the current learning rate from the optimizer
    #     weight_decay = self.optimizers().optimizer.param_groups[0]['weight_decay']
    #     if (weight_decay != self.weight_decay) or (eps != self.eps):
    #         self.configure_optimizers()
