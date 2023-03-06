import typing as T
from pathlib import Path
import json
import warnings
import logging

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from pytorch_lightning import LightningModule
from torchvision.ops import box_iou
from torchvision import transforms
import torchmetrics

from . import model_utils
from .cultio import CultioNet, GeoRefinement
from .maskcrnn import BFasterRCNN
from .base_layers import Softmax
from ..losses import TanimotoDistLoss


warnings.filterwarnings('ignore')
logging.getLogger('lightning').addHandler(logging.NullHandler())
logging.getLogger('lightning').propagate = False
logging.getLogger('lightning').setLevel(logging.ERROR)


class MaskRCNNLitModel(LightningModule):
    def __init__(
        self,
        cultionet_model_file: Path,
        cultionet_num_features: int,
        cultionet_num_time_features: int,
        cultionet_filters: int,
        cultionet_num_classes: int,
        ckpt_name: str = 'maskrcnn',
        model_name: str = 'maskrcnn',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        resize_height: int = 201,
        resize_width: int = 201,
        min_image_size: int = 100,
        max_image_size: int = 500,
        trainable_backbone_layers: int = 3
    ):
        """Lightning model

        Args:
            num_features
            num_time_features
            filters
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
        self.resize_height= resize_height
        self.resize_width = resize_width

        self.cultionet_model = CultioLitModel(
            num_features=cultionet_num_features,
            num_time_features=cultionet_num_time_features,
            filters=cultionet_filters,
            num_classes=cultionet_num_classes
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
            max_image_size=max_image_size
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
        y: T.Union[None, torch.Tensor] = None
    ):
        height = int(height) if batch is None else int(height[0])
        width = int(width) if batch is None else int(width[0])
        batch_size = 1 if batch is None else batch.unique().size(0)
        x = torch.cat(
            (
                distance_ori,
                distance,
                edge[:, 1][:, None],
                crop_r[:, 1][:, None]
            ),
            dim=1
        )
        # in x = (H*W x C)
        # new x = (B x C x H x W)
        gc = model_utils.GraphToConv()
        x = gc(x, batch_size, height, width)
        resizer = transforms.Resize((self.resize_height, self.resize_width))
        x = [resizer(image) for image in x]
        targets = None
        if y is not None:
            targets = []
            for bidx in y['image_id'].unique():
                batch_dict = {}
                batch_slice = y['image_id'] == bidx
                for k in y.keys():
                    if k == 'masks':
                        batch_dict[k] = resizer(
                            y[k][batch_slice]
                        )
                    elif k == 'boxes':
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
        height: T.Union[None, int, T.List[int], torch.Tensor]
    ):
        height = int(height) if batch is None else int(height[0])
        scale = self.resize_height / height

        return boxes * scale

    def forward(
        self, batch: Data, batch_idx: int = None, y: T.Optional[torch.Tensor] = None
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Performs a single model forward pass
        """
        with torch.no_grad():
            distance_ori, distance, edge, __, crop_r = self.cultionet_model(batch)
        estimates = self.mask_forward(
            distance_ori,
            distance,
            edge,
            crop_r,
            height=batch.height,
            width=batch.width,
            batch=batch.batch,
            y=y
        )

        return estimates

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint
        """
        ckpt_file = Path(self.logger.save_dir) / f'{self.ckpt_name}.ckpt'
        if ckpt_file.is_file():
            ckpt_file.unlink()
        torch.save(
            checkpoint, ckpt_file
        )

    def on_validation_epoch_end(self, *args, **kwargs):
        """Save the model on validation end
        """
        model_file = Path(self.logger.save_dir) / f'{self.model_name}.pt'
        if model_file.is_file():
            model_file.unlink()
        torch.save(
            self.state_dict(), model_file
        )

    def calc_loss(self, batch: T.Union[Data, T.List], y: T.Optional[torch.Tensor] = None):
        """Calculates the loss for each layer

        Returns:
            Average loss
        """
        losses = self(batch, y=y)
        loss = sum(loss for loss in losses.values())

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
        """
        y = {
            'boxes': batch.boxes,
            'labels': batch.box_labels,
            'masks': batch.box_masks,
            'image_id': batch.image_id
        }
        loss = self.calc_loss(batch, y=y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)

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
                instances[bidx]['boxes'], instances[bidx]['boxes']
            ).mean()
            # This should be high (i.e., high overlap of predictions and true boxes)
            iou_score += box_iou(
                true_boxes[batch.image_id == batch_value], instances[bidx]['boxes']
            ).mean()
            # This should be high (i.e., masks should be confident)
            box_score += instances[bidx]['scores'].mean()
        predict_iou_score /= batch.image_id.unique().size(0)
        iou_score /= batch.image_id.unique().size(0)
        box_score /= batch.image_id.unique().size(0)

        total_iou_score = (predict_iou_score + (1.0 - iou_score)) * 0.5
        box_score = 1.0 - box_score
        # Minimize intersection-over-union and maximum score
        total_score = (total_iou_score + box_score) * 0.5

        metrics = {
            'predict_iou_score': predict_iou_score,
            'iou_score': iou_score,
            'box_score': box_score,
            'mean_score': total_score
        }

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step
        """
        eval_metrics = self._shared_eval_step(batch)

        metrics = {
            'val_loss': eval_metrics['mean_score'],
            'val_piou': eval_metrics['predict_iou_score'],
            'val_iou': eval_metrics['iou_score'],
            'val_box': eval_metrics['box_score'],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def test_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one test step
        """
        eval_metrics = self._shared_eval_step(batch)

        metrics = {
            'test_loss': eval_metrics['mean_score'],
            'test_piou': eval_metrics['predict_iou_score'],
            'test_iou': eval_metrics['iou_score'],
            'test_box': eval_metrics['box_score'],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-4
        )
        lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }


def scale_logits(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return x / t


class TemperatureScaling(LightningModule):
    def __init__(
        self,
        cultionet_model: CultioNet = None,
        learning_rate_refine: float = 1e-3,
        learning_rate_lbfgs: float = 0.01,
        weight_decay: float = 0.01,
        eps: float = 1e-4,
        max_iter: float = 25,
        edge_class: int = 2,
        class_counts: T.Optional[torch.Tensor] = None
    ):
        super(TemperatureScaling, self).__init__()

        self.crop_temperature = torch.nn.Parameter(torch.ones(1))

        self.cultionet_model = cultionet_model
        self.learning_rate_refine = learning_rate_refine
        self.learning_rate_lbfgs = learning_rate_lbfgs
        self.weight_decay = weight_decay
        self.eps = eps
        self.max_iter = max_iter
        self.edge_class = edge_class
        self.class_counts = class_counts
        self.softmax = Softmax(dim=1)
        self.frozen_ = False

        self.geo_refine_model = GeoRefinement(
            # StarRNN 3 + 2
            # Distance transform x4
            # Edge sigmoid x4
            # Crop softmax x4
            in_channels=3+2+4+4+(4*2),
            n_features=16,
            out_channels=2,
            double_dilation=2
        )
        import ipdb; ipdb.set_trace()
        if self.cultionet_model is not None:
            self.cultionet_model.eval()
            self.cultionet_model.freeze()
        self.configure_loss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def freeze_params(self):
        self.geo_refine_model.eval()
        self.geo_refine_model.freeze()

        self.frozen_ = True

    def forward(
        self,
        predictions: T.Dict[str, torch.Tensor],
        batch: Data,
        crop_temperature: torch.Tensor,
        batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        if not self.frozen_:
            self.freeze_params()

        with torch.no_grad():
            predictions = self.predict_crop_probas(
                predictions=predictions,
                batch=batch,
                with_no_grad=False,
                scale_predict_logits=True,
                crop_temperature=crop_temperature
            )

        return predictions

    def calc_refined_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: torch.Tensor
    ):
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(
            (batch.y > 0) & (batch.y != self.edge_class), 1, 0
        ).long()

        # Predicted crop values are logits
        loss = self.crop_loss_refine(predictions['crop'], true_crop)

        return loss

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: T.Dict[str, torch.Tensor]
    ):
        # true_edge = batch.y.eq(self.edge_class).long()
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(
            (batch.y > 0) & (batch.y != self.edge_class), 1, 0
        ).long()

        # Predicted crop values are probabilities
        loss = self.crop_loss(predictions['crop'], true_crop)

        return loss

    def predict_crop_probas(
        self, predictions: T.Dict[str, torch.Tensor],
        batch: Data,
        with_no_grad: bool,
        scale_predict_logits: bool,
        crop_temperature: T.Optional[torch.Tensor] = None
    ) -> T.Dict[str, torch.Tensor]:
        if with_no_grad:
            with torch.no_grad():
                # The inputs from cultionet are probabilities
                # The output crop predictions are logits
                predictions['crop'] = self.geo_refine_model(
                    predictions,
                    data=batch
                )
        else:
            predictions['crop'] = self.geo_refine_model(
                predictions,
                data=batch
            )
        if scale_predict_logits:
            if crop_temperature is None:
                crop_temperature = self.crop_temperature

            predictions['crop'] = scale_logits(
                predictions['crop'], crop_temperature
            )
        predictions['crop'] = self.softmax(predictions['crop'])

        return predictions

    def training_step(
        self,
        batch: Data,
        batch_idx: int = None,
        optimizer_idx: int = None
    ):
        """Executes one training step
        """
        # Apply inference with the main cultionet model
        with torch.no_grad():
            predictions = self.cultionet_model(batch)

        if optimizer_idx == 0:
            # The first optimizer is used for the refinement model
            self.geo_refine_model.train()
            predictions = self.predict_crop_probas(
                predictions=predictions,
                batch=batch,
                with_no_grad=False,
                scale_predict_logits=False
            )

            loss = self.calc_refined_loss(
                batch,
                predictions
            )
        else:
            # The second optimizer is used for the probability calibration
            self.geo_refine_model.eval()
            predictions = self.predict_crop_probas(
                predictions=predictions,
                batch=batch,
                with_no_grad=True,
                scale_predict_logits=True
            )

            loss = self.calc_loss(
                batch,
                predictions
            )

        metrics = {
            'loss': loss,
            'crop_temp': self.crop_temperature
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def on_train_epoch_end(self, *args, **kwargs):
        """Save the scaling parameters on training end
        """
        if self.logger.save_dir is not None:
            scale_file = Path(self.logger.save_dir) / 'temperature.scales'

            temperature_scales = {
                'crop': self.crop_temperature.item()
            }
            with open(scale_file, mode='w') as f:
                f.write(json.dumps(temperature_scales))

    def configure_loss(self):
        self.crop_loss = TanimotoDistLoss(scale_pos_weight=True)
        self.crop_loss_refine = TanimotoDistLoss(
            scale_pos_weight=True
        )

    def configure_optimizers(self):
        optimizer_adamw = torch.optim.AdamW(
            list(self.geo_refine_model.parameters()),
            lr=self.learning_rate_refine,
            weight_decay=self.weight_decay,
            eps=self.eps
        )
        optimizer_lbfgs = torch.optim.LBFGS(
            [self.crop_temperature],
            lr=self.learning_rate_lbfgs,
            max_iter=self.max_iter,
            line_search_fn=None
        )
        model_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer_adamw,
            T_max=20,
            eta_min=1e-5,
            last_epoch=-1
        )

        return (
            {
                'optimizer': optimizer_adamw,
                'lr_scheduler': {
                    'scheduler': model_lr_scheduler,
                    'monitor': 'loss',
                    'interval': 'epoch',
                    'frequency': 1
                },
            },
            {
                'optimizer': optimizer_lbfgs
            }
        )


class CultioLitModel(LightningModule):
    def __init__(
        self,
        num_features: int = None,
        num_time_features: int = None,
        num_classes: int = 2,
        filters: int = 32,
        model_type: str = 'ResUNet3Psi',
        activation_type: str = 'SiLU',
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = 'resa',
        attention_weights: str = 'spatial_channel',
        optimizer: str = 'AdamW',
        learning_rate: float = 1e-3,
        lr_scheduler: str = 'CosineAnnealingLR',
        steplr_step_size: int = 5,
        weight_decay: float = 0.01,
        eps: float = 1e-4,
        ckpt_name: str = 'last',
        model_name: str = 'cultionet',
        deep_sup_dist: bool = False,
        deep_sup_edge: bool = False,
        deep_sup_mask: bool = False,
        class_counts: T.Optional[torch.Tensor] = None,
        edge_class: T.Optional[int] = None,
        crop_temperature: T.Optional[float] = None,
        temperature_lit_model: T.Optional[LightningModule] = None,
        scale_pos_weight: T.Optional[bool] = True,
        save_batch_val_metrics: T.Optional[bool] = False
    ):
        """Lightning model
        """
        super(CultioLitModel, self).__init__()

        self.save_hyperparameters()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.steplr_step_size = steplr_step_size
        self.weight_decay = weight_decay
        self.eps = eps
        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_time_features = num_time_features
        self.class_counts = class_counts
        self.crop_temperature = crop_temperature
        self.temperature_lit_model = temperature_lit_model
        self.scale_pos_weight = scale_pos_weight
        self.save_batch_val_metrics = save_batch_val_metrics
        self.deep_sup_dist = deep_sup_dist
        self.deep_sup_edge = deep_sup_edge
        self.deep_sup_mask = deep_sup_mask
        self.sigmoid = torch.nn.Sigmoid()
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes

        self.model_attr = f'{model_name}_{model_type}'
        setattr(
            self,
            self.model_attr,
            CultioNet(
                ds_features=num_features,
                ds_time_features=num_time_features,
                filters=filters,
                num_classes=self.num_classes,
                model_type=model_type,
                activation_type=activation_type,
                dilations=dilations,
                res_block_type=res_block_type,
                attention_weights=attention_weights,
                deep_sup_dist=deep_sup_dist,
                deep_sup_edge=deep_sup_edge,
                deep_sup_mask=deep_sup_mask
            )
        )
        self.configure_loss()
        self.configure_scorer()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def cultionet_model(self) -> CultioNet:
        return getattr(
            self, self.model_attr
        )

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """Performs a single model forward pass

        Returns:
            distance: Normalized distance transform (from boundaries), [0,1].
            edge: Probabilities of edge|non-edge, [0,1].
            crop: Logits of crop|non-crop.
        """
        return self.cultionet_model(batch)

    @staticmethod
    def get_cuda_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f'{t * 1e-6:.02f}MB', f'{r * 1e-6:.02f}MB', f'{a * 1e-6:.02f}MB')

    def predict_step(
        self,
        batch: Data,
        batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """A prediction step for Lightning
        """
        predictions = self.forward(batch, batch_idx)
        if self.temperature_lit_model is not None:
            predictions = self.temperature_lit_model(
                predictions,
                batch,
                self.crop_temperature,
                batch_idx
            )

        return predictions

    def get_true_labels(
        self, batch: Data, crop_type: torch.Tensor = None
    ) -> T.Dict[str, T.Union[None, torch.Tensor]]:
        """Gets true labels from the data batch
        """
        true_edge = torch.where(
            batch.y == self.edge_class, 1, 0
        ).long()
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
            (batch.y > 0) & (batch.y < self.edge_class), 1,
            torch.where(
                batch.y == self.edge_class, 2, 0
            )
        ).long()
        true_crop_type = None
        if crop_type is not None:
            # Leave all crop classes as they are
            true_crop_type = torch.where(
                batch.y == self.edge_class, 0, batch.y
            ).long()

        return {
            'true_edge': true_edge,
            'true_crop': true_crop,
            'true_crop_and_edge': true_crop_and_edge,
            'true_crop_or_edge': true_crop_or_edge,
            'true_crop_type': true_crop_type
        }

    def softmax(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return F.softmax(x, dim=dim, dtype=x.dtype)

    def probas_to_labels(
        self, x: torch.Tensor, thresh: float = 0.5
    ) -> torch.Tensor:
        if x.shape[1] == 1:
            labels = x.gt(thresh).long()
        else:
            labels = x.argmax(dim=1).long()

        return labels

    def logits_to_probas(
        self, x: torch.Tensor
    ) -> T.Union[None, torch.Tensor]:
        if x is not None:
            # Single-dimension inputs are sigmoid probabilities
            if x.shape[1] > 1:
                # Transform logits to probabilities
                x = self.softmax(x)
            else:
                x = self.sigmoid(x)
            x = x.clip(0, 1)

        return x

    # def on_train_epoch_start(self):
    #     # Get the current learning rate from the optimizer
    #     eps = self.optimizers().optimizer.param_groups[0]['eps']
    #     weight_decay = self.optimizers().optimizer.param_groups[0]['weight_decay']
    #     if (weight_decay != self.weight_decay) or (eps != self.eps):
    #         self.configure_optimizers()

    def on_validation_epoch_end(self, *args, **kwargs):
        """Save the model on validation end
        """
        if self.logger.save_dir is not None:
            model_file = Path(self.logger.save_dir) / f'{self.model_name}.pt'
            if model_file.is_file():
                model_file.unlink()
            torch.save(
                self.state_dict(), model_file
            )

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: T.Dict[str, torch.Tensor]
    ):
        """Calculates the loss

        Returns:
            Total loss
        """
        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions['crop_type']
        )

        # RNN level 2 loss (non-crop=0; crop|edge=1)
        crop_star_l2_loss = self.crop_star_l2_loss(
            predictions['crop_star_l2'], true_labels_dict['true_crop_and_edge']
        )
        # RNN final loss (non-crop=0; crop=1; edge=2)
        crop_star_loss = self.crop_star_loss(
            predictions['crop_star'], true_labels_dict['true_crop_or_edge']
        )
        # Main loss
        loss = (
            # RNN losses
            0.25 * crop_star_l2_loss
            + 0.5 * crop_star_loss
        )
        # Edge losses
        if self.deep_sup_dist:
            dist_loss_3_1 = self.dist_loss_3_1(predictions['dist_3_1'], batch.bdist)
            dist_loss_2_2 = self.dist_loss_2_2(predictions['dist_2_2'], batch.bdist)
            dist_loss_1_3 = self.dist_loss_1_3(predictions['dist_1_3'], batch.bdist)
            # Main loss
            loss = (
                loss
                + 0.1 * dist_loss_3_1
                + 0.25 * dist_loss_2_2
                + 0.5 * dist_loss_1_3
            )
        # Distance transform loss
        dist_loss = self.dist_loss(predictions['dist'], batch.bdist)
        # Main loss
        loss = loss + dist_loss
        # Distance transform losses
        if self.deep_sup_edge:
            edge_loss_3_1 = self.edge_loss_3_1(predictions['edge_3_1'], true_labels_dict['true_edge'])
            edge_loss_2_2 = self.edge_loss_2_2(predictions['edge_2_2'], true_labels_dict['true_edge'])
            edge_loss_1_3 = self.edge_loss_1_3(predictions['edge_1_3'], true_labels_dict['true_edge'])
            # Main loss
            loss = (
                loss
                + 0.1 * edge_loss_3_1
                + 0.25 * edge_loss_2_2
                + 0.5 * edge_loss_1_3
            )
        # Edge loss
        edge_loss = self.edge_loss(predictions['edge'], true_labels_dict['true_edge'])
        # Main loss
        loss = loss + edge_loss
        # Crop mask losses
        if self.deep_sup_mask:
            crop_loss_3_1 = self.crop_loss_3_1(predictions['crop_3_1'], true_labels_dict['true_crop'])
            crop_loss_2_2 = self.crop_loss_2_2(predictions['crop_2_2'], true_labels_dict['true_crop'])
            crop_loss_1_3 = self.crop_loss_1_3(predictions['crop_1_3'], true_labels_dict['true_crop'])
            # Main loss
            loss = (
                loss
                + 0.1 * crop_loss_3_1
                + 0.25 * crop_loss_2_2
                + 0.5 * crop_loss_1_3
            )
        # Crop mask loss
        crop_loss = self.crop_loss(predictions['crop'], true_labels_dict['true_crop'])
        # Main loss
        loss = loss + crop_loss

        if predictions['crop_type'] is not None:
            # Upstream (deep) loss on crop-type
            crop_type_star_loss = self.crop_type_star_loss(
                predictions['crop_type_star'], true_labels_dict['true_crop_type']
            )
            loss = loss + crop_type_star_loss
            # Loss on crop-type
            crop_type_loss = self.crop_type_loss(
                predictions['crop_type'], true_labels_dict['true_crop_type']
            )
            loss = loss + crop_type_loss

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step and logs training step metrics
        """
        predictions = self(batch)
        loss = self.calc_loss(
            batch,
            predictions
        )
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _shared_eval_step(self, batch: Data, batch_idx: int = None) -> dict:
        predictions = self(batch)
        loss = self.calc_loss(
            batch,
            predictions
        )

        dist_mae = self.dist_mae(
            predictions['dist'].contiguous().view(-1),
            batch.bdist.contiguous().view(-1)
        )
        dist_mse = self.dist_mse(
            predictions['dist'].contiguous().view(-1),
            batch.bdist.contiguous().view(-1)
        )
        # Get the class labels
        edge_ypred = self.probas_to_labels(predictions['edge'])
        crop_ypred = self.probas_to_labels(predictions['crop'])
        # Get the true edge and crop labels
        true_labels_dict = self.get_true_labels(
            batch, crop_type=predictions['crop_type']
        )
        # F1-score
        edge_score = self.edge_f1(edge_ypred, true_labels_dict['true_edge'])
        crop_score = self.crop_f1(crop_ypred, true_labels_dict['true_crop'])
        # MCC
        edge_mcc = self.edge_mcc(edge_ypred, true_labels_dict['true_edge'])
        crop_mcc = self.crop_mcc(crop_ypred, true_labels_dict['true_crop'])
        # Dice
        edge_dice = self.edge_dice(edge_ypred, true_labels_dict['true_edge'])
        crop_dice = self.crop_dice(crop_ypred, true_labels_dict['true_crop'])
        # Jaccard/IoU
        edge_jaccard = self.edge_jaccard(edge_ypred, true_labels_dict['true_edge'])
        crop_jaccard = self.crop_jaccard(crop_ypred, true_labels_dict['true_crop'])

        total_score = (
            loss
            + (1.0 - edge_score)
            + (1.0 - crop_score)
            + dist_mae
            + (1.0 - edge_mcc)
            + (1.0 - crop_mcc)
        )

        metrics = {
            'loss': loss,
            'dist_mae': dist_mae,
            'dist_mse': dist_mse,
            'edge_f1': edge_score,
            'crop_f1': crop_score,
            'edge_mcc': edge_mcc,
            'crop_mcc': crop_mcc,
            'edge_dice': edge_dice,
            'crop_dice': crop_dice,
            'edge_jaccard': edge_jaccard,
            'crop_jaccard': crop_jaccard,
            'score': total_score
        }
        if predictions['crop_type'] is not None:
            crop_type_ypred = self.probas_to_labels(
                self.logits_to_probas(
                    predictions['crop_type']
                )
            )
            crop_type_score = self.crop_type_f1(
                crop_type_ypred, true_labels_dict['true_crop_type']
            )
            metrics['crop_type_f1'] = crop_type_score

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step
        """
        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            'val_loss': eval_metrics['loss'],
            'vef1': eval_metrics['edge_f1'],
            'vcf1': eval_metrics['crop_f1'],
            'vmae': eval_metrics['dist_mae'],
            'val_score': eval_metrics['score']
        }
        if 'crop_type_f1' in eval_metrics:
            metrics['vctf1'] = eval_metrics['crop_type_f1']

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        if self.save_batch_val_metrics:
            self._save_batch_metrics(metrics, self.current_epoch, batch)

        return metrics

    def _save_batch_metrics(
        self,
        metrics: T.Dict[str, torch.Tensor],
        epoch: int,
        batch: Data
    ) -> None:
        """Saves batch metrics to a parquet file
        """
        if not self.trainer.sanity_checking:
            write_metrics = {
                'epoch': [epoch] * len(batch.train_id),
                'train_ids': batch.train_id
            }
            for k, v in metrics.items():
                write_metrics[k] = [float(v)] * len(batch.train_id)
            if self.logger.save_dir is not None:
                metrics_file = Path(self.logger.save_dir) / 'batch_metrics.parquet'
                if not metrics_file.is_file():
                    df = pd.DataFrame(write_metrics)
                    df.to_parquet(metrics_file)
                else:
                    df_new = pd.DataFrame(write_metrics)
                    df = pd.read_parquet(metrics_file)
                    df = pd.concat((df, df_new), axis=0)
                    df.to_parquet(metrics_file)

    def test_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one test step
        """
        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            'test_loss': eval_metrics['loss'],
            'tmae': eval_metrics['dist_mae'],
            'tmse': eval_metrics['dist_mse'],
            'tef1': eval_metrics['edge_f1'],
            'tcf1': eval_metrics['crop_f1'],
            'temcc': eval_metrics['edge_mcc'],
            'tcmcc': eval_metrics['crop_mcc'],
            'tedice': eval_metrics['edge_dice'],
            'tcdice': eval_metrics['crop_dice'],
            'tejaccard': eval_metrics['edge_jaccard'],
            'tcjaccard': eval_metrics['crop_jaccard'],
            'test_score': eval_metrics['score']
        }
        if 'crop_type_f1' in eval_metrics:
            metrics['tctf1'] = eval_metrics['crop_type_f1']

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        self.dist_mae = torchmetrics.MeanAbsoluteError()
        self.dist_mse = torchmetrics.MeanSquaredError()
        self.edge_f1 = torchmetrics.F1Score(num_classes=2, average='micro')
        self.crop_f1 = torchmetrics.F1Score(num_classes=2, average='micro')
        self.edge_mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)
        self.crop_mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)
        self.edge_dice = torchmetrics.Dice(num_classes=2, average='micro')
        self.crop_dice = torchmetrics.Dice(num_classes=2, average='micro')
        self.edge_jaccard = torchmetrics.JaccardIndex(
            average='micro',
            num_classes=2
        )
        self.crop_jaccard = torchmetrics.JaccardIndex(
            average='micro',
            num_classes=2
        )
        if self.num_classes > 2:
            self.crop_type_f1 = torchmetrics.F1Score(
                num_classes=self.num_classes, average='weighted', ignore_index=0
            )

    def configure_loss(self):
        self.dist_loss = TanimotoDistLoss()
        if self.deep_sup_dist:
            self.dist_loss_3_1 = TanimotoDistLoss()
            self.dist_loss_2_2 = TanimotoDistLoss()
            self.dist_loss_1_3 = TanimotoDistLoss()
        # Edge losses
        self.edge_loss = TanimotoDistLoss()
        if self.deep_sup_edge:
            self.edge_loss_3_1 = TanimotoDistLoss()
            self.edge_loss_2_2 = TanimotoDistLoss()
            self.edge_loss_1_3 = TanimotoDistLoss()
        # Crop mask losses
        self.crop_loss = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)
        if self.deep_sup_mask:
            self.crop_loss_3_1 = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)
            self.crop_loss_2_2 = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)
            self.crop_loss_1_3 = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)
        # Crop RNN losses
        self.crop_star_l2_loss = TanimotoDistLoss()
        self.crop_star_loss = TanimotoDistLoss()
        # FIXME:
        if self.num_classes > 2:
            self.crop_type_star_loss = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)
            self.crop_type_loss = TanimotoDistLoss(scale_pos_weight=self.scale_pos_weight)

    def configure_optimizers(self):
        params_list = list(self.cultionet_model.parameters())
        if self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                params_list,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.eps
            )
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                params_list,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise NameError("Choose either 'AdamW' or 'SGD'.")

        if self.lr_scheduler == 'ExponentialLR':
            model_lr_scheduler = lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.5
            )
        elif self.lr_scheduler == 'CosineAnnealingLR':
            model_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=20,
                eta_min=1e-5,
                last_epoch=-1
            )
        elif self.lr_scheduler == 'StepLR':
            model_lr_scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=self.steplr_step_size,
                gamma=0.5
            )
        else:
            raise NameError('The learning rate scheduler is not implemented in Cultionet.')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': model_lr_scheduler,
                'name': 'lr_sch',
                'monitor': 'val_score',
                'interval': 'epoch',
                'frequency': 1
            },
        }
