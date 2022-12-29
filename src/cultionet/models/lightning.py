import typing as T
from pathlib import Path
import json

from ..losses import (
    CrossEntropyLoss,
    MSELoss,
    TanimotoDistLoss
)
from .cultio import CultioNet
from .maskcrnn import BFasterRCNN
from . import model_utils

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import pytorch_lightning as pl
from torchvision.ops import box_iou
from torchvision import transforms
import torchmetrics

import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('lightning').addHandler(logging.NullHandler())
logging.getLogger('lightning').propagate = False
logging.getLogger('lightning').setLevel(logging.ERROR)


class MaskRCNNLitModel(pl.LightningModule):
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
        lr_scheduler = ReduceLROnPlateau(
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


class TemperatureScaling(pl.LightningModule):
    def __init__(
        self,
        model: CultioNet = None,
        learning_rate: float = 0.01,
        max_iter: float = 20,
        class_weights: T.Sequence[float] = None,
        edge_class: T.Optional[int] = None
    ):
        super(TemperatureScaling, self).__init__()

        self.edge_temperature = torch.nn.Parameter(torch.ones(1))
        self.crop_temperature = torch.nn.Parameter(torch.ones(1))

        self.model = model
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.class_weights = class_weights
        self.edge_class = edge_class

        self.model.eval()
        self.model.freeze()
        self.configure_loss()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return scale_logits(x, t)

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        predictions: torch.Tensor
    ):
        edge = self(predictions['edge'], self.edge_temperature)
        crop = self(predictions['crop'], self.crop_temperature)

        true_edge = batch.y.eq(self.edge_class).long()
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(
            (batch.y > 0) & (batch.y != self.edge_class), 1, 0
        ).long()

        edge_loss = self.edge_loss(edge, true_edge)
        crop_loss = self.crop_loss(crop, true_crop)
        loss = edge_loss + crop_loss

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
        """
        with torch.no_grad():
            predictions = self.model(batch)

        loss = self.calc_loss(
            batch,
            predictions
        )
        metrics = {
            'loss': loss,
            'edge_temp': self.edge_temperature,
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
                'edge': self.edge_temperature.item(),
                'crop': self.crop_temperature.item()
            }
            with open(scale_file, mode='w') as f:
                f.write(json.dumps(temperature_scales))

    def configure_loss(self):
        self.edge_loss = TanimotoDistLoss()
        self.crop_loss = TanimotoDistLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(
            [
                self.edge_temperature,
                self.crop_temperature
            ],
            lr=self.learning_rate,
            max_iter=self.max_iter,
            line_search_fn=None
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'loss',
                'interval': 'epoch',
                'frequency': 1
            },
        }


class CultioLitModel(pl.LightningModule):
    def __init__(
        self,
        num_features: int = None,
        num_time_features: int = None,
        num_classes: int = 2,
        filters: int = 32,
        star_rnn_n_layers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        depth: int = 5,
        ckpt_name: str = 'last',
        model_name: str = 'cultionet',
        model_type: str = 'ResUNet3Psi',
        class_weights: T.Sequence[float] = None,
        edge_weights: T.Sequence[float] = None,
        edge_class: T.Optional[int] = None,
        edge_temperature: T.Optional[float] = None,
        crop_temperature: T.Optional[float] = None
    ):
        """Lightning model
        """
        super(CultioLitModel, self).__init__()

        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_time_features = num_time_features
        self.class_weights = class_weights
        self.edge_weights = edge_weights
        self.edge_temperature = edge_temperature
        self.crop_temperature = crop_temperature
        if edge_class is not None:
            self.edge_class = edge_class
        else:
            self.edge_class = num_classes
        self.depth = depth

        self.cultionet_model = CultioNet(
            ds_features=num_features,
            ds_time_features=num_time_features,
            filters=filters,
            star_rnn_hidden_dim=filters,
            star_rnn_n_layers=star_rnn_n_layers,
            num_classes=self.num_classes,
            model_type=model_type
        )
        self.configure_loss()
        self.configure_scorer()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Dict[str, torch.Tensor]:
        """Performs a single model forward pass

        Returns:
            distance: Normalized distance from boundaries [0,1].
            edge: Logits of edge|non-edge.
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
        if self.edge_temperature is not None:
            predictions['edge'] = scale_logits(
                predictions['edge'],
                self.edge_temperature
            )
        if self.crop_temperature is not None:
            predictions['crop'] = scale_logits(
                predictions['crop'],
                self.crop_temperature
            )
        predictions['edge'] = self.logits_to_probas(
            predictions['edge']
        )
        predictions['crop'] = self.logits_to_probas(
            predictions['crop']
        )
        predictions['crop_type'] = self.logits_to_probas(
            predictions['crop_type']
        )

        return predictions

    def get_true_labels(
        self, batch: Data
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        edge_true = torch.where(
            batch.y == self.edge_class, 1, 0
        ).long()
        crop_true = torch.where(
            (batch.y > 0) & (batch.y != self.edge_class), 1, 0
        ).long()

        return edge_true, crop_true

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

        return x

    # def on_train_epoch_start(self):
    #     """
    #     Set the depth for the d hyperparameter in the Tanimoto loss

    #     Source:
    #         https://arxiv.org/pdf/2009.02062.pdf
    #     """
    #     # Get the current learning rate from the optimizer
    #     lr = self.optimizers().optimizer.param_groups[0]['lr']
    #     if lr == self.learning_rate:
    #         self.depth = 1
    #     elif 1e-5 < lr < 1e-3:
    #         self.depth = 10
    #     else:
    #         self.depth = 20

    #     self.configure_loss()

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
        true_edge, true_crop = self.get_true_labels(batch)

        dist_loss = self.dist_loss(predictions['dist'], batch.bdist)
        edge_loss = self.edge_loss(predictions['edge'], true_edge)
        crop_loss = self.crop_loss(predictions['crop'], true_crop)

        loss = (
            dist_loss
            + edge_loss
            + crop_loss
        )
        if predictions['crop_type'] is not None:
            true_crop_type = torch.where(
                batch.y == self.edge_class, 0, batch.y
            ).long()
            crop_type_loss = self.crop_type_loss(
                predictions['crop_type'], true_crop_type
            )
            loss = loss + crop_type_loss
        # else:
        #     true_crop = torch.where(
        #         true_crop == 1, 1,
        #         torch.where(
        #             true_edge == 1, 2, 0
        #         )
        #     )
        #     crop_loss_star = self.crop_rnn_loss(predictions['crop_star'], true_crop)
        #     loss = loss + crop_loss_star

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
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
        edge_ypred = self.probas_to_labels(
            self.logits_to_probas(
                predictions['edge']
            )
        )
        crop_ypred = self.probas_to_labels(
            self.logits_to_probas(
                predictions['crop']
            )
        )
        # Get the true edge and crop labels
        edge_ytrue, crop_ytrue = self.get_true_labels(batch)
        # F1-score
        edge_score = self.edge_f1(edge_ypred, edge_ytrue)
        crop_score = self.crop_f1(crop_ypred, crop_ytrue)
        # MCC
        edge_mcc = self.edge_mcc(edge_ypred, edge_ytrue)
        crop_mcc = self.crop_mcc(crop_ypred, crop_ytrue)
        # Dice
        edge_dice = self.edge_dice(edge_ypred, edge_ytrue)
        crop_dice = self.crop_dice(crop_ypred, crop_ytrue)
        # Jaccard/IoU
        edge_jaccard = self.edge_jaccard(edge_ypred, edge_ytrue)
        crop_jaccard = self.crop_jaccard(crop_ypred, crop_ytrue)

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
            'crop_jaccard': crop_jaccard
        }
        if predictions['crop_type'] is not None:
            crop_type_ypred = self.probas_to_labels(
                self.logits_to_probas(
                    predictions['crop_type']
                )
            )
            crop_type_ytrue = torch.where(
                batch.y == self.edge_class, 0, batch.y
            ).long()
            crop_type_score = self.crop_type_f1(crop_type_ypred, crop_type_ytrue)
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
            'vemcc': eval_metrics['edge_mcc'],
            'vcmcc': eval_metrics['crop_mcc'],
            'vmae': eval_metrics['dist_mae']
        }
        if 'crop_type_f1' in eval_metrics:
            metrics['vctf1'] = eval_metrics['crop_type_f1']
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

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
            'tcjaccard': eval_metrics['crop_jaccard']
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
        self.edge_loss = TanimotoDistLoss()
        self.crop_loss = TanimotoDistLoss()
        # self.crop_rnn_loss = TanimotoDistLoss()
        if self.num_classes > 2:
            self.crop_type_loss = CrossEntropyLoss(
                weight=self.class_weights
            )

    def configure_optimizers(self):
        params_list = list(self.cultionet_model.parameters())
        optimizer = torch.optim.AdamW(
            params_list,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.eps
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
        }
