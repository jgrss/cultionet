import typing as T
from pathlib import Path

from ..losses import (
    HuberLoss,
    TanimotoDistanceLoss,
    CrossEntropyLoss
)
from .cultio import CultioGraphNet
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
        learning_rate: float = 0.001,
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


class CultioLitModel(pl.LightningModule):
    def __init__(
        self,
        num_features: int = None,
        num_time_features: int = None,
        num_classes: int = 2,
        filters: int = 64,
        star_rnn_hidden_dim: int = 64,
        star_rnn_n_layers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ckpt_name: str = 'last',
        model_name: str = 'cultionet',
        class_weights: T.Sequence[float] = None,
        edge_weights: T.Sequence[float] = None
    ):
        """Lightning model

        Args:
            num_features
            num_time_features
            filters
            learning_rate
            weight_decay
        """
        super(CultioLitModel, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_time_features = num_time_features
        self.class_weights = class_weights
        self.edge_weights = edge_weights
        self.crop_class = num_classes - 1
        self.edge_class = num_classes

        self.model = CultioGraphNet(
            ds_features=num_features,
            ds_time_features=num_time_features,
            filters=filters,
            star_rnn_hidden_dim=star_rnn_hidden_dim,
            star_rnn_n_layers=star_rnn_n_layers,
            num_classes=self.num_classes
        )
        self.configure_loss()
        self.configure_scorer()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Performs a single model forward pass

        Returns:
            distance: Normalized distance from boundaries [0,1].
            edge: Probability of an edge [0,1].
            crop: Probability of crop [0,1].
        """
        distance_ori, distance, edge, crop, crop_type = self.model(batch)

        # Transform edge and crop logits to probabilities
        edge = F.softmax(edge, dim=1, dtype=edge.dtype)
        crop = F.softmax(crop, dim=1, dtype=crop.dtype)
        crop_type = F.softmax(crop_type, dim=1, dtype=crop.dtype)

        return distance_ori, distance, edge, crop, crop_type

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
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """A prediction step for Lightning
        """
        return self.forward(batch, batch_idx)

    def predict_labels(
        self,
        batch: Data
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Predicts edge and crop labels
        """
        with torch.no_grad():
            distance_ori, distance, edge, crop, crop_type = self(batch)

        # Take the argmax of the class probabilities
        edge_labels = edge.argmax(dim=1)
        crop_labels = crop.argmax(dim=1)
        crop_type_labels = crop_type.argmax(dim=1)

        return distance_ori, distance, edge_labels, crop_labels, crop_type_labels

    def on_validation_epoch_end(self, *args, **kwargs):
        """Save the model on validation end
        """
        model_file = Path(self.logger.save_dir) / f'{self.model_name}.pt'
        if model_file.is_file():
            model_file.unlink()
        torch.save(
            self.state_dict(), model_file
        )

    def calc_loss(self, batch: T.Union[Data, T.List]):
        """Calculates the loss for each layer

        Returns:
            Average loss

        Reference:
            @article{waldner2020deep,
                title={
                    Deep learning on edge: Extracting field boundaries from
                    satellite images with a convolutional neural network
                },
                author={Waldner, Fran{\c{c}}ois and Diakogiannis, Foivos I},
                journal={Remote Sensing of Environment},
                volume={245},
                pages={111741},
                year={2020},
                publisher={Elsevier}
            }
        """
        if self.volume.device != self.device:
            self.configure_loss()

        distance_ori, distance, edge, crop, crop_type = self(batch)

        true_edge = (batch.y == self.edge_class).long()
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(
            (batch.y > 0) & (batch.y < self.edge_class), 1, 0
        ).long()
        true_crop_type = torch.where(batch.y == self.edge_class, 0, batch.y).long()

        ori_loss = self.dist_loss(distance_ori, batch.ori)
        dist_loss = self.dist_loss(distance, batch.bdist)
        edge_loss = self.edge_loss(edge, true_edge)
        crop_loss = self.crop_loss(crop, true_crop)
        crop_type_loss = self.crop_type_loss(crop_type, true_crop_type)

        loss = ori_loss*0.75 + dist_loss + edge_loss + crop_loss + crop_type_loss

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
        """
        loss = self.calc_loss(batch)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def _shared_eval_step(self, batch: Data, batch_idx: int = None) -> dict:
        loss = self.calc_loss(batch)

        __, dist, edge, crop, crop_type = self(batch)

        dist_mae = self.dist_mae(
            dist.contiguous().view(-1), batch.bdist.contiguous().view(-1)
        )
        dist_mse = self.dist_mse(
            dist.contiguous().view(-1), batch.bdist.contiguous().view(-1)
        )

        # Take the argmax of the class probabilities
        edge_ypred = edge.argmax(dim=1).long()
        crop_ypred = crop.argmax(dim=1).long()
        crop_type_ypred = crop_type.argmax(dim=1).long()
        # Get the true edge and crop labels
        edge_ytrue = batch.y.eq(self.edge_class).long()
        crop_ytrue = torch.where(
            (batch.y > 0) & (batch.y < self.edge_class), 1, 0
        ).long()
        crop_type_ytrue = torch.where(
            batch.y == self.edge_class, 0, batch.y
        ).long()
        # F1-score
        edge_score = self.edge_f1(edge_ypred, edge_ytrue)
        crop_score = self.crop_f1(crop_ypred, crop_ytrue)
        crop_type_score = self.crop_type_f1(crop_type_ypred, crop_type_ytrue)
        # MCC
        edge_mcc = self.edge_mcc(edge_ypred, edge_ytrue)
        crop_mcc = self.crop_mcc(crop_ypred, crop_ytrue)
        # Dice
        edge_dice = self.edge_dice(edge_ypred, edge_ytrue)
        crop_dice = self.crop_dice(crop_ypred, crop_ytrue)

        metrics = {
            'loss': loss,
            'dist_mae': dist_mae,
            'dist_mse': dist_mse,
            'edge_f1': edge_score,
            'crop_f1': crop_score,
            'crop_type_f1': crop_type_score,
            'edge_mcc': edge_mcc,
            'crop_mcc': crop_mcc,
            'edge_dice': edge_dice,
            'crop_dice': crop_dice,
        }

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step
        """
        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            'val_loss': eval_metrics['loss'],
            'vef1': eval_metrics['edge_f1'],
            'vcf1': eval_metrics['crop_f1'],
            'vctf1': eval_metrics['crop_type_f1'],
            'vemcc': eval_metrics['edge_mcc'],
            'vcmcc': eval_metrics['crop_mcc']
        }
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
            'tctf1': eval_metrics['crop_type_f1'],
            'temcc': eval_metrics['edge_mcc'],
            'tcmcc': eval_metrics['crop_mcc'],
            'tedice': eval_metrics['edge_dice'],
            'tcdice': eval_metrics['crop_dice']
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        self.dist_mae = torchmetrics.MeanAbsoluteError()
        self.dist_mse = torchmetrics.MeanSquaredError()
        self.edge_f1 = torchmetrics.F1Score(num_classes=2, average='micro')
        self.crop_f1 = torchmetrics.F1Score(num_classes=2, average='micro')
        self.crop_type_f1 = torchmetrics.F1Score(num_classes=self.num_classes, average='weighted')
        self.edge_mcc = torchmetrics.MatthewsCorrCoef(num_classes=2)
        self.crop_mcc = torchmetrics.MatthewsCorrCoef(num_classes=self.num_classes)
        self.edge_dice = torchmetrics.Dice(num_classes=2, average='micro')
        self.crop_dice = torchmetrics.Dice(num_classes=self.num_classes, average='weighted')

    def configure_loss(self):
        self.volume = torch.ones(
            2, dtype=self.dtype, device=self.device
        )

        self.dist_loss = HuberLoss()
        self.edge_loss = TanimotoDistanceLoss(
            volume=self.volume,
            inputs_are_logits=True,
            apply_transform=True
        )
        self.crop_loss = TanimotoDistanceLoss(
            volume=self.volume,
            inputs_are_logits=True,
            apply_transform=True
        )
        self.crop_type_loss = CrossEntropyLoss(class_weights=self.class_weights)

    def configure_optimizers(self):
        params_list = list(self.model.parameters())
        optimizer = torch.optim.AdamW(
            params_list,
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
