import typing as T
from pathlib import Path

from ..losses import (
    HuberLoss,
    TanimotoDistanceLoss,
    F1Score,
    MatthewsCorrcoef
)
from ..data.const import CROP_CLASS, EDGE_CLASS
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
        model_name: str = 'cultionet'
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

        self.ckpt_name = ckpt_name
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # 0 = background
        # 1 = crop
        # 2 = [optional] fallow
        self.num_classes = num_classes
        self.num_time_features = num_time_features

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
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Performs a single model forward pass

        Returns:
            distance: Normalized distance from boundaries [0,1].
            edge: Probability of an edge [0,1].
            crop: Probability of crop [0,1].
        """
        distance_ori, distance, edge, crop = self.model(batch)
        # Transform edge and crop logits to probabilities
        edge = F.softmax(edge, dim=1)
        crop = F.softmax(crop, dim=1)

        return distance_ori, distance, edge, crop

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
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """A prediction step for Lightning
        """
        return self.forward(batch, batch_idx)

    def predict_labels(
        self,
        batch: Data
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Predicts edge and crop labels
        """
        with torch.no_grad():
            distance_ori, distance, edge, crop = self(batch)

        # Take the argmax of the class probabilities
        edge_labels = edge.argmax(dim=1)
        class_labels = crop.argmax(dim=1)

        return distance_ori, distance, edge_labels, class_labels

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

        distance_ori, distance, edge, crop = self(batch)

        true_edge = (batch.y == EDGE_CLASS).long()
        # in case of multi-class, `true_crop` = 1, 2, etc.
        true_crop = torch.where(batch.y == EDGE_CLASS, 0, batch.y).long()

        oloss = self.dloss(distance_ori, batch.ori)
        dloss = self.dloss(distance, batch.bdist)
        eloss = self.eloss(edge, true_edge)
        closs = self.closs(crop, true_crop)

        loss = oloss*0.5 + dloss*0.75 + eloss + closs

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
        """
        loss = self.calc_loss(batch)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # print('Train')
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # print(t, r, a)

        return loss

    def _shared_eval_step(self, batch: Data, batch_idx: int = None) -> dict:
        loss = self.calc_loss(batch)

        __, __, edge, crop = self(batch)

        # Take the argmax of the class probabilities
        edge_ypred = edge.argmax(dim=1)
        class_ypred = crop.argmax(dim=1)
        # F1-score
        edge_score = self.scorer(edge_ypred, batch.y.eq(EDGE_CLASS).long())
        class_score = self.scorer(class_ypred, batch.y.eq(CROP_CLASS).long())
        # MCC
        edge_mcc = self.mcc(edge_ypred, batch.y.eq(EDGE_CLASS).long())
        class_mcc = self.mcc(class_ypred, batch.y.eq(CROP_CLASS).long())

        metrics = {
            'loss': loss,
            'edge_score': edge_score,
            'class_score': class_score,
            'emcc': edge_mcc,
            'cmcc': class_mcc
        }
        # print('Eval')
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # print(t, r, a)

        return metrics

    def validation_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one valuation step
        """
        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            'val_loss': eval_metrics['loss'],
            'vef1': eval_metrics['edge_score'],
            'vcf1': eval_metrics['class_score'],
            'vemcc': eval_metrics['emcc'],
            'vcmcc': eval_metrics['cmcc']
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def test_step(self, batch: Data, batch_idx: int = None) -> dict:
        """Executes one test step
        """
        eval_metrics = self._shared_eval_step(batch, batch_idx)

        metrics = {
            'test_loss': eval_metrics['loss'],
            'tef1': eval_metrics['edge_score'],
            'tcf1': eval_metrics['class_score'],
            'temcc': eval_metrics['emcc'],
            'tcmcc': eval_metrics['cmcc']
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        self.scorer = F1Score(num_classes=self.num_classes)
        self.mcc = MatthewsCorrcoef(
            num_classes=self.num_classes, inputs_are_logits=False
        )

    def configure_loss(self):
        self.volume = torch.ones(
            self.num_classes, dtype=self.dtype, device=self.device
        )
        self.dloss = HuberLoss()
        self.eloss = TanimotoDistanceLoss(
            volume=self.volume, inputs_are_logits=True, apply_transform=False
        )
        self.closs = TanimotoDistanceLoss(
            volume=self.volume, inputs_are_logits=True, apply_transform=False
        )

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
