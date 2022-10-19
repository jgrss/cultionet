import typing as T

from .cultio import CultioGraphNet
from .maskcrnn import BFasterRCNN
from .refinement import RefineConv
from . import model_utils
from ..losses import (
    HuberLoss,
    TanimotoDistanceLoss,
    F1Score,
    MatthewsCorrcoef
)

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import pytorch_lightning as pl
from torchvision import transforms

import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('lightning').addHandler(logging.NullHandler())
logging.getLogger('lightning').propagate = False
logging.getLogger('lightning').setLevel(logging.ERROR)


class CultioLitModel(pl.LightningModule):
    def __init__(
        self,
        num_features: int = None,
        num_time_features: int = None,
        filters: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        rheight: int = 201,
        rwidth: int = 201
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
        self.num_classes = 2
        self.edge_value = 2
        self.crop_value = 1
        self.num_time_features = num_time_features
        self.rheight= rheight
        self.rwidth = rwidth

        self.model = CultioGraphNet(
            ds_features=num_features,
            ds_time_features=num_time_features,
            filters=filters,
            num_classes=self.num_classes,
            rheight=self.rheight,
            rwidth=self.rwidth
        )
        self.refine = RefineConv(
            in_channels=1+self.num_classes*2,
            mid_channels=256,
            out_channels=self.num_classes
        )
        self.maskrcnn = BFasterRCNN(
            in_channels=4,
            out_channels=256,
            num_classes=self.num_classes,
            sizes=(16, 32, 64, 128, 256),
            aspect_ratios=(0.5, 1.0, 3.0),
            trainable_backbone_layers=3,
            min_image_size=100,
            max_image_size=400
        )

        self.configure_loss()
        self.configure_scorer()

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
        scale = self.rheight / height
        resizer = transforms.Resize((self.rheight, self.rwidth))
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
                        batch_dict[k] = y[k][batch_slice] * scale
                    else:
                        batch_dict[k] = y[k][batch_slice]
                    targets.append(batch_dict)
        outputs = self.maskrcnn(x, targets)

        return outputs

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
            crop_r: Probability of refined crop [0,1].
        """
        distance_ori, distance, edge, crop = self.model(batch)
        height = int(batch.height) if batch.batch is None else int(batch.height[0])
        width = int(batch.width) if batch.batch is None else int(batch.width[0])
        batch_size = 1 if batch.batch is None else batch.batch.unique().size(0)

        crop_r = self.refine(
            torch.cat([
                distance,
                F.log_softmax(edge, dim=1),
                F.log_softmax(crop, dim=1)
            ], dim=1),
            batch_size,
            height,
            width
        )

        # Transform edge and crop logits to probabilities
        edge = F.softmax(edge, dim=1)
        crop = F.softmax(crop, dim=1)
        crop_r = F.softmax(crop_r, dim=1)

        return distance_ori, distance, edge, crop, crop_r

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
            distance_ori, distance, edge, crop, crop_r = self(batch)

        # Take the argmax of the class probabilities
        edge_labels = edge.argmax(dim=1)
        class_labels = crop.argmax(dim=1)
        crop_r_labels = crop_r.argmax(dim=1)

        return distance_ori, distance, edge_labels, class_labels, crop_r_labels

    def calc_loss(
        self,
        batch: T.Union[Data, T.List],
        include_mask: bool = True
    ):
        """Calculates the loss for each layer

        Returns:
            Average loss

        Reference:
            @article{waldner2020deep,
              title={Deep learning on edge: Extracting field boundaries from satellite images with a convolutional neural network},
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

        distance_ori, distance, edge, crop, crop_r = self(batch)

        oloss = self.dloss(distance_ori, batch.ori)
        dloss = self.dloss(distance, batch.bdist)
        eloss = self.eloss(edge, (batch.y == 2).long())
        closs = self.closs(crop, (batch.y == 1).long())
        crop_r_loss = self.closs(crop_r, (batch.y == 1).long())

        loss = oloss + dloss + eloss + closs + crop_r_loss
        denom = 5.0
        if include_mask:
            if hasattr(batch, 'boxes') and batch.boxes is not None:
                mask_y = {
                    'boxes': batch.boxes,
                    'labels': batch.box_labels,
                    'masks': batch.box_masks,
                    'image_id': batch.image_id
                }
                masks_losses = self.mask_forward(
                    distance_ori,
                    distance,
                    edge,
                    crop_r,
                    height=batch.height,
                    width=batch.width,
                    batch=batch.batch,
                    y=mask_y
                )
                masks_losses = sum(loss for loss in masks_losses.values())
                loss = loss + masks_losses
                denom += 1.0

        loss = loss / denom

        return loss

    def training_step(self, batch: Data, batch_idx: int = None):
        """Executes one training step
        """
        loss = self.calc_loss(batch, include_mask=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # print('Train')
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # print(t, r, a)

        return loss

    def _shared_eval_step(self, batch: Data, batch_idx: int = None) -> dict:
        loss = self.calc_loss(batch, include_mask=False)

        distance_ori, distance, edge, __, crop_r = self(batch)

        box_score = torch.tensor(0.0, device=self.device)
        if hasattr(batch, 'boxes'):
            masks = self.mask_forward(
                distance_ori,
                distance,
                edge,
                crop_r,
                height=batch.height,
                width=batch.width,
                batch=batch.batch,
                y=None
            )
            for bidx in range(0, batch.batch.unique().size(0)):
                box_score = torch.maximum(box_score, torch.nanmean(masks[bidx]['scores']))
            # box_score /= batch.batch.unique().size(0)

        # Take the argmax of the class probabilities
        edge_ypred = edge.argmax(dim=1)
        class_ypred_r = crop_r.argmax(dim=1)

        # F1-score
        edge_score = self.scorer(edge_ypred, batch.y.eq(self.edge_value).long())
        class_score = self.scorer(class_ypred_r, batch.y.eq(self.crop_value).long())

        # MCC
        edge_mcc = self.mcc(edge_ypred, batch.y.eq(self.edge_value).long())
        class_mcc = self.mcc(class_ypred_r, batch.y.eq(self.crop_value).long())

        metrics = {
            'loss': loss,
            'edge_score': edge_score,
            'class_score': class_score,
            'emcc': edge_mcc,
            'cmcc': class_mcc,
            'box_score': box_score
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
            'vcmcc': eval_metrics['cmcc'],
            'vboxs': eval_metrics['box_score']
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
            'tcmcc': eval_metrics['cmcc'],
            'tboxs': eval_metrics['box_score']
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        self.scorer = F1Score(num_classes=self.num_classes)
        self.mcc = MatthewsCorrcoef(num_classes=self.num_classes, inputs_are_logits=False)

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
        optimizer = torch.optim.AdamW(
            list(
                list(self.model.parameters())
                + list(self.refine.parameters())
                + list(self.maskrcnn.parameters())
            ),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-4
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=10
        )

        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }
