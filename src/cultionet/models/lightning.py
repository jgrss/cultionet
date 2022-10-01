import typing as T

from .cultio import CultioGraphNet
from .refinement import RefineConv
from ..losses import QuantileLoss, TanimotoDistanceLoss, F1Score, MatthewsCorrcoef

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import pytorch_lightning as pl

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
        weight_decay: float = 1e-5
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

        self.model = CultioGraphNet(
            ds_features=num_features,
            ds_time_features=num_time_features,
            filters=filters,
            num_classes=self.num_classes
        )

        self.refine = RefineConv(
            in_channels=3+self.num_classes*2,
            mid_channels=128,
            out_channels=self.num_classes
        )

        self.configure_loss()
        self.configure_scorer()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, batch: Data, batch_idx: int = None
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single model forward pass

        Returns:
            distance: Normalized distance from boundaries [0,1].
            edge: Probability of an edge [0,1].
            crop: Probability of crop [0,1].
            crop_r: Probability of refined crop [0,1].
        """
        distance, edge, crop = self.model(batch)
        height = int(batch.height) if batch.batch is None else int(batch.height[0])
        width = int(batch.width) if batch.batch is None else int(batch.width[0])
        batch_size = 1 if batch.batch is None else batch.batch.unique().size(0)

        crop_r = self.refine(
            torch.cat([
                distance, F.log_softmax(edge, dim=1), F.log_softmax(crop, dim=1)
            ], dim=1),
            batch_size,
            height,
            width
        )

        # Transform edge and crop logits to probabilities
        edge = F.softmax(edge, dim=1)
        crop = F.softmax(crop, dim=1)
        crop_r = F.softmax(crop_r, dim=1)

        return distance, edge, crop, crop_r

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
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A prediction step for Lightning
        """
        return self.forward(batch, batch_idx)

    def predict_labels(self, batch: Data) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts edge and crop labels
        """
        with torch.no_grad():
            distance, edge, crop, crop_r = self(batch)

        # Take the argmax of the class probabilities
        edge_labels = edge.argmax(dim=1)
        class_labels = crop.argmax(dim=1)
        crop_r_labels = crop_r.argmax(dim=1)

        return distance, edge_labels, class_labels, crop_r_labels

    def calc_loss(self, batch: T.Union[Data, T.List]):
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

        distance, edge, crop, crop_r = self(batch)

        qloss = self.qloss(distance, batch.bdist)
        eloss = self.eloss(edge, (batch.y == 2).long())
        closs = self.closs(crop, (batch.y == 1).long())
        crop_r_loss = self.closs(crop_r, (batch.y == 1).long())

        loss = (qloss + eloss + closs + crop_r_loss) / 4.0

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

        __, edge_ypred, class_ypred, class_ypred_r = self.predict_labels(batch)

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
            'crop_r_loss': eval_metrics['crop_r_loss'],
            'tf1': eval_metrics['class_score'],
            'temcc': eval_metrics['emcc'],
            'tcmcc': eval_metrics['cmcc']
        }

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return metrics

    def configure_scorer(self):
        self.scorer = F1Score(num_classes=self.num_classes)
        self.mcc = MatthewsCorrcoef(num_classes=self.num_classes, inputs_are_logits=False)

    def configure_loss(self):
        self.volume = torch.ones(self.num_classes, dtype=self.dtype, device=self.device)
        self.qloss = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
        self.eloss = TanimotoDistanceLoss(volume=self.volume, inputs_are_logits=True, apply_transform=False)
        self.closs = TanimotoDistanceLoss(volume=self.volume, inputs_are_logits=True, apply_transform=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.refine.parameters()),
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
