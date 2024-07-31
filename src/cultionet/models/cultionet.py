import typing as T

import torch
import torch.nn as nn

from .. import nn as cunn
from ..data.data import Data
from ..enums import AttentionTypes, ModelTypes, ResBlockTypes
from .nunet import TowerUNet
from .temporal_transformer import TemporalTransformer


class CultioNet(nn.Module):
    """The cultionet model framework.

    Args:
        in_channels (int): The total number of dataset features (bands x time).
        in_time (int): The number of dataset time features in each band/channel.
        hidden_channels (int): The number of hidden channels.
        model_type (str): The model architecture type.
        activation_type (str): The nonlinear activation.
        dropout (float): The dropout fraction / probability.
        dilations (int | list): The convolution dilation or dilations.
        res_block_type (str): The residual convolution block type.
        attention_weights (str): The attention weight type.
        deep_supervision (bool): Whether to use deep supervision.
        pool_attention (bool): Whether to apply attention along the backbone pooling layers.
        pool_by_max (bool): Whether to apply max pooling before residual block.
        repeat_resa_kernel (bool): Whether to repeat the input res-a kernel (otherwise, the first kernel is always 1x1).
        batchnorm_first (bool): Whether to apply BatchNorm2d -> Activation -> Convolution2d. Otherwise,
            apply Convolution2d -> BatchNorm2d -> Activation.
    """

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        model_type: str = ModelTypes.TOWERUNET,
        activation_type: str = "SiLU",
        dropout: float = 0.1,
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: str = AttentionTypes.SPATIAL_CHANNEL,
        deep_supervision: bool = False,
        pool_attention: bool = False,
        pool_by_max: bool = False,
        repeat_resa_kernel: bool = False,
        batchnorm_first: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_time = in_time
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.temporal_encoder = TemporalTransformer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_head=8,
            in_time=self.in_time,
            dropout=0.2,
            num_layers=2,
            d_model=128,
            time_scaler=1_000,
            num_classes_l2=self.num_classes,
            num_classes_last=self.num_classes + 1,
            activation_type=activation_type,
            final_activation=nn.Softmax(dim=1),
        )

        mask_model_kwargs = {
            "in_channels": self.in_channels,
            "in_time": self.in_time,
            "hidden_channels": self.hidden_channels,
            "num_classes": self.num_classes,
            "attention_weights": attention_weights,
            "res_block_type": res_block_type,
            "dropout": dropout,
            "dilations": dilations,
            "activation_type": activation_type,
            "deep_supervision": deep_supervision,
            "pool_attention": pool_attention,
            "mask_activation": nn.Softmax(dim=1),
            "pool_by_max": pool_by_max,
            "repeat_resa_kernel": repeat_resa_kernel,
            "batchnorm_first": batchnorm_first,
        }

        assert model_type in (
            ModelTypes.TOWERUNET
        ), "The model type is not supported."

        self.mask_model = TowerUNet(**mask_model_kwargs)

    def forward(
        self, batch: Data, training: bool = True
    ) -> T.Dict[str, torch.Tensor]:
        # Transformer attention encoder
        transformer_outputs = self.temporal_encoder(batch.x)

        latlon_coords = torch.cat(
            (batch.lon.unsqueeze(1), batch.lat.unsqueeze(1)),
            dim=1,
        ).to(
            dtype=batch.x.dtype,
            device=batch.x.device,
        )

        # Main stream
        out = self.mask_model(
            batch.x,
            temporal_encoding=transformer_outputs["encoded"],
            latlon_coords=latlon_coords,
            training=training,
        )

        out.update(
            {
                "crop_type": None,
                "classes_l2": transformer_outputs["l2"],
                "classes_l3": transformer_outputs["l3"],
            }
        )

        return out
