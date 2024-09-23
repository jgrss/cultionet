import typing as T

import einops
import torch
import torch.nn as nn

from ..data import Data
from ..enums import AttentionTypes, InferenceNames, ModelTypes, ResBlockTypes
from .nunet import TowerUNet


class CultioNet(nn.Module):
    """The cultionet model framework.

    Parameters
    ==========
    in_channels
        The total number of dataset features (bands x time).
    in_time
        The number of dataset time features in each band/channel.
    hidden_channels
        The number of hidden channels.
    model_type
        The model architecture type.
    activation_type
        The nonlinear activation.
    dropout
        The dropout fraction / probability.
    dilations
        The convolution dilation or dilations.
    res_block_type
        The residual convolution block type.
    attention_weights
        The attention weight type.
    pool_by_max
        Whether to apply max pooling before residual block.
    batchnorm_first
        Whether to apply BatchNorm2d -> Activation -> Convolution2d. Otherwise,
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
        res_block_type: str = ResBlockTypes.RESA,
        attention_weights: str = AttentionTypes.NATTEN,
        pool_by_max: bool = False,
        batchnorm_first: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_time = in_time
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        mask_model_kwargs = {
            "in_channels": self.in_channels,
            "in_time": self.in_time,
            "hidden_channels": self.hidden_channels,
            "num_classes": 1,
            "attention_weights": attention_weights,
            "res_block_type": res_block_type,
            "dropout": dropout,
            "dilations": dilations,
            "activation_type": activation_type,
            "edge_activation": True,
            "mask_activation": True,
            "pool_by_max": pool_by_max,
            "batchnorm_first": batchnorm_first,
            "use_latlon": False,
        }

        assert model_type in (
            ModelTypes.TOWERUNET
        ), "The model type is not supported."

        self.mask_model = TowerUNet(**mask_model_kwargs)

    def forward(self, batch: Data) -> T.Dict[str, torch.Tensor]:

        latlon_coords = torch.cat(
            (
                einops.rearrange(batch.lon, 'b -> b 1'),
                einops.rearrange(batch.lat, 'b -> b 1'),
            ),
            dim=1,
        )

        # Main stream
        out = self.mask_model(
            batch.x,
            latlon_coords=latlon_coords,
        )

        out.update(
            {
                InferenceNames.CROP_TYPE: None,
                InferenceNames.CLASSES_L2: None,
                InferenceNames.CLASSES_L3: None,
            }
        )

        return out
