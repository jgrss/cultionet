import typing as T
import warnings

import torch
import torch.nn as nn

from .. import nn as cunn
from ..data.data import Data
from ..enums import ModelTypes, ResBlockTypes
from .nunet import ResUNet3Psi, TowerUNet, UNet3Psi
from .temporal_transformer import TemporalTransformer


def scale_min_max(
    x: torch.Tensor,
    min_in: float,
    max_in: float,
    min_out: float,
    max_out: float,
) -> torch.Tensor:
    return (((max_out - min_out) * (x - min_in)) / (max_in - min_in)) + min_out


class GeoRefinement(nn.Module):
    def __init__(
        self,
        in_features: int,
        in_channels: int = 21,
        n_hidden: int = 32,
        out_channels: int = 2,
    ):
        super(GeoRefinement, self).__init__()

        # in_channels =
        # StarRNN 3 + 2
        # Distance transform x4
        # Edge sigmoid x4
        # Crop softmax x4

        self.gamma = nn.Parameter(torch.ones((1, out_channels, 1, 1)))
        self.geo_attention = nn.Sequential(
            cunn.ConvBlock2d(
                in_channels=2,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=False,
            ),
            nn.Sigmoid(),
        )

        self.x_res_modules = nn.ModuleList(
            [
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=2,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=3,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=4,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
            ]
        )
        self.crop_res_modules = nn.ModuleList(
            [
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=2,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=3,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
                nn.Sequential(
                    cunn.ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=4,
                        activation_type='SiLU',
                    ),
                    nn.Dropout(0.5),
                ),
            ]
        )

        self.fc = nn.Sequential(
            cunn.ConvBlock2d(
                in_channels=(
                    (n_hidden * len(self.x_res_modules))
                    + (n_hidden * len(self.crop_res_modules))
                ),
                out_channels=n_hidden,
                kernel_size=1,
                padding=0,
                activation_type="SiLU",
            ),
            nn.Conv2d(
                in_channels=n_hidden,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
        )
        self.softmax = nn.Softmax(dim=1)

    def proba_to_logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1.0 - x))

    def forward(
        self, predictions: T.Dict[str, torch.Tensor], data: Data
    ) -> T.Dict[str, torch.Tensor]:
        """A single forward pass.

        Edge and crop inputs should be probabilities
        """
        height = (
            int(data.height) if data.batch is None else int(data.height[0])
        )
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        latitude_norm = scale_min_max(
            data.top - ((data.top - data.bottom) * 0.5), -90.0, 90.0, 0.0, 1.0
        )
        longitude_norm = scale_min_max(
            data.left + ((data.right - data.left) * 0.5),
            -180.0,
            180.0,
            0.0,
            1.0,
        )
        lat_lon = torch.cat(
            [
                latitude_norm.reshape(*latitude_norm.shape, 1, 1, 1),
                longitude_norm.reshape(*longitude_norm.shape, 1, 1, 1),
            ],
            dim=1,
        )
        geo_attention = self.geo_attention(lat_lon)
        geo_attention = 1.0 + self.gamma * geo_attention

        crop_x = torch.cat(
            [
                predictions["crop_star_l2"],
                predictions["crop_star"],
                predictions["dist"],
                predictions["dist_3_1"],
                predictions["dist_2_2"],
                predictions["dist_1_3"],
                predictions["edge"],
                predictions["edge_3_1"],
                predictions["edge_2_2"],
                predictions["edge_1_3"],
                predictions["crop"],
                predictions["crop_3_1"],
                predictions["crop_2_2"],
                predictions["crop_1_3"],
            ],
            dim=1,
        )
        x = torch.cat([m(crop_x) for m in self.x_res_modules], dim=1)
        crop_x = torch.cat([m(crop_x) for m in self.crop_res_modules], dim=1)

        x = torch.cat([x, crop_x], dim=1)
        x = self.softmax(self.fc(x) * geo_attention)
        predictions["crop"] = x

        return predictions


class CropTypeFinal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, out_classes: int):
        super(CropTypeFinal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_classes = out_classes

        self.conv1 = cunn.ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation_type="ReLU",
        )
        layers1 = [
            cunn.ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type="ReLU",
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        self.seq = nn.Sequential(*layers1)

        layers_final = [
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_classes, kernel_size=1, padding=0),
        ]
        self.final = nn.Sequential(*layers_final)

    def forward(
        self, x: torch.Tensor, crop_type_star: torch.Tensor
    ) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.seq(out1)
        out = out + out1
        out = self.final(out)
        out = out + crop_type_star

        return out


class CultioNet(nn.Module):
    """The cultionet model framework.

    Args:
        ds_features (int): The total number of dataset features (bands x time).
        ds_time_features (int): The number of dataset time features in each band/channel.
        filters (int): The number of output filters for each stream.
        num_classes (int): The number of output mask/crop classes.
        model_type (str): The model architecture type.
        activation_type (str): The nonlinear activation.
        dilations (int | list): The convolution dilation or dilations.
        res_block_type (str): The residual convolution block type.
        attention_weights (str): The attention weight type.
        deep_sup_dist (bool): Whether to use deep supervision on the distance layer.
        deep_sup_edge (bool): Whether to use deep supervision on the edge layer.
        deep_sup_mask (bool): Whether to use deep supervision on the mask layer.
    """

    def __init__(
        self,
        in_channels: int,
        in_time: int,
        hidden_channels: int = 32,
        num_classes: int = 2,
        model_type: str = ModelTypes.TOWERUNET,
        activation_type: str = "SiLU",
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = ResBlockTypes.RES,
        attention_weights: str = "spatial_channel",
        deep_sup_dist: bool = False,
        deep_sup_edge: bool = False,
        deep_sup_mask: bool = False,
    ):
        super(CultioNet, self).__init__()

        self.in_channels = in_channels
        self.in_time = in_time
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.temporal_encoder = TemporalTransformer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_head=8,
            in_time=self.in_time,
            dropout=0.1,
            num_layers=2,
            d_model=128,
            time_scaler=100,
            num_classes_l2=self.num_classes,
            num_classes_last=self.num_classes + 1,
            activation_type=activation_type,
            final_activation=nn.Softmax(dim=1),
        )

        unet3_kwargs = {
            "in_channels": self.in_channels,
            "in_time": self.in_time,
            "hidden_channels": self.hidden_channels,
            "num_classes": self.num_classes,
            "activation_type": activation_type,
            # "deep_sup_dist": deep_sup_dist,
            # "deep_sup_edge": deep_sup_edge,
            # "deep_sup_mask": deep_sup_mask,
            "mask_activation": nn.Softmax(dim=1),
        }

        assert model_type in (
            ModelTypes.UNET3PSI,
            ModelTypes.RESUNET3PSI,
            ModelTypes.TOWERUNET,
        ), "The model type is not supported."
        if model_type == ModelTypes.UNET3PSI:
            unet3_kwargs["dilation"] = 2 if dilations is None else dilations
            assert isinstance(
                unet3_kwargs["dilation"], int
            ), f"The dilation for {ModelTypes.UNET3PSI} must be an integer."
            self.mask_model = UNet3Psi(**unet3_kwargs)
        elif model_type in (
            ModelTypes.RESUNET3PSI,
            ModelTypes.TOWERUNET,
        ):
            # ResUNet3Psi
            unet3_kwargs["attention_weights"] = (
                None if attention_weights == "none" else attention_weights
            )
            unet3_kwargs["res_block_type"] = res_block_type
            if res_block_type == ResBlockTypes.RES:
                unet3_kwargs["dilations"] = (
                    [2] if dilations is None else dilations
                )
                assert (
                    len(unet3_kwargs["dilations"]) == 1
                ), f"The dilations for {ModelTypes.RESUNET3PSI} must be a length-1 integer sequence."
            elif res_block_type == ResBlockTypes.RESA:
                unet3_kwargs["dilations"] = (
                    [1, 2] if dilations is None else dilations
                )
            assert isinstance(
                unet3_kwargs["dilations"], list
            ), f"The dilations for {ModelTypes.RESUNET3PSI} must be a sequence of integers."

            if model_type == ModelTypes.RESUNET3PSI:
                self.mask_model = ResUNet3Psi(**unet3_kwargs)
            else:
                self.mask_model = TowerUNet(**unet3_kwargs)

    def forward(self, data: Data) -> T.Dict[str, torch.Tensor]:
        # Transformer attention encoder
        transformer_outputs = self.temporal_encoder(data.x)

        # Main stream
        logits = self.mask_model(
            data.x,
            temporal_encoding=transformer_outputs['encoded'],
        )

        classes_l2 = transformer_outputs['l2']
        classes_l3 = transformer_outputs['l3']
        logits_distance = logits["dist"]
        logits_edges = logits["edge"]
        logits_crop = logits["mask"]

        out = {
            "dist": logits_distance,
            "edge": logits_edges,
            "crop": logits_crop,
            "crop_type": None,
            "classes_l2": classes_l2,
            "classes_l3": classes_l3,
        }

        if logits.get("dist_3_1") is not None:
            out["dist_3_1"] = logits["dist_3_1"]
            out["dist_2_2"] = logits["dist_2_2"]
            out["dist_1_3"] = logits["dist_1_3"]

        if logits.get("mask_3_1") is not None:
            out["crop_3_1"] = logits["mask_3_1"]
            out["crop_2_2"] = logits["mask_2_2"]
            out["crop_1_3"] = logits["mask_1_3"]

        if logits.get("edge_3_1") is not None:
            out["edge_3_1"] = logits["edge_3_1"]
            out["edge_2_2"] = logits["edge_2_2"]
            out["edge_1_3"] = logits["edge_1_3"]

        return out
