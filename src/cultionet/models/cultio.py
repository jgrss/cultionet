import typing as T
import warnings

import torch
from torch_geometric.data import Data

from . import model_utils
from .base_layers import ConvBlock2d, ResidualConv, Softmax
from .nunet import UNet3Psi, ResUNet3Psi
from .ltae import LightweightTemporalAttentionEncoder


def scale_min_max(
    x: torch.Tensor,
    min_in: float,
    max_in: float,
    min_out: float,
    max_out: float,
) -> torch.Tensor:
    return (((max_out - min_out) * (x - min_in)) / (max_in - min_in)) + min_out


class GeoRefinement(torch.nn.Module):
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

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        self.gamma = torch.nn.Parameter(torch.ones((1, out_channels, 1, 1)))
        self.geo_attention = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=2,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                add_activation=False,
            ),
            torch.nn.Sigmoid(),
        )

        self.x_res_modules = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=2,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=3,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_features,
                        out_channels=n_hidden,
                        dilation=4,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
            ]
        )
        self.crop_res_modules = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=2,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=3,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
                torch.nn.Sequential(
                    ResidualConv(
                        in_channels=in_channels,
                        out_channels=n_hidden,
                        dilation=4,
                        activation_type='SiLU',
                    ),
                    torch.nn.Dropout(0.5),
                ),
            ]
        )

        self.fc = torch.nn.Sequential(
            ConvBlock2d(
                in_channels=(
                    (n_hidden * len(self.x_res_modules))
                    + (n_hidden * len(self.crop_res_modules))
                ),
                out_channels=n_hidden,
                kernel_size=1,
                padding=0,
                activation_type="SiLU",
            ),
            torch.nn.Conv2d(
                in_channels=n_hidden,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
        )
        self.softmax = Softmax(dim=1)

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
        x = self.gc(data.x, batch_size, height, width)
        x = torch.cat([m(x) for m in self.x_res_modules], dim=1)

        crop_x = self.gc(crop_x, batch_size, height, width)
        crop_x = torch.cat([m(crop_x) for m in self.crop_res_modules], dim=1)

        x = torch.cat([x, crop_x], dim=1)
        x = self.softmax(self.fc(x) * geo_attention)
        predictions["crop"] = self.cg(x)

        return predictions


class CropTypeFinal(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, out_classes: int):
        super(CropTypeFinal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_classes = out_classes

        self.conv1 = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation_type="ReLU",
        )
        layers1 = [
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type="ReLU",
            ),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
        ]
        self.seq = torch.nn.Sequential(*layers1)

        layers_final = [
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                out_channels, out_classes, kernel_size=1, padding=0
            ),
        ]
        self.final = torch.nn.Sequential(*layers_final)

    def forward(
        self, x: torch.Tensor, crop_type_star: torch.Tensor
    ) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.seq(out1)
        out = out + out1
        out = self.final(out)
        out = out + crop_type_star

        return out


def check_batch_dims(batch: Data, attribute: str):
    batch_var = getattr(batch, attribute)
    if not (batch_var == batch_var[0]).all():
        invalid = batch.train_id[batch_var != torch.mode(batch_var)[0]]
        warnings.warn("The following ids do not match the batch mode.")
        warnings.warn(invalid)
        raise ValueError(f"The {attribute} dimensions do not align.")


class CultioNet(torch.nn.Module):
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
        ds_features: int,
        ds_time_features: int,
        filters: int = 32,
        num_classes: int = 2,
        model_type: str = "ResUNet3Psi",
        activation_type: str = "SiLU",
        dilations: T.Union[int, T.Sequence[int]] = None,
        res_block_type: str = "resa",
        attention_weights: str = "spatial_channel",
        deep_sup_dist: bool = False,
        deep_sup_edge: bool = False,
        deep_sup_mask: bool = False,
    ):
        super(CultioNet, self).__init__()

        # Total number of features (time x bands/indices/channels)
        self.ds_num_features = ds_features
        # Total number of time features
        self.ds_num_time = ds_time_features
        # Total number of bands
        self.ds_num_bands = int(self.ds_num_features / self.ds_num_time)
        self.filters = filters
        self.num_classes = num_classes

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()
        self.ct = model_utils.ConvToTime()

        self.temporal_encoder = LightweightTemporalAttentionEncoder(
            in_channels=self.ds_num_bands,
            hidden_size=128,
            d_model=256,
            n_head=16,
            n_time=self.ds_num_time,
            # [d_model, encoder_widths[-1]]
            mlp=[256, 128],
            return_att=False,
            d_k=4,
            num_classes_l2=self.num_classes,
            num_classes_last=self.num_classes + 1,
            activation_type=activation_type,
            final_activation=Softmax(dim=1),
        )

        unet3_kwargs = {
            "in_channels": self.ds_num_bands,
            "in_time": self.ds_num_time,
            "in_encoding_channels": 128,  # <- L-TAE; #int(self.filters * 3), <- ConvSTAR
            "init_filter": self.filters,
            "num_classes": self.num_classes,
            "activation_type": activation_type,
            "deep_sup_dist": deep_sup_dist,
            "deep_sup_edge": deep_sup_edge,
            "deep_sup_mask": deep_sup_mask,
            "mask_activation": Softmax(dim=1),
        }
        assert model_type in (
            "UNet3Psi",
            "ResUNet3Psi",
        ), "The model type is not supported."
        if model_type == "UNet3Psi":
            unet3_kwargs["dilation"] = 2 if dilations is None else dilations
            assert isinstance(
                unet3_kwargs["dilation"], int
            ), "The dilation for UNet3Psi must be an integer."
            self.mask_model = UNet3Psi(**unet3_kwargs)
        elif model_type == "ResUNet3Psi":
            # ResUNet3Psi
            unet3_kwargs["attention_weights"] = (
                None if attention_weights == "none" else attention_weights
            )
            unet3_kwargs["res_block_type"] = res_block_type
            if res_block_type == "res":
                unet3_kwargs["dilations"] = (
                    [2] if dilations is None else dilations
                )
                assert (
                    len(unet3_kwargs["dilations"]) == 1
                ), "The dilations for ResUNet3Psi must be a length-1 integer sequence."
            elif res_block_type == "resa":
                unet3_kwargs["dilations"] = (
                    [1, 2] if dilations is None else dilations
                )
            assert isinstance(
                unet3_kwargs["dilations"], list
            ), "The dilations for ResUNet3Psi must be a sequence of integers."
            self.mask_model = ResUNet3Psi(**unet3_kwargs)

    def forward(self, data: Data) -> T.Dict[str, torch.Tensor]:
        height = (
            int(data.height) if data.batch is None else int(data.height[0])
        )
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        for attribute in ("ntime", "nbands", "height", "width"):
            check_batch_dims(data, attribute)

        # Reshape from ((H*W) x (C*T)) -> (B x C x H x W)
        x = self.gc(data.x, batch_size, height, width)
        # Reshape from (B x C x H x W) -> (B x C x T|D x H x W)
        x = self.ct(x, nbands=self.ds_num_bands, ntime=self.ds_num_time)

        # Transformer attention encoder
        logits_hidden, classes_l2, classes_last = self.temporal_encoder(x)

        classes_l2 = self.cg(classes_l2)
        classes_last = self.cg(classes_last)
        # Main stream
        logits = self.mask_model(x, temporal_encoding=logits_hidden)
        logits_distance = self.cg(logits["dist"])
        logits_edges = self.cg(logits["edge"])
        logits_crop = self.cg(logits["mask"])

        out = {
            "dist": logits_distance,
            "edge": logits_edges,
            "crop": logits_crop,
            "crop_type": None,
            "classes_l2": classes_l2,
            "classes_last": classes_last,
        }

        if logits["dist_3_1"] is not None:
            out["dist_3_1"] = self.cg(logits["dist_3_1"])
            out["dist_2_2"] = self.cg(logits["dist_2_2"])
            out["dist_1_3"] = self.cg(logits["dist_1_3"])
        if logits["mask_3_1"] is not None:
            out["crop_3_1"] = self.cg(logits["mask_3_1"])
            out["crop_2_2"] = self.cg(logits["mask_2_2"])
            out["crop_1_3"] = self.cg(logits["mask_1_3"])
        if logits["edge_3_1"] is not None:
            out["edge_3_1"] = self.cg(logits["edge_3_1"])
            out["edge_2_2"] = self.cg(logits["edge_2_2"])
            out["edge_1_3"] = self.cg(logits["edge_1_3"])

        return out
