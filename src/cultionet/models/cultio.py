import typing as T

from . import model_utils
from .base_layers import ConvBlock2d, Softmax
from .nunet import UNet3, UNet3Psi, ResUNet3Psi
from .convstar import StarRNN

import torch
from torch_geometric.data import Data


class FinalRefinement(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_features: int,
        out_channels: int,
        double_dilation: int = 1
    ):
        super(FinalRefinement, self).__init__()

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        self.model = UNet3(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filter=n_features,
            double_dilation=double_dilation
        )

    def proba_to_logit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x / (1.0 - x))

    def forward(
        self,
        predictions: T.Dict[str, torch.Tensor],
        data: Data
    ) -> torch.Tensor:
        """A single forward pass

        Edge and crop inputs should be probabilities
        """
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        x = torch.cat(
            [
                predictions['crop_star_l2'],
                predictions['crop_star'],
                predictions['dist'],
                predictions['dist_3_1'],
                predictions['dist_2_2'],
                predictions['dist_1_3'],
                predictions['edge'],
                predictions['edge_3_1'],
                predictions['edge_2_2'],
                predictions['edge_1_3'],
                predictions['crop'],
                predictions['crop_3_1'],
                predictions['crop_2_2'],
                predictions['crop_1_3']
            ],
            dim=1
        )

        # Reshape
        x = self.gc(
            x, batch_size, height, width
        )
        h = self.model(x)

        return self.cg(h)


class CropTypeFinal(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_classes: int
    ):
        super(CropTypeFinal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_classes = out_classes

        self.conv1 = ConvBlock2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation_type='ReLU'
        )
        layers1 = [
            ConvBlock2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                activation_type='ReLU'
            ),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        ]
        self.seq = torch.nn.Sequential(*layers1)

        layers_final = [
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                out_channels,
                out_classes,
                kernel_size=1,
                padding=0
            )
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


class CultioNet(torch.nn.Module):
    """The cultionet model framework

    Args:
        ds_features (int): The total number of dataset features (bands x time).
        ds_time_features (int): The number of dataset time features in each band/channel.
        filters (int): The number of output filters for each stream.
        star_rnn_hidden_dim (int): The number of hidden features for the ConvSTAR layer.
        star_rnn_n_layers (int): The number of ConvSTAR layers.
        num_classes (int): The number of output classes.
    """
    def __init__(
        self,
        ds_features: int,
        ds_time_features: int,
        filters: int = 64,
        num_classes: int = 2,
        model_type: str = 'UNet3Psi',
        activation_type: str = 'LeakyReLU',
        deep_sup_dist: bool = False,
        deep_sup_edge: bool = False,
        deep_sup_mask: bool = False
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

        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=self.filters,
            n_layers=3,
            num_classes_l2=self.num_classes,
            num_classes_last=self.num_classes + 1,
            crop_type_layer=True if self.num_classes > 2 else False,
            activation_type=activation_type,
            final_activation=Softmax(dim=1)
        )
        unet3_kwargs = {
            'in_channels': self.ds_num_bands,
            'in_time': self.ds_num_time,
            'in_rnn_channels': int(self.filters * 3),
            'init_filter': self.filters,
            'num_classes': self.num_classes,
            'activation_type': activation_type,
            'deep_sup_dist': deep_sup_dist,
            'deep_sup_edge': deep_sup_edge,
            'deep_sup_mask': deep_sup_mask,
            'mask_activation': Softmax(dim=1)
        }
        assert model_type in ('UNet3Psi', 'ResUNet3Psi'), \
            'The model type is not supported.'
        if model_type == 'UNet3Psi':
            unet3_kwargs['dilation'] = 2
            self.mask_model = UNet3Psi(**unet3_kwargs)
        elif model_type == 'ResUNet3Psi':
            # ResUNet3Psi
            # spatial_channel | fractal
            unet3_kwargs['attention_weights'] = 'spatial_channel'
            # unet3_kwargs['res_block_type'] = 'res'
            # unet3_kwargs['dilations'] = [2]
            unet3_kwargs['res_block_type'] = 'resa'
            unet3_kwargs['dilations'] = [1, 2]
            self.mask_model = ResUNet3Psi(**unet3_kwargs)

    def forward(
        self, data: Data
    ) -> T.Dict[str, torch.Tensor]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        assert (data.ntime == data.ntime[0]).all(), 'The time dimension must match.'
        assert (data.nbands == data.nbands[0]).all(), 'The band dimension must match.'
        assert (data.height == data.height[0]).all(), 'The height dimension must match.'
        assert (data.width == data.width[0]).all(), 'The width dimension must match.'

        # Reshape from ((H*W) x (C*T)) -> (B x C x H x W)
        x = self.gc(
            data.x, batch_size, height, width
        )
        # Reshape from (B x C x H x W) -> (B x C x T|D x H x W)
        x = self.ct(x, nbands=self.ds_num_bands, ntime=self.ds_num_time)
        # StarRNN
        logits_star_hidden, logits_star_l2, logits_star_last = self.star_rnn(x)
        logits_star_l2 = self.cg(logits_star_l2)
        logits_star_last = self.cg(logits_star_last)
        # Main stream
        logits = self.mask_model(x, logits_star_hidden)
        logits_distance = self.cg(logits['dist'])
        logits_edges = self.cg(logits['edge'])
        logits_crop = self.cg(logits['mask'])

        out = {
            'dist': logits_distance,
            'edge': logits_edges,
            'crop': logits_crop,
            'crop_type': None,
            'crop_star_l2': logits_star_l2,
            'crop_star': logits_star_last
        }

        if logits['dist_3_1'] is not None:
            out['dist_3_1'] = self.cg(logits['dist_3_1'])
            out['dist_2_2'] = self.cg(logits['dist_2_2'])
            out['dist_1_3'] = self.cg(logits['dist_1_3'])
        if logits['mask_3_1'] is not None:
            out['crop_3_1'] = self.cg(logits['mask_3_1'])
            out['crop_2_2'] = self.cg(logits['mask_2_2'])
            out['crop_1_3'] = self.cg(logits['mask_1_3'])
        if logits['edge_3_1'] is not None:
            out['edge_3_1'] = self.cg(logits['edge_3_1'])
            out['edge_2_2'] = self.cg(logits['edge_2_2'])
            out['edge_1_3'] = self.cg(logits['edge_1_3'])

        return out
