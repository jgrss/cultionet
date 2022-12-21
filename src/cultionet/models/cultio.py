import typing as T

from . import model_utils
from .nunet import UNet3Psi, ResUNet3Psi
from .convstar import StarRNN

import torch
from torch_geometric.data import Data


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
        star_rnn_hidden_dim: int = 64,
        star_rnn_n_layers: int = 4,
        num_classes: int = 2,
        model_type: str = 'UNet3Psi'
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

        # Star RNN layer (+star_rnn_hidden_dim +num_classes_l2 +num_classes_last)
        # Crop classes
        if self.num_classes > 2:
            # Crop-type classes
            num_classes_last = self.num_classes
        else:
            num_classes_last = 2
        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=star_rnn_hidden_dim,
            n_layers=star_rnn_n_layers,
            num_classes_last=num_classes_last,
            crop_type_layer=True if self.num_classes > 2 else False
        )
        # Local 1 = hidden dimensions
        # Local 2 = crop (0|1)
        # Last = crop-type (2..N)
        base_in_channels = (
            + star_rnn_hidden_dim * (star_rnn_n_layers - 1)
            + num_classes_last
        )
        if model_type == 'UNet3Psi':
            self.mask_model = UNet3Psi(
                in_channels=base_in_channels,
                out_dist_channels=1,
                out_edge_channels=2,
                out_mask_channels=2,
                init_filter=self.filters
            )
        elif model_type == 'ResUNet3Psi':
            self.mask_model = ResUNet3Psi(
                in_channels=base_in_channels,
                out_dist_channels=1,
                out_edge_channels=2,
                out_mask_channels=2,
                init_filter=self.filters,
                attention=False
            )
        else:
            raise NameError('Model type not supported.')

    def forward(
        self, data: Data
    ) -> T.Dict[str, torch.Tensor]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        # (1) RNN ConvStar
        time_stream = self.gc(
            data.x, batch_size, height, width
        )
        # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        # nbatch, ntime, height, width
        nbatch, __, height, width = time_stream.shape
        time_stream = time_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        )
        # Crop/Non-crop and Crop types
        logits_star_l2, logits_star_last = self.star_rnn(time_stream)
        logits_star_l2 = self.cg(logits_star_l2)
        logits_star_last = self.cg(logits_star_last)

        # CONCAT
        h = torch.cat(
            [
                logits_star_l2,
                logits_star_last
            ], dim=1
        )

        # Main stream
        logits = self.mask_model(
            self.gc(
                h, batch_size, height, width
            )
        )
        logits_distance = self.cg(logits['dist'])
        logits_edges = self.cg(logits['edge'])
        logits_crop = self.cg(logits['mask'])

        out = {
            'dist': logits_distance,
            'edge': logits_edges,
            'crop': logits_crop,
            'crop_star': None,
            'crop_type': None
        }
        if self.num_classes > 2:
            # With no crop-type, return last layer (crop)
            out['crop_type'] = logits_star_last
        else:
            out['crop_star'] = logits_star_last

        return out
