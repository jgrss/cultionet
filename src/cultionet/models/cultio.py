import typing as T

from . import model_utils
from .nunet import NestedUNet2, NestedUNet3
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
        dropout (Optional[float]): The dropout fraction for the transformer stream.
    """
    def __init__(
        self,
        ds_features: int,
        ds_time_features: int,
        filters: int = 64,
        star_rnn_hidden_dim: int = 64,
        star_rnn_n_layers: int = 4,
        num_classes: int = 2
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

        # Star RNN layer (+star_rnn_hidden_dim +num_classes_last)
        num_last_channels = self.num_classes if self.num_classes > 2 else star_rnn_hidden_dim
        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=star_rnn_hidden_dim,
            n_layers=star_rnn_n_layers,
            num_classes_last=num_last_channels
        )
        base_in_channels = star_rnn_hidden_dim + num_last_channels
        # Distance layers (+5)
        self.nunet3_model = NestedUNet3(
            in_channels=base_in_channels,
            out_channels=1,
            init_filter=self.filters,
            deep_supervision=True
        )
        # Nested UNet (+2 edges +2 crops)
        self.nunet2_model = NestedUNet2(
            in_channels=base_in_channels+5,
            out_channels=2,
            out_side_channels=2,
            init_filter=self.filters,
            boundary_layer=True,
            deep_supervision=True
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, data: Data
    ) -> T.Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        # (1) RNN ConvStar
        star_stream = self.gc(
            data.x, batch_size, height, width
        )
        # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        # nbatch, ntime, height, width
        nbatch, __, height, width = star_stream.shape
        star_stream = star_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        )
        # Crop/Non-crop and Crop types
        logits_star_local, logits_star_last = self.star_rnn(star_stream)
        logits_star_local = self.cg(logits_star_local)
        logits_star_last = self.cg(logits_star_last)

        # (2) Distance stream
        logits_distance = self.nunet3_model(
            self.gc(
                torch.cat(
                    [
                        logits_star_local, logits_star_last
                    ], dim=1
                ),
                batch_size,
                height,
                width
            )
        )
        logits_distance_0 = self.cg(logits_distance['mask_0'])
        logits_distance_1 = self.cg(logits_distance['mask_1'])
        logits_distance_2 = self.cg(logits_distance['mask_2'])
        logits_distance_3 = self.cg(logits_distance['mask_3'])
        logits_distance_4 = self.cg(logits_distance['mask_4'])

        # CONCAT
        h = torch.cat(
            [
                logits_star_local,
                logits_star_last,
                logits_distance_0,
                logits_distance_1,
                logits_distance_2,
                logits_distance_3,
                logits_distance_4
            ], dim=1
        )

        # (3) Nested UNet for crop and edges
        nunet_stream = self.nunet2_model(
            self.gc(
                h, batch_size, height, width
            )
        )
        logits_crop = self.cg(nunet_stream['mask'])
        logits_edges = self.cg(nunet_stream['boundary'])

        return (
            logits_distance_0,
            logits_distance_1,
            logits_distance_2,
            logits_distance_3,
            logits_distance_4,
            logits_edges,
            logits_crop
        )
