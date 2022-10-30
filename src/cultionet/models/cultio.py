import typing as T

from . import model_utils
from .nunet import TemporalNestedUNet2
from .convstar import StarRNN

import torch
from torch_geometric.data import Data


class CultioGraphNet(torch.nn.Module):
    """The cultionet graph network model framework

    Args:
        ds_features (int): The total number of dataset features (bands x time).
        ds_time_features (int): The number of dataset time features in each band/channel.
        filters (int): The number of output filters for each stream.
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
        num_classes: int = 2,
        dropout: T.Optional[float] = 0.1
    ):
        super(CultioGraphNet, self).__init__()

        # Total number of features (time x bands/indices/channels)
        self.ds_num_features = ds_features
        # Total number of time features
        self.ds_num_time = ds_time_features
        # Total number of bands
        self.ds_num_bands = int(self.ds_num_features / self.ds_num_time)
        self.filters = filters

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        # Star RNN layer
        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=star_rnn_hidden_dim,
            n_layers=star_rnn_n_layers,
            num_classes_last=2
        )
        # Nested UNet (+self.filters x self.ds_num_bands)
        self.nunet_model = TemporalNestedUNet2(
            in_channels=self.ds_num_features+star_rnn_hidden_dim,
            out_channels=2,
            out_side_channels=2,
            init_filter=self.filters,
            boundary_layer=True
        )
        self.dist_model = TemporalNestedUNet2(
            in_channels=self.ds_num_features+star_rnn_hidden_dim+2+2,
            out_channels=1,
            out_side_channels=2,
            init_filter=self.filters,
            boundary_layer=False,
            linear_fc=True,
            dropout=dropout
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, data: Data
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        # (1) RNN ConvStar
        star_stream = self.gc(
            data.x, batch_size, height, width
        )
        # nbatch, ntime, height, width
        nbatch, __, height, width = star_stream.shape
        # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        star_stream = star_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        )
        # Crop/Non-crop and Crop types
        logits_star_last = self.star_rnn(star_stream)
        # logits_star_local_2 = self.cg(logits_star_local_2)
        logits_star_last = self.cg(logits_star_last)

        # CONCAT
        h = torch.cat([data.x, logits_star_last], dim=1)

        # (2) Nested UNet for crop and edges
        nunet_stream = self.nunet_model(
            self.gc(
                h, batch_size, height, width
            )
        )
        logits_crop = self.cg(nunet_stream['net'])
        logits_edges = self.cg(nunet_stream['side'])

        # CONCAT
        h = torch.cat([h, logits_crop, logits_edges], dim=1)

        # (3) Distance stream
        dist_stream = self.dist_model(
            self.gc(
                h, batch_size, height, width
            )
        )
        logits_distance = self.cg(dist_stream['net'])

        return (
            logits_distance,
            logits_edges,
            logits_crop
        )
