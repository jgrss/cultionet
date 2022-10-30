import typing as T

from . import model_utils
from .nunet import TemporalNestedUNet2

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
        star_rnn_n_layers: int = 3,
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

        # Nested UNet (+self.filters x self.ds_num_bands)
        # self.nunet1 = TemporalNestedUNet2(
        #     in_channels=self.ds_num_features,
        #     out_channels=1,
        #     out_side_channels=0,
        #     init_filter=int(self.filters / 2),
        #     linear_fc=True
        # )
        # self.dist_model = GraphRegressionLayer(
        #     in_channels=self.ds_num_features,
        #     mid_channels=self.filters,
        #     out_channels=1
        # )
        self.nunet_model = TemporalNestedUNet2(
            in_channels=self.ds_num_features,
            out_channels=2,
            out_side_channels=2,
            init_filter=self.filters
        )
        self.dist_model = TemporalNestedUNet2(
            in_channels=2,
            out_channels=1,
            out_side_channels=2,
            init_filter=self.filters
        )
        self.dist_linear = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Dropout2d(dropout),
            torch.nn.ReLU(inplace=False),
            self.cg,
            torch.nn.Linear(1, 1)
        )
        # Star RNN layer
        # self.star_rnn = StarRNN(
        #     input_dim=self.ds_num_bands,
        #     hidden_dim=star_rnn_hidden_dim,
        #     n_layers=star_rnn_n_layers,
        #     num_classes_last=2
        # )
        # self.star_refine = Refine(
        #     in_channels=2+2+2,
        #     mid_channels=128,
        #     out_channels=2,
        #     dropout=dropout
        # )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, data: Data
    ) -> T.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)

        # logits_distance = self.dist_model(
        #     self.gc(
        #         data.x, batch_size, height, width
        #     ),
        #     data.edge_index,
        #     data.edge_attrs
        # )
        # Nested UNet on each band time series
        nunet_stream = self.nunet_model(
            self.gc(
                data.x, batch_size, height, width
            )
        )
        logits_edges = self.cg(nunet_stream['side'])
        logits_crop = self.cg(nunet_stream['net'])
        dist_stream = self.dist_model(
            self.gc(
                logits_crop, batch_size, height, width
            ),
            side=self.gc(
                logits_edges, batch_size, height, width
            )
        )
        logits_distance = self.dist_linear(dist_stream['net'])
        # # RNN ConvStar
        # star_stream = self.gc(
        #     data.x, batch_size, height, width
        # )
        # # nbatch, ntime, height, width
        # nbatch, __, height, width = star_stream.shape
        # # Reshape from (B x C x H x W) -> (B x C x T x H x W)
        # star_stream = star_stream.reshape(
        #     nbatch, self.ds_num_bands, self.ds_num_time, height, width
        # )
        # # Crop/Non-crop and Crop types
        # star_stream = self.star_rnn(star_stream)[1]
        # logits_star = self.cg(star_stream)
        # # Concatenate time series streams
        # h = torch.cat(
        #     [
        #         logits_edges,
        #         logits_crop,
        #         logits_star
        #     ],
        #     dim=1
        # )
        # h = self.gc(
        #     h, batch_size, height, width
        # )
        # logits_star = self.cg(self.star_refine(h))

        return (
            logits_distance,
            logits_edges,
            logits_crop
        )
