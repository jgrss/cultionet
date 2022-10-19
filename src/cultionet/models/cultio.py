import typing as T

from . import model_utils
from .nunet import NestedUNet2
from .convstar import StarRNN
from .graph import GraphMid, GraphFinal
from .regression import GraphRegressionLayer

import torch
from torch_geometric.data import Data
from torch_geometric import nn


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
        filters: int = 32,
        num_classes: int = 2,
        dropout: T.Optional[float] = 0.1,
        rheight: int = 201,
        rwidth: int = 201
    ):
        super(CultioGraphNet, self).__init__()

        # Total number of features (time x bands/indices/channels)
        self.ds_num_features = ds_features
        # Total number of time features
        self.ds_num_time = ds_time_features
        # Total number of bands
        self.ds_num_bands = int(self.ds_num_features / self.ds_num_time)
        self.filters = filters
        num_distances = 2
        num_index_streams = 2
        base_in_channels = (filters * self.ds_num_bands) * num_index_streams + self.filters

        self.gc = model_utils.GraphToConv()
        self.cg = model_utils.ConvToGraph()

        # Transformer stream (+self.filters x self.ds_num_bands)
        self.transformer = GraphMid(
            nn.TransformerConv(
                self.ds_num_time, self.filters, heads=1, edge_dim=2, dropout=dropout
            )
        )
        # Nested UNet (+self.filters x self.ds_num_bands)
        self.nunet = NestedUNet2(
            in_channels=self.ds_num_time, out_channels=self.filters
        )
        # Star RNN layer
        self.star_rnn = StarRNN(
            input_dim=self.ds_num_bands,
            hidden_dim=32,
            nclasses=self.filters,
            n_layers=3
        )
        # Boundary distance orientations (+1)
        self.dist_layer_ori = GraphRegressionLayer(
            in_channels=base_in_channels,
            mid_channels=self.filters,
            out_channels=1
        )
        # Boundary distances (+1)
        self.dist_layer = GraphRegressionLayer(
            in_channels=base_in_channels+1,
            mid_channels=self.filters,
            out_channels=1
        )
        # Edges (+num_classes)
        self.edge_layer = GraphFinal(
            nn.GCNConv(base_in_channels+num_distances, self.filters, improved=True),
            nn.TransformerConv(self.filters, num_classes, heads=1, edge_dim=2, dropout=0.1),
            self.filters, num_classes
        )
        # Classes (+num_classes)
        self.class_layer = GraphFinal(
            nn.GCNConv(base_in_channels+num_distances+num_classes, self.filters, improved=True),
            nn.TransformerConv(
                self.filters, num_classes, heads=1, edge_dim=2, dropout=0.1
            ),
            self.filters, num_classes
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, data: Data
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        height = int(data.height) if data.batch is None else int(data.height[0])
        width = int(data.width) if data.batch is None else int(data.width[0])
        batch_size = 1 if data.batch is None else data.batch.unique().size(0)
        # Transformer on each band time series
        transformer_stream = []
        # Iterate over all data features, stepping over time chunks
        for band in range(0, self.ds_num_features, self.ds_num_time):
            # Get the current band through all time steps
            t = self.transformer(
                data.x[:, band:band+self.ds_num_time],
                data.edge_index,
                data.edge_attrs
            )
            transformer_stream.append(t)
        transformer_stream = torch.cat(transformer_stream, dim=1)

        # Nested UNet on each band time series
        nunet_stream = []
        for band in range(0, self.ds_num_features, self.ds_num_time):
            t = self.nunet(
                data.x[:, band:band+self.ds_num_time],
                data.edge_index,
                data.edge_attrs[:, 1],
                data.batch,
                height,
                width
            )
            nunet_stream.append(t)
        nunet_stream = torch.cat(nunet_stream, dim=1)
        # RNN ConvStar
        # Reshape from (B x C x H x W) -> (B x T x C x H x W)
        star_stream = self.gc(
            data.x, batch_size, height, width
        )
        # nbatch, ntime, height, width
        nbatch, __, height, width = star_stream.shape
        star_stream = star_stream.reshape(
            nbatch, self.ds_num_bands, self.ds_num_time, height, width
        ).permute(0, 2, 1, 3, 4)
        star_stream = self.star_rnn(star_stream)
        star_stream = self.cg(star_stream)
        # Concatenate time series streams
        h = torch.cat(
            [
                transformer_stream,
                nunet_stream,
                star_stream
            ],
            dim=1
        )

        # Estimate distance orientations
        logits_distances_ori = self.dist_layer_ori(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations
        h = torch.cat([h, logits_distances_ori], dim=1)

        # Estimate distance from edges
        logits_distances = self.dist_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations + distances
        h = torch.cat([h, logits_distances], dim=1)

        # Estimate edges
        logits_edges = self.edge_layer(
            h,
            data.edge_index,
            data.edge_attrs
        )
        # Concatenate streams + distance orientations + distances + edges
        h = torch.cat([h, logits_edges], dim=1)

        # Estimate all classes
        logits_labels = self.class_layer(
            h,
            data.edge_index,
            data.edge_attrs
        )

        return logits_distances_ori, logits_distances, logits_edges, logits_labels
