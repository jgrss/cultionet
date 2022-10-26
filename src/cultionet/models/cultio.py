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
        star_rnn_hidden_dim: int = 32,
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
        num_distances = 2
        num_index_streams = 2
        # Index streams = filters x num_bands
        index_stream_out_channels = (filters * self.ds_num_bands) * num_index_streams
        # RNN stream = 2 + num_classes
        rnn_stream_out_channels = 2
        base_in_channels = index_stream_out_channels + rnn_stream_out_channels

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
            hidden_dim=star_rnn_hidden_dim,
            # Number of output classes (0=not cropland; 1=cropland)
            num_classes=2,
            # Number of output crop type classes
            num_crop_classes=num_classes,
            n_layers=star_rnn_n_layers
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
        # Edges (+2)
        self.edge_layer = GraphFinal(
            in_channels=base_in_channels+num_distances,
            mid_channels=self.filters,
            out_channels=2
        )
        # Crops (+2)
        self.crop_layer = GraphFinal(
            in_channels=base_in_channels+num_distances+2,
            mid_channels=self.filters,
            out_channels=2
        )
        # Crop types (+num_classes)
        self.crop_type_layer = GraphFinal(
            in_channels=base_in_channels+num_distances+2+2+num_classes,
            mid_channels=self.filters,
            out_channels=num_classes
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
        # Crop/Non-crop and Crop types
        star_stream, star_stream_local_2 = self.star_rnn(star_stream)
        star_stream = self.cg(star_stream)
        star_stream_local_2 = self.cg(star_stream_local_2)
        # Concatenate time series streams
        h = torch.cat(
            [
                transformer_stream,
                nunet_stream,
                star_stream_local_2
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
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        # Concatenate streams + distance orientations + distances + edges
        h = torch.cat([h, logits_edges], dim=1)

        # Estimate crop/non-crop
        logits_crop = self.crop_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )
        h = torch.cat([h, logits_crop, star_stream], dim=1)

        # Estimate crop-type
        logits_crop_type = self.crop_type_layer(
            h,
            data.edge_index,
            data.edge_attrs,
            data.batch,
            height,
            width
        )

        return (
            logits_distances_ori,
            logits_distances,
            logits_edges,
            logits_crop,
            logits_crop_type
        )
