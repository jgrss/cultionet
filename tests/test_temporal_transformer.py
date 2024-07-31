import torch

from cultionet.models.temporal_transformer import TemporalTransformer


def test_temporal_transformer():
    batch_size = 2
    num_channels = 3
    hidden_channels = 64
    num_head = 8
    d_model = 128
    in_time = 13
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, in_time, height, width),
        dtype=torch.float32,
    )

    model = TemporalTransformer(
        in_channels=num_channels,
        hidden_channels=hidden_channels,
        num_head=num_head,
        d_model=d_model,
        in_time=in_time,
    )
    output = model(x)

    assert tuple(output.keys()) == ('l2', 'l3', 'encoded')
    output['encoded'].shape == (batch_size, hidden_channels, height, width)
