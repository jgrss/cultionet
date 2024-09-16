import torch

from cultionet.enums import AttentionTypes, ResBlockTypes
from cultionet.models.nunet import TowerUNet


def test_tower_unet():
    batch_size = 2
    num_channels = 3
    hidden_channels = 32
    num_time = 13
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, num_time, height, width),
        dtype=torch.float32,
    )
    logits_hidden = torch.rand(
        (batch_size, hidden_channels, height, width), dtype=torch.float32
    )

    model = TowerUNet(
        in_channels=num_channels,
        in_time=num_time,
        hidden_channels=hidden_channels,
        dilations=[1, 2],
        dropout=0.2,
        res_block_type=ResBlockTypes.RESA,
        attention_weights=AttentionTypes.SPATIAL_CHANNEL,
        pool_by_max=False,
        repeat_resa_kernel=False,
        batchnorm_first=True,
    )

    logits = model(x, temporal_encoding=logits_hidden)

    assert logits['dist'].shape == (batch_size, 1, height, width)
    assert logits['edge'].shape == (batch_size, 1, height, width)
    assert logits['mask'].shape == (batch_size, 2, height, width)
