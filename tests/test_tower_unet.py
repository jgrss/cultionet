import torch

from cultionet.enums import InferenceNames, ResBlockTypes
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

    model = TowerUNet(
        in_channels=num_channels,
        in_time=num_time,
        hidden_channels=hidden_channels,
        dilations=[1, 2],
        res_block_type=ResBlockTypes.RESA,
        pool_by_max=False,
    )

    logits = model(x)

    assert logits[InferenceNames.DISTANCE].shape == (
        batch_size,
        1,
        height,
        width,
    )
    assert logits[InferenceNames.EDGE].shape == (batch_size, 1, height, width)
    assert logits[InferenceNames.CROP].shape == (batch_size, 1, height, width)
