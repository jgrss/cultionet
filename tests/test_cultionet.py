import torch

from cultionet.models import model_utils
from cultionet.layers.base_layers import Softmax
from cultionet.models.nunet import ResUNet3Psi
from cultionet.models.temporal_transformer import TemporalAttention


def test_cultionet():
    cg = model_utils.ConvToGraph()

    batch_size = 2
    in_channels = 4
    in_time = 20
    height = 40
    width = 40

    hidden_size = 128
    d_model = 256
    n_head = 16
    num_classes_l2 = 2
    num_classes_last = 3
    filters = 64
    activation_type = 'SiLU'

    x = torch.rand(
        (batch_size, in_channels, in_time, height, width),
        dtype=torch.float32,
    )

    temporal_encoder = TemporalAttention(
        in_channels=in_channels,
        hidden_size=hidden_size,
        d_model=d_model,
        num_head=n_head,
        num_time=in_time,
        num_classes_l2=num_classes_l2,
        num_classes_last=num_classes_last,
    )
    unet3_kwargs = {
        "in_channels": in_channels,
        "in_time": in_time,
        "in_encoding_channels": d_model,
        "init_filter": filters,
        "num_classes": num_classes_last,
        "activation_type": activation_type,
        "deep_sup_dist": True,
        "deep_sup_edge": True,
        "deep_sup_mask": True,
        "mask_activation": Softmax(dim=1),
    }
    mask_model = ResUNet3Psi(**unet3_kwargs)

    # Transformer attention encoder
    logits_hidden, logits_l2, logits_last = temporal_encoder(x)
    logits_l2 = cg(logits_l2)
    logits_last = cg(logits_last)
    logits = mask_model(x, temporal_encoding=logits_hidden)

    assert logits_hidden.shape == (batch_size, d_model, height, width)
    assert logits_l2.shape == (batch_size * height * width, num_classes_l2)
    assert logits_last.shape == (batch_size * height * width, num_classes_last)
    assert len(logits) == 12
    assert logits.get('dist').shape == (batch_size, 1, height, width)
    assert logits.get('dist_3_1').shape == (batch_size, 1, height, width)
    assert logits.get('dist_2_2').shape == (batch_size, 1, height, width)
    assert logits.get('dist_1_3').shape == (batch_size, 1, height, width)
    assert logits.get('edge').shape == (batch_size, 1, height, width)
    assert logits.get('edge_3_1').shape == (batch_size, 1, height, width)
    assert logits.get('edge_2_2').shape == (batch_size, 1, height, width)
    assert logits.get('edge_1_3').shape == (batch_size, 1, height, width)
    assert logits.get('mask').shape == (
        batch_size,
        num_classes_last,
        height,
        width,
    )
    assert logits.get('mask_3_1').shape == (
        batch_size,
        num_classes_last,
        height,
        width,
    )
    assert logits.get('mask_2_2').shape == (
        batch_size,
        num_classes_last,
        height,
        width,
    )
    assert logits.get('mask_1_3').shape == (
        batch_size,
        num_classes_last,
        height,
        width,
    )


if __name__ == '__main__':
    test_cultionet()
