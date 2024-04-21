import torch

from cultionet.models.temporal_transformer import TemporalAttention


def test_ltae():
    batch_size = 2
    in_channels = 4
    in_time = 20
    height = 40
    width = 40

    hidden_channels = 128
    d_model = 256
    n_head = 16
    num_classes_l2 = 2
    num_classes_last = 3

    x = torch.rand(
        (batch_size, in_channels, in_time, height, width),
        dtype=torch.float32,
    )

    temporal_encoder = TemporalAttention(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        d_model=d_model,
        num_head=n_head,
        num_time=in_time,
        num_classes_l2=num_classes_l2,
        num_classes_last=num_classes_last,
    )
    # Transformer attention encoder
    logits_hidden, classes_l2, classes_last = temporal_encoder(x)

    assert logits_hidden.shape == (batch_size, d_model, height, width)
    assert classes_l2.shape == (batch_size, num_classes_l2, height, width)
    assert classes_last.shape == (batch_size, num_classes_last, height, width)


if __name__ == '__main__':
    test_ltae()
