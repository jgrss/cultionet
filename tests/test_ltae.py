import torch

from cultionet.models.ltae import LightweightTemporalAttentionEncoder


def test_ltae():
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

    x = torch.rand(
        (batch_size, in_channels, in_time, height, width),
        dtype=torch.float32,
    )

    temporal_encoder = LightweightTemporalAttentionEncoder(
        in_channels=in_channels,
        hidden_size=hidden_size,
        d_model=d_model,
        n_head=n_head,
        n_time=in_time,
        mlp=[d_model, hidden_size],
        return_att=True,
        d_k=4,
        num_classes_l2=num_classes_l2,
        num_classes_last=num_classes_last,
    )
    # Transformer attention encoder
    out, last_l2, last, attn = temporal_encoder(x)

    assert out.shape == (batch_size, hidden_size, height, width)
    assert last_l2.shape == (batch_size, num_classes_l2, height, width)
    assert last.shape == (batch_size, num_classes_last, height, width)
    assert attn.shape == (n_head, batch_size, in_time, height, width)


if __name__ == '__main__':
    test_ltae()
