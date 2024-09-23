"""
Source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py

    MIT License

    Copyright (c) 2021 VSainteuf
"""

import numpy as np
import torch


def calc_angle(position: int, hid_idx: int, d_hid: int, time_scaler: int):
    return position / np.power(time_scaler, 2 * (hid_idx // 2) / d_hid)


def get_posi_angle_vec(position, d_hid, time_scaler):
    return [
        calc_angle(position, hid_j, d_hid, time_scaler)
        for hid_j in range(d_hid)
    ]


def get_sinusoid_encoding_table(
    positions: int, d_hid: int, time_scaler: int = 1_000
):
    positions = list(range(positions))
    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i, d_hid, time_scaler) for pos_i in positions]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float32)
