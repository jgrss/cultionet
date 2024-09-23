from typing import Callable

import torch.nn as nn


def init_attention_weights(module: Callable) -> None:
    if isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
        ),
    ):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.normal_(module.bias.data)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def init_conv_weights(module: Callable) -> None:
    if isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
        ),
    ):
        nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.normal_(module.bias.data)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
