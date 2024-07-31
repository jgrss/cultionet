import torch
import torch.nn.functional as F


def check_upsample(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    if x.shape[-2:] != size:
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=True,
        )

    return x
