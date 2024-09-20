import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad
def merge_distances(
    foreground_distances: torch.Tensor,
    crop_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    inverse: bool = True,
    beta: float = 10.0,
) -> torch.Tensor:

    if len(foreground_distances.shape) == 3:
        foreground_distances = einops.rearrange(
            foreground_distances, 'b h w -> b 1 h w'
        )

    if len(crop_mask.shape) == 3:
        crop_mask = einops.rearrange(crop_mask, 'b h w -> b 1 h w')

    if len(edge_mask.shape) == 3:
        edge_mask = einops.rearrange(edge_mask, 'b h w -> b 1 h w')

    background_mask = (
        ((crop_mask == 0) & (edge_mask == 0)).detach().cpu().numpy()
    )
    background_dist = np.zeros(background_mask.shape, dtype='float32')
    for midx in range(background_mask.shape[0]):
        bdist = cv2.distanceTransform(
            background_mask[midx].squeeze(axis=0).astype('uint8'),
            cv2.DIST_L2,
            3,
        )
        bdist /= bdist.max()

        if inverse:
            bdist = 1.0 - bdist

        if beta != 1:
            bdist = bdist**beta
            bdist[np.isnan(bdist)] = 0

        background_dist[midx] = bdist[None, None]

    if inverse:
        foreground_distances = 1.0 - foreground_distances

    if beta != 1:
        foreground_distances = foreground_distances**beta
        foreground_distances[torch.isnan(foreground_distances)] = 0

    distance = np.where(
        background_mask,
        background_dist,
        foreground_distances.detach().cpu().numpy(),
    )
    targets = torch.tensor(
        distance,
        dtype=foreground_distances.dtype,
        device=foreground_distances.device,
    )

    targets[edge_mask == 1] = 1.0 if inverse else 0.0

    return targets


def check_upsample(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    if x.shape[-2:] != size:
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=True,
        )

    return x
