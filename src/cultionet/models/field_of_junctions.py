import typing as T

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldOfJunctions(nn.Module):
    """
    Source:
        https://github.com/dorverbin/fieldofjunctions
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int = 8,
        stride: int = 1,
        nvals: int = 31,
        delta: float = 0.05,
        eta: float = 0.01,
    ):
        super(FieldOfJunctions, self).__init__()

        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.stride = stride
        self.nvals = nvals
        self.delta = delta
        self.eta = eta

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(),
        )
        self.final_boundaries = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
        )
        # self.final_image = nn.Sequential(
        #     nn.Conv2d(3, in_channels, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.SiLU(),
        # )

        # Number of patches (throughout the documentation H_patches and W_patches are denoted by H' and W' resp.)
        self.h_patches = (height - patch_size) // stride + 1
        self.w_patches = (width - patch_size) // stride + 1

        self.unfold = nn.Unfold(self.patch_size, stride=self.stride)
        self.fold = nn.Fold(
            output_size=[height, width],
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        # Create local grid within each patch
        meshy, meshx = torch.meshgrid(
            [
                torch.linspace(-1.0, 1.0, self.patch_size),
                torch.linspace(-1.0, 1.0, self.patch_size),
            ],
            indexing='ij',
        )
        self.y = einops.rearrange(meshy, 'p k -> 1 p k 1 1')
        self.x = einops.rearrange(meshx, 'p k -> 1 p k 1 1')

        # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for vertex position.
        self.angle_range = torch.linspace(0.0, 2 * np.pi, self.nvals + 1)[
            : self.nvals
        ]
        self.x0_y0_range = torch.linspace(-3.0, 3.0, self.nvals)

        # Create pytorch variables for angles and vertex position for each patch
        angles = torch.zeros(
            1, 3, self.h_patches, self.w_patches, dtype=torch.float32
        )
        x0y0 = torch.zeros(
            1, 2, self.h_patches, self.w_patches, dtype=torch.float32
        )
        # self.angles.requires_grad = True
        # self.x0y0.requires_grad = True

        self.params = torch.cat([angles, x0y0], dim=1)

    def forward(self, x: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        batch_size, in_channels, in_height, in_width = x.shape

        row_pad = 0
        col_pad = 0
        if (in_height, in_width) != (self.height, self.width):
            row_pad = (self.height - in_height) // 2
            col_pad = (self.width - in_width) // 2
            x = F.pad(
                x,
                (row_pad, row_pad, col_pad, col_pad),
                mode='constant',
                value=0,
            )

        x = self.reduce(x)

        batch_size, num_channels, height, width = x.shape

        # Split image into overlapping patches, creating a tensor of shape [N, C, R, R, H', W']
        image_patches = einops.rearrange(
            self.unfold(x),
            'b (c p k) (h w) -> b c p k h w',
            p=self.patch_size,
            k=self.patch_size,
            h=self.h_patches,
            w=self.w_patches,
        )

        # Compute number of patches containing each pixel: has shape [H, W]
        num_patches = self.fold(
            torch.ones(
                batch_size,
                self.patch_size**2,
                self.h_patches * self.w_patches,
                dtype=x.dtype,
                device=x.device,
            ),
        )
        # Paper shape is (height x width)
        num_patches = einops.rearrange(num_patches, 'b 1 h w -> b h w')

        self.y = self.y.to(device=x.device)
        self.x = self.x.to(device=x.device)
        angle_range = self.angle_range.to(device=x.device)
        x0_y0_range = self.x0_y0_range.to(device=x.device)

        params = self.params.detach()

        # Run one step of Algorithm 2, sequentially improving each coordinate
        for i in range(5):
            # Repeat the set of parameters `nvals` times along 0th dimension
            params_query = params.repeat(self.nvals, 1, 1, 1)
            param_range = angle_range if i < 3 else x0_y0_range
            params_query[:, i, :, :] = params_query[
                :, i, :, :
            ] + einops.rearrange(param_range, 'l -> l 1 1')

            best_indices = self.get_best_indices(
                params_query,
                image_patches=image_patches,
                num_channels=num_channels,
            )

            # Update parameters
            params[0, i, :, :] = params_query[
                einops.rearrange(best_indices, 'h w -> 1 h w'),
                i,
                einops.rearrange(torch.arange(self.h_patches), 'l -> 1 l 1'),
                einops.rearrange(torch.arange(self.w_patches), 'l -> 1 1 l'),
            ]

        # Heuristic for accelerating convergence (not necessary but sometimes helps):
        # Update x0 and y0 along the three optimal angles (search over a line passing through current x0, y0)
        for i in range(3):
            params_query = params.repeat(self.nvals, 1, 1, 1)
            params_query[:, 3, :, :] = params[:, 3, :, :] + torch.cos(
                params[:, i, :, :]
            ) * x0_y0_range.view(-1, 1, 1)
            params_query[:, 4, :, :] = params[:, 4, :, :] + torch.sin(
                params[:, i, :, :]
            ) * x0_y0_range.view(-1, 1, 1)
            best_indices = self.get_best_indices(
                params_query,
                image_patches=image_patches,
                num_channels=num_channels,
            )

            # Update vertex positions of parameters
            for j in range(3, 5):
                params[:, j, :, :] = params_query[
                    einops.rearrange(best_indices, 'h w -> 1 h w'),
                    j,
                    einops.rearrange(
                        torch.arange(self.h_patches), 'l -> 1 l 1'
                    ),
                    einops.rearrange(
                        torch.arange(self.w_patches), 'l -> 1 1 l'
                    ),
                ]

        self.params.data = params.data

        # Update global boundaries and image
        distances, colors, patches = self.get_distances_and_patches(
            params,
            image_patches=image_patches,
            num_channels=num_channels,
        )
        # smoothed_image = self.local_to_global(
        #     patches, height, width, num_patches
        # )
        local_boundaries = self.distances_to_boundaries(distances)
        global_boundaries = self.local_to_global(
            einops.rearrange(local_boundaries, '1 1 p k h w -> 1 1 1 p k h w'),
            height,
            width,
            num_patches,
        )
        global_boundaries = self.final_boundaries(global_boundaries)
        # smoothed_image = self.final_image(smoothed_image)

        if row_pad > 0:
            global_boundaries = global_boundaries[
                :,
                :,
                row_pad : row_pad + in_height,
                col_pad : col_pad + in_width,
            ]

        return global_boundaries

    def distances_to_boundaries(self, dists: torch.Tensor) -> torch.Tensor:
        """Compute boundary map for each patch, given distance functions.

        The width of the boundary is determined by opts.delta.
        """
        # Find places where either distance transform is small, except where d1 > 0 and d2 < 0
        d1 = dists[:, 0:1, ...]
        d2 = dists[:, 1:2, ...]
        min_abs_distance = torch.where(
            d1 < 0.0,
            -d1,
            torch.where(d2 < 0.0, torch.min(d1, -d2), torch.min(d1, d2)),
        )

        return 1.0 / (1.0 + (min_abs_distance / self.delta) ** 2)

    def local_to_global(
        self,
        patches: torch.Tensor,
        height: int,
        width: int,
        num_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Compute average value for each pixel over all patches containing it.

        For example, this can be used to compute the global boundary maps, or
        the boundary-aware smoothed image.
        """
        numerator = self.fold(
            einops.rearrange(patches, 'b 1 c p k h w -> b (c p k) (h w)')
        )
        denominator = einops.rearrange(num_patches, 'b h w -> b 1 h w')

        return numerator / denominator

    def get_best_indices(
        self,
        params: torch.Tensor,
        image_patches: torch.Tensor,
        num_channels: int,
    ) -> torch.Tensor:
        distances, colors, smooth_patches = self.get_distances_and_patches(
            params,
            image_patches=image_patches,
            num_channels=num_channels,
        )
        loss_per_patch = self.get_loss(
            distances=distances,
            colors=colors,
            patches=smooth_patches,
            image_patches=image_patches,
        )
        best_indices = loss_per_patch.argmin(dim=0)

        return best_indices

    def get_distances_and_patches(
        self,
        params: torch.Tensor,
        image_patches: torch.Tensor,
        num_channels: int,
        lmbda_color: float = 0.0,
    ):
        """Compute distance functions and piecewise-constant patches given
        junction parameters."""
        # Get dists
        distances = self.params_to_distances(
            params
        )  # shape [N, 2, R, R, H', W']

        # Get wedge indicator functions
        wedges = self.distances_to_indicators(
            distances
        )  # shape [N, 3, R, R, H', W']

        # if lmbda_color >= 0 and self.global_image is not None:
        #     curr_global_image_patches = nn.Unfold(self.patch_size, stride=self.opts.stride)(
        #         self.global_image.detach()).view(1, num_channels, self.patch_size, self.patch_size, self.h_patches, self.w_patches)

        #     numerator = ((self.img_patches + lmbda_color *
        #                   curr_global_image_patches).unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3)
        #     denominator = (1.0 + lmbda_color) * wedges.sum(-3).sum(-3).unsqueeze(1)

        #     colors = numerator / (denominator + 1e-10)
        # else:

        numerator = einops.rearrange(
            image_patches, 'b c p k h w -> b 1 c 1 p k h w'
        ) * einops.rearrange(wedges, 'n c p k h w -> 1 n 1 c p k h w')
        numerator = einops.reduce(
            numerator, 'b n c l p k h w -> b n c l h w', 'sum'
        )
        denominator = (
            einops.reduce(wedges, 'n c p k h w -> 1 n 1 c h w', 'sum') + 1e-10
        )
        colors = numerator / denominator

        # Fill wedges with optimal colors
        patches = einops.rearrange(
            wedges, 'n c p k h w -> 1 n 1 c p k h w'
        ) * einops.rearrange(colors, 'b n c l h w -> b n c l 1 1 h w')
        patches = einops.reduce(
            patches, 'b n c l p k h w -> b n c p k h w', 'sum'
        )

        return distances, colors, patches

    def params_to_distances(
        self, params: torch.Tensor, tau=1e-1
    ) -> torch.Tensor:
        """Compute distance functions from field of junctions."""
        x0 = (
            params[:, 3, :, :].unsqueeze(1).unsqueeze(1)
        )  # shape [N, 1, 1, H', W']
        y0 = (
            params[:, 4, :, :].unsqueeze(1).unsqueeze(1)
        )  # shape [N, 1, 1, H', W']

        # Sort so angle1 <= angle2 <= angle3 (mod 2pi)
        angles = torch.remainder(params[:, :3, :, :], 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]

        angle1 = (
            angles[:, 0, :, :].unsqueeze(1).unsqueeze(1)
        )  # shape [N, 1, 1, H', W']
        angle2 = (
            angles[:, 1, :, :].unsqueeze(1).unsqueeze(1)
        )  # shape [N, 1, 1, H', W']
        angle3 = (
            angles[:, 2, :, :].unsqueeze(1).unsqueeze(1)
        )  # shape [N, 1, 1, H', W']

        # Define another angle halfway between angle3 and angle1, clockwise from angle3
        # This isn't critical but it seems a bit more stable for computing gradients
        angle4 = 0.5 * (angle1 + angle3) + torch.where(
            torch.remainder(0.5 * (angle1 - angle3), 2 * np.pi) >= np.pi,
            torch.ones_like(angle1) * np.pi,
            torch.zeros_like(angle1),
        )

        def _g(dtheta):
            # Map from [0, 2pi] to [-1, 1]
            return (dtheta / np.pi - 1.0) ** 35

        # Compute the two distance functions
        sgn42 = torch.where(
            torch.remainder(angle2 - angle4, 2 * np.pi) < np.pi,
            torch.ones_like(angle2),
            -torch.ones_like(angle2),
        )
        tau42 = _g(torch.remainder(angle2 - angle4, 2 * np.pi)) * tau

        dist42 = (
            sgn42
            * torch.min(
                sgn42
                * (
                    -torch.sin(angle4) * (self.x - x0)
                    + torch.cos(angle4) * (self.y - y0)
                ),
                -sgn42
                * (
                    -torch.sin(angle2) * (self.x - x0)
                    + torch.cos(angle2) * (self.y - y0)
                ),
            )
            + tau42
        )

        sgn13 = torch.where(
            torch.remainder(angle3 - angle1, 2 * np.pi) < np.pi,
            torch.ones_like(angle3),
            -torch.ones_like(angle3),
        )
        tau13 = _g(torch.remainder(angle3 - angle1, 2 * np.pi)) * tau
        dist13 = (
            sgn13
            * torch.min(
                sgn13
                * (
                    -torch.sin(angle1) * (self.x - x0)
                    + torch.cos(angle1) * (self.y - y0)
                ),
                -sgn13
                * (
                    -torch.sin(angle3) * (self.x - x0)
                    + torch.cos(angle3) * (self.y - y0)
                ),
            )
            + tau13
        )

        return torch.stack([dist13, dist42], dim=1)

    def distances_to_indicators(self, dists: torch.Tensor) -> torch.Tensor:
        """Computes the indicator functions u_1, u_2, u_3 from the distance
        functions d_{13}, d_{12}"""
        # Apply smooth Heaviside function to distance functions
        hdists = 0.5 * (1.0 + (2.0 / np.pi) * torch.atan(dists / self.eta))

        # Convert Heaviside functions into wedge indicator functions
        return torch.stack(
            [
                1.0 - hdists[:, 0, :, :, :, :],
                hdists[:, 0, :, :, :, :] * (1.0 - hdists[:, 1, :, :, :, :]),
                hdists[:, 0, :, :, :, :] * hdists[:, 1, :, :, :, :],
            ],
            dim=1,
        )

    def get_loss(
        self,
        distances: torch.Tensor,
        colors: torch.Tensor,
        patches: torch.Tensor,
        image_patches: torch.Tensor,
        lmbda_boundary: float = 0.0,
        lmbda_color: float = 0.0,
    ):
        """Compute the objective of our model (see Equation 8 of the paper)."""

        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = einops.reduce(
            (
                einops.rearrange(image_patches, 'b c p k h w -> b 1 c p k h w')
                - patches
            )
            ** 2,
            'b n c p k h w -> b n c h w',
            'mean',
        )
        loss_per_patch = einops.reduce(
            loss_per_patch, 'b n c h w -> b n h w', 'sum'
        )
        # Reduce to the batch mean
        loss_per_patch = einops.reduce(
            loss_per_patch, 'b n h w -> n h w', 'mean'
        )

        return loss_per_patch


if __name__ == '__main__':
    batch_size = 2
    num_channels = 3
    height = 100
    width = 100

    x = torch.rand(
        (batch_size, num_channels, height, width),
        dtype=torch.float32,
    )

    foj = FieldOfJunctions(
        in_channels=num_channels,
        height=110,
        width=110,
        patch_size=8,
        stride=1,
        nvals=31,
        delta=0.05,
        eta=0.01,
    )
    foj(x)
