import typing as T

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class FieldOfJunctions(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        stride: int = 1,
        nvals: int = 31,
        delta: float = 0.05,
        eta: float = 0.01,
    ):
        super(FieldOfJunctions, self).__init__()

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
            nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
        )
        self.final_image = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        x = self.reduce(x)

        batch_size, num_channels, height, width = x.shape

        self.h_patches = (height - self.patch_size) // self.stride + 1
        self.w_patches = (width - self.patch_size) // self.stride + 1

        # Split image into overlapping patches, creating a tensor of shape [N, C, R, R, H', W']
        unfold = nn.Unfold(self.patch_size, stride=self.stride)
        image_patches = rearrange(
            unfold(x),
            'b (c hps wps) (hp wp) -> b c hps wps hp wp',
            hps=self.patch_size,
            wps=self.patch_size,
            hp=self.h_patches,
            wp=self.w_patches,
        )
        angles = torch.zeros(
            batch_size,
            3,
            self.h_patches,
            self.w_patches,
            dtype=x.dtype,
            device=x.device,
        )
        x0_y0 = torch.zeros(
            batch_size,
            2,
            self.h_patches,
            self.w_patches,
            dtype=x.dtype,
            device=x.device,
        )

        # Compute number of patches containing each pixel: has shape [H, W]
        fold = nn.Fold(
            output_size=[height, width],
            kernel_size=self.patch_size,
            stride=self.stride,
        )
        num_patches = fold(
            torch.ones(
                batch_size,
                self.patch_size**2,
                self.h_patches * self.w_patches,
                dtype=x.dtype,
                device=x.device,
            ),
        ).squeeze(dim=1)

        # Create local grid within each patch
        meshy, meshx = torch.meshgrid(
            [
                torch.linspace(-1.0, 1.0, self.patch_size, device=x.device),
                torch.linspace(-1.0, 1.0, self.patch_size, device=x.device),
            ],
        )
        self.y = rearrange(meshy, 'hps wps -> 1 hps wps 1 1')
        self.x = rearrange(meshx, 'hps wps -> 1 hps wps 1 1')

        params = torch.cat([angles, x0_y0], dim=1).detach()
        # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for vertex position.
        angle_range = torch.linspace(
            0.0, 2 * np.pi, self.nvals + 1, device=x.device
        )[: self.nvals]
        x0_y0_range = torch.linspace(-3.0, 3.0, self.nvals, device=x.device)

        # Save current global image and boundary map (initially None)
        for i in range(5):
            for bidx in range(batch_size):
                # Repeat the set of parameters `nvals` times along 0th dimension
                params_query = (
                    params[bidx].unsqueeze(0).repeat(self.nvals, 1, 1, 1)
                )
                param_range = angle_range if i < 3 else x0_y0_range
                params_query[:, i, :, :] = params_query[
                    :, i, :, :
                ] + rearrange(param_range, 'l -> l 1 1')
                best_indices = self.get_best_indices(
                    params_query,
                    image_patches=image_patches[bidx].unsqueeze(0),
                    num_channels=num_channels,
                )
                # Update parameters
                params[bidx, i, :, :] = params_query[
                    best_indices.unsqueeze(0),
                    i,
                    rearrange(torch.arange(self.h_patches), 'l -> 1 l 1'),
                    rearrange(torch.arange(self.w_patches), 'l -> 1 1 l'),
                ]

        # Update angles and vertex position using the best values found
        angles.data = params[:, :3, :, :].data
        x0_y0.data = params[:, 3:, :, :].data

        # Update global boundaries and image
        global_boundaries = torch.zeros_like(x)
        smoothed_image = torch.zeros_like(x)
        for bidx in range(batch_size):
            distances, colors, patches = self.get_distances_and_patches(
                params[bidx].unsqueeze(0),
                image_patches=image_patches[bidx].unsqueeze(0),
                num_channels=num_channels,
            )
            smoothed_image[bidx] = self.local_to_global(
                patches, height, width, num_patches[bidx].unsqueeze(0)
            )
            local_boundaries = self.distances_to_boundaries(distances)
            global_boundaries[bidx] = self.local_to_global(
                local_boundaries,
                height,
                width,
                num_patches[bidx].unsqueeze(0),
            )

        global_boundaries = self.final_boundaries(global_boundaries)
        smoothed_image = self.final_image(smoothed_image)

        return {
            "boundaries": global_boundaries,
            "image": smoothed_image,
        }

    def distances_to_boundaries(self, dists: torch.Tensor) -> torch.Tensor:
        """Compute boundary map for each patch, given distance functions.

        The width of the boundary is determined by opts.delta.
        """
        # Find places where either distance transform is small, except where d1 > 0 and d2 < 0
        d1 = dists[:, 0:1, :, :, :, :]
        d2 = dists[:, 1:2, :, :, :, :]
        minabsdist = torch.where(
            d1 < 0.0,
            -d1,
            torch.where(d2 < 0.0, torch.min(d1, -d2), torch.min(d1, d2)),
        )

        return 1.0 / (1.0 + (minabsdist / self.delta) ** 2)

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
        N = patches.shape[0]
        C = patches.shape[1]
        fold = torch.nn.Fold(
            output_size=[height, width],
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        return fold(patches.view(N, C * self.patch_size**2, -1)).view(
            N, C, height, width
        ) / num_patches.unsqueeze(0).unsqueeze(0)

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
            distances, colors, smooth_patches, image_patches
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
        # Get best color for each wedge and each patch
        colors = (image_patches.unsqueeze(2) * wedges.unsqueeze(1)).sum(
            -3
        ).sum(-3) / (wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # Fill wedges with optimal colors
        patches = (
            wedges.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)
        ).sum(dim=2)

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
        dists: torch.Tensor,
        colors: torch.Tensor,
        patches: torch.Tensor,
        image_patches: torch.Tensor,
        lmbda_boundary: float = 0.0,
        lmbda_color: float = 0.0,
    ):
        """Compute the objective of our model (see Equation 8 of the paper)."""
        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = (
            ((image_patches - patches) ** 2).mean(-3).mean(-3).sum(1)
        )

        # Add spatial consistency loss for each patch, if lambda > 0
        if lmbda_boundary > 0.0:
            loss_per_patch = (
                loss_per_patch
                + lmbda_boundary * self.get_boundary_consistency_term(dists)
            )

        if lmbda_color > 0.0:
            loss_per_patch = (
                loss_per_patch
                + lmbda_color * self.get_color_consistency_term(dists, colors)
            )

        return loss_per_patch
