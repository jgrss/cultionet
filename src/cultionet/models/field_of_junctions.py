import typing as T
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cultionet.models import model_utils


@dataclass
class FieldOfJunctionsArgs:
    patch_size: int
    stride: int
    nvals: int
    delta: float
    eta: float
    lambda_boundary_final: float
    lambda_color_final: float
    num_initialization_iters: int
    num_refinement_iters: int
    num_iters: int
    greedy_step_every_iters: int
    parallel_mode: bool


class _FieldOfJunctions:
    """
    Source:
        https://github.com/dorverbin/fieldofjunctions/tree/main
    """

    def __init__(
        self,
        x: torch.Tensor,
        in_height: int,
        in_width: int,
        foj_args: FieldOfJunctionsArgs,
    ) -> T.Dict[str, torch.Tensor]:
        self.dtype = x.dtype
        self.device = x.device
        self.in_height = in_height
        self.in_width = in_width
        self.foj_args = foj_args

        x = x.clone().detach()
        batch_size, num_channels, height, width = x.shape
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width

        self.h_patches = (
            self.height - self.foj_args.patch_size
        ) // self.foj_args.stride + 1
        self.w_patches = (
            self.width - self.foj_args.patch_size
        ) // self.foj_args.stride + 1

        # Split image into overlapping patches, creating a tensor of shape [N, C, R, R, H', W']
        unfold = nn.Unfold(foj_args.patch_size, stride=foj_args.stride)
        self.image_patches = rearrange(
            unfold(x),
            'b (c hps wps) (hp wp) -> b c hps wps hp wp',
            hps=self.foj_args.patch_size,
            wps=self.foj_args.patch_size,
            hp=self.h_patches,
            wp=self.w_patches,
        )

        self.params_init = torch.zeros(
            self.batch_size,
            5,
            self.h_patches,
            self.w_patches,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )

        # Compute number of patches containing each pixel: has shape [H, W]
        fold = nn.Fold(
            output_size=[self.height, self.width],
            kernel_size=self.foj_args.patch_size,
            stride=self.foj_args.stride,
        )
        self.num_patches = fold(
            torch.ones(
                self.batch_size,
                self.foj_args.patch_size**2,
                self.h_patches * self.w_patches,
                dtype=x.dtype,
                device=x.device,
            ),
        ).view(self.height, self.width)

        # Create local grid within each patch
        meshy, meshx = torch.meshgrid(
            [
                torch.linspace(
                    -1.0, 1.0, self.foj_args.patch_size, device=x.device
                ),
                torch.linspace(
                    -1.0, 1.0, self.foj_args.patch_size, device=x.device
                ),
            ],
        )
        self.y = rearrange(meshy, 'hps wps -> 1 hps wps 1 1')
        self.x = rearrange(meshx, 'hps wps -> 1 hps wps 1 1')

        # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for vertex position.
        self.angle_range = torch.linspace(
            0.0, 2 * np.pi, self.foj_args.nvals + 1, device=self.device
        )[: self.foj_args.nvals]
        self.x0_y0_range = torch.linspace(
            -3.0, 3.0, self.foj_args.nvals, device=self.device
        )

        self.optimizer = None

    def optimize(self) -> T.Dict[str, torch.Tensor]:
        """Optimize field of junctions."""
        results = {}
        for iteration in range(self.foj_args.num_iters):
            results = self.step(
                iteration,
                image_patches=self.image_patches,
                num_patches=self.num_patches,
                **results,
            )

        return results

    def step(
        self,
        iteration: int,
        image_patches: torch.Tensor,
        num_patches: torch.Tensor,
        global_image: torch.Tensor = None,
        global_boundaries: torch.Tensor = None,
    ):
        """Perform one step (either initialization's coordinate descent, or
        refinement gradient descent)"""
        # Linearly increase lambda from 0 to lambda_boundary_final and lambda_color_final
        if self.foj_args.num_refinement_iters <= 1:
            factor = 0.0
        else:
            factor = max(
                [
                    0,
                    (iteration - self.foj_args.num_initialization_iters)
                    / (self.foj_args.num_refinement_iters - 1),
                ]
            )

        lmbda_boundary = factor * self.foj_args.lambda_boundary_final
        lmbda_color = factor * self.foj_args.lambda_color_final

        if (iteration < self.foj_args.num_initialization_iters) or (
            iteration - self.foj_args.num_initialization_iters + 1
        ) % self.foj_args.greedy_step_every_iters == 0:
            out = self.initialization_step(
                image_patches=image_patches,
                num_patches=num_patches,
                lmbda_boundary=lmbda_boundary,
                lmbda_color=lmbda_color,
                global_image=global_image,
                global_boundaries=global_boundaries,
            )
        else:
            if self.optimizer is None:
                self.params = self.params_init.clone()
                self.params.requires_grad = True
                # Create optimizers for angles and vertices
                self.optimizer = torch.optim.AdamW(
                    [self.params],
                    lr=0.001,
                    weight_decay=0.01,
                    eps=1e-4,
                )
            out = self.refinement_step(
                image_patches=image_patches,
                num_patches=num_patches,
                global_image=global_image,
                global_boundaries=global_boundaries,
                lmbda_boundary=lmbda_boundary,
                lmbda_color=lmbda_color,
            )

        return out

    def refinement_step(
        self,
        image_patches: torch.Tensor,
        num_patches: torch.Tensor,
        global_image: torch.Tensor,
        global_boundaries: torch.Tensor,
        lmbda_boundary: float = 0.0,
        lmbda_color: float = 0.0,
    ):
        """Perform a single refinement step."""
        # Compute distance functions, colors, and junction patches
        distances, colors, patches = self.get_distances_and_patches(
            self.params,
            num_channels=self.num_channels,
            image_patches=image_patches,
            lmbda_color=lmbda_color,
            global_image=global_image,
        )
        # Compute loss
        loss = self.get_loss(
            distances,
            colors,
            patches,
            num_channels=self.num_channels,
            image_patches=image_patches,
            lmbda_boundary=lmbda_boundary,
            lmbda_color=lmbda_color,
            global_image=global_image,
            global_boundaries=global_boundaries,
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            # Update global boundaries and image
            distances, colors, patches = self.get_distances_and_patches(
                self.params,
                num_channels=self.num_channels,
                image_patches=image_patches,
                lmbda_color=lmbda_color,
                global_image=global_image,
            )
            global_image = self.local_to_global(patches, num_patches)
            local_boundaries = self.distances_to_boundaries(distances)
            global_boundaries = self.local_to_global(
                local_boundaries, num_patches
            )

        return {
            'global_image': global_image,
            'global_boundaries': global_boundaries,
        }

    def initialization_step(
        self,
        image_patches: torch.Tensor,
        num_patches: torch.Tensor,
        lmbda_boundary: float = 0.0,
        lmbda_color: float = 0.0,
        global_image: torch.Tensor = None,
        global_boundaries: torch.Tensor = None,
    ):
        params = self.params_init
        # Save current global image and boundary map (initially None)
        for i in range(5):
            # Repeat the set of parameters `nvals` times along 0th dimension
            params_query = params.repeat(self.foj_args.nvals, 1, 1, 1)
            param_range = self.angle_range if i < 3 else self.x0_y0_range
            params_query[:, i, :, :] = params_query[:, i, :, :] + rearrange(
                param_range, 'l -> l 1 1'
            )
            best_indices = self.get_best_indices(
                params_query,
                image_patches=image_patches,
                num_channels=self.num_channels,
                lmbda_boundary=lmbda_boundary,
                lmbda_color=lmbda_color,
                global_image=global_image,
                global_boundaries=global_boundaries,
            )
            # Update parameters
            params[0, i, :, :] = params_query[
                best_indices.unsqueeze(0),
                i,
                rearrange(torch.arange(self.h_patches), 'l -> 1 l 1'),
                rearrange(torch.arange(self.w_patches), 'l -> 1 1 l'),
            ]

        # Update angles and vertex position using the best values found
        self.params_init[:, :3, ...] = params[:, :3, ...]
        self.params_init[:, 3:, ...] = params[:, 3:, ...]

        with torch.no_grad():
            distances, colors, patches = self.get_distances_and_patches(
                self.params_init,
                num_channels=self.num_channels,
                image_patches=image_patches,
                global_image=global_image,
            )
            global_image = self.local_to_global(patches, num_patches)
            local_boundaries = self.distances_to_boundaries(distances)
            global_boundaries = self.local_to_global(
                local_boundaries, num_patches
            )

        return {
            'global_image': global_image,
            'global_boundaries': global_boundaries,
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

        return 1.0 / (1.0 + (minabsdist / self.foj_args.delta) ** 2)

    def local_to_global(
        self, patches: torch.Tensor, num_patches: torch.Tensor
    ) -> torch.Tensor:
        """Compute average value for each pixel over all patches containing it.

        For example, this can be used to compute the global boundary maps, or
        the boundary-aware smoothed image.
        """
        num_channels = patches.shape[1]
        fold = torch.nn.Fold(
            output_size=[self.height, self.width],
            kernel_size=self.foj_args.patch_size,
            stride=self.foj_args.stride,
        )
        patches = rearrange(
            patches,
            'b c hps wps hp wp -> b (c hps wps) (hp wp)',
            b=self.batch_size,
            c=num_channels,
            hps=self.foj_args.patch_size,
            wps=self.foj_args.patch_size,
            hp=self.h_patches,
            wp=self.w_patches,
        )

        return fold(patches) / num_patches.unsqueeze(0).unsqueeze(0)

    def get_best_indices(
        self,
        params: torch.Tensor,
        num_channels: int,
        image_patches: torch.Tensor,
        lmbda_boundary: float,
        lmbda_color: float,
        global_image: torch.Tensor = None,
        global_boundaries: torch.Tensor = None,
    ) -> torch.Tensor:
        distances, colors, smooth_patches = self.get_distances_and_patches(
            params,
            num_channels=num_channels,
            image_patches=image_patches,
            lmbda_color=lmbda_color,
            global_image=global_image,
        )
        loss_per_patch = self.get_loss(
            distances,
            colors,
            smooth_patches,
            num_channels=num_channels,
            image_patches=image_patches,
            lmbda_boundary=lmbda_boundary,
            lmbda_color=lmbda_color,
            global_image=global_image,
            global_boundaries=global_boundaries,
        )
        best_indices = loss_per_patch.argmin(dim=0)

        return best_indices

    def get_distances_and_patches(
        self,
        params: torch.Tensor,
        num_channels: int,
        image_patches: torch.Tensor,
        lmbda_color: float = 0.0,
        global_image: torch.Tensor = None,
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

        if lmbda_color >= 0 and global_image is not None:
            unfold = nn.Unfold(
                self.foj_args.patch_size, stride=self.foj_args.stride
            )
            global_image = unfold(global_image.detach())
            curr_global_image_patches = rearrange(
                global_image,
                'b (c hps wps) (hp wp) -> b c hps wps hp wp',
                c=num_channels,
                hps=self.foj_args.patch_size,
                wps=self.foj_args.patch_size,
                hp=self.h_patches,
                wp=self.w_patches,
            )

            numerator = (
                (
                    (
                        image_patches + lmbda_color * curr_global_image_patches
                    ).unsqueeze(2)
                    * wedges.unsqueeze(1)
                )
                .sum(-3)
                .sum(-3)
            )
            denominator = (1.0 + lmbda_color) * wedges.sum(-3).sum(
                -3
            ).unsqueeze(1)

            colors = numerator / (denominator + 1e-10)
        else:
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
        hdists = 0.5 * (
            1.0 + (2.0 / np.pi) * torch.atan(dists / self.foj_args.eta)
        )

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
        num_channels: int,
        image_patches: torch.Tensor,
        lmbda_boundary: float = 0.0,
        lmbda_color: float = 0.0,
        global_image: torch.Tensor = None,
        global_boundaries: torch.Tensor = None,
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
                + lmbda_boundary
                * self.get_boundary_consistency_term(dists, global_boundaries)
            )

        if lmbda_color > 0.0:
            loss_per_patch = (
                loss_per_patch
                + lmbda_color
                * self.get_color_consistency_term(
                    dists, colors, num_channels, global_image
                )
            )

        return loss_per_patch

    def get_boundary_consistency_term(
        self, dists: torch.Tensor, global_boundaries: torch.Tensor
    ) -> torch.Tensor:
        """Compute the spatial consistency term."""
        # Split global boundaries into patches
        unfold = nn.Unfold(
            self.foj_args.patch_size, stride=self.foj_args.stride
        )
        global_boundaries = global_boundaries.detach()
        curr_global_boundaries_patches = rearrange(
            unfold(global_boundaries),
            'b (hps wps) (hp wp) -> b 1 hps wps hp wp',
            hps=self.foj_args.patch_size,
            wps=self.foj_args.patch_size,
            hp=self.h_patches,
            wp=self.w_patches,
        )

        # Get local boundaries defined using the queried parameters (defined by `dists`)
        local_boundaries = self.distances_to_boundaries(dists)

        # Compute consistency term
        consistency = (
            ((local_boundaries - curr_global_boundaries_patches) ** 2)
            .mean(2)
            .mean(2)
        )

        return consistency[:, 0, :, :]

    def get_color_consistency_term(
        self,
        dists: torch.Tensor,
        colors: torch.Tensor,
        num_channels: int,
        global_image: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the spatial consistency term."""
        # Split into patches
        unfold = nn.Unfold(
            self.foj_args.patch_size, stride=self.foj_args.stride
        )
        global_image = unfold(global_image.detach())
        curr_global_image_patches = rearrange(
            global_image,
            'b (c hps wps) (hp wp) -> b c hps wps hp wp',
            c=num_channels,
            hps=self.foj_args.patch_size,
            wps=self.foj_args.patch_size,
            hp=self.h_patches,
            wp=self.w_patches,
        )

        wedges = self.distances_to_indicators(
            dists
        )  # shape [N, 3, R, R, H', W']

        # Compute consistency term
        consistency = (
            (
                wedges.unsqueeze(1)
                * (
                    colors.unsqueeze(-3).unsqueeze(-3)
                    - curr_global_image_patches.unsqueeze(2)
                )
                ** 2
            )
            .mean(-3)
            .mean(-3)
            .sum(1)
            .sum(1)
        )

        return consistency


class FieldOfJunctions(nn.Module):
    """
    Source:
        https://github.com/dorverbin/fieldofjunctions/tree/main
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        stride: int = 1,
        nvals: int = 31,
        delta: float = 0.05,
        eta: float = 0.01,
        lambda_boundary_final: float = 0.5,
        lambda_color_final: float = 0.1,
        num_initialization_iters: int = 10,
        num_refinement_iters: int = 30,
        greedy_step_every_iters: int = 10,
    ):
        super(FieldOfJunctions, self).__init__()

        self.foj_args = FieldOfJunctionsArgs(
            patch_size=patch_size,
            stride=stride,
            nvals=nvals,
            delta=delta,
            eta=eta,
            lambda_boundary_final=lambda_boundary_final,
            lambda_color_final=lambda_color_final,
            num_initialization_iters=num_initialization_iters,
            num_refinement_iters=num_refinement_iters,
            num_iters=num_initialization_iters + num_refinement_iters,
            greedy_step_every_iters=greedy_step_every_iters,
            parallel_mode=True,
        )

    def forward(self, x: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        """Optimize field of junctions."""
        batch_size, num_channels, in_height, in_width = x.shape
        x = F.interpolate(
            x,
            size=(in_height // 2, in_width // 2),
            mode="bilinear",
            align_corners=True,
        )

        global_boundaries = torch.zeros_like(x)
        smoothed_image = torch.zeros_like(x)
        for i, x_batch in enumerate(x):
            foj = _FieldOfJunctions(
                x=x_batch.unsqueeze(0),
                in_height=in_height,
                in_width=in_width,
                foj_args=self.foj_args,
            )
            optimized = foj.optimize()
            global_boundaries[i] = optimized['global_boundaries']
            smoothed_image[i] = optimized['global_image']

        return {
            'boundaries': global_boundaries,
            'image': smoothed_image,
        }
