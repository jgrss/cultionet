import typing as T

import attr
import numpy as np
import torch
import torch.nn.functional as F
from rasterio.windows import Window


@attr.s
class ModelOutputs(object):
    """A class for reshaping of the model output estimates."""

    distance: torch.Tensor = attr.ib(
        validator=attr.validators.instance_of(torch.Tensor)
    )
    edge: torch.Tensor = attr.ib(
        validator=attr.validators.instance_of(torch.Tensor)
    )
    crop: torch.Tensor = attr.ib(
        validator=attr.validators.instance_of(torch.Tensor)
    )
    crop_type: T.Union[torch.Tensor, None] = attr.ib(
        validator=attr.validators.optional(
            attr.validators.instance_of(torch.Tensor)
        )
    )
    instances: T.Optional[T.Union[None, np.ndarray]] = attr.ib(
        default=None,
        validator=attr.validators.optional(
            attr.validators.instance_of(np.ndarray)
        ),
    )
    apply_softmax: T.Optional[bool] = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )

    def stack_outputs(self) -> np.ndarray:
        return (
            torch.cat((self.distance, self.edge, self.crop), dim=0)
            .detach()
            .cpu()
            .numpy()
        )

    @staticmethod
    def _clip_and_reshape(
        tarray: torch.Tensor, window_obj: Window
    ) -> np.ndarray:
        if (len(tarray.shape) == 1) or (
            (len(tarray.shape) > 1) and (tarray.shape[1] == 1)
        ):
            return (
                tarray.contiguous()
                .view(-1)
                .detach()
                .cpu()
                .numpy()
                .clip(0, 1)
                .reshape(window_obj.height, window_obj.width)
            )
        else:
            n_layers = tarray.shape[1]

            return (
                tarray.contiguous()
                .t()
                .detach()
                .cpu()
                .numpy()
                .clip(0, 1)
                .reshape(n_layers, window_obj.height, window_obj.width)
            )

    def inputs_to_probas(
        self, inputs: np.ndarray, w_pad: Window
    ) -> np.ndarray:
        if self.apply_softmax:
            inputs = F.softmax(inputs, dim=1, dtype=inputs.dtype)[:, 1]
        else:
            if len(inputs.shape) > 1:
                if inputs.shape[1] > 1:
                    # Two-class output
                    inputs = inputs[:, 1]

        inputs = self._clip_and_reshape(inputs, w_pad)

        return inputs

    def reshape(self, w: Window, w_pad: Window) -> None:
        # Get the distance from edges
        self.edge_dist = self._clip_and_reshape(self.distance, w_pad)
        # Get the edge probabilities
        self.edge_probas = self.inputs_to_probas(self.edge, w_pad)
        # Get the crop probabilities
        self.crop_probas = self.inputs_to_probas(self.crop, w_pad)

        # Get the crop-type probabilities
        self.crop_type_probas = None
        if self.crop_type is not None:
            self.crop_type_probas = self.inputs_to_probas(
                self.crop_type, w_pad
            )

        # Reshape the window chunk and slice off padding
        i = abs(w.row_off - w_pad.row_off)
        j = abs(w.col_off - w_pad.col_off)
        slicer = (slice(i, i + w.height), slice(j, j + w.width))
        slicer3d = (
            slice(0, None),
            slice(i, i + w.height),
            slice(j, j + w.width),
        )
        self.edge_dist = self.edge_dist[slicer]
        self.edge_probas = self.edge_probas[slicer]
        if len(self.crop_probas.shape) == 3:
            self.crop_probas = self.crop_probas[slicer3d]
        else:
            self.crop_probas = self.crop_probas[slicer]
        if self.crop_type_probas is not None:
            if len(self.crop_type_probas.shape) == 3:
                self.crop_type_probas = self.crop_type_probas[slicer3d]
            else:
                self.crop_type_probas = self.crop_type_probas[slicer]
        if self.instances is not None:
            self.instances = self.instances.reshape(w_pad.height, w_pad.width)
            self.instances = self.instances[slicer]

    def nan_to_num(self):
        # Convert the data type to integer and set 'no data' values
        self.edge_dist = np.nan_to_num(
            self.edge_dist, nan=-1.0, neginf=-1.0, posinf=-1.0
        ).astype('float32')

        self.edge_probas = np.nan_to_num(
            self.edge_probas, nan=-1.0, neginf=-1.0, posinf=-1.0
        ).astype('float32')

        self.crop_probas = np.nan_to_num(
            self.crop_probas, nan=-1.0, neginf=-1.0, posinf=-1.0
        ).astype('float32')

        if self.crop_type_probas is not None:
            self.crop_type_probas = np.nan_to_num(
                self.crop_type_probas, nan=-1.0, neginf=-1.0, posinf=-1.0
            ).astype('float32')
