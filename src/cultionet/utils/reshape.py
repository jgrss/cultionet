import typing as T

import numpy as np
from rasterio.windows import Window
import attr
import torch
import torch.nn.functional as F


def nd_to_columns(data, layers, rows, columns):
    """Reshapes an array from nd layout to [samples (rows*columns) x dimensions]
    """
    if layers == 1:
        return np.ascontiguousarray(data.flatten()[:, np.newaxis])
    else:
        return np.ascontiguousarray(data.transpose(1, 2, 0).reshape(rows*columns, layers))


def columns_to_nd(data, layers, rows, columns):
    """Reshapes an array from columns layout to [layers x rows x columns]
    """
    if layers == 1:
        return np.ascontiguousarray(data.reshape(columns, rows).T)
    else:
        return np.ascontiguousarray(data.T.reshape(layers, rows, columns))


@attr.s
class ModelOutputs(object):
    """A class for reshaping of the model output estimates
    """
    ori: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    distance: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    edge: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    crop: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    crop_type: T.Union[torch.Tensor, None] = attr.ib(
        validator=attr.validators.optional(attr.validators.instance_of(torch.Tensor))
    )
    instances: T.Optional[T.Union[None, np.ndarray]] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(np.ndarray))
    )
    apply_softmax: T.Optional[bool] = attr.ib(
        default=False,
        validator=attr.validators.instance_of(bool)
    )

    def stack_outputs(self, w: Window, w_pad: Window) -> np.ndarray:
        self.reshape(w, w_pad)
        self.nan_to_num()
        if (self.crop_type_probas is not None) and len(self.crop_type_probas.shape) == 3:
            stack_items = (
                self.ori[None],
                self.edge_dist[None],
                self.edge_probas[None],
                self.crop_probas[None]
            )
            if self.crop_type_probas is not None:
                stack_items += (self.crop_type_probas,)
            if self.instances is not None:
                stack_items += (self.instances[None],)

            return np.vstack(stack_items)
        else:
            stack_items = (
                self.ori,
                self.edge_dist,
                self.edge_probas,
                self.crop_probas
            )
            if self.crop_type_probas is not None:
                stack_items += (self.crop_type_probas,)
            if self.instances is not None:
                stack_items += (self.instances,)

            return np.stack(stack_items)

    @staticmethod
    def _clip_and_reshape(tarray: torch.Tensor, window_obj: Window) -> np.ndarray:
        if (len(tarray.shape) == 1) or ((len(tarray.shape) > 1) and (tarray.shape[1] == 1)):
            return (
                tarray
                .contiguous()
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
                tarray
                .contiguous()
                .t()
                .detach()
                .cpu()
                .numpy()
                .clip(0, 1)
                .reshape(n_layers, window_obj.height, window_obj.width)
            )

    def reshape(self, w: Window, w_pad: Window) -> None:
        # Get the orientations
        self.ori = self._clip_and_reshape(self.ori, w_pad)
        # Get the distance from edges
        self.edge_dist = self._clip_and_reshape(self.distance, w_pad)

        # Get the edge probabilities
        if self.apply_softmax:
            self.edge_probas = F.softmax(
                self.edge, dim=1, dtype=self.crop_type.dtype
            )[:, 1]
        else:
            self.edge_probas = self.edge[:, 1]
        self.edge_probas = self._clip_and_reshape(self.edge_probas, w_pad)

        # Get the crop probabilities
        if self.apply_softmax:
            self.crop_probas = F.softmax(
                self.crop, dim=1, dtype=self.crop_type.dtype
            )[:, 1]
        else:
            self.crop_probas = self.crop[:, 1]
        self.crop_probas = self._clip_and_reshape(self.crop_probas, w_pad)

        # Get the crop-type probabilities
        self.crop_type_probas = None
        if self.crop_type is not None:
            if self.apply_softmax:
                self.crop_type_probas = F.softmax(
                    self.crop_type, dim=1, dtype=self.crop_type.dtype
                )[:, 1:]
            else:
                self.crop_type_probas = self.crop_type[:, 1:]
            self.crop_type_probas = self._clip_and_reshape(self.crop_type_probas, w_pad)

        # Reshape the window chunk and slice off padding
        i = abs(w.row_off - w_pad.row_off)
        j = abs(w.col_off - w_pad.col_off)
        slicer = (
            slice(i, i+w.height),
            slice(j, j+w.width)
        )
        slicer3d = (
            slice(0, None),
            slice(i, i+w.height),
            slice(j, j+w.width)
        )
        self.ori = self.ori[slicer]
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
        self.ori = (
            np.nan_to_num(
                self.ori,
                nan=-1.0,
                neginf=-1.0,
                posinf=-1.0
            ).astype('float32')
        )

        self.edge_dist = (
            np.nan_to_num(
                self.edge_dist,
                nan=-1.0,
                neginf=-1.0,
                posinf=-1.0
            ).astype('float32')
        )

        self.edge_probas = (
            np.nan_to_num(
                self.edge_probas,
                nan=-1.0,
                neginf=-1.0,
                posinf=-1.0
            ).astype('float32')
        )

        self.crop_probas = (
            np.nan_to_num(
                self.crop_probas,
                nan=-1.0,
                neginf=-1.0,
                posinf=-1.0
            ).astype('float32')
        )

        if self.crop_type_probas is not None:
            self.crop_type_probas = (
                np.nan_to_num(
                    self.crop_type_probas,
                    nan=-1.0,
                    neginf=-1.0,
                    posinf=-1.0
                ).astype('float32')
            )
