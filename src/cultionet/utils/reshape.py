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
    distance: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    edge: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    crop: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    crop_r: torch.Tensor = attr.ib(validator=attr.validators.instance_of(torch.Tensor))
    apply_softmax: T.Optional[bool] = attr.ib(default=False, validator=attr.validators.instance_of(bool))

    def stack_outputs(self, w: Window, w_pad: Window) -> np.ndarray:
        self.reshape(w, w_pad)
        self.nan_to_num()

        return np.stack((
            self.edge_dist, self.edge_probas, self.crop_probas, self.crop_probas_r
        ))

    @staticmethod
    def _clip_and_reshape(tarray: torch.Tensor, window_obj: Window) -> np.ndarray:
        return (tarray
                .contiguous()
                .view(-1)
                .detach().cpu().numpy()
                .clip(0, 1)
                .reshape(window_obj.height, window_obj.width))

    def reshape(self, w: Window, w_pad: Window) -> None:
        # Get the distance from edges (1 = 0.1, 2 = 0.5, 3 = 0.9 quantiles)
        self.edge_dist = self._clip_and_reshape(self.distance[:, 1], w_pad)

        # Get the edge probabilities
        if self.apply_softmax:
            self.edge_probas = F.softmax(self.edge, dim=1)[:, 1]
        else:
            self.edge_probas = self.edge[:, 1]
        self.edge_probas = self._clip_and_reshape(self.edge_probas, w_pad)

        # Get the crop probabilities
        if self.apply_softmax:
            self.crop_probas = F.softmax(self.crop, dim=1)[:, 1]
        else:
            self.crop_probas = self.crop[:, 1]
        self.crop_probas = self._clip_and_reshape(self.crop_probas, w_pad)

        if self.apply_softmax:
            self.crop_probas_r = F.softmax(self.crop_r, dim=1)[:, 1]
        else:
            self.crop_probas_r = self.crop_r[:, 1]
        self.crop_probas_r = self._clip_and_reshape(self.crop_probas_r, w_pad)

        # Reshape the window chunk and slice off padding
        i = abs(w.row_off - w_pad.row_off)
        j = abs(w.col_off - w_pad.col_off)
        self.edge_dist = self.edge_dist[i:i+w.height, j:j+w.width]
        self.edge_probas = self.edge_probas[i:i+w.height, j:j+w.width]
        self.crop_probas = self.crop_probas[i:i+w.height, j:j+w.width]
        self.crop_probas_r = self.crop_probas_r[i:i+w.height, j:j+w.width]

    def nan_to_num(self):
        # Convert the data type to integer and set 'no data' values
        self.edge_dist = (np.nan_to_num(self.edge_dist,
                                        nan=-1.0,
                                        neginf=-1.0,
                                        posinf=-1.0)
                          .astype('float32'))

        self.edge_probas = (np.nan_to_num(self.edge_probas,
                                          nan=-1.0,
                                          neginf=-1.0,
                                          posinf=-1.0)
                            .astype('float32'))

        self.crop_probas = (np.nan_to_num(self.crop_probas,
                                          nan=-1.0,
                                          neginf=-1.0,
                                          posinf=-1.0)
                            .astype('float32'))

        self.crop_probas_r = (np.nan_to_num(self.crop_probas_r,
                                            nan=-1.0,
                                            neginf=-1.0,
                                            posinf=-1.0)
                            .astype('float32'))
