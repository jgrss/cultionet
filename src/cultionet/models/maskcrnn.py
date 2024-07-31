import typing as T

import einops
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as VF
from affine import Affine
from rasterio import features
from scipy.ndimage.measurements import label as nd_label
from shapely.geometry import shape
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import masks_to_boxes, nms
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask


def reshape_and_resize(
    x: torch.Tensor,
    dimensions: str,
    size: int,
    mode: str,
) -> torch.Tensor:
    x = einops.rearrange(x, dimensions)
    x = F.interpolate(x, size=size, mode=mode)

    return x


class ReshapeMaskData:
    def __init__(
        self, x: torch.Tensor, masks: T.Optional[torch.Tensor] = None
    ):
        self.x = x
        self.masks = masks

    @classmethod
    def prepare(
        cls,
        x: torch.Tensor,
        size: int = 256,
        y: T.Optional[torch.Tensor] = None,
    ) -> "ReshapeMaskData":
        """Pads and resizes."""

        x = reshape_and_resize(
            x=F.pad(x, pad=(1, 1, 1, 1)),
            dimensions='c h w -> 1 c h w',
            size=size,
            mode='bilinear',
        )

        if y is None:
            return cls(x=x)

        # Label segments
        y = F.pad(y, pad=(1, 1, 1, 1)).detach().cpu().numpy()
        labels = nd_label(y == 1)[0]
        labels = torch.from_numpy(labels).to(
            dtype=torch.uint8, device=y.device
        )

        labels = reshape_and_resize(
            x=labels,
            dimensions='h w -> 1 1 h w',
            size=size,
            mode='nearest',
        )

        return cls(x=x, masks=labels.long())


def pad_label_and_resize(
    x: torch.Tensor, y: T.Optional[torch.Tensor] = None
) -> tuple:
    """Pads and resizes."""
    x = F.pad(x, pad=(1, 1, 1, 1))
    x = einops.rearrange(x, 'c h w -> 1 c h w')
    x = F.interpolate(x, size=256, mode='bilinear')

    if y is None:
        return x, None

    y = F.pad(y, pad=(1, 1, 1, 1))
    labels = nd_label(y.detach().cpu().numpy() == 1)[0]
    labels = torch.from_numpy(labels).to(dtype=torch.long, device=y.device)

    labels = einops.rearrange(labels, 'h w -> 1 1 h w')
    labels = F.interpolate(
        labels.to(dtype=torch.uint8), size=256, mode='nearest'
    ).long()

    return x, labels


def mask_2d_to_3d(labels: torch.Tensor) -> torch.Tensor:
    """Converts 2d masks to 3d."""

    unique_labels = labels.unique()
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]
    num_labels = len(unique_labels)

    if num_labels == 0:
        masks = torch.zeros(1, *labels.shape[-2:]).to(
            dtype=torch.long, device=labels.device
        )
    else:
        masks = torch.ones(num_labels, *labels.shape[-2:]).to(
            dtype=torch.long, device=labels.device
        )
        for idx, label in enumerate(unique_labels):
            masks[idx] *= (labels == label).long()[0, 0]

    return masks


def create_mask_targets(x: torch.Tensor, masks: torch.Tensor) -> tuple:
    """Creates targets for Mask-RCNN."""

    if masks.max() == 0:
        bboxes = torch.tensor([[0, 0, 0, 0]])
    else:
        bboxes = masks_to_boxes(masks)

    bboxes = BoundingBoxes(
        data=bboxes.to(dtype=masks.dtype, device=x.device),
        format=BoundingBoxFormat.XYXY,
        canvas_size=VF.get_size(masks),
    )

    targets = {
        "masks": Mask(masks),
        "boxes": bboxes,
        # NOTE: these are the labels for each mask (i.e., all masks are 1)
        "labels": torch.ones(
            bboxes.shape[0], dtype=torch.int64, device=x.device
        ),
    }

    box_sanitizer = v2.SanitizeBoundingBoxes()
    x, targets = box_sanitizer(x, targets)

    return x, targets


def nms_masks(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    masks: torch.Tensor,
    iou_threshold: float,
    size: tuple,
) -> torch.Tensor:
    """Get non maximum suppression scores."""

    nms_idx = nms(
        boxes=boxes,
        scores=scores,
        iou_threshold=iou_threshold,
    )
    # Get the scores and resize
    pred_masks = F.interpolate(
        masks[scores[nms_idx] > iou_threshold],
        size=size,
        mode='bilinear',
    )

    return pred_masks


def mask_to_polygon(
    masks: torch.Tensor,
    image_left: float,
    image_top: float,
    row_off: int,
    col_off: int,
    resolution: float,
    padding: int,
) -> gpd.GeoDataFrame:
    # Set the window transform
    window_transform = Affine(
        resolution,
        0.0,
        image_left + col_off,
        0.0,
        -resolution,
        image_top - row_off,
    )
    geometry = []
    for mask_layer in (
        masks.squeeze(dim=1)[..., padding:-padding, padding:-padding]
        .detach()
        .cpu()
        .numpy()
    ):
        # Get the polygon for every box mask
        shapes = features.shapes(
            (mask_layer > 0).astype('uint8'),
            mask=(mask_layer > 0).astype('uint8'),
            transform=window_transform,
        )
        layer_geometry = [shape(polygon) for polygon, value in shapes]
        geometry.extend(layer_geometry)

    df = gpd.GeoDataFrame(geometry=geometry)

    return df


class BFasterRCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        sizes: T.Optional[T.Sequence[int]] = None,
        aspect_ratios: T.Optional[T.Sequence[int]] = None,
        trainable_backbone_layers: T.Optional[int] = 3,
        min_image_size: int = 800,
        max_image_size: int = 1333,
    ) -> None:
        super().__init__()

        if sizes is None:
            sizes = (32, 64, 128, 256, 512)
        if not isinstance(sizes, tuple):
            try:
                sizes = tuple(sizes)
            except TypeError as e:
                raise TypeError(e)

        if aspect_ratios is None:
            aspect_ratios = (0.5, 1.0, 2.0)
        if not isinstance(aspect_ratios, tuple):
            try:
                aspect_ratios = tuple(aspect_ratios)
            except TypeError as e:
                raise TypeError(e)

        # Load a pretrained model
        self.model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            trainable_backbone_layers=trainable_backbone_layers,
        )

        # Remove image normalization and add custom resizing
        self.model.transform = GeneralizedRCNNTransform(
            image_mean=(0.0,) * in_channels,
            image_std=(1.0,) * in_channels,
            min_size=min_image_size,
            max_size=max_image_size,
        )
        # Replace the first convolution
        # out_channels = self.model.backbone.body.conv1.out_channels
        # self.model.backbone.body.conv1 = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     bias=False,
        # )
        # self.model.rpn.anchor_generator = AnchorGenerator(
        #     sizes=tuple((size,) for size in sizes),
        #     aspect_ratios=(aspect_ratios,) * len(sizes),
        # )

        # Update the output classes in the predictor heads

        # Fast RCNN predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # Mask RCNN predictor
        in_features_mask = (
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        )
        out_channels = (
            self.model.roi_heads.mask_predictor.conv5_mask.out_channels
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, out_channels, num_classes
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor, y: T.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x, y)
