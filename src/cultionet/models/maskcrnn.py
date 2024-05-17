import typing as T

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as VF
from scipy.ndimage.measurements import label as nd_label
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask


def pad_label_and_resize(
    x: torch.Tensor, y: T.Optional[torch.Tensor] = None
) -> tuple:
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
        super(BFasterRCNN, self).__init__()

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
