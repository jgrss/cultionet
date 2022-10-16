"""
Backbone source:
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py
"""
import typing as T

import torch
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2


class BFasterRCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        sizes: T.Optional[T.Sequence[int]] = None,
        aspect_ratios: T.Optional[T.Sequence[int]] = None,
        min_image_size: int = 800,
        max_image_size: int = 1333
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
            weights='DEFAULT',
            trainable_backbone_layers=3
        )
        # Remove image normalization and add custom resizing
        self.model.transform = GeneralizedRCNNTransform(
            image_mean=(0.0,) * in_channels,
            image_std=(1.0,) * in_channels,
            min_size=min_image_size,
            max_size=max_image_size
        )
        # Replace the first convolution
        out_channels = self.model.backbone.body.conv1.out_channels
        self.model.backbone.body.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.rpn.anchor_generator = AnchorGenerator(
            sizes=tuple((size,) for size in sizes),
            aspect_ratios=(aspect_ratios,) * len(sizes)
        )
        # Update the output classes in the predictor heads
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            out_channels,
            num_classes
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        y: T.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x, y)
