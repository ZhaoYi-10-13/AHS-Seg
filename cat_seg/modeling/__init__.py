# Copyright (c) Facebook, Inc. and its affiliates.
from .backbone.swin import D2SwinTransformer
from .heads.cat_seg_head import CATSegHead

# Enhanced modules for ECCV 2026
from .multi_scale_fusion import (
    MultiScaleHypersphericalFPN,
    BoundaryAwareLoss,
    MultiScaleBoundaryLoss
)
from .class_aware_projection import (
    DynamicClassAwareProjection,
    ClassContrastiveLoss,
    ClassAwareLossModule
)