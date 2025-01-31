# Copyright (c) OpenMMLab. All rights reserved.
from codes.registry import LOSSES


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)