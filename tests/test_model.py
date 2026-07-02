"""
tests/test_model.py
=======================
Tests for the 5-channel YOLOv11-Nano model construction and weight transfer.
"""

import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO

from think_fast.model.yolo11n_5ch import (
    build_model_from_scratch,
    _get_first_conv,
)


def test_build_model_from_scratch():
    model = build_model_from_scratch(num_classes=10, verbose=False)
    assert isinstance(model, YOLO)

    first_conv = _get_first_conv(model)
    assert first_conv.in_channels == 5
    assert first_conv.out_channels == 16

    # Test dummy forward pass
    device = next(model.model.parameters()).device
    dummy_input = torch.randn(1, 5, 640, 640, device=device)
    
    with torch.no_grad():
        preds = model.model(dummy_input)
    
    assert len(preds) > 0
