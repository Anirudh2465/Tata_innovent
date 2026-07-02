"""
think_fast/model/yolo11n_5ch.py
=====================================
Builder and utilities for the 5-channel YOLOv11-Nano model.

Key responsibilities
--------------------
1. Build the model from the custom YAML (`yolo11n_5ch.yaml`).
2. Optionally transfer pre-trained RGB weights from a standard
   `yolo11n.pt` checkpoint — maximising the value of existing
   ImageNet/COCO pre-training while correctly initialising the
   two new channels (Depth & Velocity).
3. Verify the first conv layer has in_channels == 5.
4. Provide a convenience `forward_batch` function that accepts
   the [6, 5, 640, 640] tensor and returns per-camera detections.

Weight Transfer Strategy
------------------------
Standard YOLOv11n was trained on 3-channel RGB images.
Its first conv kernel has shape (out_ch, 3, kH, kW).

When we expand to 5 channels we:
  - Copy the 3 RGB channel weights directly (channels 0, 1, 2).
  - Initialise channels 3 (Depth) and 4 (Velocity) via
    Kaiming-uniform initialisation, as if they were newly added.

This lets the backbone immediately exploit its learned feature
detectors while the new channels train from scratch.

Alternatively, the Depth channel can be initialised as the mean
of the RGB channels (grayscale assumption), which gives a warm
start when depth is coarsely correlated with image intensity.

Usage
-----
    # Option A: from scratch
    model = build_model_from_scratch(num_classes=10)

    # Option B: with pre-trained weight transfer
    model = build_model_with_weight_transfer(
        pretrained_pt="yolo11n.pt",
        num_classes=10,
    )

    # Verify
    verify_model(model)

    # Batch forward pass
    tensor = torch.randn(6, 5, 640, 640)
    results = forward_batch(model, tensor)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Path to the YAML config (sibling file) ───────────────────
_YAML_PATH = Path(__file__).parent / "yolo11n_5ch.yaml"

# ── nuScenes class definitions for the Detect head ───────────
NUSCENES_CLASSES: List[str] = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "barrier",
    "traffic_cone",
]


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _get_first_conv(model: nn.Module) -> nn.Conv2d:
    """
    Return the first Conv2d layer in the model's backbone.
    In Ultralytics YOLO, the backbone is model.model.model[0].conv.
    """
    try:
        return model.model.model[0].conv
    except AttributeError:
        # Fallback: iterate to find first Conv2d
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                return m
    raise RuntimeError("Could not locate the first Conv2d in the model.")


def _kaiming_init_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Apply Kaiming-uniform initialisation to a weight tensor."""
    fan_in = tensor.size(1) * tensor.size(2) * tensor.size(3)
    std    = math.sqrt(2.0 / fan_in)
    bound  = math.sqrt(3.0) * std
    return tensor.uniform_(-bound, bound)


# ─────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────

def build_model_from_scratch(
    num_classes:  int  = 10,
    yaml_path:    str  = str(_YAML_PATH),
    verbose:      bool = True,
) -> "ultralytics.YOLO":
    """
    Initialise a fresh YOLOv11-Nano model with 5 input channels.
    All weights are randomly initialised — no pre-training used.

    Parameters
    ----------
    num_classes : int — number of detection classes.
    yaml_path   : str — path to the 5-channel YAML config.
    verbose     : bool — print model summary.

    Returns
    -------
    model : ultralytics.YOLO instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    logger.info("Building 5-channel YOLOv11-Nano from scratch …")
    model = YOLO(yaml_path)

    # Sanity-check
    first_conv = _get_first_conv(model)
    assert first_conv.in_channels == 5, (
        f"Expected in_channels=5, got {first_conv.in_channels}. "
        "Check that `ch: 5` is set in the YAML."
    )

    # Override nc if needed
    if hasattr(model.model, 'nc') and model.model.nc != num_classes:
        logger.warning(
            "YAML nc=%d ≠ requested num_classes=%d. "
            "Update the YAML `nc:` field to match your dataset.",
            model.model.nc, num_classes
        )

    if verbose:
        model.info()

    return model


def build_model_with_weight_transfer(
    pretrained_pt: str,
    num_classes:   int  = 10,
    yaml_path:     str  = str(_YAML_PATH),
    depth_init:    str  = "grayscale",
    verbose:       bool = True,
) -> "ultralytics.YOLO":
    """
    Build a 5-channel YOLOv11-Nano and transfer pre-trained RGB weights.

    Weight transfer steps
    ---------------------
    1. Load the 5-channel model from YAML (random init).
    2. Load the standard 3-channel YOLOv11n checkpoint.
    3. Copy all matching parameter tensors.
    4. For the first conv kernel (in_channels 3→5):
         - Copy channels 0, 1, 2  from the pre-trained weights.
         - Initialise channel 3 (Depth) per `depth_init` strategy:
             "grayscale" → mean of RGB channels (warm start).
             "kaiming"   → Kaiming-uniform random.
             "zero"      → zeros.
         - Initialise channel 4 (Velocity) with Kaiming-uniform.
    5. Load the merged state dict into the 5-channel model.

    Parameters
    ----------
    pretrained_pt : str — path to a standard yolo11n.pt checkpoint.
    num_classes   : int — detection classes.
    yaml_path     : str — path to 5-channel YAML.
    depth_init    : str — strategy for the Depth channel init:
                    "grayscale" | "kaiming" | "zero"
    verbose       : bool

    Returns
    -------
    model : ultralytics.YOLO with transferred weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    if not os.path.exists(pretrained_pt):
        raise FileNotFoundError(f"Pre-trained weights not found: {pretrained_pt}")

    # ── Step 1: Build fresh 5-channel model ──────────────────────
    logger.info("Initialising 5-channel model from YAML …")
    model_5ch = YOLO(yaml_path)

    # ── Step 2: Load 3-channel pre-trained model ──────────────────
    logger.info("Loading pre-trained weights from %s …", pretrained_pt)
    model_3ch = YOLO(pretrained_pt)

    state_5ch = model_5ch.model.state_dict()
    state_3ch = model_3ch.model.state_dict()

    # ── Step 3: Transfer all matching parameters ───────────────────
    transferred = 0
    skipped     = []

    for key, param_5ch in state_5ch.items():
        if key not in state_3ch:
            skipped.append(key)
            continue

        param_3ch = state_3ch[key]

        # ── Step 4: Special handling for the first conv kernel ─────
        #   Key format: "model.0.conv.weight"
        #   Shape 5ch: (out_ch, 5, kH, kW)
        #   Shape 3ch: (out_ch, 3, kH, kW)
        if param_5ch.shape != param_3ch.shape:
            if "0.conv.weight" in key:
                logger.info(
                    "Transferring first conv: %s → %s with depth_init='%s'",
                    tuple(param_3ch.shape),
                    tuple(param_5ch.shape),
                    depth_init,
                )
                new_weight = _merge_first_conv(
                    param_3ch, param_5ch, depth_init
                )
                state_5ch[key] = new_weight
                transferred += 1
            else:
                logger.warning(
                    "Shape mismatch for key '%s': %s vs %s — skipping.",
                    key, tuple(param_5ch.shape), tuple(param_3ch.shape)
                )
                skipped.append(key)
            continue

        state_5ch[key] = param_3ch.clone()
        transferred += 1

    model_5ch.model.load_state_dict(state_5ch, strict=False)

    logger.info(
        "Weight transfer complete: %d transferred, %d skipped.",
        transferred, len(skipped)
    )
    if verbose and skipped:
        logger.debug("Skipped keys: %s", skipped[:10])

    verify_model(model_5ch)
    if verbose:
        model_5ch.info()

    return model_5ch


def _merge_first_conv(
    param_3ch:  torch.Tensor,
    param_5ch:  torch.Tensor,
    depth_init: str,
) -> torch.Tensor:
    """
    Build the 5-channel first conv weight by expanding a 3-channel
    pre-trained weight.

    Parameters
    ----------
    param_3ch  : (out_ch, 3, kH, kW) — pre-trained RGB weights.
    param_5ch  : (out_ch, 5, kH, kW) — target shape (random init).
    depth_init : str — strategy for channel index 3 (Depth).

    Returns
    -------
    merged : (out_ch, 5, kH, kW) float32 tensor.
    """
    merged = param_5ch.clone()

    # Channels 0-2: copy RGB weights directly
    merged[:, :3, :, :] = param_3ch.clone()

    # Channel 3: Depth
    if depth_init == "grayscale":
        # Warm start: mean of R, G, B weights (intensity assumption)
        merged[:, 3, :, :] = param_3ch.mean(dim=1)
    elif depth_init == "kaiming":
        _kaiming_init_2d(merged[:, 3:4, :, :])
    elif depth_init == "zero":
        merged[:, 3, :, :] = 0.0
    else:
        raise ValueError(f"Unknown depth_init='{depth_init}'. Use 'grayscale', 'kaiming', or 'zero'.")

    # Channel 4: Velocity — always Kaiming (no reasonable warm-start)
    _kaiming_init_2d(merged[:, 4:5, :, :])

    return merged


# ─────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────

def verify_model(model: "ultralytics.YOLO") -> None:
    """
    Assert critical structural properties of the 5-channel model.

    Raises
    ------
    AssertionError if:
        - First conv in_channels ≠ 5
        - Model is not in a valid state for forward pass
    """
    first_conv = _get_first_conv(model)

    assert first_conv.in_channels == 5, (
        f"First conv in_channels={first_conv.in_channels}, expected 5."
    )
    logger.info(
        "✓ Model verified: first conv in_channels=%d, out_channels=%d",
        first_conv.in_channels, first_conv.out_channels,
    )

    # Run a dummy forward pass to catch shape errors early
    device = next(model.model.parameters()).device
    dummy  = torch.zeros(1, 5, 640, 640, device=device)
    with torch.no_grad():
        try:
            _ = model.model(dummy)
            logger.info("✓ Dummy forward pass on [1,5,640,640] succeeded.")
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed: {e}") from e


# ─────────────────────────────────────────────────────────────
# Batched forward pass
# ─────────────────────────────────────────────────────────────

def forward_batch(
    model:  "ultralytics.YOLO",
    tensor: torch.Tensor,
    conf:   float = 0.25,
    iou:    float = 0.45,
) -> List:
    """
    Run batched inference on the [6, 5, 640, 640] early-fusion tensor.

    Parameters
    ----------
    model  : ultralytics.YOLO — 5-channel model.
    tensor : torch.Tensor — shape [6, 5, 640, 640].
    conf   : float — confidence threshold for NMS.
    iou    : float — IoU threshold for NMS.

    Returns
    -------
    results : List[ultralytics.engine.results.Results]
              One Results object per camera view (6 items).
    """
    assert tensor.shape == (6, 5, 640, 640), (
        f"Expected tensor shape [6, 5, 640, 640], got {tuple(tensor.shape)}"
    )

    # Ultralytics predict() handles NMS internally when called with a raw tensor
    results = model.predict(
        source    = tensor,
        conf      = conf,
        iou       = iou,
        verbose   = False,
        save      = False,
    )

    return results
