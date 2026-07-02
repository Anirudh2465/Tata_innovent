"""
think_fast/data/channel_stacker.py
=====================================
Assembles the 5-channel early-fusion tensor for each camera view and
stacks all 6 camera views into the final batched tensor.

Output Shape
------------
    [6, 5, 640, 640]   (PyTorch float32)
    │   │   │    │
    │   │   └────┴── Spatial resolution (letterbox-padded to 640×640)
    │   └──────────── 5 channels: R, G, B, RadarDepth, RadarVelocity
    └──────────────── 6 camera views

Channel Definitions
-------------------
    Ch 0  R   — Red   (normalised to [0,1])
    Ch 1  G   — Green (normalised to [0,1])
    Ch 2  B   — Blue  (normalised to [0,1])
    Ch 3  D   — Radar depth  (normalised with max_depth=80 m → [0,1])
    Ch 4  V   — Radar radial velocity (standardised: µ=0, σ≈10 m/s → ≈[-4,4])

Letterboxing
------------
All camera images in nuScenes are 900×1600 (H×W). Letterboxing
resizes the longer edge to 640 and pads the shorter edge with zeros
so that the aspect ratio is preserved and no stretching occurs.

CAMERA_CHANNELS ordering (index → camera name):
    0  CAM_FRONT
    1  CAM_FRONT_RIGHT
    2  CAM_BACK_RIGHT
    3  CAM_BACK
    4  CAM_BACK_LEFT
    5  CAM_FRONT_LEFT
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from think_fast.data.nuscenes_dataset import (
    CAMERA_CHANNELS,
    CameraData,
    NuScenesSample,
)
from think_fast.data.radar_projector import (
    ProjectedRadarMaps,
    RadarProjector,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

TARGET_SIZE:   int   = 640      # Output resolution (square)
MAX_DEPTH_M:   float = 80.0     # Maximum radar range in metres
VEL_MEAN:      float = 0.0      # Velocity normalisation mean (m/s)
VEL_STD:       float = 10.0     # Velocity normalisation std  (m/s)


# ─────────────────────────────────────────────────────────────
# Letterbox utility
# ─────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target: int = TARGET_SIZE,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize `image` so its longest edge equals `target`, padding the
    shorter edge with `pad_value` to produce a (target × target) image.

    Parameters
    ----------
    image     : (H, W) or (H, W, C) np.ndarray.
    target    : int — output resolution (both height and width).
    pad_value : float — fill value for padding regions.

    Returns
    -------
    img_lb    : (target, target[, C]) np.ndarray — letterboxed image.
    scale     : float — the uniform scale factor applied.
    pad       : (pad_top, pad_left) — padding offsets in pixels.
    """
    h, w     = image.shape[:2]
    scale    = target / max(h, w)
    new_h    = int(round(h * scale))
    new_w    = int(round(w * scale))

    resized  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding to centre the image
    pad_top  = (target - new_h) // 2
    pad_left = (target - new_w) // 2

    if resized.ndim == 2:
        canvas = np.full((target, target), fill_value=pad_value, dtype=resized.dtype)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    else:
        C = resized.shape[2]
        canvas = np.full((target, target, C), fill_value=pad_value, dtype=resized.dtype)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, (pad_top, pad_left)


# ─────────────────────────────────────────────────────────────
# ChannelStacker
# ─────────────────────────────────────────────────────────────

class ChannelStacker:
    """
    Converts a NuScenesSample into the batched 5-channel tensor
    `[6, 5, TARGET_SIZE, TARGET_SIZE]` ready for YOLOv11 inference.

    Parameters
    ----------
    target_size   : int — spatial resolution of the output tensor.
    max_depth_m   : float — max radar depth for [0,1] normalisation.
    vel_mean      : float — mean for velocity standardisation.
    vel_std       : float — std  for velocity standardisation.
    splat_radius  : int — disk radius for radar splatting.

    Example
    -------
    >>> stacker = ChannelStacker()
    >>> tensor, meta = stacker.build_batch_tensor(sample)
    >>> print(tensor.shape)     # torch.Size([6, 5, 640, 640])
    >>> print(tensor.dtype)     # torch.float32
    """

    def __init__(
        self,
        target_size:  int   = TARGET_SIZE,
        max_depth_m:  float = MAX_DEPTH_M,
        vel_mean:     float = VEL_MEAN,
        vel_std:      float = VEL_STD,
        splat_radius: int   = 5,
    ) -> None:
        self.target_size  = target_size
        self.max_depth_m  = max_depth_m
        self.vel_mean     = vel_mean
        self.vel_std      = vel_std
        self.projector    = RadarProjector(splat_radius=splat_radius)

    # ── Public API ──────────────────────────────────────────────

    def build_batch_tensor(
        self,
        sample: NuScenesSample,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Build the final [6, 5, H, W] batched tensor from a NuScenesSample.

        All 5 radar sensors are fused onto every camera view.
        (Points outside a camera's FOV are automatically discarded
        during the projection step.)

        Parameters
        ----------
        sample : NuScenesSample

        Returns
        -------
        tensor : torch.Tensor — shape [6, 5, 640, 640], dtype float32.
        meta   : dict — per-camera letterbox scale and padding info,
                 useful for bounding-box coordinate rescaling.
        """
        all_radars = list(sample.radars.values())
        batch_list: List[np.ndarray] = []
        meta: Dict = {}

        for cam_name in CAMERA_CHANNELS:
            cam_data = sample.cameras.get(cam_name)
            if cam_data is None:
                # If a camera is missing, fill with zeros
                logger.warning("Camera %s missing — using zero tensor.", cam_name)
                batch_list.append(np.zeros((5, self.target_size, self.target_size), dtype=np.float32))
                meta[cam_name] = {"scale": 1.0, "pad": (0, 0)}
                continue

            channel_tensor, cam_meta = self._build_camera_tensor(cam_data, all_radars)
            batch_list.append(channel_tensor)
            meta[cam_name] = cam_meta

        # Stack to [6, 5, H, W]
        batch_np = np.stack(batch_list, axis=0)   # (6, 5, H, W)
        tensor   = torch.from_numpy(batch_np)

        return tensor, meta

    def build_single_camera_tensor(
        self,
        cam_data:   CameraData,
        all_radars: List,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Build a single [5, H, W] tensor for one camera view.
        Useful for debugging individual camera-radar fusion.
        """
        ch, meta = self._build_camera_tensor(cam_data, all_radars)
        return torch.from_numpy(ch), meta

    # ── Private Methods ─────────────────────────────────────────

    def _build_camera_tensor(
        self,
        cam_data:   CameraData,
        all_radars: List,
    ) -> Tuple[np.ndarray, Dict]:
        """
        For one camera:
          1. Project all radars → depth_map, vel_map (camera resolution).
          2. Letterbox the RGB image and both radar maps.
          3. Normalise each channel.
          4. Stack to (5, H, W) float32 numpy array.

        Returns
        -------
        arr  : (5, TARGET, TARGET) float32 numpy array.
        meta : dict with 'scale' and 'pad' for coord rescaling.
        """
        H_orig, W_orig = cam_data.image_bgr.shape[:2]

        # ── 1. Project all radars onto this camera ─────────────────
        maps: ProjectedRadarMaps = self.projector.project(all_radars, cam_data)

        # ── 2. Letterbox RGB ───────────────────────────────────────
        rgb_lb, scale, pad = letterbox(cam_data.image_bgr, self.target_size, pad_value=0)
        # rgb_lb is (H_lb, W_lb, 3) BGR  →  convert to RGB float [0,1]
        rgb_f  = (rgb_lb[..., ::-1].astype(np.float32) / 255.0)  # (H, W, 3) RGB

        # ── 3. Letterbox depth and velocity maps ───────────────────
        depth_lb, _, _ = letterbox(maps.depth_map, self.target_size, pad_value=0.0)
        vel_lb, _, _   = letterbox(maps.vel_map,   self.target_size, pad_value=0.0)

        # ── 4. Normalise depth [0, 1] ──────────────────────────────
        depth_norm = np.clip(depth_lb / self.max_depth_m, 0.0, 1.0)

        # ── 5. Standardise velocity (zero-mean, unit-std) ─────────
        vel_norm = (vel_lb - self.vel_mean) / (self.vel_std + 1e-8)

        # ── 6. Stack to (5, H, W) — channels: R, G, B, D, V ───────
        r = rgb_f[:, :, 0]                             # (H, W)
        g = rgb_f[:, :, 1]
        b = rgb_f[:, :, 2]
        arr = np.stack([r, g, b, depth_norm, vel_norm], axis=0)  # (5, H, W)
        arr = arr.astype(np.float32)

        meta = {
            "scale": scale,
            "pad":   pad,
            "orig_hw": (H_orig, W_orig),
        }

        return arr, meta


# ─────────────────────────────────────────────────────────────
# Quick-test utility
# ─────────────────────────────────────────────────────────────

def stack_sample(sample: NuScenesSample) -> torch.Tensor:
    """
    Convenience one-liner: convert a NuScenesSample to a [6,5,640,640] tensor.

    Example
    -------
    >>> tensor = stack_sample(sample)
    >>> assert tensor.shape == (6, 5, 640, 640)
    """
    stacker = ChannelStacker()
    tensor, _ = stacker.build_batch_tensor(sample)
    return tensor
