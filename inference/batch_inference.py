"""
think_fast/inference/batch_inference.py
==========================================
Pure-PyTorch batched inference for the 5-channel YOLOv11-Nano.

This is the "development path" — used when TensorRT is not available
(e.g., workstation development, CI/CD). For production edge deployment,
use tensorrt_engine.py instead.

Input
-----
    tensor : torch.Tensor — shape [6, 5, 640, 640], dtype float32.

Output
------
    List[CameraDetections] — one per camera view (6 entries).
    Each CameraDetections holds:
        - camera_idx : int
        - boxes      : (N, 4) float32 — [x1, y1, x2, y2] in pixel coords
        - scores     : (N,)   float32 — confidence scores
        - classes    : (N,)   int     — class indices
        - depths     : (N,)   float32 — radar depth (metres) at box centre
        - velocities : (N,)   float32 — radar velocity (m/s) at box centre
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from think_fast.model.yolo11n_5ch import NUSCENES_CLASSES

logger = logging.getLogger(__name__)

# ── Camera channel names (index ↔ name) ──────────────────────
CAMERA_NAMES: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

# ── NMS thresholds ────────────────────────────────────────────
DEFAULT_CONF: float = 0.25
DEFAULT_IOU:  float = 0.45

# ── Radar channel indices in the 5-ch tensor ─────────────────
CH_DEPTH: int = 3   # Depth channel index
CH_VEL:   int = 4   # Velocity channel index

# ── Depth de-normalisation ────────────────────────────────────
MAX_DEPTH_M: float = 80.0
VEL_MEAN:    float = 0.0
VEL_STD:     float = 10.0


# ─────────────────────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class DetectionBox:
    """
    A single detected object with fused radar measurements.

    Attributes
    ----------
    x1, y1, x2, y2 : float — bounding box corners in [0, 640] pixel space.
    score           : float — detection confidence.
    class_id        : int   — class index.
    class_name      : str   — class label.
    depth_m         : float — radar depth at box centre (metres). 0 if no radar hit.
    velocity_ms     : float — radar radial velocity (m/s). Negative = approaching.
    """
    x1:          float
    y1:          float
    x2:          float
    y2:          float
    score:       float
    class_id:    int
    class_name:  str
    depth_m:     float = 0.0
    velocity_ms: float = 0.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class CameraDetections:
    """
    All detections for a single camera view.

    Attributes
    ----------
    camera_idx  : int — index in [0, 5].
    camera_name : str — e.g. "CAM_FRONT".
    boxes       : List[DetectionBox] — detected objects.
    latency_ms  : float — inference latency for this camera (informational).
    """
    camera_idx:  int
    camera_name: str
    boxes:       List[DetectionBox] = field(default_factory=list)
    latency_ms:  float = 0.0

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class BatchInferenceResult:
    """
    Full inference result for one 360° snapshot (all 6 cameras).

    Attributes
    ----------
    cameras         : List[CameraDetections] — one per camera (length 6).
    total_latency_ms: float — wall-clock time for the full batch inference.
    """
    cameras:          List[CameraDetections]
    total_latency_ms: float = 0.0

    def all_boxes(self) -> List[Tuple[int, DetectionBox]]:
        """Return all boxes across cameras as (camera_idx, box) tuples."""
        out = []
        for cam in self.cameras:
            for box in cam.boxes:
                out.append((cam.camera_idx, box))
        return out

    def total_detections(self) -> int:
        return sum(len(c) for c in self.cameras)


# ─────────────────────────────────────────────────────────────
# Radar value extraction from the input tensor
# ─────────────────────────────────────────────────────────────

def _extract_radar_at_box(
    input_tensor: torch.Tensor,
    cam_idx:      int,
    x1: float, y1: float, x2: float, y2: float,
) -> Tuple[float, float]:
    """
    Extract the mean radar depth and velocity within a detected bounding box.

    Samples from the stored depth (ch 3) and velocity (ch 4) channels
    of the input tensor, which carry the fused radar measurements.

    Parameters
    ----------
    input_tensor : [6, 5, 640, 640] float32 tensor (normalised).
    cam_idx      : camera index (0–5).
    x1..y2       : bounding box corners in [0, 640] pixel space.

    Returns
    -------
    depth_m   : float — de-normalised depth in metres.
    velocity  : float — de-standardised velocity in m/s.
    """
    H = W = 640
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(W, int(x2))
    y2i = min(H, int(y2))

    if x2i <= x1i or y2i <= y1i:
        return 0.0, 0.0

    # Extract depth and velocity patches
    depth_patch = input_tensor[cam_idx, CH_DEPTH, y1i:y2i, x1i:x2i]
    vel_patch   = input_tensor[cam_idx, CH_VEL,   y1i:y2i, x1i:x2i]

    # Only use pixels with actual radar hits (depth > 0 in normalised space)
    valid = depth_patch > 0

    if not valid.any():
        return 0.0, 0.0

    depth_norm = depth_patch[valid].mean().item()
    vel_norm   = vel_patch[valid].mean().item()

    # De-normalise
    depth_m  = depth_norm * MAX_DEPTH_M
    velocity = vel_norm * VEL_STD + VEL_MEAN

    return depth_m, velocity


# ─────────────────────────────────────────────────────────────
# BatchInferenceEngine
# ─────────────────────────────────────────────────────────────

class BatchInferenceEngine:
    """
    PyTorch-based batched inference engine for the 5-channel model.

    Accepts the [6, 5, 640, 640] tensor, runs a single batched
    forward pass through the YOLO model, applies NMS per camera,
    then extracts radar depth + velocity at each box centre.

    Parameters
    ----------
    model : ultralytics.YOLO — 5-channel model (loaded & verified).
    conf  : float — confidence threshold.
    iou   : float — IoU threshold for NMS.
    device: str   — "cuda" or "cpu".

    Example
    -------
    >>> engine = BatchInferenceEngine(model, device="cuda")
    >>> result = engine.run(tensor)
    >>> print(result.total_detections())
    """

    def __init__(
        self,
        model:  "ultralytics.YOLO",
        conf:   float = DEFAULT_CONF,
        iou:    float = DEFAULT_IOU,
        device: str   = "cuda",
    ) -> None:
        self.conf   = conf
        self.iou    = iou
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = model
        self.model.model.to(self.device)
        self.model.model.eval()

        logger.info(
            "BatchInferenceEngine ready on %s (conf=%.2f, iou=%.2f)",
            self.device, conf, iou
        )

    @torch.no_grad()
    def run(self, tensor: torch.Tensor) -> BatchInferenceResult:
        """
        Run batched inference on a [6, 5, 640, 640] tensor.

        Parameters
        ----------
        tensor : torch.Tensor — shape [6, 5, 640, 640], float32.

        Returns
        -------
        BatchInferenceResult
        """
        assert tensor.shape == (6, 5, 640, 640), (
            f"Expected [6,5,640,640], got {tuple(tensor.shape)}"
        )

        input_cpu = tensor.float()                       # Keep CPU copy for radar lookup
        tensor_gpu = input_cpu.to(self.device)

        t0 = time.perf_counter()

        # ── Single batched forward pass ────────────────────────────
        # Ultralytics model.predict() handles batches natively
        results = self.model.predict(
            source  = tensor_gpu,
            conf    = self.conf,
            iou     = self.iou,
            verbose = False,
            save    = False,
        )

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # ── Parse per-camera results ───────────────────────────────
        cam_detections: List[CameraDetections] = []

        for cam_idx, result in enumerate(results):
            cam_name = CAMERA_NAMES[cam_idx]
            boxes:    List[DetectionBox] = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy    = result.boxes.xyxy.cpu().numpy()     # (N, 4)
                scores  = result.boxes.conf.cpu().numpy()     # (N,)
                classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

                for i in range(len(scores)):
                    x1, y1, x2, y2 = xyxy[i]
                    depth_m, vel = _extract_radar_at_box(
                        input_cpu, cam_idx, x1, y1, x2, y2
                    )
                    cls_id   = int(classes[i])
                    cls_name = (
                        NUSCENES_CLASSES[cls_id]
                        if cls_id < len(NUSCENES_CLASSES)
                        else f"cls_{cls_id}"
                    )
                    boxes.append(DetectionBox(
                        x1          = float(x1),
                        y1          = float(y1),
                        x2          = float(x2),
                        y2          = float(y2),
                        score       = float(scores[i]),
                        class_id    = cls_id,
                        class_name  = cls_name,
                        depth_m     = depth_m,
                        velocity_ms = vel,
                    ))

            cam_detections.append(CameraDetections(
                camera_idx  = cam_idx,
                camera_name = cam_name,
                boxes       = boxes,
                latency_ms  = total_ms / 6.0,
            ))

        return BatchInferenceResult(
            cameras          = cam_detections,
            total_latency_ms = total_ms,
        )
