"""
think_fast/threat/threat_matrix.py
=====================================
Threat Matrix Evaluator — Python implementation.

Iterates over all detections from all 6 camera views and computes
Time-to-Collision (TTC) for each object. Objects that cross safety
thresholds generate ThreatEvents that are dispatched to:
  A. The Physical Reflex actuator (hard brake / steer)
  B. The MCP Semantic Dispatcher (async VLA handoff)

TTC Calculation
---------------
Two formulas are used depending on available data:

1. Simple TTC (constant velocity):
       TTC = distance / |relative_velocity|

2. Enhanced TTC (with acceleration, kinematically accurate):
       ETTC = [-Vr - sqrt(Vr² - 2·Ar·D)] / Ar
   where Vr = relative velocity, Ar = relative acceleration, D = distance.

Threat Levels
-------------
    SAFE      : TTC > 3.0 s
    WARNING   : 2.0 ≤ TTC < 3.0 s  → dashboard alert
    PRE_FILL  : 1.5 ≤ TTC < 2.0 s  → brake pre-fill
    PARTIAL   : 1.0 ≤ TTC < 1.5 s  → partial braking
    EMERGENCY : TTC < 1.0 s         → full emergency brake
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from think_fast.inference.batch_inference import (
    BatchInferenceResult,
    CameraDetections,
    DetectionBox,
    CAMERA_NAMES,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Threat Level Enum
# ─────────────────────────────────────────────────────────────

class ThreatLevel(IntEnum):
    """
    Ordered severity levels for detected collision threats.
    Higher integer = higher severity.
    """
    SAFE       = 0   # TTC > 3.0 s — no action
    WARNING    = 1   # 2.0 – 3.0 s — dashboard warning
    PRE_FILL   = 2   # 1.5 – 2.0 s — brake pre-fill
    PARTIAL    = 3   # 1.0 – 1.5 s — partial braking
    EMERGENCY  = 4   # < 1.0 s     — full emergency brake

    def __str__(self) -> str:
        return self.name


# ── TTC thresholds (seconds) ──────────────────────────────────
TTC_THRESHOLDS: Dict[ThreatLevel, float] = {
    ThreatLevel.EMERGENCY: 1.0,
    ThreatLevel.PARTIAL:   1.5,
    ThreatLevel.PRE_FILL:  2.0,
    ThreatLevel.WARNING:   3.0,
    ThreatLevel.SAFE:      float("inf"),
}


def ttc_to_level(ttc: float) -> ThreatLevel:
    """Map a TTC value (seconds) to its ThreatLevel."""
    if ttc < TTC_THRESHOLDS[ThreatLevel.EMERGENCY]:
        return ThreatLevel.EMERGENCY
    elif ttc < TTC_THRESHOLDS[ThreatLevel.PARTIAL]:
        return ThreatLevel.PARTIAL
    elif ttc < TTC_THRESHOLDS[ThreatLevel.PRE_FILL]:
        return ThreatLevel.PRE_FILL
    elif ttc < TTC_THRESHOLDS[ThreatLevel.WARNING]:
        return ThreatLevel.WARNING
    else:
        return ThreatLevel.SAFE


# ─────────────────────────────────────────────────────────────
# ThreatEvent
# ─────────────────────────────────────────────────────────────

@dataclass
class ThreatEvent:
    """
    A single collision threat detected in one camera view.

    Attributes
    ----------
    camera_idx    : int — camera index (0–5).
    camera_name   : str — e.g. "CAM_FRONT".
    box           : DetectionBox — the detected object.
    ttc_s         : float — Time-to-Collision in seconds.
    threat_level  : ThreatLevel — severity classification.
    distance_m    : float — current distance to object (metres).
    velocity_ms   : float — closing velocity (m/s, negative = approaching).
    timestamp_us  : int   — UNIX microsecond timestamp.
    frame_tensor  : Optional[torch.Tensor] — [5,640,640] camera slice
                    for semantic dispatch to VLA. None if not captured.
    """
    camera_idx:   int
    camera_name:  str
    box:          DetectionBox
    ttc_s:        float
    threat_level: ThreatLevel
    distance_m:   float
    velocity_ms:  float
    timestamp_us: int             = field(default_factory=lambda: int(time.time() * 1e6))
    frame_tensor: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dict for MCP dispatch."""
        return {
            "camera_idx":   self.camera_idx,
            "camera_name":  self.camera_name,
            "ttc_s":        round(self.ttc_s, 4),
            "threat_level": str(self.threat_level),
            "distance_m":   round(self.distance_m, 2),
            "velocity_ms":  round(self.velocity_ms, 2),
            "timestamp_us": self.timestamp_us,
            "bbox": {
                "x1": round(self.box.x1, 1),
                "y1": round(self.box.y1, 1),
                "x2": round(self.box.x2, 1),
                "y2": round(self.box.y2, 1),
            },
            "class_name":  self.box.class_name,
            "confidence":  round(self.box.score, 4),
        }

    def __str__(self) -> str:
        return (
            f"[{self.threat_level}] cam={self.camera_name} "
            f"class={self.box.class_name} "
            f"TTC={self.ttc_s:.2f}s dist={self.distance_m:.1f}m "
            f"vel={self.velocity_ms:.1f}m/s"
        )


# ─────────────────────────────────────────────────────────────
# Kinematics helpers
# ─────────────────────────────────────────────────────────────

_EPS: float = 1e-6   # Small value to avoid division by zero


def compute_ttc_simple(distance_m: float, closing_vel_ms: float) -> float:
    """
    Constant-velocity TTC:  TTC = D / |Vr|

    Parameters
    ----------
    distance_m    : float — range to target (metres). Must be > 0.
    closing_vel_ms: float — closing speed (m/s).
                    Negative = approaching, positive = receding.

    Returns
    -------
    float — TTC in seconds. Returns inf if not closing (safe).
    """
    if closing_vel_ms >= 0:
        # Object is moving away — not a collision threat
        return float("inf")

    if distance_m <= 0:
        return 0.0

    return distance_m / abs(closing_vel_ms)


def compute_ettc(
    distance_m:    float,
    closing_vel_ms: float,
    rel_accel_ms2:  float,
) -> float:
    """
    Enhanced TTC (ETTC) accounting for relative acceleration.

        ETTC = [-Vr - sqrt(Vr² - 2·Ar·D)] / Ar

    Falls back to simple TTC when Ar ≈ 0.

    Parameters
    ----------
    distance_m     : float — range to target (metres).
    closing_vel_ms : float — closing speed (m/s, negative = approaching).
    rel_accel_ms2  : float — relative acceleration (m/s², negative = closing faster).

    Returns
    -------
    float — ETTC in seconds. inf if not a threat. 0 if already colliding.
    """
    Vr = closing_vel_ms   # negative for approaching
    Ar = rel_accel_ms2
    D  = distance_m

    if D <= 0:
        return 0.0

    if abs(Ar) < _EPS:
        # Degenerate: use simple TTC
        return compute_ttc_simple(D, Vr)

    discriminant = Vr**2 - 2.0 * Ar * D

    if discriminant < 0:
        # No real solution: paths never cross
        return float("inf")

    ttc = (-Vr - math.sqrt(discriminant)) / Ar

    return max(ttc, 0.0) if ttc > 0 else float("inf")


# ─────────────────────────────────────────────────────────────
# ThreatMatrixEvaluator
# ─────────────────────────────────────────────────────────────

class ThreatMatrixEvaluator:
    """
    Evaluate collision threats from a BatchInferenceResult.

    For each detected object across all 6 cameras, this module:
    1. Extracts range (depth_m) and closing velocity from the DetectionBox.
    2. Computes TTC (simple or enhanced if acceleration data is available).
    3. Classifies the threat level.
    4. Returns only threats above a minimum level.

    Parameters
    ----------
    min_threat_level   : ThreatLevel — minimum level to report.
                         Default: WARNING (TTC < 3 s).
    min_distance_m     : float — ignore objects closer than this
                         (likely false positives from own vehicle).
    max_distance_m     : float — ignore objects beyond this range.
    ego_speed_ms       : float — current ego-vehicle speed (m/s).
                         Used to compute absolute closing speed.
    capture_frames     : bool — attach camera tensor slice to ThreatEvents
                         for MCP semantic dispatch. Requires `batch_tensor`.

    Example
    -------
    >>> evaluator = ThreatMatrixEvaluator(ego_speed_ms=15.0)
    >>> threats = evaluator.evaluate(inference_result, batch_tensor)
    >>> for t in threats:
    ...     print(t)
    """

    def __init__(
        self,
        min_threat_level:  ThreatLevel = ThreatLevel.WARNING,
        min_distance_m:    float       = 0.5,
        max_distance_m:    float       = 80.0,
        ego_speed_ms:      float       = 0.0,
        capture_frames:    bool        = True,
    ) -> None:
        self.min_threat_level = min_threat_level
        self.min_distance_m   = min_distance_m
        self.max_distance_m   = max_distance_m
        self.ego_speed_ms     = ego_speed_ms
        self.capture_frames   = capture_frames

        # Per-object velocity history for acceleration estimation
        self._velocity_history: Dict[str, List[Tuple[float, float]]] = {}

    def evaluate(
        self,
        result:          BatchInferenceResult,
        batch_tensor:    Optional[torch.Tensor] = None,
        timestamp_us:    Optional[int]          = None,
    ) -> List[ThreatEvent]:
        """
        Evaluate all detections and return threats above the minimum level.

        Parameters
        ----------
        result       : BatchInferenceResult from batch_inference.py.
        batch_tensor : Optional[torch.Tensor] — [6,5,640,640] input tensor.
                       If provided and capture_frames=True, the camera slice
                       [cam_idx, :, :, :] is attached to each ThreatEvent.
        timestamp_us : Optional[int] — current timestamp in microseconds.

        Returns
        -------
        List[ThreatEvent] — sorted by TTC ascending (most critical first).
        """
        ts = timestamp_us or int(time.time() * 1e6)
        threats: List[ThreatEvent] = []

        for cam in result.cameras:
            cam_idx  = cam.camera_idx
            cam_name = cam.camera_name

            # Extract camera frame for semantic dispatch
            frame = None
            if self.capture_frames and batch_tensor is not None:
                frame = batch_tensor[cam_idx].cpu()   # [5, 640, 640]

            for box in cam.boxes:
                threat = self._evaluate_box(
                    box, cam_idx, cam_name, frame, ts
                )
                if threat is not None:
                    threats.append(threat)

        # Sort by TTC ascending — most dangerous first
        threats.sort(key=lambda t: t.ttc_s)

        if threats:
            logger.info(
                "ThreatMatrix: %d threats detected. Most critical: %s",
                len(threats), threats[0]
            )

        return threats

    def _evaluate_box(
        self,
        box:       DetectionBox,
        cam_idx:   int,
        cam_name:  str,
        frame:     Optional[torch.Tensor],
        ts:        int,
    ) -> Optional[ThreatEvent]:
        """
        Evaluate a single DetectionBox and return a ThreatEvent
        if it represents a threat, or None otherwise.
        """
        depth_m  = box.depth_m
        vel_ms   = box.velocity_ms   # negative = approaching

        # ── Filter: ignore invalid measurements ───────────────────
        if depth_m <= 0 or not self.min_distance_m <= depth_m <= self.max_distance_m:
            return None

        # ── Compute closing velocity ───────────────────────────────
        # Radar velocity_ms is the target's radial velocity in the
        # camera sensor frame. Negative = approaching the sensor.
        # Add ego speed: if ego moves at +15 m/s forward and target
        # is stationary, radar sees -15 m/s (approaching).
        # Here we rely directly on the ego-motion-compensated radar
        # channel (vx_comp), which already accounts for ego motion.
        closing_vel_ms = vel_ms   # Negative = approaching (convention)

        # ── Estimate relative acceleration ────────────────────────
        obj_key = f"{cam_idx}_{box.class_name}_{int(box.cx)}_{int(box.cy)}"
        rel_accel = self._estimate_acceleration(obj_key, depth_m, closing_vel_ms, ts)

        # ── Compute TTC ───────────────────────────────────────────
        if abs(rel_accel) > 0.1:
            ttc = compute_ettc(depth_m, closing_vel_ms, rel_accel)
        else:
            ttc = compute_ttc_simple(depth_m, closing_vel_ms)

        level = ttc_to_level(ttc)

        # ── Filter by minimum threat level ────────────────────────
        if level < self.min_threat_level:
            return None

        return ThreatEvent(
            camera_idx   = cam_idx,
            camera_name  = cam_name,
            box          = box,
            ttc_s        = ttc,
            threat_level = level,
            distance_m   = depth_m,
            velocity_ms  = closing_vel_ms,
            timestamp_us = ts,
            frame_tensor = frame,
        )

    def _estimate_acceleration(
        self,
        obj_key:    str,
        distance_m: float,
        vel_ms:     float,
        ts:         int,
    ) -> float:
        """
        Estimate relative acceleration from consecutive velocity observations.

        Uses a simple finite difference over the last 2 measurements.
        Returns 0.0 if insufficient history.
        """
        hist = self._velocity_history.setdefault(obj_key, [])
        hist.append((ts, vel_ms))

        # Keep only recent history (last 5 frames)
        if len(hist) > 5:
            self._velocity_history[obj_key] = hist[-5:]
            hist = self._velocity_history[obj_key]

        if len(hist) < 2:
            return 0.0

        t0, v0 = hist[-2]
        t1, v1 = hist[-1]
        dt = (t1 - t0) / 1e6   # microseconds → seconds

        if dt < _EPS:
            return 0.0

        return (v1 - v0) / dt


# ─────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────

def evaluate_threats(
    result:       BatchInferenceResult,
    batch_tensor: Optional[torch.Tensor] = None,
    ego_speed_ms: float                  = 0.0,
    min_level:    ThreatLevel            = ThreatLevel.WARNING,
) -> List[ThreatEvent]:
    """
    One-shot convenience wrapper to evaluate threats from a batch result.

    Returns
    -------
    List[ThreatEvent] sorted by TTC (most critical first).
    """
    evaluator = ThreatMatrixEvaluator(
        min_threat_level = min_level,
        ego_speed_ms     = ego_speed_ms,
        capture_frames   = (batch_tensor is not None),
    )
    return evaluator.evaluate(result, batch_tensor)
