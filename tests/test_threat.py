"""
tests/test_threat.py
=======================
Tests for the Threat Matrix calculations and evaluations.
"""

import math
import pytest

from think_fast.threat.threat_matrix import (
    compute_ttc_simple,
    compute_ettc,
    ttc_to_level,
    ThreatLevel,
    ThreatMatrixEvaluator,
)
from think_fast.inference.batch_inference import (
    DetectionBox,
    CameraDetections,
    BatchInferenceResult,
)


def test_ttc_simple():
    # 20m distance, approaching at 10m/s -> TTC = 2s
    assert compute_ttc_simple(20.0, -10.0) == 2.0

    # Moving away -> safe (inf)
    assert math.isinf(compute_ttc_simple(20.0, 10.0))

    # Already crashed
    assert compute_ttc_simple(0.0, -10.0) == 0.0


def test_ettc():
    # Constant velocity fallback
    assert compute_ettc(20.0, -10.0, 0.0) == 2.0

    # Accelerating towards us (Ar = -2 m/s^2)
    # Discriminant = 100 - 2(-2)(20) = 180
    # ttc = (10 - sqrt(180)) / -2 = (10 - 13.416) / -2 = 1.708s
    ettc = compute_ettc(20.0, -10.0, -2.0)
    assert 1.7 < ettc < 1.71


def test_ttc_to_level():
    assert ttc_to_level(0.5) == ThreatLevel.EMERGENCY
    assert ttc_to_level(1.2) == ThreatLevel.PARTIAL
    assert ttc_to_level(1.8) == ThreatLevel.PRE_FILL
    assert ttc_to_level(2.5) == ThreatLevel.WARNING
    assert ttc_to_level(5.0) == ThreatLevel.SAFE


def test_evaluator_filtering():
    evaluator = ThreatMatrixEvaluator(min_threat_level=ThreatLevel.WARNING)

    # Box 1: EMERGENCY (TTC=0.5s)
    box1 = DetectionBox(0, 0, 10, 10, 0.9, 0, "car", 5.0, -10.0)
    # Box 2: SAFE (TTC=10s)
    box2 = DetectionBox(20, 20, 30, 30, 0.9, 0, "car", 50.0, -5.0)

    cam_dets = CameraDetections(0, "CAM_FRONT", [box1, box2])
    result = BatchInferenceResult([cam_dets])

    threats = evaluator.evaluate(result)

    assert len(threats) == 1
    assert threats[0].threat_level == ThreatLevel.EMERGENCY
    assert threats[0].box == box1
