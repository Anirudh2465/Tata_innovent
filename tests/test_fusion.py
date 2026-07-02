"""
tests/test_fusion.py
=======================
Tests for the Early Fusion data pipeline (sensor ingestion, radar
projection, and channel stacking).
"""

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from think_fast.data.nuscenes_dataset import CameraData, RadarData, NuScenesSample
from think_fast.data.radar_projector import RadarProjector
from think_fast.data.channel_stacker import ChannelStacker


def _dummy_camera() -> CameraData:
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[450, 800, :] = 255  # center pixel white

    intrinsic = np.array([
        [1000.0, 0.0, 800.0],
        [0.0, 1000.0, 450.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return CameraData(
        channel="CAM_FRONT",
        image_bgr=image,
        timestamp=1000,
        token="cam_token",
        intrinsic=intrinsic,
        ego_pose={
            "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
            "timestamp": 1000,
        },
        calibration={
            "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
            "camera_intrinsic": intrinsic.tolist(),
        }
    )


def _dummy_radar() -> RadarData:
    # 3 points: 
    #  p1: in front of camera (Z=20)
    #  p2: behind camera (Z=-10) -> should be filtered
    #  p3: in front, approaching (Vr=-15)
    pts = np.array([
        [0.0, 0.0, 20.0, 10.0, -5.0],
        [0.0, 0.0, -10.0, 5.0, 0.0],
        [5.0, 0.0, 30.0, 15.0, -15.0],
    ], dtype=np.float32)

    return RadarData(
        channel="RADAR_FRONT",
        points=pts,
        timestamp=990,
        token="radar_token",
        ego_pose={
            "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
            "timestamp": 990,
        },
        calibration={
            "translation": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
        }
    )


def test_radar_projection():
    cam = _dummy_camera()
    radar = _dummy_radar()
    projector = RadarProjector(splat_radius=2, min_depth=1.0, max_depth=80.0)

    maps = projector.project([radar], cam)

    assert maps.depth_map.shape == (900, 1600)
    assert maps.vel_map.shape == (900, 1600)

    # 2 points should be valid (p1, p3), p2 is behind camera
    assert maps.n_points == 2

    # Check that depth map has some non-zero values
    assert maps.depth_map.max() > 0


def test_channel_stacker():
    cam = _dummy_camera()
    radar = _dummy_radar()
    sample = NuScenesSample(
        scene_token="scene",
        sample_token="sample",
        timestamp=1000,
        cameras={"CAM_FRONT": cam},
        radars={"RADAR_FRONT": radar},
    )

    stacker = ChannelStacker(target_size=640)
    tensor, meta = stacker.build_batch_tensor(sample)

    assert tensor.shape == (6, 5, 640, 640)
    assert tensor.dtype == torch.float32

    # Only CAM_FRONT has data; others should be zero
    assert torch.max(tensor[1]) == 0.0

    # CAM_FRONT (idx 0) should have valid image data
    assert torch.max(tensor[0, 0]) > 0.0  # R channel

    # CAM_FRONT depth channel (idx 3) should have valid radar data
    assert torch.max(tensor[0, 3]) > 0.0
