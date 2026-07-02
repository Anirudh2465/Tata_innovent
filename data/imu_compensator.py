"""
think_fast/data/imu_compensator.py
=====================================
Ego-Motion Compensation for temporal radar–camera alignment.

Problem
-------
Cameras and radar sensors poll at different rates and timestamps.
In the time gap (Δt) between a radar sweep and the camera frame,
the ego-vehicle physically moves through space, causing the radar
point cloud to be "stale" — i.e., it describes the world as it
was at t_radar, not t_camera.

Solution
--------
Using the ego-vehicle's pose records from the nuScenes dataset
(which effectively encodes IMU + GPS integration), we compute
the 4×4 rigid-body transform that maps the radar sensor frame
at t_radar into the camera ego frame at t_camera.

This module provides:
  - `EgoMotionCompensator`      — main class
  - `compute_radar_to_camera_transform` — standalone helper

The compensated radar points can then be safely projected onto
the camera image plane without "ghosting" artifacts.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helper: Build 4×4 rigid transform from translation + rotation
# ─────────────────────────────────────────────────────────────

def _pose_to_mat(translation: np.ndarray, rotation: Quaternion) -> np.ndarray:
    """
    Build a 4×4 homogeneous rigid-body transform matrix.

    Parameters
    ----------
    translation : (3,) np.ndarray — x, y, z in metres.
    rotation    : pyquaternion.Quaternion

    Returns
    -------
    T : (4, 4) np.ndarray — SE(3) transform matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation.rotation_matrix
    T[:3,  3] = translation
    return T


def _mat_from_record(record: Dict) -> np.ndarray:
    """Convert a nuScenes pose/calibration record to a 4×4 matrix."""
    return _pose_to_mat(
        np.array(record["translation"], dtype=np.float64),
        Quaternion(record["rotation"]),
    )


# ─────────────────────────────────────────────────────────────
# EgoMotionCompensator
# ─────────────────────────────────────────────────────────────

class EgoMotionCompensator:
    """
    Computes the rigid-body transform to warp radar points from the
    radar sensor frame (at t_radar) into a target camera frame
    (at t_camera), accounting for ego-vehicle motion between timestamps.

    The transform chain is:

        radar_sensor → ego_radar → global → ego_camera → camera_sensor

    Step by step:
        1. T_radar_to_ego   = calibrated_sensor extrinsics (radar)
        2. T_ego_radar_glob = ego_pose at t_radar (radar pose → global)
        3. T_glob_ego_cam   = inverse of ego_pose at t_camera (global → camera ego)
        4. T_ego_to_cam     = calibrated_sensor extrinsics (camera)
        5. T_full           = T_ego_to_cam @ T_glob_ego_cam @ T_ego_radar_glob @ T_radar_to_ego

    The result T_full maps a 3D point in the RADAR sensor frame at
    t_radar into the CAMERA sensor frame at t_camera.

    Parameters
    ----------
    None — all calibration / pose records are passed per call.

    Example
    -------
    >>> comp = EgoMotionCompensator()
    >>> T = comp.compute_transform(
    ...     radar_cs_record=radar_data.calibration,
    ...     radar_ego_record=radar_data.ego_pose,
    ...     camera_cs_record=camera_data.calibration,
    ...     camera_ego_record=camera_data.ego_pose,
    ... )
    >>> pts_cam = comp.apply(radar_data.points, T)  # (N, 3) in camera frame
    """

    def compute_transform(
        self,
        radar_cs_record:  Dict,
        radar_ego_record: Dict,
        camera_cs_record: Dict,
        camera_ego_record: Dict,
    ) -> np.ndarray:
        """
        Compute the 4×4 transform: radar_sensor → camera_sensor
        with full ego-motion compensation.

        Parameters
        ----------
        radar_cs_record   : calibrated_sensor record for the radar.
        radar_ego_record  : ego_pose record at radar timestamp.
        camera_cs_record  : calibrated_sensor record for the camera.
        camera_ego_record : ego_pose record at camera timestamp.

        Returns
        -------
        T : (4, 4) float64 np.ndarray
        """
        # 1. Radar sensor → ego frame at t_radar
        T_radar_sensor_to_ego = _mat_from_record(radar_cs_record)

        # 2. Ego at t_radar → global frame
        T_ego_radar_to_global = _mat_from_record(radar_ego_record)

        # 3. Global → ego frame at t_camera (inverse)
        T_ego_camera_to_global = _mat_from_record(camera_ego_record)
        T_global_to_ego_camera = np.linalg.inv(T_ego_camera_to_global)

        # 4. Ego at t_camera → camera sensor frame (inverse of camera extrinsics)
        T_camera_sensor_to_ego = _mat_from_record(camera_cs_record)
        T_ego_to_camera_sensor = np.linalg.inv(T_camera_sensor_to_ego)

        # 5. Full chain: radar_sensor → camera_sensor
        T_full = (
            T_ego_to_camera_sensor
            @ T_global_to_ego_camera
            @ T_ego_radar_to_global
            @ T_radar_sensor_to_ego
        )

        return T_full

    def apply(
        self,
        points_xyz: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """
        Apply a 4×4 rigid transform to a set of 3D points.

        Parameters
        ----------
        points_xyz : (N, 3) or (N, ≥3) np.ndarray
            3D points in the source frame (only first 3 columns used).
        transform  : (4, 4) np.ndarray

        Returns
        -------
        pts_out : (N, 3) np.ndarray — transformed 3D points.
        """
        if points_xyz.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        N = points_xyz.shape[0]
        pts = points_xyz[:, :3].astype(np.float64)     # (N, 3)

        # Homogeneous coordinates: (N, 4)
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.concatenate([pts, ones], axis=1)    # (N, 4)

        # Apply transform: (4,4) @ (4,N) = (4,N) → (N,4)
        pts_transformed = (transform @ pts_h.T).T       # (N, 4)

        return pts_transformed[:, :3].astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Standalone convenience function
# ─────────────────────────────────────────────────────────────

def compute_radar_to_camera_transform(
    radar_cs_record:   Dict,
    radar_ego_record:  Dict,
    camera_cs_record:  Dict,
    camera_ego_record: Dict,
) -> np.ndarray:
    """
    Convenience wrapper around EgoMotionCompensator.compute_transform().

    Returns the (4×4) rigid transform that maps 3D radar sensor-frame
    points (at t_radar) into 3D camera sensor-frame coordinates (at t_camera).
    """
    comp = EgoMotionCompensator()
    return comp.compute_transform(
        radar_cs_record   = radar_cs_record,
        radar_ego_record  = radar_ego_record,
        camera_cs_record  = camera_cs_record,
        camera_ego_record = camera_ego_record,
    )


# ─────────────────────────────────────────────────────────────
# Delta-time diagnostic helper
# ─────────────────────────────────────────────────────────────

def compute_temporal_offset_ms(radar_timestamp_us: int, camera_timestamp_us: int) -> float:
    """
    Return the temporal offset between a radar sweep and a camera frame in milliseconds.

    Positive value → radar is older than camera (most common: radar lags).
    Negative value → radar is newer than camera.

    Parameters
    ----------
    radar_timestamp_us  : int — radar sample_data timestamp in microseconds.
    camera_timestamp_us : int — camera sample_data timestamp in microseconds.

    Returns
    -------
    delta_ms : float — time difference in milliseconds.
    """
    delta_us = camera_timestamp_us - radar_timestamp_us
    return delta_us / 1_000.0
