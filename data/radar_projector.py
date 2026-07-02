"""
think_fast/data/radar_projector.py
=====================================
Projects ego-motion-compensated 3D radar points onto a 2D camera
image plane, producing dense Depth and Velocity maps.

Pipeline per (radar, camera) pair
----------------------------------
1. Apply ego-motion transform (radar_sensor → camera_sensor) to get
   3D points in the camera frame.
2. Filter points behind the camera (z ≤ 0) and outside the image FOV.
3. Apply camera intrinsic projection to get 2D pixel coordinates.
4. Splat each projected radar point with a Gaussian disk to fill
   the sparse measurement into a dense 2D map.
5. Return:
     - depth_map  : (H, W) float32 — metric range in metres (0 = no data).
     - vel_map    : (H, W) float32 — compensated radial velocity in m/s.

Splatting strategy
------------------
Radar measurements are inherently sparse (~100–300 points per sweep).
A naive single-pixel assignment leaves >99% of the image with zero depth.
We use a small filled disk of radius `splat_radius` pixels centred on
each projected point; for velocity, the disk is filled with the point's
measured velocity value.  This is a practical approximation — more
sophisticated methods (e.g., Gaussian RBF, depth completion networks)
can replace this step downstream.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from think_fast.data.imu_compensator import EgoMotionCompensator
from think_fast.data.nuscenes_dataset import CameraData, RadarData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DEFAULT_SPLAT_RADIUS: int = 5      # pixels — disk radius for each radar point
MIN_DEPTH: float           = 0.5   # metres — discard radar points closer than this
MAX_DEPTH: float           = 80.0  # metres — discard points farther than this
MAX_VEL: float             = 40.0  # m/s  — clamp velocity channel to ±40 m/s


# ─────────────────────────────────────────────────────────────
# ProjectedRadarMaps
# ─────────────────────────────────────────────────────────────

class ProjectedRadarMaps:
    """
    Container for the output of radar-to-camera projection.

    Attributes
    ----------
    depth_map  : (H, W) float32 — radar range values in metres.
                  0 = no radar measurement at this pixel.
    vel_map    : (H, W) float32 — radar compensated radial velocity (m/s).
                  Signed: negative = approaching, positive = receding.
                  0 = no radar measurement at this pixel.
    n_points   : int — number of valid radar points projected.
    """

    def __init__(
        self,
        depth_map: np.ndarray,
        vel_map: np.ndarray,
        n_points: int,
    ) -> None:
        self.depth_map = depth_map   # (H, W)
        self.vel_map   = vel_map     # (H, W)
        self.n_points  = n_points


# ─────────────────────────────────────────────────────────────
# RadarProjector
# ─────────────────────────────────────────────────────────────

class RadarProjector:
    """
    Projects one or more radar point clouds onto a target camera plane.

    Handles:
      - Ego-motion compensation via EgoMotionCompensator.
      - Perspective projection via the camera intrinsic matrix.
      - Gaussian disk splatting to create dense depth & velocity maps.
      - Multi-radar aggregation: radar points from all 5 sensors
        can be fused onto a single camera image when their coverage
        overlaps the camera's field of view.

    Parameters
    ----------
    splat_radius : int
        Radius of the filled disk (in pixels) used to "spread" each
        sparse radar measurement into a small local neighbourhood.
    min_depth    : float
        Points closer than this (metres) are discarded.
    max_depth    : float
        Points farther than this (metres) are discarded.

    Example
    -------
    >>> projector = RadarProjector()
    >>> maps = projector.project(
    ...     radar_list  = list(sample.radars.values()),
    ...     camera_data = sample.cameras["CAM_FRONT"],
    ... )
    >>> print(maps.depth_map.shape)   # (900, 1600)
    """

    def __init__(
        self,
        splat_radius: int = DEFAULT_SPLAT_RADIUS,
        min_depth: float  = MIN_DEPTH,
        max_depth: float  = MAX_DEPTH,
    ) -> None:
        self.splat_radius = splat_radius
        self.min_depth    = min_depth
        self.max_depth    = max_depth
        self._compensator = EgoMotionCompensator()

    def project(
        self,
        radar_list:   List[RadarData],
        camera_data:  CameraData,
    ) -> ProjectedRadarMaps:
        """
        Project all radar point clouds in `radar_list` onto `camera_data`.

        When a radar sensor has no geometric overlap with the camera FOV,
        all its points will fall outside the image bounds and be silently
        discarded.

        Parameters
        ----------
        radar_list   : List[RadarData] — typically all 5 nuScenes radars.
        camera_data  : CameraData — target camera frame.

        Returns
        -------
        ProjectedRadarMaps with depth_map and vel_map in camera image resolution.
        """
        H, W, _ = camera_data.image_bgr.shape
        depth_map = np.zeros((H, W), dtype=np.float32)
        vel_map   = np.zeros((H, W), dtype=np.float32)
        total_pts = 0

        for radar_data in radar_list:
            if radar_data.points.shape[0] == 0:
                continue

            # ── 1. Ego-motion-compensated transform ──────────────────
            T = self._compensator.compute_transform(
                radar_cs_record   = radar_data.calibration,
                radar_ego_record  = radar_data.ego_pose,
                camera_cs_record  = camera_data.calibration,
                camera_ego_record = camera_data.ego_pose,
            )

            # ── 2. Transform radar points into camera frame ───────────
            pts_cam = self._compensator.apply(radar_data.points, T)  # (N, 3)

            # ── 3. Filter points behind camera and out-of-range ───────
            valid_mask = (
                (pts_cam[:, 2] > self.min_depth) &
                (pts_cam[:, 2] < self.max_depth)
            )
            pts_cam   = pts_cam[valid_mask]
            vx_comp   = radar_data.points[valid_mask, 4]   # radial velocity (m/s)

            if pts_cam.shape[0] == 0:
                continue

            # ── 4. Perspective projection: 3D → 2D ────────────────────
            #    uv = K @ (X/Z, Y/Z, 1)^T
            K  = camera_data.intrinsic   # (3, 3)
            Z  = pts_cam[:, 2]
            uv = (K @ pts_cam.T).T       # (N, 3)
            u  = (uv[:, 0] / Z).astype(np.float32)
            v  = (uv[:, 1] / Z).astype(np.float32)
            z  = Z.astype(np.float32)

            # ── 5. Filter to image bounds ─────────────────────────────
            in_bounds = (
                (u >= 0) & (u < W) &
                (v >= 0) & (v < H)
            )
            u, v, z, vx_comp = u[in_bounds], v[in_bounds], z[in_bounds], vx_comp[in_bounds]

            if u.shape[0] == 0:
                continue

            # ── 6. Splat each point as a filled disk ──────────────────
            n = u.shape[0]
            total_pts += n
            self._splat_points(depth_map, vel_map, u, v, z, vx_comp, H, W)

        return ProjectedRadarMaps(
            depth_map = depth_map,
            vel_map   = vel_map,
            n_points  = total_pts,
        )

    # ── Private Helpers ─────────────────────────────────────────

    def _splat_points(
        self,
        depth_map: np.ndarray,
        vel_map:   np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
        vx_comp: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        """
        Paint each radar point onto depth_map / vel_map as a filled disk.

        When multiple points overlap, the closest (smallest Z) wins
        for the depth_map, and its associated velocity is used in vel_map.

        Strategy: iterate over each point and use cv2.circle on a
        temporary mask, then use np.minimum to handle overlaps for depth.
        """
        r = self.splat_radius

        for i in range(len(u)):
            px = int(round(u[i]))
            py = int(round(v[i]))

            # Clip disk to image boundaries
            x0, x1 = max(0, px - r), min(W, px + r + 1)
            y0, y1 = max(0, py - r), min(H, py + r + 1)

            # Create coordinate grids for the disk patch
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            if len(xs) == 0 or len(ys) == 0:
                continue

            xx, yy   = np.meshgrid(xs, ys)              # (patch_H, patch_W)
            dist_sq  = (xx - px)**2 + (yy - py)**2
            in_disk  = dist_sq <= r**2

            # Where the disk lands and the current depth is 0 (no data)
            # OR the new point is closer → update
            patch_depth = depth_map[y0:y1, x0:x1]
            update_mask = in_disk & ((patch_depth == 0) | (z[i] < patch_depth))

            depth_map[y0:y1, x0:x1][update_mask] = z[i]
            vel_map[y0:y1, x0:x1][update_mask]   = float(
                np.clip(vx_comp[i], -MAX_VEL, MAX_VEL)
            )


# ─────────────────────────────────────────────────────────────
# Visualization helper (debug only)
# ─────────────────────────────────────────────────────────────

def visualize_radar_projection(
    camera_data: CameraData,
    maps: ProjectedRadarMaps,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay the radar depth map on the camera image for debugging.

    Returns a BGR image (same resolution as camera_data.image_bgr)
    where detected radar points are coloured by range (blue=far, red=close).
    """
    img = camera_data.image_bgr.copy()
    depth = maps.depth_map

    # Normalise depth to 0–255 for colormap
    valid = depth > 0
    if not np.any(valid):
        return img

    d_norm = np.zeros_like(depth, dtype=np.uint8)
    d_min, d_max = depth[valid].min(), depth[valid].max()
    d_range = max(d_max - d_min, 1e-3)
    d_norm[valid] = (
        255 * (1.0 - (depth[valid] - d_min) / d_range)
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    heatmap[~valid] = 0

    overlay = cv2.addWeighted(img, 1.0, heatmap, alpha, 0)
    return overlay
