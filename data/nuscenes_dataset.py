"""
think_fast/data/nuscenes_dataset.py
====================================
nuScenes DevKit wrapper that provides synchronized access to:
  - 6 RGB camera feeds (CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT,
                         CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT)
  - 5 Radar point clouds (RADAR_FRONT, RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT,
                           RADAR_BACK_LEFT, RADAR_BACK_RIGHT)
  - IMU / ego-pose data for temporal alignment
  - Ground-truth annotations projected per camera

Yields a NuScenesSample dataclass for each annotated sample in the dataset.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import RadarPointCloud
    from nuscenes.utils.geometry_utils import view_points, transform_matrix
    from nuscenes.utils.splits import create_splits_scenes
except ImportError as e:
    raise ImportError(
        "nuscenes-devkit is required. Install it with:\n"
        "  pip install nuscenes-devkit"
    ) from e

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants: nuScenes sensor channel names
# ─────────────────────────────────────────────────────────────

CAMERA_CHANNELS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

RADAR_CHANNELS: List[str] = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class RadarData:
    """
    Parsed radar point cloud for a single sensor.

    Attributes
    ----------
    channel   : Sensor channel name (e.g. "RADAR_FRONT").
    points    : np.ndarray of shape (N, 5) — columns:
                  [x, y, z, rcs, compensated_velocity_radial]
                  x/y/z in RADAR sensor frame (metres).
    timestamp : UNIX microsecond timestamp.
    token     : nuScenes sample_data token.
    """
    channel: str
    points: np.ndarray          # (N, 5) — x, y, z, rcs, vx_comp
    timestamp: int
    token: str
    ego_pose: Dict              # Raw ego_pose record at this timestamp
    calibration: Dict           # calibrated_sensor record


@dataclass
class CameraData:
    """
    Loaded camera frame for a single sensor.

    Attributes
    ----------
    channel      : Sensor channel name (e.g. "CAM_FRONT").
    image_bgr    : np.ndarray of shape (H, W, 3) — BGR, original resolution.
    timestamp    : UNIX microsecond timestamp.
    token        : nuScenes sample_data token.
    intrinsic    : 3×3 camera intrinsic matrix.
    ego_pose     : Ego pose at this camera timestamp.
    calibration  : calibrated_sensor record (extrinsics + intrinsics).
    """
    channel: str
    image_bgr: np.ndarray
    timestamp: int
    token: str
    intrinsic: np.ndarray       # (3, 3)
    ego_pose: Dict
    calibration: Dict


@dataclass
class AnnotationBox:
    """
    A single ground-truth 3D bounding box from nuScenes annotations.

    Attributes
    ----------
    token         : annotation token.
    category      : e.g. "vehicle.car".
    translation   : (3,) np.ndarray — 3D box centre in global frame.
    size          : (3,) np.ndarray — width, length, height (metres).
    rotation      : Quaternion — box orientation in global frame.
    velocity      : (2,) np.ndarray — vx, vy in global frame (m/s).
    num_lidar_pts : lidar points inside box (quality indicator).
    visibility    : visibility token ("1"–"4").
    """
    token: str
    category: str
    translation: np.ndarray
    size: np.ndarray
    rotation: Quaternion
    velocity: np.ndarray        # (2,) vx, vy
    num_lidar_pts: int
    visibility: str


@dataclass
class EgoPose:
    """
    Ego-vehicle pose at a given timestamp.

    Attributes
    ----------
    translation  : (3,) position in global frame.
    rotation     : Quaternion orientation in global frame.
    timestamp    : UNIX microsecond timestamp.
    """
    translation: np.ndarray
    rotation: Quaternion
    timestamp: int


@dataclass
class NuScenesSample:
    """
    A fully synchronized multi-sensor snapshot from the nuScenes dataset.

    This is the primary unit of data consumed downstream by the fusion
    and training pipeline.

    Attributes
    ----------
    scene_token   : Token of the scene this sample belongs to.
    sample_token  : Token of this specific annotated sample.
    timestamp     : UNIX microsecond timestamp of this sample.
    cameras       : Dict mapping channel name → CameraData (6 entries).
    radars        : Dict mapping channel name → RadarData  (5 entries).
    ego_pose      : Ego pose at the sample timestamp.
    annotations   : List of 3D GT bounding boxes in global frame.
    """
    scene_token: str
    sample_token: str
    timestamp: int
    cameras: Dict[str, CameraData] = field(default_factory=dict)
    radars: Dict[str, RadarData]   = field(default_factory=dict)
    ego_pose: Optional[EgoPose]    = None
    annotations: List[AnnotationBox] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# NuScenesLoader
# ─────────────────────────────────────────────────────────────

class NuScenesLoader:
    """
    High-level loader for the nuScenes dataset.

    Parameters
    ----------
    dataroot : str
        Path to the nuScenes data directory that contains the
        `v1.0-trainval` (or `v1.0-mini`) metadata folder.
    version : str
        nuScenes version string, e.g. `"v1.0-trainval"` or `"v1.0-mini"`.
    split : str
        Dataset split — `"train"`, `"val"`, or `"mini_train"`, `"mini_val"`.
    verbose : bool
        Whether to print nuScenes devkit loading messages.

    Example
    -------
    >>> loader = NuScenesLoader(dataroot="/data/nuscenes", version="v1.0-trainval")
    >>> for sample in loader:
    ...     print(sample.sample_token, len(sample.cameras), len(sample.radars))
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        verbose: bool = True,
    ) -> None:
        self.dataroot = dataroot
        self.version  = version
        self.split    = split

        logger.info("Loading nuScenes %s from %s …", version, dataroot)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)

        # Collect sample tokens belonging to the requested split
        self.sample_tokens: List[str] = self._get_split_tokens(split)
        logger.info(
            "NuScenesLoader ready: %d samples in split '%s'",
            len(self.sample_tokens),
            split,
        )

    # ── Public Interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __iter__(self):
        for token in self.sample_tokens:
            yield self.load_sample(token)

    def __getitem__(self, idx: int) -> NuScenesSample:
        return self.load_sample(self.sample_tokens[idx])

    def load_sample(self, sample_token: str) -> NuScenesSample:
        """
        Load a fully synchronized NuScenesSample from a given sample token.

        Steps
        -----
        1. Resolve sample record and scene token.
        2. Load all 6 camera images + calibrations.
        3. Load all 5 radar point clouds + calibrations.
        4. Load ego pose at sample timestamp.
        5. Load and parse 3D annotations.

        Parameters
        ----------
        sample_token : str
            nuScenes sample token.

        Returns
        -------
        NuScenesSample
            Fully populated dataclass.
        """
        sample_rec = self.nusc.get("sample", sample_token)
        scene_token = sample_rec["scene_token"]
        timestamp   = sample_rec["timestamp"]

        # ── Cameras ──────────────────────────────────────────
        cameras: Dict[str, CameraData] = {}
        for ch in CAMERA_CHANNELS:
            cam_token = sample_rec["data"].get(ch)
            if cam_token is None:
                logger.warning("Camera channel %s not found in sample %s", ch, sample_token)
                continue
            cameras[ch] = self._load_camera(ch, cam_token)

        # ── Radars ───────────────────────────────────────────
        radars: Dict[str, RadarData] = {}
        for ch in RADAR_CHANNELS:
            radar_token = sample_rec["data"].get(ch)
            if radar_token is None:
                logger.warning("Radar channel %s not found in sample %s", ch, sample_token)
                continue
            radars[ch] = self._load_radar(ch, radar_token)

        # ── Ego Pose ─────────────────────────────────────────
        ego_pose = self._load_ego_pose(sample_rec)

        # ── Annotations ──────────────────────────────────────
        annotations = self._load_annotations(sample_rec)

        return NuScenesSample(
            scene_token  = scene_token,
            sample_token = sample_token,
            timestamp    = timestamp,
            cameras      = cameras,
            radars       = radars,
            ego_pose     = ego_pose,
            annotations  = annotations,
        )

    # ── Private Helpers ─────────────────────────────────────────

    def _get_split_tokens(self, split: str) -> List[str]:
        """Return sample tokens for the requested split using nuScenes splits."""
        splits = create_splits_scenes()

        if split not in splits:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available: {list(splits.keys())}"
            )

        split_scenes = set(splits[split])
        tokens: List[str] = []

        for sample in self.nusc.sample:
            scene_rec = self.nusc.get("scene", sample["scene_token"])
            if scene_rec["name"] in split_scenes:
                tokens.append(sample["token"])

        return tokens

    def _load_camera(self, channel: str, sd_token: str) -> CameraData:
        """Load a single camera frame, calibration, and ego pose."""
        sd_rec   = self.nusc.get("sample_data", sd_token)
        img_path = os.path.join(self.dataroot, sd_rec["filename"])
        image    = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(
                f"Camera image not found: {img_path}\n"
                "Ensure the nuScenes blobs are fully extracted."
            )

        cs_rec    = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        ego_rec   = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        intrinsic = np.array(cs_rec["camera_intrinsic"], dtype=np.float64)  # (3,3)

        return CameraData(
            channel     = channel,
            image_bgr   = image,
            timestamp   = sd_rec["timestamp"],
            token       = sd_token,
            intrinsic   = intrinsic,
            ego_pose    = ego_rec,
            calibration = cs_rec,
        )

    def _load_radar(self, channel: str, sd_token: str) -> RadarData:
        """
        Load a single radar point cloud.

        Returns points in the RADAR sensor frame as (N, 5):
            [x, y, z, rcs, vx_compensated]

        The nuScenes RadarPointCloud provides 18 features per point;
        we extract only the 5 most relevant for early fusion.

        Feature indices in nuScenes RadarPointCloud.points (18 rows):
            0  x (m)
            1  y (m)
            2  z (m)
            3  dyn_prop
            4  id
            5  rcs (dBsm)
            6  vx (m/s)
            7  vy (m/s)
            8  vx_comp (m/s) — ego-motion compensated
            9  vy_comp (m/s)
            ...
        """
        sd_rec     = self.nusc.get("sample_data", sd_token)
        pcl_path   = os.path.join(self.dataroot, sd_rec["filename"])
        cs_rec     = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        ego_rec    = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        try:
            pcl = RadarPointCloud.from_file(pcl_path)
        except FileNotFoundError:
            logger.warning("Radar file not found: %s — returning empty cloud.", pcl_path)
            return RadarData(
                channel     = channel,
                points      = np.zeros((0, 5), dtype=np.float32),
                timestamp   = sd_rec["timestamp"],
                token       = sd_token,
                ego_pose    = ego_rec,
                calibration = cs_rec,
            )

        # Extract the 5 channels: x, y, z, rcs, vx_comp
        # pcl.points shape: (18, N)
        pts = pcl.points  # (18, N)
        N   = pts.shape[1]

        if N == 0:
            arr = np.zeros((0, 5), dtype=np.float32)
        else:
            arr = np.stack([
                pts[0],   # x
                pts[1],   # y
                pts[2],   # z
                pts[5],   # rcs
                pts[8],   # vx_comp (ego-motion compensated radial velocity)
            ], axis=1).astype(np.float32)  # (N, 5)

        return RadarData(
            channel     = channel,
            points      = arr,
            timestamp   = sd_rec["timestamp"],
            token       = sd_token,
            ego_pose    = ego_rec,
            calibration = cs_rec,
        )

    def _load_ego_pose(self, sample_rec: Dict) -> EgoPose:
        """
        Load ego-vehicle pose at the sample timestamp.
        Uses the CAM_FRONT sample_data ego_pose as the reference frame.
        """
        cam_token = sample_rec["data"].get("CAM_FRONT")
        if cam_token is None:
            return None

        sd_rec  = self.nusc.get("sample_data", cam_token)
        ep_rec  = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        return EgoPose(
            translation = np.array(ep_rec["translation"], dtype=np.float64),
            rotation    = Quaternion(ep_rec["rotation"]),
            timestamp   = ep_rec["timestamp"],
        )

    def _load_annotations(self, sample_rec: Dict) -> List[AnnotationBox]:
        """
        Load 3D bounding box annotations for a sample.
        Boxes are in the global frame.
        """
        boxes: List[AnnotationBox] = []
        for ann_token in sample_rec["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Object velocity in global frame (may be (nan, nan) for static objects)
            velocity = self.nusc.box_velocity(ann_token)  # (3,) vx,vy,vz
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(2, dtype=np.float32)
            else:
                velocity = velocity[:2].astype(np.float32)  # only vx, vy

            boxes.append(AnnotationBox(
                token         = ann_token,
                category      = ann["category_name"],
                translation   = np.array(ann["translation"], dtype=np.float64),
                size          = np.array(ann["size"], dtype=np.float64),
                rotation      = Quaternion(ann["rotation"]),
                velocity      = velocity,
                num_lidar_pts = ann.get("num_lidar_pts", 0),
                visibility    = ann.get("visibility_token", "0"),
            ))

        return boxes
