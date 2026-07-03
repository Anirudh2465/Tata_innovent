"""
think_fast/model/train.py
============================
Training script for the 5-channel YOLOv11-Nano model on nuScenes.

Architecture
------------
  NuScenesDataset (torch.utils.data.Dataset)
      ↓  yields (tensor[6,5,640,640], labels[List[per_camera_boxes]])
  DataLoader (batch_size=B)
      ↓  tensor[B*6, 5, 640, 640]  (flatten 6 cameras into batch dim)
  YOLOv11n-5ch (Ultralytics YOLO)
      ↓  standard Ultralytics training loop

Labels Format (YOLO)
---------------------
Each bounding box is [class_id, cx, cy, w, h] in normalised [0,1] coordinates
relative to the 640×640 letterboxed image.

3D annotations from nuScenes are projected onto each camera's 2D image plane.
Boxes that fall outside the camera FOV are discarded.

Running
-------
    # From scratch:
    python -m think_fast.model.train \
        --dataroot /data/nuscenes \
        --version  v1.0-trainval \
        --epochs   100 \
        --batch    8

    # With pre-trained weight transfer:
    python -m think_fast.model.train \
        --dataroot   /data/nuscenes \
        --pretrained yolo11n.pt \
        --epochs     100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pyquaternion import Quaternion

# ── Think Fast imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from think_fast.data.nuscenes_dataset import (
    CAMERA_CHANNELS,
    AnnotationBox,
    CameraData,
    NuScenesLoader,
    NuScenesSample,
)
from think_fast.data.channel_stacker import ChannelStacker, letterbox
from think_fast.model.yolo11n_5ch import (
    NUSCENES_CLASSES,
    build_model_from_scratch,
    build_model_with_weight_transfer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE:   int   = 640
NUM_CLASSES: int  = len(NUSCENES_CLASSES)   # 10

# nuScenes category → YOLO class index mapping
CATEGORY_TO_IDX: Dict[str, int] = {
    "vehicle.car":                   0,
    "vehicle.truck":                 1,
    "vehicle.bus.bendy":             2,
    "vehicle.bus.rigid":             2,
    "vehicle.trailer":               3,
    "vehicle.construction":          4,
    "human.pedestrian.adult":        5,
    "human.pedestrian.child":        5,
    "human.pedestrian.wheelchair":   5,
    "human.pedestrian.stroller":     5,
    "human.pedestrian.personal_mobility": 5,
    "human.pedestrian.police_officer": 5,
    "human.pedestrian.construction_worker": 5,
    "vehicle.motorcycle":            6,
    "vehicle.bicycle":               7,
    "movable_object.barrier":        8,
    "movable_object.trafficcone":    9,
    "movable_object.pushable_pullable": 9,
    "movable_object.debris":         9,
}


# ─────────────────────────────────────────────────────────────
# Label Utilities
# ─────────────────────────────────────────────────────────────

def project_3d_box_to_2d(
    ann:        AnnotationBox,
    cam_data:   CameraData,
    scale:      float,
    pad:        Tuple[int, int],
    img_size:   int = IMG_SIZE,
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Project a nuScenes 3D bounding box onto a camera image plane.

    Uses the 8 corners of the 3D box, projects them via the camera
    intrinsic + extrinsic chain, and takes the 2D axis-aligned
    bounding box of the visible corners.

    Parameters
    ----------
    ann       : AnnotationBox in global frame.
    cam_data  : CameraData with calibration.
    scale     : letterbox scale factor.
    pad       : (pad_top, pad_left) from letterboxing.
    img_size  : target image size (default 640).

    Returns
    -------
    (class_id, cx, cy, w, h) normalised to [0,1], or None if
    the box is not visible in this camera.
    """
    class_id = CATEGORY_TO_IDX.get(ann.category)
    if class_id is None:
        return None

    # ── Build 3D box corners in object frame ─────────────────────
    # size: [width, length, height] (nuScenes convention)
    width, length, height = ann.size
    corners_obj = np.array([
        [ length/2,  width/2, 0],
        [-length/2,  width/2, 0],
        [-length/2, -width/2, 0],
        [ length/2, -width/2, 0],
        [ length/2,  width/2, height],
        [-length/2,  width/2, height],
        [-length/2, -width/2, height],
        [ length/2, -width/2, height],
    ], dtype=np.float64)  # (8, 3)

    # ── Transform corners from object → global frame ──────────────
    R_obj = ann.rotation.rotation_matrix
    corners_global = (R_obj @ corners_obj.T).T + ann.translation  # (8, 3)

    # ── Global → ego frame (camera timestamp) ──────────────────────
    ego_rec  = cam_data.ego_pose
    T_ego    = np.eye(4, dtype=np.float64)
    T_ego[:3,:3] = Quaternion(ego_rec["rotation"]).rotation_matrix
    T_ego[:3, 3] = np.array(ego_rec["translation"])
    T_ego_inv = np.linalg.inv(T_ego)

    corners_h = np.hstack([corners_global, np.ones((8,1))])  # (8, 4)
    corners_ego = (T_ego_inv @ corners_h.T).T[:, :3]         # (8, 3)

    # ── Ego → camera sensor frame ─────────────────────────────────
    cs_rec   = cam_data.calibration
    T_cs     = np.eye(4, dtype=np.float64)
    T_cs[:3,:3] = Quaternion(cs_rec["rotation"]).rotation_matrix
    T_cs[:3, 3] = np.array(cs_rec["translation"])
    T_cs_inv = np.linalg.inv(T_cs)

    corners_ego_h = np.hstack([corners_ego, np.ones((8,1))])  # (8, 4)
    corners_cam   = (T_cs_inv @ corners_ego_h.T).T[:, :3]     # (8, 3)

    # ── Filter: keep only corners in front of camera ───────────────
    in_front = corners_cam[:, 2] > 0
    if in_front.sum() < 1:
        return None

    # ── Perspective projection ─────────────────────────────────────
    K = cam_data.intrinsic  # (3, 3)
    uv = (K @ corners_cam[in_front].T).T  # (N, 3)
    u  = uv[:, 0] / uv[:, 2]
    v  = uv[:, 1] / uv[:, 2]

    # ── Apply letterbox transform to pixel coordinates ─────────────
    pad_top, pad_left = pad
    u_lb = u * scale + pad_left
    v_lb = v * scale + pad_top

    # ── Filter corners inside image ────────────────────────────────
    visible = (u_lb >= 0) & (u_lb < img_size) & (v_lb >= 0) & (v_lb < img_size)
    if visible.sum() < 2:
        return None

    u_lb = np.clip(u_lb, 0, img_size - 1)
    v_lb = np.clip(v_lb, 0, img_size - 1)

    # ── 2D axis-aligned bounding box ──────────────────────────────
    x_min, x_max = u_lb.min(), u_lb.max()
    y_min, y_max = v_lb.min(), v_lb.max()

    # Normalise to [0, 1]
    cx = (x_min + x_max) / 2.0 / img_size
    cy = (y_min + y_max) / 2.0 / img_size
    w  = (x_max - x_min) / img_size
    h  = (y_max - y_min) / img_size

    # Discard degenerate boxes
    if w < 0.005 or h < 0.005:
        return None

    return (class_id, cx, cy, w, h)


# ─────────────────────────────────────────────────────────────
# NuScenesDataset
# ─────────────────────────────────────────────────────────────

class NuScenesDataset(Dataset):
    """
    PyTorch Dataset that yields:
        tensor : [6, 5, 640, 640] float32 early-fusion tensor
        labels : List of (camera_idx, class_id, cx, cy, w, h) tuples
                 One list entry per ground-truth box across all 6 cameras.

    To integrate with Ultralytics' training loop we flatten the 6 camera
    views into the batch dimension. Each camera view + its labels is
    treated as an independent sample. Consequently the effective batch
    size seen by YOLO is  DataLoader.batch_size × 6.

    Parameters
    ----------
    dataroot : str — nuScenes data root.
    version  : str — nuScenes version.
    split    : str — "train" or "val".
    """

    def __init__(
        self,
        dataroot: str,
        version:  str = "v1.0-trainval",
        split:    str = "train",
    ) -> None:
        self.loader  = NuScenesLoader(dataroot=dataroot, version=version, split=split)
        self.stacker = ChannelStacker()

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List]:
        sample: NuScenesSample = self.loader[idx]
        tensor, meta = self.stacker.build_batch_tensor(sample)  # [6, 5, 640, 640]

        # Build labels for all 6 cameras
        labels: List[Tuple[int, int, float, float, float, float]] = []

        for cam_idx, cam_name in enumerate(CAMERA_CHANNELS):
            cam_data = sample.cameras.get(cam_name)
            if cam_data is None:
                continue

            cam_meta = meta[cam_name]
            scale    = cam_meta["scale"]
            pad      = cam_meta["pad"]

            for ann in sample.annotations:
                result = project_3d_box_to_2d(ann, cam_data, scale, pad)
                if result is not None:
                    class_id, cx, cy, w, h = result
                    labels.append((cam_idx, class_id, cx, cy, w, h))

        return tensor, labels


def collate_fn(batch):
    """
    Custom collate: stack tensors, keep labels as list-of-lists.
    """
    tensors = torch.stack([item[0] for item in batch], dim=0)  # (B, 6, 5, H, W)
    labels  = [item[1] for item in batch]
    return tensors, labels


# ─────────────────────────────────────────────────────────────
# Ultralytics-compatible data YAML writer
# ─────────────────────────────────────────────────────────────

def write_data_yaml(
    dataroot:   str,
    version:    str,
    output_dir: str,
) -> str:
    """
    Write a data.yaml file compatible with Ultralytics YOLO training.

    Ultralytics' trainer requires a YAML that specifies:
        - train / val directories (or lists of image paths)
        - nc: number of classes
        - names: list of class names

    Because nuScenes does not store images in a flat folder, we write
    a custom dataset wrapper and point the YAML at placeholder paths.
    The actual data loading is handled by the NuScenesDataset above.

    For direct Ultralytics trainer integration, images + labels must
    be written to disk as standard YOLO-format .txt files. This
    function writes a summary YAML that can be passed to model.train().

    Returns
    -------
    str — path to the written data.yaml file.
    """
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    content = f"""# Think Fast — nuScenes 5-channel data config
# Generated by think_fast/model/train.py

path: {dataroot}
train: images/train
val:   images/val

nc: {NUM_CLASSES}
names:
"""
    for name in NUSCENES_CLASSES:
        content += f"  - {name}\n"

    os.makedirs(output_dir, exist_ok=True)
    with open(data_yaml_path, "w") as f:
        f.write(content)

    logger.info("Data YAML written to: %s", data_yaml_path)
    return data_yaml_path


# ─────────────────────────────────────────────────────────────
# Custom training loop (bypasses Ultralytics trainer for 5-ch)
# ─────────────────────────────────────────────────────────────

class ThinkFastTrainer:
    """
    Training loop for the 5-channel YOLOv11-Nano.

    Why not use model.train() directly?
    ------------------------------------
    Ultralytics' built-in trainer assumes 3-channel PNG/JPEG images
    loaded from disk. Our 5-channel fused tensors require a custom
    DataLoader. This class wraps the raw model (model.model) and
    trains it with standard PyTorch + Ultralytics loss functions.

    Parameters
    ----------
    model_5ch    : ultralytics.YOLO — 5-channel model.
    dataset_train: NuScenesDataset
    dataset_val  : NuScenesDataset
    epochs       : int
    batch_size   : int — number of nuScenes *samples* per batch.
                   Effective GPU batch = batch_size * 6 cameras.
    lr0          : float — initial learning rate.
    output_dir   : str — directory for checkpoints and logs.
    device       : str — "cuda" or "cpu".
    """

    def __init__(
        self,
        model_5ch:     "ultralytics.YOLO",
        dataset_train: NuScenesDataset,
        dataset_val:   NuScenesDataset,
        epochs:        int   = 100,
        batch_size:    int   = 4,
        lr0:           float = 1e-3,
        output_dir:    str   = "runs/think_fast",
        device:        str   = "cuda",
    ) -> None:
        self.model       = model_5ch
        
        # Populate ultralytics model args with default training hyperparameters 
        # so the loss functions (e.g., v8DetectionLoss) have attributes like hyp.box
        from ultralytics.cfg import get_cfg
        self.model.model.args = get_cfg()

        self.epochs      = epochs
        self.batch_size  = batch_size
        self.output_dir  = output_dir
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model.model.to(self.device)

        self.train_loader = DataLoader(
            dataset_train,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = collate_fn,
        )
        self.val_loader = DataLoader(
            dataset_val,
            batch_size  = 1,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = collate_fn,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr           = lr0,
            weight_decay = 5e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        os.makedirs(output_dir, exist_ok=True)
        self.log: List[Dict] = []

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(
            "Starting training: %d epochs, batch=%d, device=%s",
            self.epochs, self.batch_size, self.device
        )

        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss   = self._val_epoch(epoch)
            self.scheduler.step()

            log_entry = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         self.scheduler.get_last_lr()[0],
            }
            self.log.append(log_entry)
            self._save_log()

            logger.info(
                "Epoch %3d/%d | train=%.4f | val=%.4f | lr=%.2e",
                epoch, self.epochs, train_loss, val_loss,
                self.scheduler.get_last_lr()[0],
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best.pt")
                logger.info("  ↑ New best val_loss=%.4f — checkpoint saved.", val_loss)

            # Save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(f"epoch{epoch:04d}.pt")

        self._save_checkpoint("last.pt")
        logger.info("Training complete. Best val_loss=%.4f", best_val_loss)

    def _train_epoch(self, epoch: int) -> float:
        """One training epoch. Returns mean loss."""
        self.model.model.train()
        total_loss = 0.0
        n_batches  = 0

        for tensors, labels in self.train_loader:
            # tensors: (B, 6, 5, 640, 640)
            # Flatten B*6 into batch dimension
            B = tensors.shape[0]
            flat = tensors.view(B * 6, 5, 640, 640).to(self.device)

            # Build YOLO-format label tensor
            # Format: [batch_idx, class_id, cx, cy, w, h]
            label_list = []
            for sample_idx, sample_labels in enumerate(labels):
                for cam_idx, cls, cx, cy, w, h in sample_labels:
                    flat_idx = sample_idx * 6 + cam_idx
                    label_list.append([flat_idx, cls, cx, cy, w, h])

            if len(label_list) == 0:
                continue

            label_tensor = torch.tensor(
                label_list, dtype=torch.float32
            ).to(self.device)

            self.optimizer.zero_grad()

            # Ultralytics model returns loss when inputs are wrapped in a dict
            batch_dict = {
                "img": flat,
                "batch_idx": label_tensor[:, 0],
                "cls": label_tensor[:, 1:2],
                "bboxes": label_tensor[:, 2:]
            }
            loss_result = self.model.model(batch_dict)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
            
            if loss.numel() > 1:
                loss = loss.sum()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), max_norm=10.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if n_batches % 10 == 0:
                logger.info("Epoch %d | Batch %4d | Loss: %.4f", epoch, n_batches, loss.item())

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, epoch: int) -> float:
        """One validation epoch. Returns mean loss."""
        self.model.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for tensors, labels in self.val_loader:
                B    = tensors.shape[0]
                flat = tensors.view(B * 6, 5, 640, 640).to(self.device)

                label_list = []
                for sample_idx, sample_labels in enumerate(labels):
                    for cam_idx, cls, cx, cy, w, h in sample_labels:
                        flat_idx = sample_idx * 6 + cam_idx
                        label_list.append([flat_idx, cls, cx, cy, w, h])

                if len(label_list) == 0:
                    continue

                label_tensor = torch.tensor(label_list, dtype=torch.float32).to(self.device)
                batch_dict = {
                    "img": flat,
                    "batch_idx": label_tensor[:, 0],
                    "cls": label_tensor[:, 1:2],
                    "bboxes": label_tensor[:, 2:]
                }
                
                # In val, we might need to compute loss if we wrap in train mode or we call loss manually.
                # Actually, during eval(), DetectionModel might not compute loss natively from a dict.
                # Wait, DetectionModel.forward(batch) calls self.loss if isinstance(batch, dict)
                # But it depends if we are in eval mode.
                # Wait, we might need to manually call model.loss() if model is in eval mode!
                # Let's check how we handle it: 
                if isinstance(batch_dict, dict):
                    loss_result = self.model.model.loss(batch_dict)
                    loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
                    if loss.numel() > 1:
                        loss = loss.sum()
                else:
                    loss = torch.tensor(0.0)

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        torch.save(self.model.model.state_dict(), path)
        logger.debug("Checkpoint saved: %s", path)

    def _save_log(self) -> None:
        path = os.path.join(self.output_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 5-channel YOLOv11-Nano on nuScenes"
    )
    p.add_argument("--dataroot",   type=str, required=True,
                   help="Path to nuScenes data root")
    p.add_argument("--version",    type=str, default="v1.0-trainval",
                   help="nuScenes version string")
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to yolo11n.pt for weight transfer (optional)")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch",      type=int, default=4,
                   help="Batch size (number of nuScenes samples; GPU batch = batch*6)")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--output_dir", type=str, default="runs/think_fast")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--depth_init", type=str, default="grayscale",
                   choices=["grayscale", "kaiming", "zero"],
                   help="Init strategy for the new Depth channel (weight transfer only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build model ───────────────────────────────────────────────
    if args.pretrained:
        model = build_model_with_weight_transfer(
            pretrained_pt = args.pretrained,
            num_classes   = NUM_CLASSES,
            depth_init    = args.depth_init,
        )
    else:
        model = build_model_from_scratch(num_classes=NUM_CLASSES)

    # ── Build datasets ────────────────────────────────────────────
    train_ds = NuScenesDataset(
        dataroot = args.dataroot,
        version  = args.version,
        split    = "train",
    )
    val_ds = NuScenesDataset(
        dataroot = args.dataroot,
        version  = args.version,
        split    = "val",
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # ── Train ─────────────────────────────────────────────────────
    trainer = ThinkFastTrainer(
        model_5ch     = model,
        dataset_train = train_ds,
        dataset_val   = val_ds,
        epochs        = args.epochs,
        batch_size    = args.batch,
        lr0           = args.lr,
        output_dir    = args.output_dir,
        device        = args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
