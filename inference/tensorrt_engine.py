"""
think_fast/inference/tensorrt_engine.py
==========================================
TensorRT FP16 inference engine for the 5-channel YOLOv11-Nano.

This is the production inference path for NVIDIA edge hardware
(Jetson Orin, AGX) or server-class GPUs (A100, RTX 4090).

Architecture
------------
  TRTEngineBuilder   → Build / load a serialised FP16 .trt engine
  TRTInferenceSession → CUDA-stream async inference with zero-copy I/O
  post_process()      → ONNX output → DetectionBox list (NMS included)

TensorRT Build (one-time, offline)
-----------------------------------
  Call TRTEngineBuilder.build_from_onnx(onnx_path, engine_path)
  once on your target hardware. This compiles and serialises an
  FP16 engine optimised for the exact GPU compute capability.

TensorRT Runtime (per inference)
----------------------------------
  Load the .trt file with TRTEngineBuilder.load(engine_path) →
  TRTInferenceSession.infer(tensor) → BatchInferenceResult

Prerequisites
-------------
    pip install tensorrt pycuda
    # TensorRT must be installed from NVIDIA:
    # https://developer.nvidia.com/tensorrt

Usage
-----
    # Build engine (once)
    from think_fast.inference.tensorrt_engine import TRTEngineBuilder
    TRTEngineBuilder.build_from_onnx("model_5ch.onnx", "model_5ch_fp16.trt")

    # Inference
    session = TRTInferenceSession.load("model_5ch_fp16.trt")
    result  = session.infer(tensor)      # tensor: [6,5,640,640]
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from think_fast.inference.batch_inference import (
    BatchInferenceResult,
    CAMERA_NAMES,
    CameraDetections,
    DetectionBox,
    _extract_radar_at_box,
)
from think_fast.model.yolo11n_5ch import NUSCENES_CLASSES

logger = logging.getLogger(__name__)

# ── Minimum TRT version ───────────────────────────────────────
_TRT_MIN_VERSION = (8, 6, 0)


# ─────────────────────────────────────────────────────────────
# TensorRT availability guard
# ─────────────────────────────────────────────────────────────

def _check_trt() -> bool:
    """Return True if tensorrt and pycuda are importable."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────
# TRTEngineBuilder
# ─────────────────────────────────────────────────────────────

class TRTEngineBuilder:
    """
    Build and serialise a TensorRT FP16 engine from an ONNX model.

    Usage
    -----
    >>> TRTEngineBuilder.build_from_onnx(
    ...     onnx_path    = "model_5ch.onnx",
    ...     engine_path  = "model_5ch_fp16.trt",
    ...     fp16         = True,
    ...     workspace_gb = 4,
    ...     opt_batch    = 6,
    ...     max_batch    = 6,
    ... )
    """

    @staticmethod
    def build_from_onnx(
        onnx_path:    str,
        engine_path:  str,
        fp16:         bool  = True,
        workspace_gb: float = 4.0,
        opt_batch:    int   = 6,
        max_batch:    int   = 6,
        min_batch:    int   = 1,
        img_size:     int   = 640,
        verbose:      bool  = False,
    ) -> None:
        """
        Build a TensorRT engine from an ONNX file and serialise to disk.

        Parameters
        ----------
        onnx_path    : Path to the 5-channel ONNX model.
        engine_path  : Output path for the serialised .trt engine.
        fp16         : Enable FP16 precision (default True).
        workspace_gb : Max GPU memory for TRT optimisation workspace (GB).
        opt_batch    : Optimal batch size for TRT profile (default 6).
        max_batch    : Maximum batch size (default 6).
        min_batch    : Minimum batch size (default 1 for dynamic batch).
        img_size     : Spatial resolution (default 640).
        verbose      : Enable TRT verbose logging.
        """
        if not _check_trt():
            raise ImportError(
                "TensorRT and/or PyCUDA not found.\n"
                "Install from: https://developer.nvidia.com/tensorrt\n"
                "and: pip install pycuda"
            )

        import tensorrt as trt

        logger.info(
            "Building TRT engine: %s → %s (FP16=%s, workspace=%gGB)",
            onnx_path, engine_path, fp16, workspace_gb
        )

        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

        with (
            trt.Builder(TRT_LOGGER)     as builder,
            builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )                           as network,
            trt.OnnxParser(network, TRT_LOGGER) as parser,
            builder.create_builder_config() as config,
        ):
            # ── Workspace ─────────────────────────────────────────
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                int(workspace_gb * (1 << 30))
            )

            # ── FP16 ──────────────────────────────────────────────
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 mode enabled.")
            elif fp16:
                logger.warning("FP16 not supported on this GPU — falling back to FP32.")

            # ── Parse ONNX ────────────────────────────────────────
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX parser error: %s", parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX model.")

            logger.info("ONNX parsed successfully. Num layers: %d", network.num_layers)

            # ── Dynamic shape profile ─────────────────────────────
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name

            profile.set_shape(
                input_name,
                min = (min_batch, 5, img_size, img_size),
                opt = (opt_batch, 5, img_size, img_size),
                max = (max_batch, 5, img_size, img_size),
            )
            config.add_optimization_profile(profile)

            # ── Build & serialise ─────────────────────────────────
            logger.info("Building TRT engine (this may take several minutes) …")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("TRT engine build failed. Check GPU memory and ONNX validity.")

            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.info("✓ TRT engine saved: %s", engine_path)


# ─────────────────────────────────────────────────────────────
# TRTInferenceSession
# ─────────────────────────────────────────────────────────────

class TRTInferenceSession:
    """
    Async TensorRT inference session using CUDA streams.

    Provides zero-copy Host→Device transfer for the input tensor
    and async Device→Host copy for outputs, overlapping compute
    with data movement to maximise GPU utilisation.

    Parameters
    ----------
    engine_path : str — path to serialised .trt engine file.
    device_id   : int — CUDA device index.
    conf        : float — confidence threshold for post-processing.
    iou         : float — IoU threshold for NMS.

    Example
    -------
    >>> session = TRTInferenceSession("model_5ch_fp16.trt")
    >>> result  = session.infer(tensor)  # tensor: [6,5,640,640]
    """

    def __init__(
        self,
        engine_path: str,
        device_id:   int   = 0,
        conf:        float = 0.25,
        iou:         float = 0.45,
    ) -> None:
        if not _check_trt():
            raise ImportError("TensorRT / PyCUDA not available.")

        import tensorrt  as trt
        import pycuda.driver as cuda

        self.conf       = conf
        self.iou        = iou
        self._trt       = trt
        self._cuda      = cuda

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(TRT_LOGGER)

        logger.info("Loading TRT engine from: %s", engine_path)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # ── Allocate I/O buffers ──────────────────────────────────
        self._allocate_buffers()

        # ── CUDA stream for async execution ───────────────────────
        self.stream = cuda.Stream()

        logger.info("✓ TRTInferenceSession ready.")

    def _allocate_buffers(self) -> None:
        """Allocate pinned host memory + device memory for I/O."""
        import pycuda.driver as cuda

        self.inputs:  List[dict] = []
        self.outputs: List[dict] = []
        self.bindings: List[int] = []

        for i in range(self.engine.num_io_tensors):
            name    = self.engine.get_tensor_name(i)
            dtype   = self.engine.get_tensor_dtype(name)
            shape   = self.engine.get_tensor_shape(name)
            mode    = self.engine.get_tensor_mode(name)

            # Resolve dynamic (-1) dims to max profile shape
            if -1 in shape:
                shape = self.engine.get_tensor_profile_shape(name, 0)[2]  # max

            np_dtype = self._trt_dtype_to_np(dtype)
            size     = int(np.prod(shape))

            host_mem   = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            entry = {
                "name":       name,
                "shape":      shape,
                "host_mem":   host_mem,
                "device_mem": device_mem,
            }

            if mode == self._trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    @staticmethod
    def _trt_dtype_to_np(trt_dtype) -> type:
        import tensorrt as trt
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
        }
        return mapping.get(trt_dtype, np.float32)

    def infer(self, tensor: torch.Tensor) -> BatchInferenceResult:
        """
        Run async FP16 TensorRT inference on a [6, 5, 640, 640] tensor.

        Parameters
        ----------
        tensor : torch.Tensor — [6, 5, 640, 640] float32.

        Returns
        -------
        BatchInferenceResult
        """
        import pycuda.driver as cuda

        assert tensor.shape == (6, 5, 640, 640), (
            f"Expected [6,5,640,640], got {tuple(tensor.shape)}"
        )

        input_np  = tensor.float().numpy()  # Keep for radar extraction
        input_cpu = input_np.ravel().astype(self.inputs[0]["host_mem"].dtype)

        t0 = time.perf_counter()

        # ── H2D copy ───────────────────────────────────────────────
        np.copyto(self.inputs[0]["host_mem"], input_cpu)
        cuda.memcpy_htod_async(
            self.inputs[0]["device_mem"],
            self.inputs[0]["host_mem"],
            self.stream,
        )

        # ── Set input shape for dynamic batch ─────────────────────
        self.context.set_input_shape(self.inputs[0]["name"], (6, 5, 640, 640))

        # ── Execute ────────────────────────────────────────────────
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # ── D2H copy ───────────────────────────────────────────────
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host_mem"], out["device_mem"], self.stream)

        self.stream.synchronize()

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # ── Post-process outputs ───────────────────────────────────
        raw_output = self.outputs[0]["host_mem"].reshape(self.outputs[0]["shape"])
        result     = self._post_process(raw_output, tensor, total_ms)

        return result

    def _post_process(
        self,
        raw:       np.ndarray,
        input_tensor: torch.Tensor,
        total_ms:  float,
    ) -> BatchInferenceResult:
        """
        Convert raw TRT output to BatchInferenceResult.

        The YOLOv11 ONNX output shape is typically:
            [batch, num_preds, 4 + nc]  — (x_c, y_c, w, h, cls_scores...)

        We apply confidence filtering and per-camera NMS here.
        """
        cam_detections: List[CameraDetections] = []

        # Batch dimension = 6 (one per camera)
        for cam_idx in range(6):
            cam_name = CAMERA_NAMES[cam_idx]
            boxes_out: List[DetectionBox] = []

            if raw.ndim == 3:
                preds = raw[cam_idx]   # (num_preds, 4+nc)
            else:
                preds = raw             # fallback

            # ── Decode predictions ─────────────────────────────────
            xywh       = preds[:, :4]
            cls_scores = preds[:, 4:]

            # Convert xywh → xyxy (in 640px space)
            x1 = xywh[:, 0] - xywh[:, 2] / 2
            y1 = xywh[:, 1] - xywh[:, 3] / 2
            x2 = xywh[:, 0] + xywh[:, 2] / 2
            y2 = xywh[:, 1] + xywh[:, 3] / 2

            max_scores = cls_scores.max(axis=1)
            max_cls    = cls_scores.argmax(axis=1)

            # ── Confidence filter ──────────────────────────────────
            mask = max_scores > self.conf
            x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
            scores = max_scores[mask]
            classes= max_cls[mask]

            # ── Simple IoU NMS ─────────────────────────────────────
            keep = _nms_numpy(
                np.stack([x1, y1, x2, y2], axis=1), scores, self.iou
            )

            for idx in keep:
                d_m, vel = _extract_radar_at_box(
                    input_tensor.float(), cam_idx,
                    x1[idx], y1[idx], x2[idx], y2[idx]
                )
                cls_id   = int(classes[idx])
                cls_name = (
                    NUSCENES_CLASSES[cls_id]
                    if cls_id < len(NUSCENES_CLASSES)
                    else f"cls_{cls_id}"
                )
                boxes_out.append(DetectionBox(
                    x1          = float(x1[idx]),
                    y1          = float(y1[idx]),
                    x2          = float(x2[idx]),
                    y2          = float(y2[idx]),
                    score       = float(scores[idx]),
                    class_id    = cls_id,
                    class_name  = cls_name,
                    depth_m     = d_m,
                    velocity_ms = vel,
                ))

            cam_detections.append(CameraDetections(
                camera_idx  = cam_idx,
                camera_name = cam_name,
                boxes       = boxes_out,
                latency_ms  = total_ms,
            ))

        return BatchInferenceResult(
            cameras          = cam_detections,
            total_latency_ms = total_ms,
        )

    @classmethod
    def load(cls, engine_path: str, **kwargs) -> "TRTInferenceSession":
        """Convenience factory method."""
        return cls(engine_path, **kwargs)


# ─────────────────────────────────────────────────────────────
# Lightweight NumPy NMS (for TRT post-processing)
# ─────────────────────────────────────────────────────────────

def _nms_numpy(
    boxes:  np.ndarray,   # (N, 4) [x1, y1, x2, y2]
    scores: np.ndarray,   # (N,)
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Fast NumPy implementation of Non-Maximum Suppression.

    Returns indices of kept boxes in descending score order.
    """
    if len(scores) == 0:
        return []

    order = scores.argsort()[::-1]
    keep  = []

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou   = inter / np.maximum(union, 1e-8)

        order = rest[iou <= iou_threshold]

    return keep
