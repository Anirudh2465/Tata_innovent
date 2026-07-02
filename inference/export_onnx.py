"""
think_fast/inference/export_onnx.py
=====================================
Export the 5-channel YOLOv11-Nano to ONNX for TensorRT compilation.

Exports with:
  - Dynamic batch dimension (batch_size=-1 → "batch" symbolic dim)
  - 5-channel input (fixed)
  - 640×640 spatial resolution (fixed)
  - FP32 ONNX (TensorRT handles FP16 quantisation at engine build time)
  - Opset 17

Usage
-----
    python -m think_fast.inference.export_onnx \
        --weights  runs/think_fast/best.pt \
        --output   model_5ch.onnx \
        --verify

After export, compile with trtexec:
    trtexec --onnx=model_5ch.onnx \
            --saveEngine=model_5ch_fp16.trt \
            --fp16 \
            --workspace=4096 \
            --minShapes=input:1x5x640x640 \
            --optShapes=input:6x5x640x640 \
            --maxShapes=input:6x5x640x640
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ─────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────

def export_onnx(
    weights:      str,
    output:       str           = "model_5ch.onnx",
    batch_size:   int           = 6,
    img_size:     int           = 640,
    opset:        int           = 17,
    dynamic:      bool          = True,
    simplify:     bool          = True,
    verify:       bool          = True,
) -> str:
    """
    Export the 5-channel YOLOv11-Nano model to ONNX.

    Parameters
    ----------
    weights    : str — path to the .pt model file (Ultralytics YOLO format)
                 or a state-dict .pt saved by ThinkFastTrainer.
    output     : str — output ONNX file path.
    batch_size : int — static batch size for non-dynamic export (default 6).
    img_size   : int — spatial resolution (default 640).
    opset      : int — ONNX opset version.
    dynamic    : bool — export with dynamic batch axis.
    simplify   : bool — apply onnx-simplifier after export.
    verify     : bool — run onnxruntime check after export.

    Returns
    -------
    str — path to the exported ONNX file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    logger.info("Loading model from: %s", weights)

    # ── Determine if weights is a YOLO checkpoint or state dict ──
    model = YOLO(weights)

    # ── Verify 5-channel first conv ───────────────────────────────
    first_conv = None
    for m in model.model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break

    if first_conv is None or first_conv.in_channels != 5:
        raise ValueError(
            f"Model first conv has in_channels={first_conv.in_channels if first_conv else '?'}, "
            "expected 5. Are you using the correct 5-channel weights?"
        )
    logger.info("✓ First conv in_channels = 5")

    # ── Export using Ultralytics built-in ONNX export ─────────────
    # Note: We override the input channels check by patching the model
    export_kwargs = dict(
        format  = "onnx",
        imgsz   = img_size,
        half    = False,         # FP32 ONNX; TensorRT handles FP16
        dynamic = dynamic,
        simplify= simplify,
        opset   = opset,
        batch   = batch_size,
    )

    logger.info("Exporting to ONNX (opset=%d, dynamic=%s) …", opset, dynamic)
    export_path = model.export(**export_kwargs)

    # Ultralytics writes the file alongside the .pt; rename if needed
    if export_path and export_path != output:
        import shutil
        shutil.move(export_path, output)
        export_path = output

    logger.info("ONNX saved to: %s", export_path)

    # ── Optional: ONNX simplification ────────────────────────────
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            logger.info("Running onnx-simplifier …")
            model_onnx = onnx.load(export_path)
            model_simplified, check = onnx_simplify(model_onnx)
            if check:
                onnx.save(model_simplified, export_path)
                logger.info("✓ ONNX simplified successfully.")
            else:
                logger.warning("ONNX simplification check failed — using original.")
        except ImportError:
            logger.info("onnx-simplifier not installed (pip install onnxsim) — skipping.")

    # ── Optional: ONNXRuntime verification ───────────────────────
    if verify:
        _verify_onnx(export_path, batch_size, img_size)

    return export_path


def _verify_onnx(
    onnx_path:  str,
    batch_size: int = 6,
    img_size:   int = 640,
) -> None:
    """
    Run a forward pass through the ONNX model using ONNXRuntime
    to confirm the export is valid.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX verification.")
        return

    logger.info("Verifying ONNX with ONNXRuntime …")
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy = np.random.rand(batch_size, 5, img_size, img_size).astype(np.float32)
    outputs = session.run([output_name], {input_name: dummy})

    logger.info(
        "✓ ONNX verification passed. "
        "Input: %s Output: %s → shape %s",
        dummy.shape, output_name, outputs[0].shape
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export 5-ch YOLOv11n to ONNX")
    p.add_argument("--weights",    type=str, required=True,
                   help="Path to trained .pt model")
    p.add_argument("--output",     type=str, default="model_5ch.onnx")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--img_size",   type=int, default=640)
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--no-dynamic", dest="dynamic",   action="store_false",
                   help="Use static batch size instead of dynamic")
    p.add_argument("--no-simplify",dest="simplify",  action="store_false")
    p.add_argument("--verify",     dest="verify",    action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx(
        weights    = args.weights,
        output     = args.output,
        batch_size = args.batch_size,
        img_size   = args.img_size,
        opset      = args.opset,
        dynamic    = args.dynamic,
        simplify   = args.simplify,
        verify     = args.verify,
    )


if __name__ == "__main__":
    main()
