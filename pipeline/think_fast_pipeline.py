"""
think_fast/pipeline/think_fast_pipeline.py
=============================================
End-to-End "Think Fast" Reflex Pipeline Orchestrator.

Wires together all 5 stages:
  Stage 1 → Sensor Ingestion (nuScenes loader)
  Stage 2 → Early Fusion Preprocessing (radar projection + channel stacking)
  Stage 3 → Batched YOLO Inference (PyTorch dev or TensorRT prod)
  Stage 4 → Threat Matrix Evaluation
  Stage 5 → Parallel Dispatch:
             A. Physical Reflex (PhysicalReflex.trigger)
             B. Semantic Dispatch (MCPDispatcher.dispatch)

Usage
-----
    # Development mode (PyTorch inference):
    python -m think_fast.pipeline.think_fast_pipeline \
        --dataroot /data/nuscenes \
        --weights  runs/think_fast/best.pt \
        --mode     dev

    # Production mode (TensorRT):
    python -m think_fast.pipeline.think_fast_pipeline \
        --dataroot /data/nuscenes \
        --engine   model_5ch_fp16.trt \
        --mode     prod \
        --sample_token <token>

    # Demo with a single sample token:
    python -m think_fast.pipeline.think_fast_pipeline \
        --dataroot /data/nuscenes \
        --weights  runs/think_fast/best.pt \
        --mode     dev \
        --demo

Latency Targets (per 360° frame)
----------------------------------
    Dev mode  (PyTorch FP32, RTX 4090):  ~30–60 ms
    Prod mode (TensorRT FP16, Orin NX):  ~8–15 ms
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Think Fast imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from think_fast.data.nuscenes_dataset import NuScenesLoader, NuScenesSample
from think_fast.data.channel_stacker  import ChannelStacker
from think_fast.threat.threat_matrix  import (
    ThreatEvent,
    ThreatLevel,
    ThreatMatrixEvaluator,
)
from think_fast.actuation.reflex_actuator import PhysicalReflex
from think_fast.mcp.mcp_dispatcher        import MCPDispatcher


# ─────────────────────────────────────────────────────────────
# Per-frame latency breakdown
# ─────────────────────────────────────────────────────────────

@dataclass
class FrameTimings:
    """Wall-clock time spent in each pipeline stage (milliseconds)."""
    fusion_ms:    float = 0.0
    inference_ms: float = 0.0
    threat_ms:    float = 0.0
    actuation_ms: float = 0.0
    mcp_ms:       float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.fusion_ms + self.inference_ms +
            self.threat_ms + self.actuation_ms
            # Note: mcp_ms is async and does not add to critical path
        )

    def to_dict(self) -> Dict:
        return {
            "fusion_ms":    round(self.fusion_ms,    2),
            "inference_ms": round(self.inference_ms, 2),
            "threat_ms":    round(self.threat_ms,    2),
            "actuation_ms": round(self.actuation_ms, 2),
            "total_ms":     round(self.total_ms,     2),
        }


@dataclass
class FrameResult:
    """Full result from processing one 360° nuScenes sample."""
    sample_token:  str
    timings:       FrameTimings
    n_detections:  int
    threats:       List[ThreatEvent]
    actuations:    int
    mcp_dispatches: int

    def summary(self) -> str:
        return (
            f"sample={self.sample_token[:8]}… "
            f"det={self.n_detections} "
            f"threats={len(self.threats)} "
            f"actuations={self.actuations} "
            f"latency={self.timings.total_ms:.1f}ms"
        )


# ─────────────────────────────────────────────────────────────
# ThinkFastPipeline
# ─────────────────────────────────────────────────────────────

class ThinkFastPipeline:
    """
    End-to-end Think Fast pipeline.

    Parameters
    ----------
    weights_or_engine : str — path to:
        - dev mode:  .pt Ultralytics model file
        - prod mode: .trt TensorRT engine file
    mode              : str — "dev" or "prod"
    dataroot          : str — nuScenes data root directory
    nuscenes_version  : str — e.g. "v1.0-trainval"
    ego_speed_ms      : float — current ego-vehicle speed [m/s]
    mcp_url           : str — MCP server URL
    min_threat_level  : ThreatLevel — minimum level to trigger
    conf              : float — detection confidence threshold
    iou               : float — NMS IoU threshold
    device            : str — "cuda" or "cpu"
    log_dir           : str — directory for pipeline logs
    """

    def __init__(
        self,
        weights_or_engine: str,
        mode:              str         = "dev",
        dataroot:          Optional[str] = None,
        nuscenes_version:  str         = "v1.0-trainval",
        ego_speed_ms:      float       = 0.0,
        mcp_url:           str         = "http://localhost:8765/mcp",
        min_threat_level:  ThreatLevel = ThreatLevel.WARNING,
        conf:              float       = 0.25,
        iou:               float       = 0.45,
        device:            str         = "cuda",
        log_dir:           str         = "logs/pipeline",
    ) -> None:

        self.mode           = mode
        self.ego_speed_ms   = ego_speed_ms
        self.log_dir        = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ── Stage 2: Channel Stacker ──────────────────────────────
        self.stacker = ChannelStacker()

        # ── Stage 3: Inference Engine ─────────────────────────────
        self.inference_engine = self._build_inference_engine(
            weights_or_engine, mode, conf, iou, device
        )

        # ── Stage 4: Threat Matrix ────────────────────────────────
        self.threat_evaluator = ThreatMatrixEvaluator(
            min_threat_level = min_threat_level,
            ego_speed_ms     = ego_speed_ms,
            capture_frames   = True,
        )

        # ── Stage 5A: Physical Reflex ─────────────────────────────
        self.reflex = PhysicalReflex(
            log_dir        = os.path.join(log_dir, "actuation"),
            sim_latency_ms = 2.0,
        )

        # ── Stage 5B: MCP Dispatcher ──────────────────────────────
        self.mcp = MCPDispatcher(
            server_url  = mcp_url,
            min_level   = min_threat_level,
        )

        # ── nuScenes Loader (optional — for batch demo mode) ──────
        self.loader: Optional[NuScenesLoader] = None
        if dataroot:
            self.loader = NuScenesLoader(
                dataroot = dataroot,
                version  = nuscenes_version,
                split    = "val",
                verbose  = False,
            )

        self._frame_log: List[Dict] = []

        logger.info(
            "ThinkFastPipeline ready | mode=%s | device=%s | threats≥%s",
            mode, device, min_threat_level
        )

    # ── Public Interface ─────────────────────────────────────────

    def run_sample(self, sample: NuScenesSample) -> FrameResult:
        """
        Process one nuScenes NuScenesSample through the full pipeline.

        Parameters
        ----------
        sample : NuScenesSample — loaded sensor data.

        Returns
        -------
        FrameResult — detections, threats, actuation counts, and timings.
        """
        timings = FrameTimings()

        # ── Stage 2: Early Fusion ─────────────────────────────────
        t0 = time.perf_counter()
        tensor, _meta = self.stacker.build_batch_tensor(sample)
        timings.fusion_ms = (time.perf_counter() - t0) * 1000.0

        return self._run_tensor(
            tensor        = tensor,
            sample_token  = sample.sample_token,
            timings       = timings,
        )

    def run_tensor(self, tensor: torch.Tensor, sample_token: str = "manual") -> FrameResult:
        """
        Process a pre-built [6, 5, 640, 640] tensor directly.
        Useful for simulation or when loading from a file.
        """
        return self._run_tensor(tensor, sample_token, FrameTimings())

    def run_demo(self, n_samples: int = 5) -> List[FrameResult]:
        """
        Run the pipeline on N validation samples from nuScenes.

        Parameters
        ----------
        n_samples : int — number of samples to process.

        Returns
        -------
        List[FrameResult]
        """
        if self.loader is None:
            raise RuntimeError("No dataroot provided. Pass dataroot= to ThinkFastPipeline.")

        results = []
        for i in range(min(n_samples, len(self.loader))):
            sample = self.loader[i]
            logger.info("Processing sample %d/%d: %s", i + 1, n_samples, sample.sample_token[:16])
            result = self.run_sample(sample)
            results.append(result)
            logger.info("  → %s", result.summary())

        self._log_summary(results)
        return results

    # ── Private ──────────────────────────────────────────────────

    def _run_tensor(
        self,
        tensor:       torch.Tensor,
        sample_token: str,
        timings:      FrameTimings,
    ) -> FrameResult:
        """Core pipeline: inference → threats → actuation → MCP."""

        # ── Stage 3: Batched Inference ────────────────────────────
        t0 = time.perf_counter()
        inference_result = self.inference_engine.run(tensor)
        timings.inference_ms = (time.perf_counter() - t0) * 1000.0

        n_det = inference_result.total_detections()

        # ── Stage 4: Threat Matrix ────────────────────────────────
        t0 = time.perf_counter()
        ts_us  = int(time.time() * 1e6)
        threats = self.threat_evaluator.evaluate(inference_result, tensor, ts_us)
        timings.threat_ms = (time.perf_counter() - t0) * 1000.0

        # ── Stage 5: Parallel Dispatch ────────────────────────────
        n_actuations    = 0
        n_mcp_dispatches = 0

        if threats:
            # 5A: Physical Reflex (synchronous trigger → daemon thread)
            t0 = time.perf_counter()
            events = self.reflex.trigger_many(threats)
            n_actuations = len(events)
            timings.actuation_ms = (time.perf_counter() - t0) * 1000.0

            # 5B: MCP Dispatch (async, non-blocking — use thread dispatch)
            t0 = time.perf_counter()
            for threat in threats:
                if threat.threat_level >= ThreatLevel.PARTIAL:
                    self.mcp.dispatch_sync(threat)
                    n_mcp_dispatches += 1
            timings.mcp_ms = (time.perf_counter() - t0) * 1000.0

        result = FrameResult(
            sample_token   = sample_token,
            timings        = timings,
            n_detections   = n_det,
            threats        = threats,
            actuations     = n_actuations,
            mcp_dispatches = n_mcp_dispatches,
        )

        # Log frame result
        self._frame_log.append({
            "sample_token": sample_token,
            **timings.to_dict(),
            "n_detections": n_det,
            "n_threats":    len(threats),
            "n_actuations": n_actuations,
        })

        return result

    def _build_inference_engine(
        self,
        path:   str,
        mode:   str,
        conf:   float,
        iou:    float,
        device: str,
    ):
        """Build the appropriate inference engine (PyTorch or TensorRT)."""
        if mode == "dev":
            from think_fast.inference.batch_inference import BatchInferenceEngine
            from think_fast.model.yolo11n_5ch import build_model_from_scratch

            logger.info("Dev mode: loading PyTorch model from %s", path)
            if not os.path.exists(path):
                logger.warning("Weights not found at %s — initialising from scratch.", path)
                model = build_model_from_scratch()
            else:
                from ultralytics import YOLO
                model = YOLO(path)

            return BatchInferenceEngine(model, conf=conf, iou=iou, device=device)

        elif mode == "prod":
            from think_fast.inference.tensorrt_engine import TRTInferenceSession

            logger.info("Prod mode: loading TensorRT engine from %s", path)
            return TRTInferenceSession.load(path, conf=conf, iou=iou)

        else:
            raise ValueError(f"Unknown mode='{mode}'. Use 'dev' or 'prod'.")

    def _log_summary(self, results: List[FrameResult]) -> None:
        """Write a summary JSON of all processed frames."""
        if not results:
            return

        total_det   = sum(r.n_detections   for r in results)
        total_thr   = sum(len(r.threats)   for r in results)
        total_act   = sum(r.actuations     for r in results)
        avg_latency = sum(r.timings.total_ms for r in results) / len(results)

        summary = {
            "n_samples":       len(results),
            "total_detections": total_det,
            "total_threats":   total_thr,
            "total_actuations": total_act,
            "avg_latency_ms":  round(avg_latency, 2),
        }

        path = os.path.join(self.log_dir, "pipeline_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Pipeline summary: %s", json.dumps(summary))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Think Fast — End-to-End Reflex Pipeline"
    )
    p.add_argument("--mode",          type=str, default="dev",
                   choices=["dev", "prod"])
    p.add_argument("--weights",       type=str, default=None,
                   help="Path to .pt model (dev mode)")
    p.add_argument("--engine",        type=str, default=None,
                   help="Path to .trt engine (prod mode)")
    p.add_argument("--dataroot",      type=str, default=None,
                   help="nuScenes data root")
    p.add_argument("--version",       type=str, default="v1.0-trainval")
    p.add_argument("--sample_token",  type=str, default=None,
                   help="Run on a specific sample token")
    p.add_argument("--demo",          action="store_true",
                   help="Run demo on N validation samples")
    p.add_argument("--n_samples",     type=int, default=5)
    p.add_argument("--mcp_url",       type=str, default="http://localhost:8765/mcp")
    p.add_argument("--ego_speed",     type=float, default=0.0)
    p.add_argument("--conf",          type=float, default=0.25)
    p.add_argument("--iou",           type=float, default=0.45)
    p.add_argument("--device",        type=str, default="cuda")
    p.add_argument("--log_dir",       type=str, default="logs/pipeline")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = args.weights if args.mode == "dev" else args.engine
    if weights_path is None:
        # Fall back to scratch model for quick demo
        weights_path = "yolo11n_5ch_scratch.yaml"

    pipeline = ThinkFastPipeline(
        weights_or_engine = weights_path,
        mode              = args.mode,
        dataroot          = args.dataroot,
        nuscenes_version  = args.version,
        ego_speed_ms      = args.ego_speed,
        mcp_url           = args.mcp_url,
        conf              = args.conf,
        iou               = args.iou,
        device            = args.device,
        log_dir           = args.log_dir,
    )

    if args.demo and args.dataroot:
        pipeline.run_demo(n_samples=args.n_samples)

    elif args.sample_token and args.dataroot:
        loader = NuScenesLoader(
            dataroot = args.dataroot,
            version  = args.version,
            split    = "val",
            verbose  = False,
        )
        sample = loader.load_sample(args.sample_token)
        result = pipeline.run_sample(sample)
        logger.info("Result: %s", result.summary())

    else:
        # Quick smoke test with a random tensor
        logger.info("No dataroot or sample_token provided — running tensor smoke test …")
        dummy   = torch.randn(6, 5, 640, 640)
        result  = pipeline.run_tensor(dummy, "smoke_test")
        logger.info("Smoke test complete: %s", result.summary())
        logger.info("Timings: %s", result.timings.to_dict())


if __name__ == "__main__":
    main()
