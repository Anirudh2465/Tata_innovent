"""
think_fast/mcp/mcp_server_stub.py
=====================================
FastAPI stub server simulating the "Think Slow" VLA receiver.

This server acts as a local stand-in for the production VLA endpoint.
It receives MCP threat dispatch payloads, logs them to disk, and
returns a mock response. Replace this with your real VLA integration.

Simulated capabilities:
  - Receives JSON-RPC 2.0 payloads from MCPDispatcher
  - Decodes and saves JPEG frames to disk (flagged_frames/)
  - Logs all telemetry to structured JSONL files
  - Exposes a /status endpoint to inspect recent events

Run the stub:
    python -m think_fast.mcp.mcp_server_stub

Or with uvicorn directly:
    uvicorn think_fast.mcp.mcp_server_stub:app --host 0.0.0.0 --port 8765

Endpoints:
    POST /mcp          — Receive MCP threat dispatch
    GET  /status       — Recent events summary (last N events)
    GET  /health       — Health check
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("mcp_server_stub")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required for the MCP stub server.\n"
        "Install with: pip install fastapi uvicorn"
    )

# ── Config ────────────────────────────────────────────────────
HOST:           str = "0.0.0.0"
PORT:           int = 8765
FRAMES_DIR:     str = "flagged_frames"
EVENTS_LOG:     str = "logs/mcp_events.jsonl"
MAX_HISTORY:    int = 100      # max events to keep in memory

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Think Fast — MCP Stub Server (Think Slow VLA)",
    description = (
        "Receives flagged camera frames and telemetry from the "
        "Think Fast reflex system. Simulates the Think Slow VLA "
        "until the production model is integrated."
    ),
    version = "1.0.0",
)

# ── In-memory event store ─────────────────────────────────────
_event_history: List[Dict] = []
_start_time:    float      = time.time()

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EVENTS_LOG), exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/mcp")
async def receive_mcp_dispatch(request: Request) -> JSONResponse:
    """
    Receive a JSON-RPC 2.0 MCP threat dispatch payload.

    Expected format:
    {
      "jsonrpc": "2.0",
      "method":  "think_fast/threat_dispatch",
      "id":      "<uuid>",
      "params": { ... telemetry + optional frame_jpeg_b64 ... }
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # ── Validate JSON-RPC structure ────────────────────────────
    if body.get("jsonrpc") != "2.0" or "params" not in body:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON-RPC 2.0 format. Expected jsonrpc=2.0 and params."
        )

    params     = body["params"]
    request_id = body.get("id", str(uuid.uuid4()))
    method     = body.get("method", "unknown")

    logger.info(
        "MCP dispatch received | method=%s | camera=%s | level=%s | TTC=%.2fs",
        method,
        params.get("camera_name", "?"),
        params.get("threat_level", "?"),
        params.get("ttc_s", -1),
    )

    # ── Save JPEG frame if present ─────────────────────────────
    frame_path = None
    if "frame_jpeg_b64" in params:
        frame_path = _save_frame(
            params.pop("frame_jpeg_b64"),
            camera_name  = params.get("camera_name", "unknown"),
            threat_level = params.get("threat_level", "UNKNOWN"),
            timestamp_us = params.get("timestamp_us", 0),
        )

    # ── Build event record ────────────────────────────────────
    event = {
        "received_at":  datetime.utcnow().isoformat(),
        "request_id":   request_id,
        "method":       method,
        "frame_saved":  frame_path,
        **params,
    }

    # ── Store in memory & log to disk ─────────────────────────
    _event_history.append(event)
    if len(_event_history) > MAX_HISTORY:
        _event_history.pop(0)

    _write_event_log(event)

    # ── Simulate VLA processing ────────────────────────────────
    # In production: forward to VLA model here and return grad-CAM
    mock_analysis = _mock_vla_analysis(params)

    return JSONResponse({
        "jsonrpc": "2.0",
        "id":      request_id,
        "result": {
            "status":       "received",
            "frame_saved":  frame_path,
            "vla_analysis": mock_analysis,
        }
    })


@app.get("/status")
async def get_status(n: int = 10) -> JSONResponse:
    """Return the last N received events."""
    recent = _event_history[-n:] if _event_history else []
    return JSONResponse({
        "uptime_s":     round(time.time() - _start_time, 1),
        "total_events": len(_event_history),
        "recent":       recent[-n:],
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "server": "think_fast_mcp_stub"})


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def _save_frame(
    b64_jpeg:    str,
    camera_name: str,
    threat_level: str,
    timestamp_us: int,
) -> Optional[str]:
    """Decode and save a base64 JPEG frame to disk."""
    try:
        img_bytes = base64.b64decode(b64_jpeg)
        ts_str    = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename  = f"{ts_str}_{camera_name}_{threat_level}.jpg"
        filepath  = os.path.join(FRAMES_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        logger.info("Frame saved: %s (%.1f KB)", filepath, len(img_bytes) / 1024.0)
        return filepath

    except Exception as e:
        logger.error("Failed to save frame: %s", e)
        return None


def _write_event_log(event: Dict) -> None:
    """Append event to JSONL log file."""
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        logger.error("Failed to write event log: %s", e)


def _mock_vla_analysis(params: Dict) -> Dict:
    """
    Mock VLA analysis response.

    In production, this would:
    1. Pass the frame to a VLM (e.g., Gemini Vision, LLaVA-Next)
    2. Generate a Grad-CAM saliency map
    3. Return a semantic scene description

    Currently returns a structured stub.
    """
    threat_level = params.get("threat_level", "UNKNOWN")
    class_name   = params.get("class_name",   "object")
    ttc          = params.get("ttc_s",         -1.0)
    camera       = params.get("camera_name",   "UNKNOWN")

    descriptions = {
        "EMERGENCY": (
            f"CRITICAL: A {class_name} is on immediate collision course "
            f"({ttc:.1f}s to impact) in the {camera} view. "
            "Emergency braking is active. Recommend full stop."
        ),
        "PARTIAL": (
            f"WARNING: Rapidly approaching {class_name} detected in {camera}. "
            f"TTC: {ttc:.1f}s. Partial braking engaged."
        ),
        "PRE_FILL": (
            f"CAUTION: {class_name} closing at unsafe rate via {camera}. "
            f"TTC: {ttc:.1f}s. Brake system pre-filled."
        ),
        "WARNING": (
            f"ALERT: {class_name} detected within safety threshold "
            f"({ttc:.1f}s) in {camera}. Monitoring."
        ),
    }

    return {
        "description":  descriptions.get(threat_level, "Scene under analysis."),
        "grad_cam_url": None,   # Would point to a saliency map image in production
        "recommended_action": threat_level,
        "confidence":   0.95,
        "processed_by": "think_slow_vla_stub_v1.0",
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Starting MCP Stub Server on %s:%d", HOST, PORT)
    logger.info("Flagged frames → %s/", os.path.abspath(FRAMES_DIR))
    logger.info("Event log      → %s",  os.path.abspath(EVENTS_LOG))
    logger.info("Endpoints:")
    logger.info("  POST http://%s:%d/mcp     — Receive dispatch", HOST, PORT)
    logger.info("  GET  http://%s:%d/status  — View events",      HOST, PORT)
    logger.info("  GET  http://%s:%d/health  — Health check",     HOST, PORT)

    uvicorn.run(
        "think_fast.mcp.mcp_server_stub:app",
        host    = HOST,
        port    = PORT,
        reload  = False,
        log_level = "info",
    )


if __name__ == "__main__":
    main()
