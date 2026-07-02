"""
think_fast/mcp/mcp_dispatcher.py
=====================================
Asynchronous MCP (Model Context Protocol) Semantic Dispatcher.

When the Threat Matrix detects a critical event, this module
asynchronously packages and sends the flagged camera frame and
telemetry to the "Think Slow" VLA (Vision-Language-Action) model.

Key Design Constraints
----------------------
1. MUST NOT block the physical reflex path.
   dispatch() runs on asyncio and is fire-and-forget.

2. The VLA response is never awaited by the main pipeline.
   The pipeline continues processing the next frame immediately.

3. The frame is JPEG-encoded (quality=85) and base64-embedded in
   the JSON payload for transport. Alternatively, the MCP server can
   be co-located in shared memory for higher throughput.

MCP Payload Schema
------------------
{
  "jsonrpc": "2.0",
  "method":  "think_fast/threat_dispatch",
  "id":      <uuid>,
  "params": {
    "timestamp_us": <int>,
    "camera_name":  <str>,
    "threat_level": <str>,
    "ttc_s":        <float>,
    "distance_m":   <float>,
    "velocity_ms":  <float>,
    "bbox":         { "x1", "y1", "x2", "y2" },
    "class_name":   <str>,
    "confidence":   <float>,
    "frame_jpeg_b64": <base64-encoded JPEG string>
  }
}

Configuration
-------------
Set the VLA server URL via:
  - Constructor argument `server_url`
  - Environment variable THINK_FAST_MCP_URL (overrides constructor)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from think_fast.threat.threat_matrix import ThreatEvent, ThreatLevel

logger = logging.getLogger(__name__)

# ── Default MCP server URL ────────────────────────────────────
_DEFAULT_MCP_URL = "http://localhost:8765/mcp"
_ENV_MCP_URL     = "THINK_FAST_MCP_URL"


# ─────────────────────────────────────────────────────────────
# Frame encoder
# ─────────────────────────────────────────────────────────────

def _encode_frame(frame_tensor: torch.Tensor) -> Optional[str]:
    """
    Encode a [5, 640, 640] camera tensor as a base64 JPEG string.

    Only the RGB channels (0, 1, 2) are used for the JPEG.
    The Depth and Velocity channels are omitted from the visual frame
    (they are transmitted as scalar values in the telemetry payload).

    Parameters
    ----------
    frame_tensor : torch.Tensor — shape [5, 640, 640], float32.

    Returns
    -------
    str — base64-encoded JPEG bytes. None on failure.
    """
    try:
        # Extract RGB channels → (H, W, 3) uint8
        rgb = frame_tensor[:3].permute(1, 2, 0).cpu().numpy()   # (640, 640, 3)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        bgr = rgb[..., ::-1]   # RGB → BGR for OpenCV

        # JPEG encode
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None

        return base64.b64encode(buf.tobytes()).decode("ascii")

    except Exception as e:
        logger.error("Frame encoding failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────
# MCPDispatcher
# ─────────────────────────────────────────────────────────────

class MCPDispatcher:
    """
    Asynchronous MCP client that sends flagged frames and telemetry
    to the "Think Slow" VLA model via the Model Context Protocol.

    Parameters
    ----------
    server_url   : str — MCP server endpoint URL.
                   Overridden by env var THINK_FAST_MCP_URL if set.
    timeout_s    : float — HTTP request timeout in seconds.
    max_retries  : int — number of retry attempts on network failure.
    min_level    : ThreatLevel — minimum threat level to dispatch.
                   Dispatching every WARNING-level event may overload
                   the VLA; PARTIAL or higher is recommended.
    encode_frames: bool — if True, attach JPEG frame to each payload.

    Example
    -------
    >>> dispatcher = MCPDispatcher("http://localhost:8765/mcp")
    >>> asyncio.create_task(dispatcher.dispatch(threat_event))
    """

    def __init__(
        self,
        server_url:    str         = _DEFAULT_MCP_URL,
        timeout_s:     float       = 5.0,
        max_retries:   int         = 2,
        min_level:     ThreatLevel = ThreatLevel.PARTIAL,
        encode_frames: bool        = True,
    ) -> None:
        # Allow environment override
        self.server_url    = os.environ.get(_ENV_MCP_URL, server_url)
        self.timeout_s     = timeout_s
        self.max_retries   = max_retries
        self.min_level     = min_level
        self.encode_frames = encode_frames

        self._dispatch_count: int = 0
        self._fail_count:     int = 0

        logger.info(
            "MCPDispatcher initialised → %s (min_level=%s, frames=%s)",
            self.server_url, min_level, encode_frames
        )

    async def dispatch(self, threat: ThreatEvent) -> bool:
        """
        Asynchronously send a ThreatEvent to the Think Slow VLA.

        Designed to be called via asyncio.create_task() from the
        main pipeline — it returns immediately without blocking.

        Parameters
        ----------
        threat : ThreatEvent — the flagged threat event.

        Returns
        -------
        bool — True if the payload was accepted by the MCP server.
        """
        # ── Filter by minimum level ──────────────────────────────
        if threat.threat_level < self.min_level:
            return False

        # ── Build payload ────────────────────────────────────────
        payload = self._build_payload(threat)

        # ── HTTP POST (with retries) ─────────────────────────────
        success = await self._post_with_retries(payload)

        self._dispatch_count += 1
        if not success:
            self._fail_count += 1

        return success

    def dispatch_sync(self, threat: ThreatEvent) -> bool:
        """
        Synchronous wrapper for use in non-async contexts.
        Runs the dispatch coroutine on a new event loop in a daemon thread.
        """
        import threading

        result = [False]

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(self.dispatch(threat))
            loop.close()

        t = threading.Thread(target=_run, daemon=True, name="mcp_dispatch")
        t.start()
        return True   # Fire-and-forget: don't wait for the thread

    def stats(self) -> Dict:
        """Return dispatch statistics."""
        return {
            "dispatched": self._dispatch_count,
            "failed":     self._fail_count,
            "success_rate": (
                (self._dispatch_count - self._fail_count) / max(self._dispatch_count, 1)
            ),
        }

    # ── Private ──────────────────────────────────────────────────

    def _build_payload(self, threat: ThreatEvent) -> Dict:
        """Build the JSON-RPC 2.0 MCP payload."""
        params = threat.to_dict()

        # ── Attach JPEG frame if available ────────────────────────
        if self.encode_frames and threat.frame_tensor is not None:
            encoded = _encode_frame(threat.frame_tensor)
            if encoded:
                params["frame_jpeg_b64"] = encoded
                params["frame_size_kb"]  = round(len(encoded) * 3 / 4 / 1024, 1)

        return {
            "jsonrpc": "2.0",
            "method":  "think_fast/threat_dispatch",
            "id":      str(uuid.uuid4()),
            "params":  params,
        }

    async def _post_with_retries(self, payload: Dict) -> bool:
        """POST the payload to the MCP server with retry logic."""
        # Try to import httpx; fall back to urllib if unavailable
        try:
            import httpx
            return await self._post_httpx(payload)
        except ImportError:
            pass

        # Fallback: synchronous urllib (blocking, but acceptable for a background task)
        try:
            import urllib.request
            body  = json.dumps(payload).encode("utf-8")
            req   = urllib.request.Request(
                self.server_url,
                data    = body,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("MCP dispatch failed (urllib): %s", e)
            return False

    async def _post_httpx(self, payload: Dict) -> bool:
        """POST using async httpx client."""
        import httpx

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(
                        self.server_url,
                        json = payload,
                        headers = {"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        logger.debug(
                            "MCP dispatch ✓ (attempt %d, level=%s)",
                            attempt + 1, payload["params"].get("threat_level")
                        )
                        return True
                    else:
                        logger.warning(
                            "MCP server returned HTTP %d", resp.status_code
                        )

            except Exception as e:
                logger.warning(
                    "MCP dispatch attempt %d failed: %s", attempt + 1, e
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.05 * (attempt + 1))   # exponential backoff

        return False
