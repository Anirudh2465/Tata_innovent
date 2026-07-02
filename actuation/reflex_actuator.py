"""
think_fast/actuation/reflex_actuator.py
==========================================
Physical Reflex Actuator — Python implementation.

Dispatches emergency actuation commands (brake, steer) in response to
ThreatEvents that cross EMERGENCY or PARTIAL thresholds.

In simulation / development:
  - Logs the command to console and a structured JSON log.
  - Introduces a configurable simulated latency to model hardware delay.

In production:
  - Replace `_execute_command()` with your platform-specific interface:
    * CAN bus message (via python-can)
    * ROS2 topic publish
    * GPIO hardware interrupt (Jetson Orin GPIO)
    * Serial protocol (AutoSAR / AUTOSAR adaptive)

Design Principle
----------------
The actuator must NEVER block the main pipeline thread. Each call to
`trigger()` returns immediately — the hardware command is fire-and-forget.
The actuator uses a dedicated asyncio task (or daemon thread) to execute
the command, ensuring the pipeline can continue processing frames.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from think_fast.threat.threat_matrix import ThreatEvent, ThreatLevel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Actuation Command Types
# ─────────────────────────────────────────────────────────────

class ActuationCommand(Enum):
    """
    High-level actuation commands dispatched to the vehicle.
    """
    NONE             = "none"
    WARNING_ALERT    = "warning_alert"    # Dashboard warning only
    BRAKE_PRE_FILL   = "brake_pre_fill"   # Prepare braking system
    PARTIAL_BRAKE    = "partial_brake"    # Apply ~40% braking force
    EMERGENCY_BRAKE  = "emergency_brake"  # Full AEB — 100% braking force
    EVASIVE_STEER    = "evasive_steer"    # Lateral avoidance manoeuvre


@dataclass
class ActuationEvent:
    """
    Record of an actuation command dispatched by the reflex system.

    Attributes
    ----------
    command      : ActuationCommand — the actuation type.
    threat       : ThreatEvent — the triggering threat.
    timestamp_us : int — time of dispatch in microseconds.
    latency_ms   : float — measured dispatch latency.
    success      : bool — whether hardware confirmed the command.
    """
    command:      ActuationCommand
    threat:       ThreatEvent
    timestamp_us: int
    latency_ms:   float = 0.0
    success:      bool  = False


# ─────────────────────────────────────────────────────────────
# Threat → Command mapping
# ─────────────────────────────────────────────────────────────

def _threat_to_command(threat: ThreatEvent) -> ActuationCommand:
    """
    Map a ThreatLevel to the appropriate actuation command.

    The camera position (front vs rear) influences the command type:
    - Side/rear EMERGENCY threats may prefer EVASIVE_STEER over full brake.
    """
    level      = threat.threat_level
    cam_name   = threat.camera_name

    if level == ThreatLevel.EMERGENCY:
        # Rear collision → pre-brace but steer available
        if "BACK" in cam_name:
            return ActuationCommand.EMERGENCY_BRAKE
        return ActuationCommand.EMERGENCY_BRAKE

    elif level == ThreatLevel.PARTIAL:
        return ActuationCommand.PARTIAL_BRAKE

    elif level == ThreatLevel.PRE_FILL:
        return ActuationCommand.BRAKE_PRE_FILL

    elif level == ThreatLevel.WARNING:
        return ActuationCommand.WARNING_ALERT

    return ActuationCommand.NONE


# ─────────────────────────────────────────────────────────────
# PhysicalReflex
# ─────────────────────────────────────────────────────────────

class PhysicalReflex:
    """
    Physical reflex actuator that dispatches braking / steering commands.

    Parameters
    ----------
    log_dir          : str — directory to write actuation event logs.
    sim_latency_ms   : float — simulated hardware round-trip latency
                       (development/testing only). Set to 0 for production.
    on_actuate       : Optional callable — hook for testing or custom
                       platform integration. Signature:
                         on_actuate(command: ActuationCommand, threat: ThreatEvent) -> bool
    dedupe_window_ms : float — suppress duplicate commands for the same
                       camera/class within this window (avoids command flooding).

    Example
    -------
    >>> actuator = PhysicalReflex(log_dir="logs/actuation")
    >>> actuator.trigger(threat_event)
    """

    def __init__(
        self,
        log_dir:          str   = "logs/actuation",
        sim_latency_ms:   float = 2.0,
        on_actuate:       Optional[Callable] = None,
        dedupe_window_ms: float = 200.0,
    ) -> None:
        self.log_dir          = log_dir
        self.sim_latency_ms   = sim_latency_ms
        self.on_actuate       = on_actuate
        self.dedupe_window_ms = dedupe_window_ms

        os.makedirs(log_dir, exist_ok=True)

        self._log:        List[Dict]   = []
        self._last_cmd:   Dict[str, float] = {}   # key → last dispatch timestamp_ms
        self._lock:       threading.Lock   = threading.Lock()

        logger.info(
            "PhysicalReflex ready. log_dir=%s sim_latency=%.1fms",
            log_dir, sim_latency_ms
        )

    def trigger(self, threat: ThreatEvent) -> Optional[ActuationEvent]:
        """
        Fire-and-forget actuation trigger.

        Returns None immediately if the threat is deduplicated.
        Spawns a daemon thread to execute the command without blocking
        the inference pipeline.

        Parameters
        ----------
        threat : ThreatEvent — the triggering threat.

        Returns
        -------
        ActuationEvent or None (if deduplicated / no-op command).
        """
        command = _threat_to_command(threat)

        if command == ActuationCommand.NONE:
            return None

        # ── Deduplication ─────────────────────────────────────────
        dedup_key = f"{threat.camera_name}_{threat.box.class_name}"
        now_ms    = time.time() * 1000.0

        with self._lock:
            last_ms = self._last_cmd.get(dedup_key, 0.0)
            if now_ms - last_ms < self.dedupe_window_ms:
                logger.debug(
                    "Deduplicating actuation: %s (%.0fms since last)",
                    dedup_key, now_ms - last_ms
                )
                return None
            self._last_cmd[dedup_key] = now_ms

        # ── Construct event ───────────────────────────────────────
        event = ActuationEvent(
            command      = command,
            threat       = threat,
            timestamp_us = int(time.time() * 1e6),
        )

        # ── Fire-and-forget on daemon thread ──────────────────────
        t = threading.Thread(
            target  = self._execute,
            args    = (event,),
            daemon  = True,
            name    = f"reflex_{command.value}",
        )
        t.start()

        logger.info(
            "⚡ ACTUATION DISPATCHED: %s | cam=%s TTC=%.2fs",
            command.value.upper(),
            threat.camera_name,
            threat.ttc_s,
        )

        return event

    def trigger_many(self, threats: List[ThreatEvent]) -> List[ActuationEvent]:
        """
        Trigger actuation for the most critical threat in a list.
        Only the single highest-severity threat is actuated per call
        to avoid conflicting commands.

        Parameters
        ----------
        threats : List[ThreatEvent] — sorted by TTC (most critical first).

        Returns
        -------
        List of dispatched ActuationEvents (typically 0 or 1).
        """
        if not threats:
            return []

        # Threats are pre-sorted by TTC — take the most critical
        most_critical = threats[0]
        event = self.trigger(most_critical)
        return [event] if event else []

    def _execute(self, event: ActuationEvent) -> None:
        """
        Internal: execute the actuation command.
        Called on a daemon thread — never on the main pipeline thread.
        """
        t0 = time.perf_counter()

        # ── Platform hook (production override) ───────────────────
        if self.on_actuate is not None:
            try:
                success = self.on_actuate(event.command, event.threat)
            except Exception as e:
                logger.error("on_actuate hook raised: %s", e)
                success = False
        else:
            # ── Simulation: log and sleep for latency model ────────
            success = self._simulate_actuate(event)

        t1 = time.perf_counter()
        event.latency_ms = (t1 - t0) * 1000.0
        event.success    = success

        # ── Write log entry ───────────────────────────────────────
        self._log_event(event)

    def _simulate_actuate(self, event: ActuationEvent) -> bool:
        """
        Simulation-mode actuator: print the command and pause.
        """
        print(
            f"\n{'='*60}\n"
            f"  🚨 THINK FAST — PHYSICAL REFLEX TRIGGERED\n"
            f"  Command     : {event.command.value.upper()}\n"
            f"  Camera      : {event.threat.camera_name}\n"
            f"  Object      : {event.threat.box.class_name} "
            f"(conf={event.threat.box.score:.2f})\n"
            f"  TTC         : {event.threat.ttc_s:.3f} s\n"
            f"  Distance    : {event.threat.distance_m:.1f} m\n"
            f"  Closing vel : {event.threat.velocity_ms:.1f} m/s\n"
            f"  Threat level: {event.threat.threat_level}\n"
            f"{'='*60}\n"
        )

        if self.sim_latency_ms > 0:
            time.sleep(self.sim_latency_ms / 1000.0)

        return True

    def _log_event(self, event: ActuationEvent) -> None:
        """Append actuation event to the structured JSON log."""
        entry = {
            "command":      event.command.value,
            "camera":       event.threat.camera_name,
            "class":        event.threat.box.class_name,
            "ttc_s":        round(event.threat.ttc_s, 4),
            "distance_m":   round(event.threat.distance_m, 2),
            "threat_level": str(event.threat.threat_level),
            "timestamp_us": event.timestamp_us,
            "latency_ms":   round(event.latency_ms, 2),
            "success":      event.success,
        }

        with self._lock:
            self._log.append(entry)

        log_path = os.path.join(self.log_dir, "actuation_log.jsonl")
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("Failed to write actuation log: %s", e)
