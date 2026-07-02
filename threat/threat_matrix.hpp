/**
 * think_fast/threat/threat_matrix.hpp
 * ======================================
 * C++17 Threat Matrix Evaluator — header-only interface.
 *
 * Self-contained: no external dependencies beyond the C++ standard
 * library. Designed to be dropped into any ROS2 node, bare-metal
 * embedded system, or CAN-bus interrupt handler.
 *
 * Mirrors the Python threat_matrix.py in behaviour.
 *
 * Usage
 * -----
 *   #include "threat_matrix.hpp"
 *
 *   think_fast::ThreatMatrixEvaluator evaluator;
 *   evaluator.set_ego_speed(15.0f);
 *
 *   std::vector<think_fast::DetectedObject> objects = { ... };
 *   auto threats = evaluator.evaluate(objects);
 *
 *   for (const auto& t : threats) {
 *       if (t.level >= think_fast::ThreatLevel::EMERGENCY) {
 *           trigger_emergency_brake();
 *       }
 *   }
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace think_fast {

// ─────────────────────────────────────────────────────────────────────
// ThreatLevel
// ─────────────────────────────────────────────────────────────────────

/**
 * Ordered severity levels matching the Python ThreatLevel enum.
 * Higher numeric value = higher severity.
 */
enum class ThreatLevel : uint8_t {
    SAFE      = 0,  ///< TTC > 3.0 s  — no action
    WARNING   = 1,  ///< 2.0–3.0 s   — dashboard alert
    PRE_FILL  = 2,  ///< 1.5–2.0 s   — brake pre-fill
    PARTIAL   = 3,  ///< 1.0–1.5 s   — partial braking
    EMERGENCY = 4,  ///< < 1.0 s     — full emergency brake
};

/**
 * Convert ThreatLevel to a human-readable string.
 */
inline const char* threat_level_str(ThreatLevel level) {
    switch (level) {
        case ThreatLevel::SAFE:      return "SAFE";
        case ThreatLevel::WARNING:   return "WARNING";
        case ThreatLevel::PRE_FILL:  return "PRE_FILL";
        case ThreatLevel::PARTIAL:   return "PARTIAL";
        case ThreatLevel::EMERGENCY: return "EMERGENCY";
        default:                     return "UNKNOWN";
    }
}

// ─────────────────────────────────────────────────────────────────────
// BoundingBox
// ─────────────────────────────────────────────────────────────────────

/** 2D bounding box in pixel space [0, 640]. */
struct BoundingBox {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float cx() const { return (x1 + x2) * 0.5f; }
    float cy() const { return (y1 + y2) * 0.5f; }
};

// ─────────────────────────────────────────────────────────────────────
// DetectedObject
// ─────────────────────────────────────────────────────────────────────

/**
 * A single detected object from the YOLO batch inference output.
 * Populated by the C++ inference wrapper (TensorRT binding).
 */
struct DetectedObject {
    int         camera_idx  = 0;       ///< Camera index [0,5]
    int         class_id    = 0;       ///< Class index
    std::string class_name;            ///< Class label (e.g. "car")
    float       confidence  = 0.f;
    BoundingBox bbox;
    float       depth_m     = 0.f;    ///< Radar range [metres]. 0 = no measurement.
    float       velocity_ms = 0.f;    ///< Radial velocity [m/s]. Neg = approaching.
    int64_t     timestamp_us = 0;     ///< UNIX microseconds
};

// ─────────────────────────────────────────────────────────────────────
// ThreatEvent
// ─────────────────────────────────────────────────────────────────────

/**
 * A collision threat derived from a DetectedObject.
 */
struct ThreatEvent {
    DetectedObject object;
    float          ttc_s        = 0.f;   ///< Time-to-Collision [seconds]
    ThreatLevel    level        = ThreatLevel::SAFE;
    float          distance_m   = 0.f;
    float          closing_vel  = 0.f;   ///< Closing speed [m/s], positive = approaching
    int64_t        timestamp_us = 0;
};

// ─────────────────────────────────────────────────────────────────────
// Kinematics
// ─────────────────────────────────────────────────────────────────────

/**
 * Compute simple TTC: TTC = D / |Vr|
 *
 * @param distance_m     Range to target [m].
 * @param closing_vel_ms Closing speed [m/s]. Negative = approaching.
 * @return TTC in seconds. +inf if not approaching.
 */
inline float compute_ttc_simple(float distance_m, float closing_vel_ms) {
    constexpr float EPS = 1e-6f;
    constexpr float INF = std::numeric_limits<float>::infinity();

    // Positive closing_vel means approaching (sign convention for C++ version)
    if (closing_vel_ms <= EPS) return INF;    // Not closing
    if (distance_m     <= 0.f) return 0.f;   // Already in contact

    return distance_m / closing_vel_ms;
}

/**
 * Compute Enhanced TTC (ETTC) with relative acceleration.
 *
 *   ETTC = [-Vr - sqrt(Vr² - 2·Ar·D)] / Ar
 *
 * Falls back to simple TTC when |Ar| < 0.01 m/s².
 *
 * @param distance_m    Range [m].
 * @param closing_vel   Closing speed [m/s], positive = approaching.
 * @param rel_accel     Relative acceleration [m/s²], positive = closing faster.
 * @return ETTC in seconds. +inf if not a threat.
 */
inline float compute_ettc(float distance_m, float closing_vel, float rel_accel) {
    constexpr float EPS = 1e-2f;
    constexpr float INF = std::numeric_limits<float>::infinity();

    if (std::abs(rel_accel) < EPS) {
        return compute_ttc_simple(distance_m, closing_vel);
    }

    float discriminant = closing_vel * closing_vel - 2.0f * rel_accel * distance_m;
    if (discriminant < 0.f) return INF;   // Paths don't converge

    float ttc = (-closing_vel - std::sqrt(discriminant)) / rel_accel;
    return (ttc > 0.f) ? ttc : INF;
}

/**
 * Map a TTC value to its ThreatLevel.
 */
inline ThreatLevel ttc_to_level(float ttc) {
    if (ttc < 1.0f) return ThreatLevel::EMERGENCY;
    if (ttc < 1.5f) return ThreatLevel::PARTIAL;
    if (ttc < 2.0f) return ThreatLevel::PRE_FILL;
    if (ttc < 3.0f) return ThreatLevel::WARNING;
    return ThreatLevel::SAFE;
}

// ─────────────────────────────────────────────────────────────────────
// ThreatMatrixConfig
// ─────────────────────────────────────────────────────────────────────

/** Configuration for ThreatMatrixEvaluator. */
struct ThreatMatrixConfig {
    ThreatLevel min_threat_level = ThreatLevel::WARNING;
    float min_distance_m         = 0.5f;   ///< Ignore objects closer than this [m]
    float max_distance_m         = 80.0f;  ///< Ignore objects farther than this [m]
    float ego_speed_ms           = 0.0f;   ///< Ego vehicle speed [m/s]
};

// ─────────────────────────────────────────────────────────────────────
// ThreatMatrixEvaluator
// ─────────────────────────────────────────────────────────────────────

/**
 * Evaluates TTC for all detected objects and returns critical ThreatEvents.
 *
 * Thread-safety: NOT thread-safe by default (call from a single thread).
 * To use from multiple threads, guard with a mutex.
 */
class ThreatMatrixEvaluator {
public:
    explicit ThreatMatrixEvaluator(
        ThreatMatrixConfig cfg = {}
    ) : m_cfg(cfg) {}

    /** Update ego-vehicle speed (call each frame before evaluate). */
    void set_ego_speed(float speed_ms) { m_cfg.ego_speed_ms = speed_ms; }

    /** Set minimum threat level to report. */
    void set_min_level(ThreatLevel level) { m_cfg.min_threat_level = level; }

    /**
     * Evaluate a batch of detected objects.
     *
     * @param objects   Objects from all 6 cameras (combined).
     * @param timestamp_us Current timestamp in microseconds.
     * @return Threats sorted by TTC ascending (most critical first).
     */
    std::vector<ThreatEvent> evaluate(
        const std::vector<DetectedObject>& objects,
        int64_t timestamp_us = 0
    ) {
        std::vector<ThreatEvent> threats;
        threats.reserve(objects.size());

        for (const auto& obj : objects) {
            auto maybe = evaluate_object(obj, timestamp_us);
            if (maybe.has_value()) {
                threats.push_back(std::move(maybe.value()));
            }
        }

        // Sort by TTC ascending
        std::sort(threats.begin(), threats.end(),
            [](const ThreatEvent& a, const ThreatEvent& b) {
                return a.ttc_s < b.ttc_s;
            }
        );

        return threats;
    }

    /**
     * Evaluate a single object. Returns nullopt if not a threat.
     */
    std::optional<ThreatEvent> evaluate_object(
        const DetectedObject& obj,
        int64_t timestamp_us
    ) {
        float depth_m = obj.depth_m;
        float vel_ms  = obj.velocity_ms;  // negative = approaching (nuScenes convention)

        // ── Filter: invalid or out-of-range ───────────────────────
        if (depth_m <= 0.f ||
            depth_m < m_cfg.min_distance_m ||
            depth_m > m_cfg.max_distance_m) {
            return std::nullopt;
        }

        // ── Convert to closing speed (positive = approaching) ─────
        float closing_vel = -vel_ms;    // flip sign: neg velocity → approaching

        // ── Compute TTC ───────────────────────────────────────────
        float ttc = compute_ttc_simple(depth_m, closing_vel);

        // ── Classify ──────────────────────────────────────────────
        ThreatLevel level = ttc_to_level(ttc);

        if (static_cast<uint8_t>(level) <
            static_cast<uint8_t>(m_cfg.min_threat_level)) {
            return std::nullopt;
        }

        ThreatEvent evt;
        evt.object       = obj;
        evt.ttc_s        = ttc;
        evt.level        = level;
        evt.distance_m   = depth_m;
        evt.closing_vel  = closing_vel;
        evt.timestamp_us = timestamp_us;

        return evt;
    }

private:
    ThreatMatrixConfig m_cfg;
};

}  // namespace think_fast
