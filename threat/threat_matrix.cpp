/**
 * think_fast/threat/threat_matrix.cpp
 * ======================================
 * C++17 Threat Matrix implementation file.
 *
 * Since threat_matrix.hpp is header-only for the core logic,
 * this .cpp file provides:
 *   1. A standalone CLI demo / test harness.
 *   2. A C-compatible extern "C" ABI for Python ctypes binding.
 *
 * Compile (standalone demo):
 *   g++ -std=c++17 -O2 -o threat_matrix_demo threat_matrix.cpp
 *
 * Compile (shared library for Python binding):
 *   g++ -std=c++17 -O2 -shared -fPIC \
 *       -o libthinkfast_threat.so threat_matrix.cpp
 */

#include "threat_matrix.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────
// C ABI: flat structs for extern "C" interface
// ─────────────────────────────────────────────────────────────────────

/**
 * Flat C-compatible representation of a detected object.
 * Used by the Python ctypes / cffi binding.
 */
extern "C" {

struct CThreatInput {
    int     camera_idx;
    int     class_id;
    float   depth_m;
    float   velocity_ms;   // negative = approaching
    float   x1, y1, x2, y2;
    float   confidence;
    int64_t timestamp_us;
};

struct CThreatOutput {
    int     camera_idx;
    float   ttc_s;
    int     threat_level;   // 0=SAFE..4=EMERGENCY
    float   distance_m;
    float   closing_vel;
    float   x1, y1, x2, y2;
    int     class_id;
    int64_t timestamp_us;
};

/**
 * Evaluate threats from a C array of CThreatInput.
 *
 * @param inputs        Pointer to array of CThreatInput.
 * @param n_inputs      Number of inputs.
 * @param outputs       Caller-allocated output array (at least n_inputs entries).
 * @param min_level     Minimum ThreatLevel to report (0–4).
 * @param ego_speed_ms  Ego vehicle speed in m/s.
 * @return Number of threats written to outputs.
 */
int think_fast_evaluate_threats(
    const CThreatInput* inputs,
    int                 n_inputs,
    CThreatOutput*      outputs,
    int                 min_level,
    float               ego_speed_ms
) {
    think_fast::ThreatMatrixConfig cfg;
    cfg.min_threat_level = static_cast<think_fast::ThreatLevel>(min_level);
    cfg.ego_speed_ms     = ego_speed_ms;

    think_fast::ThreatMatrixEvaluator evaluator(cfg);

    // Convert C inputs → C++ DetectedObject
    std::vector<think_fast::DetectedObject> objects;
    objects.reserve(static_cast<size_t>(n_inputs));

    for (int i = 0; i < n_inputs; ++i) {
        const auto& in = inputs[i];
        think_fast::DetectedObject obj;
        obj.camera_idx   = in.camera_idx;
        obj.class_id     = in.class_id;
        obj.depth_m      = in.depth_m;
        obj.velocity_ms  = in.velocity_ms;
        obj.confidence   = in.confidence;
        obj.timestamp_us = in.timestamp_us;
        obj.bbox         = { in.x1, in.y1, in.x2, in.y2 };
        objects.push_back(std::move(obj));
    }

    // Evaluate
    auto threats = evaluator.evaluate(objects, 0);

    // Write results
    int n_written = 0;
    for (const auto& t : threats) {
        if (n_written >= n_inputs) break;   // bounds guard

        auto& out       = outputs[n_written];
        out.camera_idx  = t.object.camera_idx;
        out.ttc_s       = t.ttc_s;
        out.threat_level= static_cast<int>(t.level);
        out.distance_m  = t.distance_m;
        out.closing_vel = t.closing_vel;
        out.x1          = t.object.bbox.x1;
        out.y1          = t.object.bbox.y1;
        out.x2          = t.object.bbox.x2;
        out.y2          = t.object.bbox.y2;
        out.class_id    = t.object.class_id;
        out.timestamp_us= t.timestamp_us;

        ++n_written;
    }

    return n_written;
}

}  // extern "C"


// ─────────────────────────────────────────────────────────────────────
// Standalone demo / unit test
// ─────────────────────────────────────────────────────────────────────

#ifdef THINK_FAST_STANDALONE_DEMO

int main() {
    using namespace think_fast;

    std::cout << "=== Think Fast — Threat Matrix Demo ===\n\n";

    // Simulate 3 detected objects across different cameras
    std::vector<DetectedObject> objects;

    // 1. Emergency: pedestrian 5m away, closing at 20 m/s
    {
        DetectedObject o;
        o.camera_idx   = 0;   // CAM_FRONT
        o.class_name   = "pedestrian";
        o.depth_m      = 5.0f;
        o.velocity_ms  = -20.0f;   // approaching
        o.confidence   = 0.91f;
        objects.push_back(o);
    }

    // 2. Warning: car 35m away, closing at 8 m/s → TTC ≈ 4.3 s
    {
        DetectedObject o;
        o.camera_idx   = 0;
        o.class_name   = "car";
        o.depth_m      = 35.0f;
        o.velocity_ms  = -8.0f;
        o.confidence   = 0.88f;
        objects.push_back(o);
    }

    // 3. PRE_FILL: truck 24m away, closing at 14 m/s → TTC ≈ 1.7 s
    {
        DetectedObject o;
        o.camera_idx   = 2;   // CAM_BACK_RIGHT
        o.class_name   = "truck";
        o.depth_m      = 24.0f;
        o.velocity_ms  = -14.0f;
        o.confidence   = 0.79f;
        objects.push_back(o);
    }

    // 4. Receding object (safe)
    {
        DetectedObject o;
        o.camera_idx   = 3;
        o.class_name   = "car";
        o.depth_m      = 15.0f;
        o.velocity_ms  = +5.0f;   // moving away
        o.confidence   = 0.65f;
        objects.push_back(o);
    }

    ThreatMatrixEvaluator evaluator;
    evaluator.set_ego_speed(13.9f);   // ~50 km/h
    evaluator.set_min_level(ThreatLevel::WARNING);

    auto threats = evaluator.evaluate(objects, 0);

    std::cout << "Detected objects : " << objects.size() << "\n";
    std::cout << "Threats reported : " << threats.size() << "\n\n";

    for (const auto& t : threats) {
        std::printf(
            "[%s] cam=%d  class=%s  TTC=%.2fs  dist=%.1fm  close_vel=%.1fm/s\n",
            threat_level_str(t.level),
            t.object.camera_idx,
            t.object.class_name.c_str(),
            t.ttc_s,
            t.distance_m,
            t.closing_vel
        );
    }

    // Verify EMERGENCY is first
    bool ok = (!threats.empty() &&
               threats[0].level == ThreatLevel::EMERGENCY &&
               threats[0].ttc_s < 1.0f);
    std::cout << "\nSelf-test: " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";

    return ok ? 0 : 1;
}

#endif   // THINK_FAST_STANDALONE_DEMO
