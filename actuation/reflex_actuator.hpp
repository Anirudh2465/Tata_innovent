/**
 * think_fast/actuation/reflex_actuator.hpp
 * ==========================================
 * C++ abstract interface for the physical reflex actuator.
 *
 * Provides:
 *   1. IReflexActuator — pure abstract base class.
 *      Override in your platform-specific implementation
 *      (ROS2 topic, CAN bus, GPIO, hardware interrupt).
 *
 *   2. SimulatedReflex — concrete stub that prints to stdout.
 *      Used in development / CI, or as a reference implementation.
 *
 *   3. ReflexDispatcher — thread-safe wrapper that runs trigger()
 *      on a dedicated std::thread, never blocking the inference loop.
 *
 * Design Principle
 * ----------------
 * trigger() MUST return in < 100 µs. The actual hardware command
 * is dispatched on a dedicated thread via a lock-free command queue.
 *
 * Usage
 * -----
 *   // Production
 *   class MyCANActuator : public think_fast::IReflexActuator { ... };
 *   auto actuator = std::make_shared<MyCANActuator>();
 *   think_fast::ReflexDispatcher dispatcher(actuator);
 *   dispatcher.trigger(threat_event);
 *
 *   // Simulation
 *   auto sim = std::make_shared<think_fast::SimulatedReflex>();
 *   think_fast::ReflexDispatcher dispatcher(sim);
 *   dispatcher.trigger(threat_event);
 */

#pragma once

#include "threat_matrix.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

namespace think_fast {

// ─────────────────────────────────────────────────────────────────────
// Actuation Commands
// ─────────────────────────────────────────────────────────────────────

enum class ActuationCommand : uint8_t {
    NONE            = 0,
    WARNING_ALERT   = 1,   ///< Dashboard warning only
    BRAKE_PRE_FILL  = 2,   ///< Prepare hydraulic braking
    PARTIAL_BRAKE   = 3,   ///< ~40% braking force
    EMERGENCY_BRAKE = 4,   ///< 100% Autonomous Emergency Braking (AEB)
    EVASIVE_STEER   = 5,   ///< Lateral avoidance
};

inline const char* actuation_command_str(ActuationCommand cmd) {
    switch (cmd) {
        case ActuationCommand::NONE:            return "NONE";
        case ActuationCommand::WARNING_ALERT:   return "WARNING_ALERT";
        case ActuationCommand::BRAKE_PRE_FILL:  return "BRAKE_PRE_FILL";
        case ActuationCommand::PARTIAL_BRAKE:   return "PARTIAL_BRAKE";
        case ActuationCommand::EMERGENCY_BRAKE: return "EMERGENCY_BRAKE";
        case ActuationCommand::EVASIVE_STEER:   return "EVASIVE_STEER";
        default:                                return "UNKNOWN";
    }
}

/** Map a ThreatEvent to an ActuationCommand. */
inline ActuationCommand threat_to_command(const ThreatEvent& threat) {
    switch (threat.level) {
        case ThreatLevel::EMERGENCY: return ActuationCommand::EMERGENCY_BRAKE;
        case ThreatLevel::PARTIAL:   return ActuationCommand::PARTIAL_BRAKE;
        case ThreatLevel::PRE_FILL:  return ActuationCommand::BRAKE_PRE_FILL;
        case ThreatLevel::WARNING:   return ActuationCommand::WARNING_ALERT;
        default:                     return ActuationCommand::NONE;
    }
}

// ─────────────────────────────────────────────────────────────────────
// IReflexActuator — pure abstract interface
// ─────────────────────────────────────────────────────────────────────

/**
 * Pure abstract base class for platform-specific reflex actuators.
 *
 * Derive from this class and implement:
 *   - execute(command, threat) → bool
 *
 * The execute() method is called on a dedicated dispatcher thread
 * and is allowed to block (e.g., for a CAN ACK or GPIO confirmation).
 * It must NOT hold locks that the main pipeline thread also acquires.
 */
class IReflexActuator {
public:
    virtual ~IReflexActuator() = default;

    /**
     * Execute a physical actuation command.
     *
     * @param command  The command to execute.
     * @param threat   The triggering threat event (for context/logging).
     * @return true if the command was successfully dispatched to hardware.
     */
    virtual bool execute(ActuationCommand command, const ThreatEvent& threat) = 0;

    /** Optional: called once during system startup to initialise hardware. */
    virtual bool init()     { return true; }

    /** Optional: called on system shutdown. */
    virtual void shutdown() {}

    /** Human-readable name of this actuator (for logging). */
    virtual std::string name() const { return "IReflexActuator"; }
};


// ─────────────────────────────────────────────────────────────────────
// SimulatedReflex — stdout stub for development
// ─────────────────────────────────────────────────────────────────────

/**
 * Development/simulation actuator. Prints command info to stdout.
 * Safe to use in any environment without hardware dependencies.
 */
class SimulatedReflex : public IReflexActuator {
public:
    explicit SimulatedReflex(float sim_latency_ms = 1.0f)
        : m_latency_ms(sim_latency_ms) {}

    std::string name() const override { return "SimulatedReflex"; }

    bool execute(ActuationCommand command, const ThreatEvent& threat) override {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        auto ts  = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

        std::cout
            << "\n" << std::string(60, '=') << "\n"
            << "  THINK FAST — PHYSICAL REFLEX TRIGGERED\n"
            << "  Timestamp   : " << ts << " ms\n"
            << "  Command     : " << actuation_command_str(command) << "\n"
            << "  Camera      : cam_idx=" << threat.object.camera_idx << "\n"
            << "  Object      : " << threat.object.class_name
                                  << " (conf=" << threat.object.confidence << ")\n"
            << "  TTC         : " << threat.ttc_s << " s\n"
            << "  Distance    : " << threat.distance_m << " m\n"
            << "  Closing vel : " << threat.closing_vel << " m/s\n"
            << "  Level       : " << threat_level_str(threat.level) << "\n"
            << std::string(60, '=') << "\n\n";

        if (m_latency_ms > 0.f) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(static_cast<int>(m_latency_ms * 1000.f))
            );
        }

        return true;
    }

private:
    float m_latency_ms;
};


// ─────────────────────────────────────────────────────────────────────
// ReflexDispatcher — non-blocking, thread-safe dispatch wrapper
// ─────────────────────────────────────────────────────────────────────

/**
 * Thread-safe dispatcher that queues ThreatEvents and executes them
 * on a dedicated background thread, ensuring the main pipeline is
 * never blocked.
 *
 * Uses a bounded command queue (max_queue_size) to prevent memory
 * growth under sustained high-frequency threats.
 *
 * Usage
 * -----
 *   auto sim = std::make_shared<SimulatedReflex>();
 *   ReflexDispatcher dispatcher(sim);
 *   dispatcher.start();
 *
 *   // From the main pipeline loop:
 *   dispatcher.trigger(threat);
 *
 *   // On shutdown:
 *   dispatcher.stop();
 */
class ReflexDispatcher {
public:
    explicit ReflexDispatcher(
        std::shared_ptr<IReflexActuator> actuator,
        size_t max_queue_size = 8
    ) : m_actuator(std::move(actuator)),
        m_max_queue(max_queue_size),
        m_running(false) {}

    ~ReflexDispatcher() { stop(); }

    /** Initialise the actuator and start the background dispatch thread. */
    bool start() {
        if (!m_actuator->init()) {
            std::cerr << "[ReflexDispatcher] Actuator init() failed.\n";
            return false;
        }
        m_running = true;
        m_thread  = std::thread(&ReflexDispatcher::_worker, this);
        return true;
    }

    /** Stop the dispatch thread gracefully. */
    void stop() {
        m_running = false;
        m_cv.notify_all();
        if (m_thread.joinable()) m_thread.join();
        m_actuator->shutdown();
    }

    /**
     * Queue a ThreatEvent for actuation (non-blocking).
     *
     * @return true if queued. false if queue is full (threat dropped).
     */
    bool trigger(const ThreatEvent& threat) {
        ActuationCommand cmd = threat_to_command(threat);
        if (cmd == ActuationCommand::NONE) return false;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queue.size() >= m_max_queue) {
                // Drop lowest-severity command to make room
                return false;
            }
            m_queue.push({ cmd, threat });
        }
        m_cv.notify_one();
        return true;
    }

private:
    struct QueueEntry {
        ActuationCommand command;
        ThreatEvent      threat;
    };

    void _worker() {
        while (m_running) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] {
                return !m_queue.empty() || !m_running;
            });

            while (!m_queue.empty()) {
                auto entry = m_queue.front();
                m_queue.pop();
                lock.unlock();

                m_actuator->execute(entry.command, entry.threat);

                lock.lock();
            }
        }
    }

    std::shared_ptr<IReflexActuator> m_actuator;
    std::queue<QueueEntry>           m_queue;
    std::mutex                       m_mutex;
    std::condition_variable          m_cv;
    std::thread                      m_thread;
    std::atomic<bool>                m_running;
    size_t                           m_max_queue;
};

}  // namespace think_fast
