// Per-op timing: Chrome trace-event JSON (open in ui.perfetto.dev / chrome://tracing) and a
// per-op-type breakdown table.
#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "edgelm/graph.hpp"

namespace edgelm {

struct ProfEvent {
  std::string name;
  std::string phase;    // "prefill" / "decode"
  const char* backend;  // nullptr for phase spans
  OpType type;
  int64_t ts_ns, dur_ns;
  int T, pos0;
};

class Profiler {
 public:
  bool enabled = false;
  std::vector<ProfEvent> events;

  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void clear() { events.clear(); }
  bool write_trace(const std::string& path) const;
  std::string stats() const;  // per phase: op type, calls, total ms, % of phase
};

}  // namespace edgelm
