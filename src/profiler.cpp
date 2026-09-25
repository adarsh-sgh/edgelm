#include "edgelm/profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace edgelm {

namespace {
std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}
}  // namespace

bool Profiler::write_trace(const std::string& path) const {
  std::ofstream f(path);
  if (!f) return false;
  const int64_t t0 = events.empty() ? 0 : std::min_element(events.begin(), events.end(), [](auto& a, auto& b) {
                                            return a.ts_ns < b.ts_ns;
                                          })->ts_ns;
  f << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n";
  f << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":1,\"args\":{\"name\":\"edgelm main\"}}";
  char buf[64];
  for (const ProfEvent& e : events) {
    f << ",\n{\"name\":\"" << json_escape(e.name) << "\",\"cat\":\"" << e.phase << "\",\"ph\":\"X\",\"pid\":1,\"tid\":1";
    std::snprintf(buf, sizeof buf, ",\"ts\":%.3f,\"dur\":%.3f", (e.ts_ns - t0) / 1e3, e.dur_ns / 1e3);
    f << buf << ",\"args\":{\"T\":" << e.T << ",\"pos0\":" << e.pos0;
    if (e.backend) f << ",\"op\":\"" << op_name(e.type) << "\",\"backend\":\"" << e.backend << "\"";
    f << "}}";
  }
  f << "\n]}\n";
  return static_cast<bool>(f);
}

std::string Profiler::stats() const {
  struct Agg {
    int64_t ns = 0;
    int calls = 0;
  };
  std::map<std::string, std::map<std::string, Agg>> by_phase;  // phase -> "OP (backend)" -> agg
  std::map<std::string, Agg> phase_total;
  for (const ProfEvent& e : events) {
    if (!e.backend) {
      auto& a = phase_total[e.phase];
      a.ns += e.dur_ns;
      a.calls++;
      continue;
    }
    auto& a = by_phase[e.phase][std::string(op_name(e.type)) + " (" + e.backend + ")"];
    a.ns += e.dur_ns;
    a.calls++;
  }
  std::ostringstream os;
  char line[160];
  for (auto& [phase, ops] : by_phase) {
    int64_t sum = 0;
    for (auto& kv : ops) sum += kv.second.ns;
    const Agg pt = phase_total[phase];
    std::snprintf(line, sizeof line, "%s: %d run(s), %.2f ms wall, %.2f ms in ops\n", phase.c_str(), pt.calls,
                  pt.ns / 1e6, sum / 1e6);
    os << line;
    std::vector<std::pair<std::string, Agg>> v(ops.begin(), ops.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.ns > b.second.ns; });
    for (auto& [name, a] : v) {
      std::snprintf(line, sizeof line, "  %-28s %7d calls %10.3f ms %6.1f%%\n", name.c_str(), a.calls, a.ns / 1e6,
                    sum ? 100.0 * a.ns / sum : 0.0);
      os << line;
    }
  }
  return os.str();
}

}  // namespace edgelm
