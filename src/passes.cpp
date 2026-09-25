#include "edgelm/passes.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace edgelm {
namespace {

std::vector<int> producers(const Graph& g) {
  std::vector<int> p(g.tensors.size(), -1);
  for (size_t i = 0; i < g.ops.size(); ++i)
    for (int t : g.ops[i].out) p[t] = static_cast<int>(i);
  return p;
}

void compact(Graph& g, const std::vector<bool>& dead) {
  std::vector<Op> ops;
  for (size_t i = 0; i < g.ops.size(); ++i)
    if (!dead[i]) ops.push_back(std::move(g.ops[i]));
  g.ops = std::move(ops);
}

bool only_consumer(const std::vector<std::vector<int>>& cons, int tensor, int op) {
  return cons[tensor].size() == 1 && cons[tensor][0] == op;
}

}  // namespace

int fuse_swiglu(Graph& g) {
  auto prod = producers(g);
  auto cons = g.consumers();
  std::vector<bool> dead(g.ops.size(), false);
  int n = 0;
  for (size_t i = 0; i < g.ops.size(); ++i) {
    Op& mul = g.ops[i];
    if (mul.type != OpType::MUL) continue;
    for (int s = 0; s < 2; ++s) {
      const int a = mul.in[s], b = mul.in[1 - s];
      const int si = prod[a], ui = prod[b];
      if (si < 0 || ui < 0 || g.ops[si].type != OpType::SILU || g.ops[ui].type != OpType::MATMUL) continue;
      const int gi = prod[g.ops[si].in[0]];
      if (gi < 0 || g.ops[gi].type != OpType::MATMUL || g.ops[gi].in[0] != g.ops[ui].in[0]) continue;
      if (!only_consumer(cons, a, int(i)) || !only_consumer(cons, b, int(i)) ||
          !only_consumer(cons, g.ops[si].in[0], si) || dead[si] || dead[ui] || dead[gi])
        continue;
      Op f;
      f.type = OpType::FFN_SWIGLU;
      f.name = mul.name + "+swiglu";
      f.in = {g.ops[gi].in[0], g.ops[gi].in[1], g.ops[ui].in[1]};
      f.out = mul.out;
      dead[si] = dead[ui] = dead[gi] = true;
      mul = f;
      ++n;
      break;
    }
  }
  compact(g, dead);
  return n;
}

int fuse_shared_input(Graph& g) {
  std::vector<bool> dead(g.ops.size(), false);
  int n = 0;
  for (size_t i = 0; i < g.ops.size(); ++i) {
    if (dead[i] || g.ops[i].type != OpType::MATMUL) continue;
    const int x = g.ops[i].in[0];
    std::vector<size_t> group{i};
    for (size_t j = i + 1; j < g.ops.size(); ++j)
      if (!dead[j] && g.ops[j].type == OpType::MATMUL && g.ops[j].in[0] == x) group.push_back(j);
    if (group.size() < 2) continue;
    Op f;
    f.type = OpType::MATMUL_N;
    f.in = {x};
    for (size_t j : group) {
      f.in.push_back(g.ops[j].in[1]);
      f.out.push_back(g.ops[j].out[0]);
      f.name += (f.name.empty() ? "" : "+") + g.ops[j].name;
      if (j != i) dead[j] = true;
    }
    g.ops[i] = f;  // earliest position: every output is defined before any of its readers
    ++n;
  }
  compact(g, dead);
  return n;
}

int fuse_residual(Graph& g) {
  auto prod = producers(g);
  auto cons = g.consumers();
  std::vector<bool> dead(g.ops.size(), false);
  int n = 0;
  for (size_t i = 0; i < g.ops.size(); ++i) {
    Op& add = g.ops[i];
    if (add.type != OpType::ADD) continue;
    for (int s = 0; s < 2; ++s) {
      const int y = add.in[s], r = add.in[1 - s];
      const int mi = prod[y];
      if (mi < 0 || dead[mi] || g.ops[mi].type != OpType::MATMUL || !only_consumer(cons, y, int(i))) continue;
      Op f;
      f.type = OpType::MATMUL_ADD;
      f.name = g.ops[mi].name + "+residual";
      f.in = {g.ops[mi].in[0], g.ops[mi].in[1], r};
      f.out = add.out;
      dead[mi] = true;
      add = f;
      ++n;
      break;
    }
  }
  compact(g, dead);
  return n;
}

int fuse_rope_attention(Graph& g) {
  auto prod = producers(g);
  auto cons = g.consumers();
  std::vector<bool> dead(g.ops.size(), false);
  int n = 0;
  for (size_t i = 0; i < g.ops.size(); ++i) {
    Op& at = g.ops[i];
    if (at.type != OpType::ATTENTION || at.iattr[3]) continue;
    const int qi = prod[at.in[0]], ki = prod[at.in[1]];
    if (qi < 0 || ki < 0 || g.ops[qi].type != OpType::ROPE || g.ops[ki].type != OpType::ROPE) continue;
    if (!only_consumer(cons, at.in[0], int(i)) || !only_consumer(cons, at.in[1], int(i))) continue;
    at.in[0] = g.ops[qi].in[0];
    at.in[1] = g.ops[ki].in[0];
    at.iattr[3] = 1;
    at.name += "+rope";
    dead[qi] = dead[ki] = true;
    ++n;
  }
  compact(g, dead);
  return n;
}

std::string FusionStats::str() const {
  std::ostringstream os;
  os << ops_before << " -> " << ops_after << " ops (swiglu " << swiglu << ", qkv/shared-input " << shared_input
     << ", residual " << residual << ", rope->attention " << rope << ")";
  return os.str();
}

FusionStats optimize(Graph& g) {
  FusionStats s;
  s.ops_before = g.ops.size();
  s.swiglu = fuse_swiglu(g);  // before shared-input so gate/up become one SwiGLU, not a MATMUL_N
  s.shared_input = fuse_shared_input(g);
  s.residual = fuse_residual(g);
  s.rope = fuse_rope_attention(g);
  s.ops_after = g.ops.size();
  return s;
}

size_t activation_bytes(const Tensor& t, int T, int L) {
  const int64_t rows = t.rows_sym == RowSym::T ? T : t.rows_sym == RowSym::L ? L : t.rows;
  return static_cast<size_t>(rows * t.cols) * 4;  // f32 / i32
}

MemoryPlan plan_memory(const Graph& g, int T, int L) {
  constexpr int64_t kAlign = 64;
  const size_t nt = g.tensors.size();
  const int nops = static_cast<int>(g.ops.size());
  MemoryPlan p;
  p.offset.assign(nt, -1);
  p.bytes.assign(nt, 0);
  p.first.assign(nt, -1);
  p.last.assign(nt, -1);
  for (int i = 0; i < nops; ++i) {
    for (int t : g.ops[i].out)
      if (p.first[t] < 0) p.first[t] = i;
    for (int t : g.ops[i].in) p.last[t] = std::max(p.last[t], i);
  }
  std::vector<int> order;
  for (size_t t = 0; t < nt; ++t) {
    const Tensor& ts = g.tensors[t];
    if (ts.is_weight()) continue;
    if (ts.kind == TensorKind::Input) p.first[t] = 0;
    if (ts.kind == TensorKind::Output) p.last[t] = nops;  // read by the caller after the run
    if (p.first[t] < 0 || p.last[t] < 0) continue;       // dead after fusion
    p.bytes[t] = (static_cast<int64_t>(activation_bytes(ts, T, L)) + kAlign - 1) / kAlign * kAlign;
    p.naive_bytes += p.bytes[t];
    order.push_back(static_cast<int>(t));
  }
  // Greedy by size (TFLite's arena planner strategy): biggest first, best-fitting gap among
  // tensors whose live ranges overlap.
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p.bytes[a] > p.bytes[b]; });
  std::vector<int> placed;
  for (int t : order) {
    std::vector<std::pair<int64_t, int64_t>> busy;
    for (int o : placed)
      if (p.first[o] <= p.last[t] && p.first[t] <= p.last[o]) busy.push_back({p.offset[o], p.offset[o] + p.bytes[o]});
    std::sort(busy.begin(), busy.end());
    int64_t best = -1, best_gap = INT64_MAX, cur = 0;
    for (auto& [b, e] : busy) {
      if (b - cur >= p.bytes[t] && b - cur < best_gap) {
        best = cur;
        best_gap = b - cur;
      }
      cur = std::max(cur, e);
    }
    p.offset[t] = best >= 0 ? best : cur;
    p.arena_bytes = std::max<size_t>(p.arena_bytes, p.offset[t] + p.bytes[t]);
    placed.push_back(t);
    ++p.n_planned;
  }
  return p;
}

bool plan_is_valid(const MemoryPlan& p) {
  const size_t n = p.offset.size();
  for (size_t a = 0; a < n; ++a) {
    if (p.offset[a] < 0) continue;
    if (p.offset[a] % 64 || p.offset[a] + p.bytes[a] > static_cast<int64_t>(p.arena_bytes)) return false;
    for (size_t b = a + 1; b < n; ++b) {
      if (p.offset[b] < 0) continue;
      const bool live = p.first[a] <= p.last[b] && p.first[b] <= p.last[a];
      const bool mem = p.offset[a] < p.offset[b] + p.bytes[b] && p.offset[b] < p.offset[a] + p.bytes[a];
      if (live && mem) return false;
    }
  }
  return true;
}

}  // namespace edgelm
