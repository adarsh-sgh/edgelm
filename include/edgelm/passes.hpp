// Graph passes: op fusion and the liveness-based activation memory planner.
#pragma once
#include <string>
#include <vector>

#include "edgelm/graph.hpp"

namespace edgelm {

// Each returns the number of fusions applied. Fused ops take the position of the op that
// consumed the last fused result, so the op list stays a valid topological order.
int fuse_swiglu(Graph& g);          // MATMUL(gate), MATMUL(up), SILU, MUL -> FFN_SWIGLU
int fuse_shared_input(Graph& g);    // MATMULs reading the same tensor -> MATMUL_N (QKV)
int fuse_residual(Graph& g);        // MATMUL then ADD(r, .) -> MATMUL_ADD
int fuse_rope_attention(Graph& g);  // ROPE(q), ROPE(k) -> ATTENTION(fused_rope=1)

struct FusionStats {
  int swiglu = 0, shared_input = 0, residual = 0, rope = 0;
  size_t ops_before = 0, ops_after = 0;
  std::string str() const;
};
FusionStats optimize(Graph& g);

struct MemoryPlan {
  std::vector<int64_t> offset;  // per tensor, -1 = not in arena (weights, unused)
  std::vector<int64_t> bytes;   // per tensor planned size
  std::vector<int> first, last; // live range in op indices (inclusive)
  size_t arena_bytes = 0;       // packed arena
  size_t naive_bytes = 0;       // one buffer per activation
  int n_planned = 0;
};

// Plans every non-weight tensor for T tokens per step and L logit rows.
MemoryPlan plan_memory(const Graph& g, int T, int L);
// True if no two tensors with overlapping live ranges overlap in the arena.
bool plan_is_valid(const MemoryPlan& p);

size_t activation_bytes(const Tensor& t, int T, int L);

}  // namespace edgelm
