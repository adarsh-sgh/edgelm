// Session: an optimized, partitioned, memory-planned instance of a model's graph with a
// preallocated KV cache. prefill() runs the prompt as batched GEMMs (chunks of max_batch
// tokens); decode() runs one token as GEMVs against the cache.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "edgelm/backend.hpp"
#include "edgelm/model.hpp"
#include "edgelm/passes.hpp"
#include "edgelm/profiler.hpp"
#include "edgelm/thread_pool.hpp"

namespace edgelm {

struct SessionOptions {
  std::vector<std::string> backends{"cpu"};  // preference order; reference is always the fallback
  int threads = 1;
  int max_ctx = 1024;
  int max_batch = 128;      // tokens per forward (prefill chunk); plans the arena
  bool all_logits = false;  // logits for every position (eval) instead of the last one
  bool fuse = true;
  bool profile = false;
  bool act_quant = false;   // W8A8 / W4A8: int8 activations for q8/q4 matmuls (no effect on f32 weights)
};

class Session {
 public:
  Session(const Model& model, SessionOptions opt);
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // n in [1, max_batch]. Returns logits [L, vocab]; L = all_logits ? n : 1.
  const float* forward(const int32_t* tokens, int n, const char* phase = "prefill");
  const float* prefill(const std::vector<int32_t>& tokens);  // chunked; logits after the last chunk
  const float* decode(int32_t token) { return forward(&token, 1, "decode"); }
  void reset() { pos_ = 0; }
  int pos() const { return pos_; }

  const Graph& graph() const { return graph_; }
  const MemoryPlan& plan() const { return plan_; }
  const FusionStats& fusion() const { return fusion_; }
  const std::vector<Partition>& partitions() const { return parts_; }
  Profiler& profiler() { return prof_; }
  const SessionOptions& options() const { return opt_; }
  size_t kv_bytes() const { return (kcache_.size() + vcache_.size()) * sizeof(float); }
  size_t scratch_bytes() const { return scratch_.size() * sizeof(float); }
  int vocab() const { return model_.cfg.vocab; }

 private:
  const Model& model_;
  SessionOptions opt_;
  Graph graph_;
  FusionStats fusion_;
  MemoryPlan plan_;
  std::vector<std::unique_ptr<Backend>> backends_;
  std::vector<Partition> parts_;
  std::unique_ptr<ThreadPool> pool_;
  std::unique_ptr<uint8_t[]> arena_raw_;
  std::vector<float> kcache_, vcache_, scratch_, act_scales_;
  std::vector<int8_t> act_q_;
  RopeTable rope_;
  ExecContext ctx_;
  Profiler prof_;
  int pos_ = 0;
};

}  // namespace edgelm
