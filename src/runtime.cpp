#include "edgelm/runtime.hpp"

#include <algorithm>
#include <cstring>

namespace edgelm {

Session::Session(const Model& model, SessionOptions opt) : model_(model), opt_(std::move(opt)), graph_(model.graph) {
  const Config& c = model.cfg;
  EDGELM_CHECK(opt_.max_batch >= 1 && opt_.max_ctx >= 1, "bad session sizes");
  opt_.max_ctx = std::min(opt_.max_ctx, c.max_seq);
  opt_.max_batch = std::min(opt_.max_batch, opt_.max_ctx);
  if (opt_.fuse) fusion_ = optimize(graph_);
  else fusion_.ops_before = fusion_.ops_after = graph_.ops.size();

  for (const std::string& b : opt_.backends)
    if (b != "reference" && b != "ref") backends_.push_back(make_backend(b));
  backends_.push_back(make_reference_backend());
  std::vector<Backend*> pref;
  for (auto& b : backends_) pref.push_back(b.get());
  parts_ = partition_graph(graph_, pref);

  const int L = opt_.all_logits ? opt_.max_batch : 1;
  plan_ = plan_memory(graph_, opt_.max_batch, L);
  arena_raw_.reset(new uint8_t[plan_.arena_bytes + 64]);
  uint8_t* arena = arena_raw_.get() + ((64 - reinterpret_cast<uintptr_t>(arena_raw_.get()) % 64) % 64);
  ctx_.graph = &graph_;
  ctx_.ptr.assign(graph_.tensors.size(), nullptr);
  for (size_t t = 0; t < graph_.tensors.size(); ++t)
    if (plan_.offset[t] >= 0) ctx_.ptr[t] = reinterpret_cast<float*>(arena + plan_.offset[t]);

  const size_t per_layer = static_cast<size_t>(c.n_kv_heads) * opt_.max_ctx * c.head_dim;
  kcache_.assign(per_layer * c.n_layers, 0.f);
  vcache_.assign(per_layer * c.n_layers, 0.f);
  for (int l = 0; l < c.n_layers; ++l)
    ctx_.kv.push_back({kcache_.data() + l * per_layer, vcache_.data() + l * per_layer, opt_.max_ctx});
  rope_ = RopeTable(c.head_dim, opt_.max_ctx, c.rope_theta);
  ctx_.rope = &rope_;

  pool_ = std::make_unique<ThreadPool>(opt_.threads);
  ctx_.pool = pool_.get();
  int64_t max_k = 0;
  for (const Tensor& t : graph_.tensors)
    if (t.is_weight()) max_k = std::max(max_k, t.cols);
  const size_t sf = cpu::scratch_floats(opt_.max_batch, max_k, opt_.max_ctx, c.head_dim);
  scratch_.assign(sf * pool_->size(), 0.f);
  for (int i = 0; i < pool_->size(); ++i) ctx_.scratch.push_back(scratch_.data() + i * sf);
  prof_.enabled = opt_.profile;
}

const float* Session::forward(const int32_t* tokens, int n, const char* phase) {
  EDGELM_CHECK(n >= 1 && n <= opt_.max_batch, "forward: n must be in [1, max_batch]");
  EDGELM_CHECK(pos_ + n <= opt_.max_ctx, "forward: context full");
  ctx_.T = n;
  ctx_.L = opt_.all_logits ? n : 1;
  ctx_.pos0 = pos_;
  std::memcpy(ctx_.ptr[graph_.input], tokens, n * sizeof(int32_t));
  const bool prof = prof_.enabled;
  const int64_t t_phase = prof ? Profiler::now_ns() : 0;
  for (const Partition& p : parts_)
    for (int i = p.begin; i < p.end; ++i) {
      const Op& op = graph_.ops[i];
      if (!prof) {
        p.backend->run(op, ctx_);
        continue;
      }
      const int64_t t0 = Profiler::now_ns();
      p.backend->run(op, ctx_);
      prof_.events.push_back({op.name, phase, p.backend->name(), op.type, t0, Profiler::now_ns() - t0, n, pos_});
    }
  if (prof)
    prof_.events.push_back({phase, phase, nullptr, OpType::COUNT, t_phase, Profiler::now_ns() - t_phase, n, pos_});
  pos_ += n;
  return ctx_.ptr[graph_.output];
}

const float* Session::prefill(const std::vector<int32_t>& tokens) {
  EDGELM_CHECK(!tokens.empty(), "prefill: empty prompt");
  const float* out = nullptr;
  for (size_t i = 0; i < tokens.size(); i += opt_.max_batch) {
    const int n = static_cast<int>(std::min<size_t>(opt_.max_batch, tokens.size() - i));
    out = forward(tokens.data() + i, n, "prefill");
  }
  return out;
}

}  // namespace edgelm
