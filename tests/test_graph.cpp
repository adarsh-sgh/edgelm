// Loader, load-time quantization, fusion passes, memory planner and partitioner.
#include <algorithm>
#include <cstdio>

#include "common.hpp"
#include "edgelm/backend.hpp"
#include "edgelm/passes.hpp"

using namespace edgelm;
using namespace edgelm_test;

TEST_CASE("loader: config, graph, tokenizer and zero-copy weights from the exporter's file") {
  Model m = Model::load(testdata("tiny.f32.elm"));
  CHECK(m.cfg.vocab == 320);
  CHECK(m.cfg.dim == 64);
  CHECK(m.cfg.n_layers == 2);
  CHECK(m.cfg.n_kv_heads == 2);
  CHECK(m.cfg.tied);
  CHECK(m.graph.ops.size() == 1 + 2 * 16 + 3);  // embed, 16 ops per layer, last_rows/norm/head
  CHECK(m.graph.tensors[m.graph.input].dtype == DType::I32);
  CHECK(m.graph.tensors[m.graph.output].cols == 320);
  CHECK(m.tok.tokens.size() == 320);
  CHECK(m.tok.merges.size() == 63);
  for (const Tensor& t : m.graph.tensors)
    if (t.is_weight()) CHECK(reinterpret_cast<uintptr_t>(t.data) % 64 == 0);  // aligned blobs in the mapping

  const std::string bad = testdata("not_a_model.elm");
  if (FILE* f = std::fopen(bad.c_str(), "wb")) {
    std::fputs("definitely not an elm file, but long enough to have a header............................"
               "...................................................",
               f);
    std::fclose(f);
  }
  CHECK_THROWS(Model::load(bad));
  CHECK_THROWS(Model::load(testdata("missing.elm")));
}

TEST_CASE("load-time quantization is bit-identical to the Python exporter's q8/q4") {
  for (DType dt : {DType::Q8, DType::Q4}) {
    Model conv = Model::load(testdata("tiny.f32.elm"));
    conv.convert_weights(dt);
    Model exp = Model::load(testdata(std::string("tiny.") + dtype_name(dt) + ".elm"));
    REQUIRE(conv.graph.tensors.size() == exp.graph.tensors.size());
    int n = 0;
    for (size_t i = 0; i < conv.graph.tensors.size(); ++i) {
      const Tensor &a = conv.graph.tensors[i], &b = exp.graph.tensors[i];
      if (!a.is_weight()) continue;
      REQUIRE(a.dtype == b.dtype);
      REQUIRE(a.data_bytes == b.data_bytes);
      CHECK(std::memcmp(a.data, b.data, a.data_bytes) == 0);
      REQUIRE(a.scale_bytes == b.scale_bytes);
      if (a.scale_bytes) CHECK(std::memcmp(a.scales, b.scales, a.scale_bytes) == 0);
      n += a.dtype == dt;
    }
    CHECK(n == 1 + 2 * 7);  // tied embedding + 7 linears per layer
    CHECK(conv.weight_bytes() == exp.weight_bytes());
  }
}

TEST_CASE("fusion passes rewrite each layer to 7 ops and keep a valid topological order") {
  Model m = Model::load(testdata("tiny.f32.elm"));
  Graph g = m.graph;
  const FusionStats s = optimize(g);
  CHECK(s.swiglu == 2);
  CHECK(s.shared_input == 2);  // q/k/v per layer (gate/up went to SwiGLU first)
  CHECK(s.residual == 4);
  CHECK(s.rope == 2);
  CHECK(g.ops.size() == 1 + 2 * 7 + 3);
  // every input is produced before it is read
  std::vector<bool> ready(g.tensors.size(), false);
  for (size_t t = 0; t < g.tensors.size(); ++t) ready[t] = g.tensors[t].is_weight() || int(t) == g.input;
  for (const Op& op : g.ops) {
    for (int t : op.in) CHECK(ready[t]);
    for (int t : op.out) ready[t] = true;
  }
  CHECK(ready[g.output]);
  // idempotent
  Graph g2 = g;
  CHECK(optimize(g2).ops_after == g.ops.size());
}

TEST_CASE("memory planner: valid packing on the model graph and on random DAGs, far below naive") {
  Model m = Model::load(testdata("tiny.f32.elm"));
  for (bool fuse : {false, true}) {
    Graph g = m.graph;
    if (fuse) optimize(g);
    for (int T : {1, 16, 128}) {
      const MemoryPlan p = plan_memory(g, T, 1);
      CHECK(plan_is_valid(p));
      CHECK(p.arena_bytes < p.naive_bytes);
      CHECK(p.arena_bytes >= static_cast<size_t>(T) * 64 * 4);  // at least the residual stream
    }
  }
  // random chains with skip connections (liveness spans many ops)
  std::mt19937 rng(7);
  for (int trial = 0; trial < 50; ++trial) {
    Graph g;
    Tensor in;
    in.kind = TensorKind::Input;
    in.rows_sym = RowSym::T;
    in.cols = 1;
    g.tensors.push_back(in);
    const int nops = 5 + trial % 20;
    for (int i = 0; i < nops; ++i) {
      Tensor t;
      t.kind = i + 1 == nops ? TensorKind::Output : TensorKind::Activation;
      t.rows_sym = RowSym::T;
      t.cols = 1 + rng() % 300;
      g.tensors.push_back(t);
      Op op;
      op.type = OpType::ADD;
      const int prev = static_cast<int>(g.tensors.size()) - 2;
      op.in = {prev, static_cast<int>(rng() % (prev + 1))};
      op.out = {static_cast<int>(g.tensors.size()) - 1};
      g.ops.push_back(op);
    }
    g.index_io();
    const MemoryPlan p = plan_memory(g, 1 + trial % 7, 1);
    CHECK(plan_is_valid(p));
    CHECK(p.arena_bytes <= p.naive_bytes);
  }
  // a plan that overlaps two live tensors must be rejected by the checker
  MemoryPlan bad;
  bad.offset = {0, 0};
  bad.bytes = {64, 64};
  bad.first = {0, 1};
  bad.last = {2, 3};
  bad.arena_bytes = 64;
  CHECK_FALSE(plan_is_valid(bad));
}

TEST_CASE("partitioner: a narrow backend claims only its ops, the rest fall back") {
  auto npu = make_matmul_only_backend();
  auto cpu = make_cpu_backend();
  auto ref = make_reference_backend();
  for (const char* file : {"tiny.q8.elm", "tiny.q4.elm"}) {
    Model m = Model::load(testdata(file));
    Graph g = m.graph;
    optimize(g);
    auto parts = partition_graph(g, {npu.get(), ref.get()});
    int claimed = 0;
    for (const Partition& p : parts) {
      for (int i = p.begin; i < p.end; ++i) {
        const OpType t = g.ops[i].type;
        const bool mm = t == OpType::MATMUL || t == OpType::MATMUL_N || t == OpType::MATMUL_ADD;
        if (p.backend == npu.get()) {
          CHECK(mm);
          ++claimed;
        } else if (m.cfg.weight_dtype == DType::Q8) {
          CHECK_FALSE(mm);  // q8 matmuls are all claimed
        }
      }
    }
    if (m.cfg.weight_dtype == DType::Q8) {
      CHECK(claimed == 2 * 3 + 1);  // per layer: qkv, o+res, down+res; plus lm_head
      CHECK(parts.size() > 4);      // interleaved with reference partitions
    } else {
      CHECK(claimed == 0);  // int4 not supported by this "device": whole graph falls back
      CHECK(parts.size() == 1);
    }
    // cpu first: only EMBED and LAST_ROWS fall back
    auto parts2 = partition_graph(g, {cpu.get(), ref.get()});
    int fallback = 0;
    for (const Partition& p : parts2)
      if (p.backend == ref.get()) fallback += p.end - p.begin;
    CHECK(fallback == 2);
    CHECK(describe_partitions(g, parts2).find("reference 2 ops") != std::string::npos);
  }
}
