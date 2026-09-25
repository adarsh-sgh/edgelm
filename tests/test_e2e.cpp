// End-to-end: logits from the whole runtime vs the NumPy reference, across dtypes, backends,
// fusion on/off, thread counts, chunked prefill and prefill+decode through the KV cache.
#include "common.hpp"
#include "edgelm/runtime.hpp"
#include "edgelm/tokenizer.hpp"

using namespace edgelm;
using namespace edgelm_test;

namespace {

std::vector<float> run_all_logits(const Model& m, SessionOptions o, const std::vector<int32_t>& ids, int prefill) {
  o.all_logits = true;
  o.max_ctx = 128;
  Session s(m, o);
  std::vector<float> out;
  const int V = m.cfg.vocab;
  for (int i = 0; i < prefill; i += o.max_batch) {
    const int n = std::min(o.max_batch, prefill - i);
    const float* lg = s.forward(ids.data() + i, n);
    out.insert(out.end(), lg, lg + static_cast<size_t>(n) * V);
  }
  for (size_t i = prefill; i < ids.size(); ++i) {
    const float* lg = s.decode(ids[i]);
    out.insert(out.end(), lg, lg + V);
  }
  CHECK(s.pos() == static_cast<int>(ids.size()));
  return out;
}

}  // namespace

TEST_CASE("fp32/q8/q4 logits match the NumPy reference (quantized vs its own dequantized weights)") {
  const GArray& tok = G("tokens");
  const std::vector<int32_t> ids(tok.i.begin(), tok.i.end());
  const int S = static_cast<int>(ids.size());
  for (const char* dt : {"f32", "q8", "q4"}) {
    Model m = Model::load(testdata(std::string("tiny.") + dt + ".elm"));
    const GArray& want = G(std::string("logits.") + dt);
    struct Variant {
      const char* name;
      std::vector<std::string> backends;
      int threads, batch, prefill;
      bool fuse;
    };
    const Variant vs[] = {
        {"cpu fused 4t full prefill", {"cpu"}, 4, 128, S, true},
        {"cpu unfused 1t", {"cpu"}, 1, 128, S, false},
        {"reference only", {"reference"}, 1, 128, S, true},
        {"matmul-only + cpu fallback", {"matmul-only", "cpu"}, 3, 128, S, true},
        {"chunked prefill (batch 7)", {"cpu"}, 2, 7, S, true},
        {"prefill 5 then decode 35", {"cpu"}, 4, 128, 5, true},
        {"decode only", {"cpu"}, 2, 128, 0, true},
    };
    for (const Variant& v : vs) {
      CAPTURE(dt);
      CAPTURE(v.name);
      SessionOptions o;
      o.backends = v.backends;
      o.threads = v.threads;
      o.max_batch = v.batch;
      o.fuse = v.fuse;
      auto got = run_all_logits(m, o, ids, v.prefill);
      REQUIRE(got.size() == want.f.size());
      CHECK(rel_err(got.data(), want.f.data(), got.size()) < 2e-5);
    }
  }
}

TEST_CASE("results are bit-identical across thread counts; last-token logits match the all-logits row") {
  Model m = Model::load(testdata("tiny.q4.elm"));
  const GArray& tok = G("tokens");
  const std::vector<int32_t> ids(tok.i.begin(), tok.i.end());
  std::vector<float> base;
  for (int th : {1, 2, 3, 8}) {
    SessionOptions o;
    o.threads = th;
    auto got = run_all_logits(m, o, ids, 20);
    if (base.empty()) base = got;
    else CHECK(std::memcmp(got.data(), base.data(), got.size() * sizeof(float)) == 0);
  }
  SessionOptions o;
  o.threads = 2;
  o.max_ctx = 128;
  Session s(m, o);  // all_logits = false: plans a 1-row logits tensor
  const float* last = s.prefill(std::vector<int32_t>(ids.begin(), ids.begin() + 20));
  // lm_head runs as GEMV here (1 row) vs GEMM above (20 rows): same math, different summation order
  CHECK(rel_err(last, base.data() + 19 * m.cfg.vocab, m.cfg.vocab) < 1e-6);
  CHECK_THROWS(s.forward(ids.data(), 200));  // > max_batch
  s.reset();
  CHECK(s.pos() == 0);
}

TEST_CASE("generation loop: greedy is deterministic, sampling stays in top-k, context overflow throws") {
  Model m = Model::load(testdata("tiny.f32.elm"));
  Tokenizer tok(m.tok);
  SessionOptions o;
  o.max_ctx = 32;
  o.threads = 2;
  Session s(m, o);
  auto generate = [&](SamplerConfig sc, int n) {
    s.reset();
    Sampler smp(sc);
    std::vector<int32_t> out;
    const float* lg = s.prefill(tok.encode("the fox"));
    for (int i = 0; i < n; ++i) {
      const int32_t t = smp.sample(lg, m.cfg.vocab);
      if (sc.temperature <= 0) CHECK(t == argmax(lg, m.cfg.vocab));
      if (sc.top_k > 0 && sc.temperature > 0) {  // t must be among the k largest logits
        int larger = 0;
        for (int v = 0; v < m.cfg.vocab; ++v) larger += lg[v] > lg[t];
        CHECK(larger < sc.top_k);
      }
      out.push_back(t);
      lg = s.decode(t);
    }
    return out;
  };
  CHECK(generate({0.f, 40, 1}, 10) == generate({0.f, 40, 2}, 10));
  CHECK(generate({0.9f, 5, 3}, 10) == generate({0.9f, 5, 3}, 10));  // seeded
  CHECK(generate({1.0f, 1, 3}, 10) == generate({0.f, 40, 3}, 10));  // top-k 1 == greedy
  s.reset();
  std::vector<int32_t> big(33, 5);
  CHECK_THROWS(s.prefill(big));
}

TEST_CASE("profiler: one span per executed op per step, valid trace file and stats") {
  Model m = Model::load(testdata("tiny.q8.elm"));
  SessionOptions o;
  o.profile = true;
  o.threads = 2;
  o.max_ctx = 64;
  Session s(m, o);
  s.prefill({1, 2, 3, 4, 5, 6});
  s.decode(7);
  s.decode(8);
  const size_t nops = s.graph().ops.size();
  size_t pre = 0, dec = 0, phases = 0;
  for (const ProfEvent& e : s.profiler().events) {
    if (!e.backend) {
      ++phases;
      continue;
    }
    CHECK(e.dur_ns >= 0);
    (e.phase == "prefill" ? pre : dec)++;
  }
  CHECK(pre == nops);
  CHECK(dec == 2 * nops);
  CHECK(phases == 3);
  const std::string path = testdata("trace_test.json");
  REQUIRE(s.profiler().write_trace(path));
  std::ifstream f(path);
  std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  CHECK(js.find("\"traceEvents\"") != std::string::npos);
  CHECK(js.find("\"ph\":\"X\"") != std::string::npos);
  CHECK(js.find("FFN_SWIGLU") != std::string::npos);
  CHECK(js.rfind("]}") != std::string::npos);
  const std::string st = s.profiler().stats();
  CHECK(st.find("prefill: 1 run(s)") != std::string::npos);
  CHECK(st.find("decode: 2 run(s)") != std::string::npos);
  CHECK(st.find("MATMUL_N (cpu)") != std::string::npos);
  CHECK(st.find("EMBED (reference)") != std::string::npos);
}
