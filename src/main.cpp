// edgelm CLI: run | bench | eval | info | logits
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "edgelm/runtime.hpp"
#include "edgelm/tokenizer.hpp"

using namespace edgelm;

namespace {

struct Args {
  std::string cmd;
  std::map<std::string, std::string> kv;
  bool has(const std::string& k) const { return kv.count(k) > 0; }
  std::string get(const std::string& k, const std::string& d = "") const { return has(k) ? kv.at(k) : d; }
  int geti(const std::string& k, int d) const { return has(k) ? std::stoi(kv.at(k)) : d; }
  float getf(const std::string& k, float d) const { return has(k) ? std::stof(kv.at(k)) : d; }
};

Args parse(int argc, char** argv) {
  static const std::map<std::string, std::string> alias{{"-m", "--model"}, {"-p", "--prompt"}, {"-n", "--n"},
                                                        {"-t", "--threads"}};
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    if (alias.count(k)) k = alias.at(k);
    EDGELM_CHECK(k.rfind("--", 0) == 0, "unexpected argument " + k);
    k = k.substr(2);
    if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0 && !(argv[i + 1][0] == '-' && std::strlen(argv[i + 1]) == 2))
      a.kv[k] = argv[++i];
    else
      a.kv[k] = "1";
  }
  return a;
}

DType parse_dtype(const std::string& s) {
  if (s == "f32") return DType::F32;
  if (s == "q8") return DType::Q8;
  if (s == "q4") return DType::Q4;
  throw std::runtime_error("edgelm: --dtype must be f32|q8|q4");
}

Model load_model(const Args& a, const std::string& key = "model", const std::string& dkey = "dtype") {
  Model m = Model::load(a.get(key));
  if (a.has(dkey)) {
    const DType d = parse_dtype(a.get(dkey));
    if (d != m.cfg.weight_dtype) m.convert_weights(d);
  }
  return m;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, sep))
    if (!x.empty()) out.push_back(x);
  return out;
}

SessionOptions session_opts(const Args& a) {
  SessionOptions o;
  if (a.has("backend")) o.backends = split(a.get("backend"), ',');
  o.threads = a.geti("threads", 4);
  o.max_ctx = a.geti("ctx", 1024);
  o.max_batch = a.geti("batch", 128);
  o.fuse = !a.has("no-fuse");
  o.act_quant = a.get("act", "f32") == "int8";
  EDGELM_CHECK(a.get("act", "f32") == "f32" || a.get("act") == "int8", "--act must be f32|int8");
  o.profile = a.has("profile") || a.has("stats");
  if (a.has("no-simd")) cpu::set_simd(false);
  return o;
}

double peak_rss_mb() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return ru.ru_maxrss / 1048576.0;
#else
  return ru.ru_maxrss / 1024.0;
#endif
}

double ms_since(int64_t t0) { return (Profiler::now_ns() - t0) / 1e6; }

void print_session(const Model& m, const Session& s) {
  const MemoryPlan& p = s.plan();
  std::printf("model   : %s weights, %.1f MB, vocab %d, dim %d, %d layers, %d/%d heads, ffn %d\n",
              dtype_name(m.cfg.weight_dtype), m.weight_bytes() / 1e6, m.cfg.vocab, m.cfg.dim, m.cfg.n_layers,
              m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.hidden);
  std::printf("fusion  : %s\n", s.fusion().str().c_str());
  std::printf("graph   : %s\n", s.graph().summary().c_str());
  std::printf("backends: %s\n", describe_partitions(s.graph(), s.partitions()).c_str());
  std::printf("arena   : %d activations planned for T=%d: %.2f MB packed vs %.2f MB naive (%.1fx)\n", p.n_planned,
              s.options().max_batch, p.arena_bytes / 1e6, p.naive_bytes / 1e6,
              p.arena_bytes ? double(p.naive_bytes) / p.arena_bytes : 0.0);
  std::printf("kv cache: %.2f MB (ctx %d), scratch %.2f MB, threads %d, simd %s\n", s.kv_bytes() / 1e6,
              s.options().max_ctx, s.scratch_bytes() / 1e6, s.options().threads, cpu::simd_enabled() ? "neon" : "off");
}

int cmd_info(const Args& a) {
  Model m = load_model(a);
  Session s(m, session_opts(a));
  print_session(m, s);
  if (a.has("ops"))
    for (size_t i = 0; i < s.graph().ops.size(); ++i)
      std::printf("  %3zu %-11s %s\n", i, op_name(s.graph().ops[i].type), s.graph().ops[i].name.c_str());
  return 0;
}

void finish_profile(const Args& a, Session& s) {
  if (a.has("stats")) std::printf("\n%s", s.profiler().stats().c_str());
  if (a.has("profile") && a.get("profile") != "1") {
    EDGELM_CHECK(s.profiler().write_trace(a.get("profile")), "cannot write trace");
    std::printf("trace: %s (%zu events; open in ui.perfetto.dev)\n", a.get("profile").c_str(), s.profiler().events.size());
  }
}

int cmd_run(const Args& a) {
  Model m = load_model(a);
  Tokenizer tok(m.tok);
  SessionOptions o = session_opts(a);
  Session s(m, o);
  if (a.has("verbose")) print_session(m, s);
  std::vector<int32_t> prompt = tok.encode(a.get("prompt", "Once upon a time"));
  if (a.has("bos")) prompt.insert(prompt.begin(), m.cfg.bos);
  const int n = a.geti("n", 64);
  EDGELM_CHECK(static_cast<int>(prompt.size()) + n <= s.options().max_ctx, "prompt + n exceeds --ctx");
  Sampler sampler({a.getf("temp", 0.f), a.geti("top-k", 40), static_cast<uint64_t>(a.geti("seed", 42))});
  std::printf("%s", tok.decode(prompt).c_str());
  std::fflush(stdout);
  const int64_t t0 = Profiler::now_ns();
  const float* logits = s.prefill(prompt);
  int32_t next = sampler.sample(logits, m.cfg.vocab);
  const double ttft = ms_since(t0);
  int generated = 0;
  const int64_t t1 = Profiler::now_ns();
  for (int i = 0; i < n; ++i) {
    if (next == m.cfg.eos && !a.has("ignore-eos")) break;
    std::printf("%s", tok.decode(next).c_str());
    std::fflush(stdout);
    ++generated;
    if (i + 1 == n) break;
    logits = s.decode(next);
    next = sampler.sample(logits, m.cfg.vocab);
  }
  const double dec_ms = ms_since(t1);
  std::printf("\n\n[%s, %d threads] prompt %zu tok, TTFT %.1f ms (prefill %.1f tok/s), decode %d tok %.1f tok/s, peak RSS %.0f MB\n",
              dtype_name(m.cfg.weight_dtype), o.threads, prompt.size(), ttft, prompt.size() / (ttft / 1e3), generated,
              generated > 1 ? (generated - 1) / (dec_ms / 1e3) : 0.0, peak_rss_mb());
  finish_profile(a, s);
  return 0;
}

int cmd_bench(const Args& a) {
  Model m = load_model(a);
  const int P = a.geti("prompt-len", 128), G = a.geti("gen", 64), reps = a.geti("reps", 3);
  std::vector<int> threads;
  for (auto& t : split(a.get("threads", "1,4"), ',')) threads.push_back(std::stoi(t));
  std::vector<int32_t> prompt(P);
  uint64_t x = 12345;
  for (auto& t : prompt) {  // fixed pseudo-random prompt, skipping the low (special) ids
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint64_t lo = m.cfg.vocab > 2000 ? 1000 : 0;
    t = static_cast<int32_t>(lo + (x >> 33) % (m.cfg.vocab - lo));
  }
  const bool json = a.has("json");
  const std::string backend = a.get("backend", "cpu");
  if (!json)
    std::printf("%-5s %-10s %-4s %7s | %10s %9s | %10s | %9s %9s\n", "dtype", "backend", "simd", "threads",
                "prefill/s", "TTFT ms", "decode/s", "arena MB", "peakRSS");
  for (int th : threads) {
    Args b = a;
    b.kv["threads"] = std::to_string(th);
    SessionOptions o = session_opts(b);
    o.max_ctx = P + G + 8;
    o.max_batch = std::min(P, a.geti("batch", P));
    o.profile = false;
    Session s(m, o);
    std::vector<double> pre, ttft, dec;
    for (int r = -1; r < reps; ++r) {  // r = -1: warm-up (page in weights, spin up workers)
      s.reset();
      int64_t t0 = Profiler::now_ns();
      const float* lg = s.prefill(prompt);
      const double p_ms = ms_since(t0);
      int32_t tok = argmax(lg, m.cfg.vocab);
      const double f_ms = ms_since(t0);
      int64_t t1 = Profiler::now_ns();
      for (int i = 0; i < G; ++i) tok = argmax(s.decode(tok), m.cfg.vocab);
      const double d_ms = ms_since(t1);
      if (r < 0) continue;
      pre.push_back(P / (p_ms / 1e3));
      ttft.push_back(f_ms);
      dec.push_back(G / (d_ms / 1e3));
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const char* simd = cpu::simd_enabled() ? "neon" : "off";
    if (json)
      std::printf("{\"dtype\":\"%s\",\"backend\":\"%s\",\"simd\":\"%s\",\"threads\":%d,\"prompt\":%d,\"gen\":%d,"
                  "\"prefill_tok_s\":%.2f,\"ttft_ms\":%.2f,\"decode_tok_s\":%.2f,\"weights_mb\":%.2f,\"arena_mb\":%.3f,"
                  "\"naive_mb\":%.3f,\"kv_mb\":%.2f,\"peak_rss_mb\":%.1f}\n",
                  dtype_name(m.cfg.weight_dtype), backend.c_str(), simd, th, P, G, med(pre), med(ttft), med(dec),
                  m.weight_bytes() / 1e6, s.plan().arena_bytes / 1e6, s.plan().naive_bytes / 1e6, s.kv_bytes() / 1e6,
                  peak_rss_mb());
    else
      std::printf("%-5s %-10s %-4s %7d | %10.1f %9.1f | %10.1f | %9.2f %8.0fM\n", dtype_name(m.cfg.weight_dtype),
                  backend.c_str(), simd, th, med(pre), med(ttft), med(dec), s.plan().arena_bytes / 1e6, peak_rss_mb());
    std::fflush(stdout);
  }
  return 0;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EDGELM_CHECK(f, "cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int cmd_eval(const Args& a) {
  Model cand = load_model(a);
  const bool with_ref = a.has("ref");
  Model refm = with_ref ? load_model(a, "ref", "ref-dtype") : Model{};
  Tokenizer tok(cand.tok);
  std::vector<int32_t> ids = tok.encode(read_file(a.get("text", "data/eval.txt")));
  const int W = a.geti("window", 256);
  const size_t max_tokens = static_cast<size_t>(a.geti("max-tokens", 1 << 30));
  if (ids.size() > max_tokens) ids.resize(max_tokens);
  SessionOptions o = session_opts(a);
  o.max_ctx = W;
  o.max_batch = W;
  o.all_logits = true;
  Session sc(cand, o);
  std::unique_ptr<Session> sr = with_ref ? std::make_unique<Session>(refm, o) : nullptr;
  const int V = cand.cfg.vocab;
  double nll_c = 0, nll_r = 0, cos_sum = 0, cos_min = 1;
  size_t n = 0, agree = 0;
  auto nll = [&](const float* lg, int target) {
    const float mx = *std::max_element(lg, lg + V);
    double s = 0;
    for (int i = 0; i < V; ++i) s += std::exp(static_cast<double>(lg[i] - mx));
    return std::log(s) + mx - lg[target];
  };
  const int64_t t0 = Profiler::now_ns();
  for (size_t w0 = 0; w0 + 1 < ids.size(); w0 += W) {
    const int len = static_cast<int>(std::min<size_t>(W, ids.size() - w0));
    if (len < 2) break;
    sc.reset();
    const float* lc = sc.forward(ids.data() + w0, len);
    const float* lr = nullptr;
    if (sr) {
      sr->reset();
      lr = sr->forward(ids.data() + w0, len);
    }
    for (int i = 0; i + 1 < len; ++i) {
      const float* rc = lc + static_cast<size_t>(i) * V;
      nll_c += nll(rc, ids[w0 + i + 1]);
      if (lr) {
        const float* rr = lr + static_cast<size_t>(i) * V;
        nll_r += nll(rr, ids[w0 + i + 1]);
        agree += argmax(rc, V) == argmax(rr, V);
        double d = 0, na = 0, nb = 0;
        for (int v = 0; v < V; ++v) {
          d += double(rc[v]) * rr[v];
          na += double(rc[v]) * rc[v];
          nb += double(rr[v]) * rr[v];
        }
        const double c = d / std::sqrt(na * nb);
        cos_sum += c;
        cos_min = std::min(cos_min, c);
      }
      ++n;
    }
  }
  std::printf("eval: %zu tokens in windows of %d, %.1f s\n", n, W, ms_since(t0) / 1e3);
  std::printf("  %s: perplexity %.4f\n", dtype_name(cand.cfg.weight_dtype), std::exp(nll_c / n));
  if (sr) {
    std::printf("  ref %s: perplexity %.4f\n", dtype_name(refm.cfg.weight_dtype), std::exp(nll_r / n));
    std::printf("  top-1 agreement %.2f%% (%zu/%zu), logits cosine mean %.6f min %.6f\n", 100.0 * agree / n, agree, n,
                cos_sum / n, cos_min);
  }
  std::printf("  weights %.1f MB\n", cand.weight_bytes() / 1e6);
  return 0;
}

// Dump logits for every position (prefill of --prefill tokens, then one decode step per remaining
// token) to a raw f32 file; used by tools/check_real.py against NumPy / HF transformers.
int cmd_logits(const Args& a) {
  Model m = load_model(a);
  Tokenizer tok(m.tok);
  std::vector<int32_t> ids;
  if (a.has("tokens"))
    for (auto& t : split(a.get("tokens"), ',')) ids.push_back(std::stoi(t));
  else
    ids = tok.encode(a.get("text", "Hello world"));
  const int k = std::min<int>(a.geti("prefill", static_cast<int>(ids.size())), static_cast<int>(ids.size()));
  SessionOptions o = session_opts(a);
  o.all_logits = true;
  o.max_ctx = static_cast<int>(ids.size()) + 1;
  o.max_batch = std::max(1, k);
  Session s(m, o);
  const int V = m.cfg.vocab;
  std::vector<float> out;
  if (k > 0) {
    const float* lg = s.forward(ids.data(), k);
    out.insert(out.end(), lg, lg + static_cast<size_t>(k) * V);
  }
  for (size_t i = k; i < ids.size(); ++i) {
    const float* lg = s.decode(ids[i]);
    out.insert(out.end(), lg, lg + V);
  }
  std::ofstream f(a.get("out", "logits.f32"), std::ios::binary);
  f.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
  std::printf("%zu tokens x %d logits -> %s\n", ids.size(), V, a.get("out", "logits.f32").c_str());
  return 0;
}

int cmd_tokenize(const Args& a) {
  Model m = Model::load(a.get("model"));
  Tokenizer tok(m.tok);
  for (int32_t id : tok.encode(a.has("file") ? read_file(a.get("file")) : a.get("text"))) std::printf("%d ", id);
  std::printf("\n");
  return 0;
}

void usage() {
  std::puts(
      "edgelm - C++17 on-device LLM inference runtime\n"
      "  edgelm run   -m model.elm -p \"prompt\" [-n 64] [--dtype f32|q8|q4] [--threads N] [--temp 0.8 --top-k 40 --seed 1]\n"
      "               [--backend cpu|reference|matmul-only,cpu] [--no-fuse] [--no-simd] [--profile trace.json] [--stats]\n"
      "  edgelm bench -m model.elm [--dtype ..] [--threads 1,4,10] [--prompt-len 128] [--gen 64] [--reps 3] [--json]\n"
      "  edgelm eval  -m model.elm [--dtype ..] [--ref f32.elm] [--text data/eval.txt] [--window 256]\n"
      "  edgelm info  -m model.elm [--dtype ..] [--backend ..] [--no-fuse] [--ops]\n"
      "  edgelm logits -m model.elm --tokens 1,2,3 [--prefill k] --out logits.f32\n"
      "  edgelm tokenize -m model.elm --text \"...\" | --file path");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    if (a.cmd == "run") return cmd_run(a);
    if (a.cmd == "bench") return cmd_bench(a);
    if (a.cmd == "eval") return cmd_eval(a);
    if (a.cmd == "info") return cmd_info(a);
    if (a.cmd == "logits") return cmd_logits(a);
    if (a.cmd == "tokenize") return cmd_tokenize(a);
    usage();
    return a.cmd.empty() || a.cmd == "help" || a.cmd == "--help" ? 0 : 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
