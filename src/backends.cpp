#include <cstring>
#include <map>
#include <sstream>

#include "edgelm/backend.hpp"
#include "edgelm/thread_pool.hpp"

namespace edgelm {

int64_t ExecContext::rows(int i) const {
  const Tensor& t = graph->tensors[i];
  return t.rows_sym == RowSym::T ? T : t.rows_sym == RowSym::L ? L : t.rows;
}

namespace {

AttnArgs attn_args(const Op& op, ExecContext& c) {
  AttnArgs a{};
  a.q = c.f(op.in[0]);
  a.k = c.f(op.in[1]);
  a.v = c.f(op.in[2]);
  a.out = c.f(op.out[0]);
  a.T = static_cast<int>(c.rows(op.in[0]));
  a.pos0 = c.pos0;
  a.n_heads = op.iattr[1];
  a.n_kv = op.iattr[2];
  a.head_dim = static_cast<int>(c.tensor(op.in[0]).cols) / a.n_heads;
  EDGELM_CHECK(op.iattr[0] >= 0 && op.iattr[0] < static_cast<int>(c.kv.size()), "attention: bad layer");
  a.cache = c.kv[op.iattr[0]];
  a.rope = op.iattr[3] ? c.rope : nullptr;
  return a;
}

const float* wf32(const ExecContext& c, int i) { return static_cast<const float*>(c.tensor(i).data); }

class ReferenceBackend : public Backend {
 public:
  const char* name() const override { return "reference"; }
  bool supports(const Graph&, const Op&) const override { return true; }
  void run(const Op& op, ExecContext& c) override {
    const int64_t n = c.rows(op.out[0]) * c.tensor(op.out[0]).cols;
    switch (op.type) {
      case OpType::EMBED:
        ref::embed(reinterpret_cast<const int32_t*>(c.f(op.in[0])), c.T, c.tensor(op.in[1]), c.f(op.out[0]));
        break;
      case OpType::RMSNORM:
        ref::rmsnorm(c.f(op.in[0]), wf32(c, op.in[1]), c.f(op.out[0]), static_cast<int>(c.rows(op.in[0])),
                     static_cast<int>(c.tensor(op.in[0]).cols), op.fattr[0]);
        break;
      case OpType::MATMUL:
        ref::matmul(c.f(op.in[0]), static_cast<int>(c.rows(op.in[0])), c.tensor(op.in[1]), c.f(op.out[0]));
        break;
      case OpType::MATMUL_N:
        for (size_t i = 0; i < op.out.size(); ++i)
          ref::matmul(c.f(op.in[0]), static_cast<int>(c.rows(op.in[0])), c.tensor(op.in[1 + i]), c.f(op.out[i]));
        break;
      case OpType::MATMUL_ADD:
        ref::matmul(c.f(op.in[0]), static_cast<int>(c.rows(op.in[0])), c.tensor(op.in[1]), c.f(op.out[0]), c.f(op.in[2]));
        break;
      case OpType::FFN_SWIGLU:
        ref::swiglu(c.f(op.in[0]), static_cast<int>(c.rows(op.in[0])), c.tensor(op.in[1]), c.tensor(op.in[2]), c.f(op.out[0]));
        break;
      case OpType::ROPE:
        ref::rope(c.f(op.in[0]), c.f(op.out[0]), static_cast<int>(c.rows(op.in[0])), op.iattr[0], op.iattr[1], c.pos0, *c.rope);
        break;
      case OpType::ATTENTION:
        ref::attention(attn_args(op, c));
        break;
      case OpType::ADD: ref::add(c.f(op.in[0]), c.f(op.in[1]), c.f(op.out[0]), n); break;
      case OpType::SILU: ref::silu(c.f(op.in[0]), c.f(op.out[0]), n); break;
      case OpType::MUL: ref::mul(c.f(op.in[0]), c.f(op.in[1]), c.f(op.out[0]), n); break;
      case OpType::LAST_ROWS: {
        const int64_t cols = c.tensor(op.in[0]).cols, skip = c.rows(op.in[0]) - c.rows(op.out[0]);
        EDGELM_CHECK(skip >= 0, "LAST_ROWS: L > T");
        std::memmove(c.f(op.out[0]), c.f(op.in[0]) + skip * cols, n * sizeof(float));
        break;
      }
      default:
        EDGELM_CHECK(false, std::string("reference: unhandled op ") + op_name(op.type));
    }
  }
};

void run_cpu_matmuls(const Op& op, ExecContext& c, ThreadPool* pool) {
  std::vector<MatmulTarget> tg;
  const int T = static_cast<int>(c.rows(op.in[0]));
  if (op.type == OpType::MATMUL_N) {
    for (size_t i = 0; i < op.out.size(); ++i) tg.push_back({&c.tensor(op.in[1 + i]), c.f(op.out[i]), nullptr});
  } else {
    tg.push_back({&c.tensor(op.in[1]), c.f(op.out[0]), op.type == OpType::MATMUL_ADD ? c.f(op.in[2]) : nullptr});
  }
  cpu::matmul(c.f(op.in[0]), T, tg, pool, c.scratch.data());
}

class CpuBackend : public Backend {
 public:
  const char* name() const override { return "cpu"; }
  // Data-movement ops (EMBED, LAST_ROWS) are left to the reference backend.
  bool supports(const Graph&, const Op& op) const override {
    return op.type != OpType::EMBED && op.type != OpType::LAST_ROWS;
  }
  void run(const Op& op, ExecContext& c) override {
    const int64_t n = c.rows(op.out[0]) * c.tensor(op.out[0]).cols;
    switch (op.type) {
      case OpType::RMSNORM:
        cpu::rmsnorm(c.f(op.in[0]), wf32(c, op.in[1]), c.f(op.out[0]), static_cast<int>(c.rows(op.in[0])),
                     static_cast<int>(c.tensor(op.in[0]).cols), op.fattr[0], c.pool);
        break;
      case OpType::MATMUL:
      case OpType::MATMUL_N:
      case OpType::MATMUL_ADD:
        run_cpu_matmuls(op, c, c.pool);
        break;
      case OpType::FFN_SWIGLU:
        cpu::swiglu(c.f(op.in[0]), static_cast<int>(c.rows(op.in[0])), c.tensor(op.in[1]), c.tensor(op.in[2]),
                    c.f(op.out[0]), c.pool, c.scratch.data());
        break;
      case OpType::ROPE:
        cpu::rope(c.f(op.in[0]), c.f(op.out[0]), static_cast<int>(c.rows(op.in[0])), op.iattr[0], op.iattr[1], c.pos0, *c.rope);
        break;
      case OpType::ATTENTION:
        cpu::attention(attn_args(op, c), c.pool, c.scratch.data());
        break;
      case OpType::ADD: cpu::add(c.f(op.in[0]), c.f(op.in[1]), c.f(op.out[0]), n); break;
      case OpType::SILU: cpu::silu(c.f(op.in[0]), c.f(op.out[0]), n); break;
      case OpType::MUL: cpu::mul(c.f(op.in[0]), c.f(op.in[1]), c.f(op.out[0]), n); break;
      default:
        EDGELM_CHECK(false, std::string("cpu: unsupported op ") + op_name(op.type));
    }
  }
};

class MatmulOnlyBackend : public Backend {
 public:
  const char* name() const override { return "matmul-only"; }
  bool supports(const Graph& g, const Op& op) const override {
    if (op.type != OpType::MATMUL && op.type != OpType::MATMUL_N && op.type != OpType::MATMUL_ADD) return false;
    const size_t nw = op.type == OpType::MATMUL_N ? op.in.size() - 1 : 1;
    for (size_t i = 1; i <= nw; ++i) {
      const DType d = g.tensors[op.in[i]].dtype;
      if (d != DType::F32 && d != DType::Q8) return false;
    }
    return true;
  }
  void run(const Op& op, ExecContext& c) override { run_cpu_matmuls(op, c, nullptr); }  // single queue
};

}  // namespace

std::unique_ptr<Backend> make_reference_backend() { return std::make_unique<ReferenceBackend>(); }
std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }
std::unique_ptr<Backend> make_matmul_only_backend() { return std::make_unique<MatmulOnlyBackend>(); }

std::unique_ptr<Backend> make_backend(const std::string& name) {
  if (name == "reference" || name == "ref") return make_reference_backend();
  if (name == "cpu") return make_cpu_backend();
  if (name == "matmul-only") return make_matmul_only_backend();
  throw std::runtime_error("edgelm: unknown backend " + name);
}

std::vector<Partition> partition_graph(const Graph& g, const std::vector<Backend*>& preference) {
  std::vector<Partition> parts;
  for (size_t i = 0; i < g.ops.size(); ++i) {
    Backend* chosen = nullptr;
    for (Backend* b : preference)
      if (b->supports(g, g.ops[i])) {
        chosen = b;
        break;
      }
    EDGELM_CHECK(chosen, std::string("no backend supports ") + op_name(g.ops[i].type));
    if (!parts.empty() && parts.back().backend == chosen) {
      parts.back().end = static_cast<int>(i) + 1;
    } else {
      parts.push_back({chosen, static_cast<int>(i), static_cast<int>(i) + 1});
    }
  }
  return parts;
}

std::string describe_partitions(const Graph& g, const std::vector<Partition>& parts) {
  std::map<std::string, std::pair<int, int>> per;  // backend -> (partitions, ops)
  for (const Partition& p : parts) {
    auto& e = per[p.backend->name()];
    e.first++;
    e.second += p.end - p.begin;
  }
  std::ostringstream os;
  os << parts.size() << " partitions:";
  for (auto& [name, e] : per) os << " " << name << " " << e.second << " ops in " << e.first << ";";
  if (parts.size() <= 8) {
    for (const Partition& p : parts) {
      os << "\n  [" << p.begin << ", " << p.end << ") " << p.backend->name() << ": " << op_name(g.ops[p.begin].type);
      if (p.end - p.begin > 1) os << " .. " << op_name(g.ops[p.end - 1].type);
    }
  }
  return os.str();
}

}  // namespace edgelm
