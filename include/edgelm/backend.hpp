// Delegate-style backend interface. Like a TFLite delegate, a backend declares which ops it
// supports; the partitioner hands each op to the first backend in preference order that claims
// it and groups consecutive ops into partitions. Anything unclaimed falls back to `reference`,
// which supports every op.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "edgelm/graph.hpp"
#include "edgelm/kernels.hpp"

namespace edgelm {

struct ExecContext {
  const Graph* graph = nullptr;
  std::vector<float*> ptr;  // activation tensor -> arena address (weights use Tensor::data)
  int T = 0, L = 0, pos0 = 0;
  std::vector<KVLayer> kv;  // per layer
  const RopeTable* rope = nullptr;
  ThreadPool* pool = nullptr;
  std::vector<float*> scratch;  // per thread
  const Tensor& tensor(int i) const { return graph->tensors[i]; }
  float* f(int i) const { return ptr[i]; }
  int64_t rows(int i) const;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  virtual bool supports(const Graph& g, const Op& op) const = 0;
  virtual void run(const Op& op, ExecContext& ctx) = 0;
};

std::unique_ptr<Backend> make_reference_backend();
std::unique_ptr<Backend> make_cpu_backend();
// Stand-in for an accelerator with a narrow op set (think int8-only NPU): claims only MATMUL /
// MATMUL_N / MATMUL_ADD with f32 or q8 weights; everything else must fall back.
std::unique_ptr<Backend> make_matmul_only_backend();
std::unique_ptr<Backend> make_backend(const std::string& name);

struct Partition {
  Backend* backend = nullptr;
  int begin = 0, end = 0;  // op range [begin, end)
};

std::vector<Partition> partition_graph(const Graph& g, const std::vector<Backend*>& preference);
std::string describe_partitions(const Graph& g, const std::vector<Partition>& parts);

}  // namespace edgelm
