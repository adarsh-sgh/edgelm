// Op kernels. `ref` is the scalar, serial, obviously-correct oracle for every op and dtype;
// `cpu` is the optimized path (NEON on arm64 with a scalar fallback, threaded). Matmuls are
// y[T, N] = x[T, K] . W[N, K]^T with W in f32, q8 (per-row scale) or q4 (group-32 scale).
#pragma once
#include <cstdint>
#include <vector>

#include "edgelm/graph.hpp"

namespace edgelm {

class ThreadPool;

struct RopeTable {
  int half = 0, max_pos = 0;
  std::vector<float> cos, sin;  // [pos][half], computed in double
  RopeTable() = default;
  RopeTable(int head_dim, int max_pos, float theta);
};

// KV cache for one layer: k/v laid out [kv_head][pos][head_dim] so a head's history is contiguous.
struct KVLayer {
  float* k = nullptr;
  float* v = nullptr;
  int max_ctx = 0;
};

struct AttnArgs {
  const float* q;  // [T, nh*hd]
  const float* k;  // [T, nkv*hd] new keys (pre-rope if rope != nullptr)
  const float* v;  // [T, nkv*hd]
  float* out;      // [T, nh*hd]
  int T, pos0, n_heads, n_kv, head_dim;
  KVLayer cache;
  const RopeTable* rope;  // non-null: apply RoPE to q and k inside the kernel
};

// Dynamic int8 activations (W8A8 / W4A8): each row of x is quantized on the fly in blocks of 32
// (scale = amax/127, like GGML Q8_0) so q8/q4 matmuls run as int8 dot products (SDOT).
constexpr int kActBlock = 32;
struct ActBuf {
  int8_t* q = nullptr;  // [T, K]
  float* d = nullptr;   // [T, K/32]
};
void quantize_act_row(const float* x, int64_t k, int8_t* q, float* d);
inline bool act_quant_applies(const Tensor& w) {
  return (w.dtype == DType::Q8 || w.dtype == DType::Q4) && w.cols % kActBlock == 0;
}

struct MatmulTarget {
  const Tensor* w;
  float* y;                  // [T, w->rows]
  const float* residual;     // optional, same shape as y
};

namespace ref {
void embed(const int32_t* tokens, int T, const Tensor& table, float* y);
void rmsnorm(const float* x, const float* w, float* y, int rows, int dim, float eps);
// act_quant: fake-quantize x the way the int8-activation kernels do, then f32 math.
void matmul(const float* x, int T, const Tensor& w, float* y, const float* residual = nullptr, bool act_quant = false);
void rope(const float* x, float* y, int T, int n_heads, int head_dim, int pos0, const RopeTable& rt);
void attention(const AttnArgs& a);
void add(const float* a, const float* b, float* y, int64_t n);
void silu(const float* x, float* y, int64_t n);
void mul(const float* a, const float* b, float* y, int64_t n);
void swiglu(const float* x, int T, const Tensor& wg, const Tensor& wu, float* y, bool act_quant = false);
}  // namespace ref

namespace cpu {
// Toggle the NEON microkernels (scalar fallback otherwise); for the scalar-vs-SIMD benchmark.
void set_simd(bool on);
bool simd_enabled();
bool simd_compiled();

constexpr int kRowBlock = 16;  // output rows per task / per dequantized weight tile
// Per-thread scratch floats needed for T tokens, max K, max context.
size_t scratch_floats(int T, int64_t max_k, int max_ctx, int head_dim);

void rmsnorm(const float* x, const float* w, float* y, int rows, int dim, float eps, ThreadPool* pool);
// One parallel dispatch over the rows of every target (fused QKV = 3 targets). With `aq`, q8/q4
// targets use int8 activations (x quantized once into aq, shared by all targets).
void matmul(const float* x, int T, const std::vector<MatmulTarget>& targets, ThreadPool* pool, float* const* scratch,
            const ActBuf* aq = nullptr);
void swiglu(const float* x, int T, const Tensor& wg, const Tensor& wu, float* y, ThreadPool* pool, float* const* scratch,
            const ActBuf* aq = nullptr);
void rope(const float* x, float* y, int T, int n_heads, int head_dim, int pos0, const RopeTable& rt);
void attention(const AttnArgs& a, ThreadPool* pool, float* const* scratch);
void add(const float* a, const float* b, float* y, int64_t n);
void silu(const float* x, float* y, int64_t n);
void mul(const float* a, const float* b, float* y, int64_t n);

// exposed for tests
float dot_f32(const float* a, const float* b, int64_t k);
}  // namespace cpu

}  // namespace edgelm
