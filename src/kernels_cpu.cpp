// Optimized CPU kernels: NEON microkernels on arm64 (scalar fallback elsewhere / when disabled),
// parallel over output-row blocks on the thread pool.
//   decode  (T == 1): GEMV, weights dequantized in registers (memory bound: one pass over W)
//   prefill (T  > 1): a 16-row weight tile is dequantized once into per-thread scratch and reused
//                     for every token through a 4x4 (tokens x rows) register-blocked f32 GEMM.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "edgelm/kernels.hpp"
#include "edgelm/model.hpp"
#include "edgelm/thread_pool.hpp"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define EDGELM_NEON 1
#else
#define EDGELM_NEON 0
#endif

namespace edgelm {
namespace cpu {
namespace {

std::atomic<bool> g_simd{true};

inline bool use_neon() { return EDGELM_NEON && g_simd.load(std::memory_order_relaxed); }

// 4 independent accumulators, like the NEON path, so the scalar baseline is not latency-bound on
// one FMA chain (the compiler may not reassociate float sums into SIMD without -ffast-math).
float dot_f32_scalar(const float* a, const float* b, int64_t k) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t i = 0;
  for (; i + 4 <= k; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < k; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

float dot_q8(const float* x, const int8_t* w, int64_t k) {  // unscaled
  int64_t i = 0;
  float s = 0.f;
#if EDGELM_NEON
  if (use_neon()) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 16 <= k; i += 16) {
      const int8x16_t q = vld1q_s8(w + i);
      const int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_high_s8(q);
      a0 = vfmaq_f32(a0, vld1q_f32(x + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
      a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vcvtq_f32_s32(vmovl_high_s16(lo)));
      a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
      a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vcvtq_f32_s32(vmovl_high_s16(hi)));
    }
    s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  }
#endif
  float s1 = 0.f, s2 = 0.f, s3 = 0.f;
  for (; i + 4 <= k; i += 4) {
    s += x[i] * static_cast<float>(w[i]);
    s1 += x[i + 1] * static_cast<float>(w[i + 1]);
    s2 += x[i + 2] * static_cast<float>(w[i + 2]);
    s3 += x[i + 3] * static_cast<float>(w[i + 3]);
  }
  for (; i < k; ++i) s += x[i] * static_cast<float>(w[i]);
  return (s + s1) + (s2 + s3);
}

float dot_q4(const float* x, const uint8_t* w, const float* scales, int64_t k) {  // scaled
  const int64_t groups = k / 32;
  int64_t g = 0;
  float s = 0.f;
#if EDGELM_NEON
  if (use_neon()) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (; g < groups; ++g) {
      const uint8x16_t b = vld1q_u8(w + g * 16);
      const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, mask)), eight);
      const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);
      const float* xg = x + g * 32;
      const int16x8_t l0 = vmovl_s8(vget_low_s8(lo)), l1 = vmovl_high_s8(lo);
      const int16x8_t h0 = vmovl_s8(vget_low_s8(hi)), h1 = vmovl_high_s8(hi);
      float32x4_t p = vmulq_f32(vld1q_f32(xg), vcvtq_f32_s32(vmovl_s16(vget_low_s16(l0))));
      float32x4_t q = vmulq_f32(vld1q_f32(xg + 4), vcvtq_f32_s32(vmovl_high_s16(l0)));
      p = vfmaq_f32(p, vld1q_f32(xg + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(l1))));
      q = vfmaq_f32(q, vld1q_f32(xg + 12), vcvtq_f32_s32(vmovl_high_s16(l1)));
      p = vfmaq_f32(p, vld1q_f32(xg + 16), vcvtq_f32_s32(vmovl_s16(vget_low_s16(h0))));
      q = vfmaq_f32(q, vld1q_f32(xg + 20), vcvtq_f32_s32(vmovl_high_s16(h0)));
      p = vfmaq_f32(p, vld1q_f32(xg + 24), vcvtq_f32_s32(vmovl_s16(vget_low_s16(h1))));
      q = vfmaq_f32(q, vld1q_f32(xg + 28), vcvtq_f32_s32(vmovl_high_s16(h1)));
      acc = vfmaq_n_f32(acc, vaddq_f32(p, q), scales[g]);
    }
    s = vaddvq_f32(acc);
  }
#endif
  for (; g < groups; ++g) {
    float g0 = 0.f, g1 = 0.f;
    for (int i = 0; i < 16; ++i) {
      const uint8_t b = w[g * 16 + i];
      g0 += x[g * 32 + i] * static_cast<float>(static_cast<int>(b & 0x0F) - 8);
      g1 += x[g * 32 + 16 + i] * static_cast<float>(static_cast<int>(b >> 4) - 8);
    }
    s += (g0 + g1) * scales[g];
  }
  return s;
}

inline float dot_row(const float* x, const Tensor& w, int64_t r) {
  const int64_t K = w.cols;
  switch (w.dtype) {
    case DType::F32: return dot_f32(x, static_cast<const float*>(w.data) + r * K, K);
    case DType::Q8: return dot_q8(x, static_cast<const int8_t*>(w.data) + r * K, K) * w.scales[r];
    case DType::Q4: return dot_q4(x, static_cast<const uint8_t*>(w.data) + r * (K / 2), w.scales + r * (K / 32), K);
    default: EDGELM_CHECK(false, "matmul: bad weight dtype");
  }
  return 0.f;
}

#if EDGELM_NEON
inline int32x4_t sdot16(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
  return vdotq_s32(acc, a, b);
#else  // armv8.0 (e.g. baseline Android arm64-v8a): widen-multiply + pairwise accumulate
  return vpadalq_s16(vpadalq_s16(acc, vmull_s8(vget_low_s8(a), vget_low_s8(b))), vmull_high_s8(a, b));
#endif
}
#endif

// out[t] = dot(dequant(xq_t), dequant(w_r)) for NT tokens (rows of xq/xd) against weight row r,
// as int8 x int8 -> int32 per 32-block, scaled in f32. The weight block is loaded/unpacked once
// and reused for all NT tokens.
template <int NT>
void dot_a8(const int8_t* xq, const float* xd, int64_t K, const Tensor& w, int64_t r, float* out) {
  const int64_t nb = K / kActBlock;
  if (w.dtype == DType::Q8) {
    const int8_t* wr = static_cast<const int8_t*>(w.data) + r * K;
#if EDGELM_NEON
    if (use_neon()) {
      float32x4_t acc[NT];
      for (auto& a : acc) a = vdupq_n_f32(0);
      for (int64_t b = 0; b < nb; ++b) {
        const int8x16_t w0 = vld1q_s8(wr + b * 32), w1 = vld1q_s8(wr + b * 32 + 16);
        for (int t = 0; t < NT; ++t) {
          const int8_t* xb = xq + t * K + b * 32;
          const int32x4_t s = sdot16(sdot16(vdupq_n_s32(0), w0, vld1q_s8(xb)), w1, vld1q_s8(xb + 16));
          acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(s), xd[t * nb + b]);
        }
      }
      for (int t = 0; t < NT; ++t) out[t] = vaddvq_f32(acc[t]) * w.scales[r];
      return;
    }
#endif
    for (int t = 0; t < NT; ++t) {
      float acc = 0.f;
      for (int64_t b = 0; b < nb; ++b) {
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) s += int32_t(wr[b * 32 + i]) * xq[t * K + b * 32 + i];
        acc += static_cast<float>(s) * xd[t * nb + b];
      }
      out[t] = acc * w.scales[r];
    }
    return;
  }
  // Q4: group of 32 == activation block, combined scale ws[g] * xd[g]
  const uint8_t* wr = static_cast<const uint8_t*>(w.data) + r * (K / 2);
  const float* ws = w.scales + r * nb;
#if EDGELM_NEON
  if (use_neon()) {
    float32x4_t acc[NT];
    for (auto& a : acc) a = vdupq_n_f32(0);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int64_t g = 0; g < nb; ++g) {
      const uint8x16_t bq = vld1q_u8(wr + g * 16);
      const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bq, mask)), eight);
      const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bq, 4)), eight);
      for (int t = 0; t < NT; ++t) {
        const int8_t* xb = xq + t * K + g * 32;
        const int32x4_t s = sdot16(sdot16(vdupq_n_s32(0), lo, vld1q_s8(xb)), hi, vld1q_s8(xb + 16));
        acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(s), ws[g] * xd[t * nb + g]);
      }
    }
    for (int t = 0; t < NT; ++t) out[t] = vaddvq_f32(acc[t]);
    return;
  }
#endif
  for (int t = 0; t < NT; ++t) {
    float acc = 0.f;
    for (int64_t g = 0; g < nb; ++g) {
      int32_t s = 0;
      for (int i = 0; i < 16; ++i) {
        const uint8_t b = wr[g * 16 + i];
        s += (int32_t(b & 0x0F) - 8) * xq[t * K + g * 32 + i] + (int32_t(b >> 4) - 8) * xq[t * K + g * 32 + 16 + i];
      }
      acc += static_cast<float>(s) * (ws[g] * xd[t * nb + g]);
    }
    out[t] = acc;
  }
}

// out[t * ldo] for all T tokens against weight row r (4 tokens at a time).
void row_a8(const ActBuf& aq, int T, const Tensor& w, int64_t r, float* out, int64_t ldo) {
  const int64_t K = w.cols, nb = K / kActBlock;
  float v[4];
  int t = 0;
  for (; t + 4 <= T; t += 4) {
    dot_a8<4>(aq.q + t * K, aq.d + t * nb, K, w, r, v);
    for (int i = 0; i < 4; ++i) out[(t + i) * ldo] = v[i];
  }
  for (; t < T; ++t) {
    dot_a8<1>(aq.q + t * K, aq.d + t * nb, K, w, r, v);
    out[t * ldo] = v[0];
  }
}

void quantize_rows(const float* x, int T, int64_t K, const ActBuf& aq, ThreadPool* pool) {
  auto body = [&](int64_t b, int64_t e, int) {
    for (int64_t t = b; t < e; ++t) quantize_act_row(x + t * K, K, aq.q + t * K, aq.d + t * (K / kActBlock));
  };
  if (pool && T > 8) pool->parallel_for(T, 4, body);
  else body(0, T, 0);
}

#if EDGELM_NEON
// y[i][j] = dot(x_i, w_j) for a 4x4 tile; 16 accumulators + 8 operand registers.
inline void tile4x4(const float* x, int64_t ldx, const float* w, int64_t ldw, int64_t K, float out[4][4]) {
  float32x4_t c[4][4];
  for (auto& r : c)
    for (auto& v : r) v = vdupq_n_f32(0);
  const float *x0 = x, *x1 = x + ldx, *x2 = x + 2 * ldx, *x3 = x + 3 * ldx;
  const float *w0 = w, *w1 = w + ldw, *w2 = w + 2 * ldw, *w3 = w + 3 * ldw;
  int64_t k = 0;
  for (; k + 4 <= K; k += 4) {
    const float32x4_t a[4] = {vld1q_f32(x0 + k), vld1q_f32(x1 + k), vld1q_f32(x2 + k), vld1q_f32(x3 + k)};
    const float32x4_t b[4] = {vld1q_f32(w0 + k), vld1q_f32(w1 + k), vld1q_f32(w2 + k), vld1q_f32(w3 + k)};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) c[i][j] = vfmaq_f32(c[i][j], a[i], b[j]);
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float s = vaddvq_f32(c[i][j]);
      for (int64_t kk = k; kk < K; ++kk) s += x[i * ldx + kk] * w[j * ldw + kk];
      out[i][j] = s;
    }
}
#endif

// y[t*ldy + j] = (res ? res[t*ldy + j] : 0) + dot(x_t, w_j), t < T, j < R; w is R x K f32.
void gemm_f32(const float* x, int T, const float* w, int R, int64_t K, float* y, int64_t ldy, const float* res) {
  int t0 = 0;
#if EDGELM_NEON
  if (use_neon()) {
    for (; t0 + 4 <= T; t0 += 4) {
      int j0 = 0;
      for (; j0 + 4 <= R; j0 += 4) {
        float o[4][4];
        tile4x4(x + static_cast<int64_t>(t0) * K, K, w + static_cast<int64_t>(j0) * K, K, K, o);
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            const int64_t idx = static_cast<int64_t>(t0 + i) * ldy + j0 + j;
            y[idx] = (res ? res[idx] : 0.f) + o[i][j];
          }
      }
      for (; j0 < R; ++j0)
        for (int i = 0; i < 4; ++i) {
          const int64_t idx = static_cast<int64_t>(t0 + i) * ldy + j0;
          y[idx] = (res ? res[idx] : 0.f) + dot_f32(x + static_cast<int64_t>(t0 + i) * K, w + static_cast<int64_t>(j0) * K, K);
        }
    }
  }
#endif
  for (; t0 < T; ++t0)
    for (int j = 0; j < R; ++j) {
      const int64_t idx = static_cast<int64_t>(t0) * ldy + j;
      y[idx] = (res ? res[idx] : 0.f) + dot_f32(x + static_cast<int64_t>(t0) * K, w + static_cast<int64_t>(j) * K, K);
    }
}

// Pointer to rows [r0, r0+nr) of w as f32: direct for f32, else dequantized into buf.
const float* weight_tile(const Tensor& w, int64_t r0, int nr, float* buf) {
  if (w.dtype == DType::F32) return static_cast<const float*>(w.data) + r0 * w.cols;
  for (int i = 0; i < nr; ++i) dequantize_row(w, r0 + i, buf + static_cast<int64_t>(i) * w.cols);
  return buf;
}

inline float silu1(float v) { return v / (1.0f + std::exp(-v)); }

}  // namespace

void set_simd(bool on) { g_simd.store(on); }
bool simd_enabled() { return use_neon(); }
bool simd_compiled() { return EDGELM_NEON; }

size_t scratch_floats(int T, int64_t max_k, int max_ctx, int head_dim) {
  const size_t mm = static_cast<size_t>(kRowBlock) * max_k + 2 * static_cast<size_t>(T) * kRowBlock;
  const size_t at = static_cast<size_t>(max_ctx) + head_dim;
  return std::max(mm, at) + 16;
}

float dot_f32(const float* a, const float* b, int64_t k) {
#if EDGELM_NEON
  if (use_neon()) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int64_t i = 0;
    for (; i + 16 <= k; i += 16) {
      s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
      s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
      s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
      s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= k; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < k; ++i) s += a[i] * b[i];
    return s;
  }
#endif
  return dot_f32_scalar(a, b, k);
}

void rmsnorm(const float* x, const float* w, float* y, int rows, int dim, float eps, ThreadPool* pool) {
  auto body = [&](int64_t b, int64_t e, int) {
    for (int64_t r = b; r < e; ++r) {
      const float* xr = x + r * dim;
      float* yr = y + r * dim;
      const float inv = 1.0f / std::sqrt(dot_f32(xr, xr, dim) / dim + eps);
      for (int i = 0; i < dim; ++i) yr[i] = xr[i] * inv * w[i];
    }
  };
  if (pool) pool->parallel_for(rows, 8, body);
  else body(0, rows, 0);
}

void matmul(const float* x, int T, const std::vector<MatmulTarget>& targets, ThreadPool* pool, float* const* scratch,
            const ActBuf* aq) {
  bool a8 = false;
  if (aq)
    for (const MatmulTarget& tg : targets) a8 |= act_quant_applies(*tg.w);
  if (a8) quantize_rows(x, T, targets[0].w->cols, *aq, pool);
  struct Block {
    int target;
    int64_t r0;
    int nr;
  };
  std::vector<Block> blocks;
  for (size_t i = 0; i < targets.size(); ++i)
    for (int64_t r = 0; r < targets[i].w->rows; r += kRowBlock)
      blocks.push_back({static_cast<int>(i), r, static_cast<int>(std::min<int64_t>(kRowBlock, targets[i].w->rows - r))});
  auto body = [&](int64_t b, int64_t e, int tid) {
    for (int64_t bi = b; bi < e; ++bi) {
      const Block& blk = blocks[bi];
      const MatmulTarget& tg = targets[blk.target];
      const Tensor& w = *tg.w;
      const int64_t N = w.rows;
      if (a8 && act_quant_applies(w)) {
        for (int64_t r = blk.r0; r < blk.r0 + blk.nr; ++r) {
          row_a8(*aq, T, w, r, tg.y + r, N);
          if (tg.residual)
            for (int t = 0; t < T; ++t) tg.y[t * N + r] += tg.residual[t * N + r];
        }
      } else if (T == 1) {
        for (int64_t r = blk.r0; r < blk.r0 + blk.nr; ++r)
          tg.y[r] = (tg.residual ? tg.residual[r] : 0.f) + dot_row(x, w, r);
      } else {
        const float* wt = weight_tile(w, blk.r0, blk.nr, scratch[tid]);
        gemm_f32(x, T, wt, blk.nr, w.cols, tg.y + blk.r0, N, tg.residual ? tg.residual + blk.r0 : nullptr);
      }
    }
  };
  if (pool) pool->parallel_for(static_cast<int64_t>(blocks.size()), 1, body);
  else body(0, static_cast<int64_t>(blocks.size()), 0);
}

void swiglu(const float* x, int T, const Tensor& wg, const Tensor& wu, float* y, ThreadPool* pool, float* const* scratch,
            const ActBuf* aq) {
  const int64_t N = wg.rows, K = wg.cols;
  const bool a8 = aq && act_quant_applies(wg) && act_quant_applies(wu);
  if (a8) quantize_rows(x, T, K, *aq, pool);
  const int64_t nblocks = (N + kRowBlock - 1) / kRowBlock;
  auto body = [&](int64_t b, int64_t e, int tid) {
    float* deq = scratch[tid];
    float* G = deq + kRowBlock * K;
    float* U = G + static_cast<int64_t>(T) * kRowBlock;
    for (int64_t bi = b; bi < e; ++bi) {
      const int64_t r0 = bi * kRowBlock;
      const int nr = static_cast<int>(std::min<int64_t>(kRowBlock, N - r0));
      if (a8) {  // G/U hold one column per row of the block: [T, kRowBlock]
        for (int j = 0; j < nr; ++j) {
          row_a8(*aq, T, wg, r0 + j, G + j, kRowBlock);
          row_a8(*aq, T, wu, r0 + j, U + j, kRowBlock);
        }
      } else if (T == 1) {
        for (int64_t r = r0; r < r0 + nr; ++r) y[r] = silu1(dot_row(x, wg, r)) * dot_row(x, wu, r);
        continue;
      } else {
        gemm_f32(x, T, weight_tile(wg, r0, nr, deq), nr, K, G, kRowBlock, nullptr);
        gemm_f32(x, T, weight_tile(wu, r0, nr, deq), nr, K, U, kRowBlock, nullptr);
      }
      for (int t = 0; t < T; ++t)
        for (int j = 0; j < nr; ++j)
          y[t * N + r0 + j] = silu1(G[t * kRowBlock + j]) * U[t * kRowBlock + j];
    }
  };
  if (pool) pool->parallel_for(nblocks, 1, body);
  else body(0, nblocks, 0);
}

void rope(const float* x, float* y, int T, int n_heads, int hd, int pos0, const RopeTable& rt) {
  ref::rope(x, y, T, n_heads, hd, pos0, rt);  // O(T*d), not worth a separate kernel
}

void attention(const AttnArgs& a, ThreadPool* pool, float* const* scratch) {
  const int hd = a.head_dim, h2 = hd / 2, kvd = a.n_kv * hd, qd = a.n_heads * hd, grp = a.n_heads / a.n_kv;
  const int64_t mc = a.cache.max_ctx;
  EDGELM_CHECK(a.pos0 + a.T <= mc, "context overflow");
  // 1. append this step's keys (rotated) and values to the cache
  for (int t = 0; t < a.T; ++t) {
    const int pos = a.pos0 + t;
    for (int h = 0; h < a.n_kv; ++h) {
      const float* ks = a.k + static_cast<int64_t>(t) * kvd + h * hd;
      float* kd = a.cache.k + (h * mc + pos) * hd;
      if (a.rope) {
        const float* c = &a.rope->cos[static_cast<size_t>(pos) * h2];
        const float* s = &a.rope->sin[static_cast<size_t>(pos) * h2];
        for (int i = 0; i < h2; ++i) {
          kd[i] = ks[i] * c[i] - ks[i + h2] * s[i];
          kd[i + h2] = ks[i + h2] * c[i] + ks[i] * s[i];
        }
      } else {
        std::memcpy(kd, ks, hd * sizeof(float));
      }
      std::memcpy(a.cache.v + (h * mc + pos) * hd, a.v + static_cast<int64_t>(t) * kvd + h * hd, hd * sizeof(float));
    }
  }
  // 2. one task per (query token, head)
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  auto body = [&](int64_t b, int64_t e, int tid) {
    float* qbuf = scratch[tid];
    float* sc = qbuf + hd;
    for (int64_t idx = b; idx < e; ++idx) {
      const int t = static_cast<int>(idx / a.n_heads), h = static_cast<int>(idx % a.n_heads);
      const int pos = a.pos0 + t;
      const float* q = a.q + static_cast<int64_t>(t) * qd + h * hd;
      if (a.rope) {
        const float* c = &a.rope->cos[static_cast<size_t>(pos) * h2];
        const float* s = &a.rope->sin[static_cast<size_t>(pos) * h2];
        for (int i = 0; i < h2; ++i) {
          qbuf[i] = q[i] * c[i] - q[i + h2] * s[i];
          qbuf[i + h2] = q[i + h2] * c[i] + q[i] * s[i];
        }
        q = qbuf;
      }
      const float* kc = a.cache.k + (h / grp) * mc * hd;
      const float* vc = a.cache.v + (h / grp) * mc * hd;
      const int n = pos + 1;
      float mx = -INFINITY;
      for (int j = 0; j < n; ++j) {
        sc[j] = dot_f32(q, kc + static_cast<int64_t>(j) * hd, hd) * scale;
        mx = std::max(mx, sc[j]);
      }
      float sum = 0.f;
      for (int j = 0; j < n; ++j) sum += (sc[j] = std::exp(sc[j] - mx));
      const float inv = 1.0f / sum;
      float* o = a.out + static_cast<int64_t>(t) * qd + h * hd;
      std::fill(o, o + hd, 0.f);
      for (int j = 0; j < n; ++j) {
        const float p = sc[j] * inv;
        const float* vj = vc + static_cast<int64_t>(j) * hd;
        int i = 0;
#if EDGELM_NEON
        if (use_neon()) {
          const float32x4_t pv = vdupq_n_f32(p);
          for (; i + 4 <= hd; i += 4) vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), pv, vld1q_f32(vj + i)));
        }
#endif
        for (; i < hd; ++i) o[i] += p * vj[i];
      }
    }
  };
  const int64_t items = static_cast<int64_t>(a.T) * a.n_heads;
  if (pool) pool->parallel_for(items, 1, body);
  else body(0, items, 0);
}

void add(const float* a, const float* b, float* y, int64_t n) { ref::add(a, b, y, n); }
void silu(const float* x, float* y, int64_t n) { ref::silu(x, y, n); }
void mul(const float* a, const float* b, float* y, int64_t n) { ref::mul(a, b, y, n); }

}  // namespace cpu
}  // namespace edgelm
