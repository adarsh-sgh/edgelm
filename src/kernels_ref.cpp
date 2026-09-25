// Scalar reference kernels: serial, no blocking, dequantize-then-dot. The oracle for cpu::.
#include <algorithm>
#include <cmath>
#include <vector>

#include "edgelm/kernels.hpp"
#include "edgelm/model.hpp"

namespace edgelm {

RopeTable::RopeTable(int head_dim, int max_pos_, float theta) : half(head_dim / 2), max_pos(max_pos_) {
  cos.resize(static_cast<size_t>(max_pos) * half);
  sin.resize(cos.size());
  for (int p = 0; p < max_pos; ++p)
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(static_cast<double>(theta), -2.0 * i / head_dim);
      const double a = p * inv;
      cos[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(a));
      sin[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(a));
    }
}

void quantize_act_row(const float* x, int64_t k, int8_t* q, float* d) {
  for (int64_t b = 0; b < k / kActBlock; ++b) {
    const float* xb = x + b * kActBlock;
    float amax = 0.f;
    for (int i = 0; i < kActBlock; ++i) amax = std::max(amax, std::fabs(xb[i]));
    const float s = amax / 127.0f;
    const float safe = s == 0.f ? 1.f : s;
    d[b] = s;
    for (int i = 0; i < kActBlock; ++i) {
      const float v = xb[i] / safe;
      const float r = std::copysign(std::floor(std::fabs(v) + 0.5f), v);
      q[b * kActBlock + i] = static_cast<int8_t>(std::min(127.f, std::max(-127.f, r)));
    }
  }
}

namespace ref {

void embed(const int32_t* tokens, int T, const Tensor& table, float* y) {
  for (int t = 0; t < T; ++t) {
    EDGELM_CHECK(tokens[t] >= 0 && tokens[t] < table.rows, "token id out of range");
    dequantize_row(table, tokens[t], y + static_cast<int64_t>(t) * table.cols);
  }
}

void rmsnorm(const float* x, const float* w, float* y, int rows, int dim, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<int64_t>(r) * dim;
    float ss = 0.f;
    for (int i = 0; i < dim; ++i) ss += xr[i] * xr[i];
    const float inv = 1.0f / std::sqrt(ss / dim + eps);
    for (int i = 0; i < dim; ++i) y[static_cast<int64_t>(r) * dim + i] = xr[i] * inv * w[i];
  }
}

void matmul(const float* x, int T, const Tensor& w, float* y, const float* residual, bool act_quant) {
  const int64_t N = w.rows, K = w.cols;
  std::vector<float> xf;
  if (act_quant && act_quant_applies(w)) {
    xf.resize(static_cast<size_t>(T) * K);
    std::vector<int8_t> q(K);
    std::vector<float> d(K / kActBlock);
    for (int t = 0; t < T; ++t) {
      quantize_act_row(x + t * K, K, q.data(), d.data());
      for (int64_t i = 0; i < K; ++i) xf[t * K + i] = static_cast<float>(q[i]) * d[i / kActBlock];
    }
    x = xf.data();
  }
  std::vector<float> row(K);
  for (int64_t n = 0; n < N; ++n) {
    dequantize_row(w, n, row.data());
    for (int t = 0; t < T; ++t) {
      float acc = 0.f;
      for (int64_t k = 0; k < K; ++k) acc += x[t * K + k] * row[k];
      y[t * N + n] = residual ? residual[t * N + n] + acc : acc;
    }
  }
}

void rope(const float* x, float* y, int T, int n_heads, int hd, int pos0, const RopeTable& rt) {
  const int h2 = hd / 2;
  for (int t = 0; t < T; ++t) {
    EDGELM_CHECK(pos0 + t < rt.max_pos, "position beyond rope table");
    const float* c = &rt.cos[static_cast<size_t>(pos0 + t) * h2];
    const float* s = &rt.sin[static_cast<size_t>(pos0 + t) * h2];
    for (int h = 0; h < n_heads; ++h) {
      const float* xi = x + (static_cast<int64_t>(t) * n_heads + h) * hd;
      float* yo = y + (static_cast<int64_t>(t) * n_heads + h) * hd;
      for (int i = 0; i < h2; ++i) {
        const float a = xi[i], b = xi[i + h2];
        yo[i] = a * c[i] - b * s[i];
        yo[i + h2] = b * c[i] + a * s[i];
      }
    }
  }
}

void attention(const AttnArgs& a) {
  const int hd = a.head_dim, kvd = a.n_kv * hd, qd = a.n_heads * hd, grp = a.n_heads / a.n_kv;
  EDGELM_CHECK(a.pos0 + a.T <= a.cache.max_ctx, "context overflow");
  std::vector<float> kr(static_cast<size_t>(a.T) * kvd), qr(static_cast<size_t>(a.T) * qd);
  const float* k = a.k;
  const float* q = a.q;
  if (a.rope) {
    rope(a.k, kr.data(), a.T, a.n_kv, hd, a.pos0, *a.rope);
    rope(a.q, qr.data(), a.T, a.n_heads, hd, a.pos0, *a.rope);
    k = kr.data();
    q = qr.data();
  }
  for (int t = 0; t < a.T; ++t)
    for (int h = 0; h < a.n_kv; ++h)
      for (int i = 0; i < hd; ++i) {
        const size_t dst = (static_cast<size_t>(h) * a.cache.max_ctx + a.pos0 + t) * hd + i;
        a.cache.k[dst] = k[static_cast<size_t>(t) * kvd + h * hd + i];
        a.cache.v[dst] = a.v[static_cast<size_t>(t) * kvd + h * hd + i];
      }
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  std::vector<float> sc(a.pos0 + a.T);
  for (int t = 0; t < a.T; ++t)
    for (int h = 0; h < a.n_heads; ++h) {
      const float* qv = q + static_cast<size_t>(t) * qd + h * hd;
      const float* kc = a.cache.k + static_cast<size_t>(h / grp) * a.cache.max_ctx * hd;
      const float* vc = a.cache.v + static_cast<size_t>(h / grp) * a.cache.max_ctx * hd;
      const int n = a.pos0 + t + 1;  // causal
      float mx = -INFINITY;
      for (int j = 0; j < n; ++j) {
        float d = 0.f;
        for (int i = 0; i < hd; ++i) d += qv[i] * kc[static_cast<size_t>(j) * hd + i];
        sc[j] = d * scale;
        mx = std::max(mx, sc[j]);
      }
      float sum = 0.f;
      for (int j = 0; j < n; ++j) sum += (sc[j] = std::exp(sc[j] - mx));
      float* o = a.out + static_cast<size_t>(t) * qd + h * hd;
      for (int i = 0; i < hd; ++i) o[i] = 0.f;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < hd; ++i) o[i] += sc[j] / sum * vc[static_cast<size_t>(j) * hd + i];
    }
}

void add(const float* a, const float* b, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
}
void silu(const float* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = x[i] / (1.0f + std::exp(-x[i]));
}
void mul(const float* a, const float* b, float* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = a[i] * b[i];
}

void swiglu(const float* x, int T, const Tensor& wg, const Tensor& wu, float* y, bool act_quant) {
  const int64_t n = static_cast<int64_t>(T) * wg.rows;
  std::vector<float> g(n), u(n);
  matmul(x, T, wg, g.data(), nullptr, act_quant);
  matmul(x, T, wu, u.data(), nullptr, act_quant);
  silu(g.data(), g.data(), n);
  mul(g.data(), u.data(), y, n);
}

}  // namespace ref
}  // namespace edgelm
