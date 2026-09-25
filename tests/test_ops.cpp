// Per-op numerics: NumPy goldens for both backends, and the optimized CPU kernels against the
// scalar reference over f32/q8/q4 weights, decode (T=1) and prefill (T>1) shapes with ragged tails.
#include <memory>

#include "common.hpp"
#include "edgelm/kernels.hpp"
#include "edgelm/thread_pool.hpp"

using namespace edgelm;
using namespace edgelm_test;

namespace {

struct OwnedWeight {
  Tensor t;
  std::vector<float> f;
  std::vector<uint8_t> q;
  std::vector<float> s;
};

std::unique_ptr<OwnedWeight> make_weight(const std::vector<float>& w, int64_t n, int64_t k, DType dt) {
  auto o = std::make_unique<OwnedWeight>();
  o->t.kind = TensorKind::Weight;
  o->t.rows = n;
  o->t.cols = k;
  o->t.dtype = dt;
  if (dt == DType::F32) {
    o->f = w;
    o->t.data = o->f.data();
  } else if (dt == DType::Q8) {
    o->q.resize(n * k);
    o->s.resize(n);
    quantize_q8(w.data(), n, k, reinterpret_cast<int8_t*>(o->q.data()), o->s.data());
  } else {
    o->q.resize(n * k / 2);
    o->s.resize(n * k / 32);
    quantize_q4(w.data(), n, k, o->q.data(), o->s.data());
  }
  if (dt != DType::F32) {
    o->t.data = o->q.data();
    o->t.scales = o->s.data();
  }
  return o;
}

// Dequantized copy as an f32 weight: "the quantized path's own reference".
std::unique_ptr<OwnedWeight> dequantized(const OwnedWeight& w) {
  std::vector<float> f(w.t.rows * w.t.cols);
  for (int64_t r = 0; r < w.t.rows; ++r) dequantize_row(w.t, r, f.data() + r * w.t.cols);
  return make_weight(f, w.t.rows, w.t.cols, DType::F32);
}

struct Pool {
  ThreadPool pool{4};
  std::vector<std::vector<float>> bufs;
  std::vector<float*> ptrs;
  Pool(int T, int64_t K, int ctx = 64, int hd = 16) {
    for (int i = 0; i < pool.size(); ++i) bufs.emplace_back(cpu::scratch_floats(T, K, ctx, hd));
    for (auto& b : bufs) ptrs.push_back(b.data());
  }
};

}  // namespace

TEST_CASE("per-op goldens from NumPy: rmsnorm, matmul, rope, attention, swiglu (reference and cpu)") {
  for (bool simd : {true, false}) {
    cpu::set_simd(simd);
    const GArray &x = G("op.rmsnorm.x"), &w = G("op.rmsnorm.w"), &y = G("op.rmsnorm.y");
    std::vector<float> out(y.f.size());
    ref::rmsnorm(x.f.data(), w.f.data(), out.data(), 5, 64, 1e-5f);
    CHECK(rel_err(out.data(), y.f.data(), out.size()) < 1e-6);
    Pool p(8, 128);
    cpu::rmsnorm(x.f.data(), w.f.data(), out.data(), 5, 64, 1e-5f, &p.pool);
    CHECK(rel_err(out.data(), y.f.data(), out.size()) < 1e-6);

    const GArray &wm = G("op.matmul.w"), &ym = G("op.matmul.y");
    auto W = make_weight(wm.f, 48, 64, DType::F32);
    std::vector<float> mo(ym.f.size());
    ref::matmul(x.f.data(), 5, W->t, mo.data());
    CHECK(rel_err(mo.data(), ym.f.data(), mo.size()) < 1e-6);
    for (int T : {1, 5}) {  // GEMV and GEMM paths
      std::fill(mo.begin(), mo.end(), 0.f);
      cpu::matmul(x.f.data(), T, {{&W->t, mo.data(), nullptr}}, &p.pool, p.ptrs.data());
      CHECK(rel_err(mo.data(), ym.f.data(), static_cast<size_t>(T) * 48) < 1e-6);
    }

    const GArray &xr = G("op.rope.x"), &yr = G("op.rope.y");
    RopeTable rt(16, 64, 10000.f);
    std::vector<float> ro(yr.f.size());
    ref::rope(xr.f.data(), ro.data(), 5, 4, 16, 7, rt);
    CHECK(rel_err(ro.data(), yr.f.data(), ro.size()) < 1e-6);

    // attention: keys 0..4 already cached, queries at positions 5..7 bring keys 5..7
    const GArray &q = G("op.attn.q"), &k = G("op.attn.k"), &v = G("op.attn.v"), &ya = G("op.attn.y");
    for (int backend = 0; backend < 2; ++backend) {
      std::vector<float> kc(2 * 64 * 16, 0.f), vc(kc.size(), 0.f), ao(ya.f.size());
      for (int pos = 0; pos < 5; ++pos)
        for (int h = 0; h < 2; ++h)
          for (int i = 0; i < 16; ++i) {
            kc[(h * 64 + pos) * 16 + i] = k.f[pos * 32 + h * 16 + i];
            vc[(h * 64 + pos) * 16 + i] = v.f[pos * 32 + h * 16 + i];
          }
      AttnArgs a{q.f.data(), k.f.data() + 5 * 32, v.f.data() + 5 * 32, ao.data(), 3, 5, 4, 2, 16,
                 KVLayer{kc.data(), vc.data(), 64}, nullptr};
      if (backend == 0) ref::attention(a);
      else cpu::attention(a, &p.pool, p.ptrs.data());
      CHECK(rel_err(ao.data(), ya.f.data(), ao.size()) < 1e-5);
    }

    const GArray &gt = G("op.swiglu.g"), &up = G("op.swiglu.u"), &ys = G("op.swiglu.y");
    std::vector<float> so(ys.f.size());
    ref::silu(gt.f.data(), so.data(), so.size());
    ref::mul(so.data(), up.f.data(), so.data(), so.size());
    CHECK(rel_err(so.data(), ys.f.data(), so.size()) < 1e-6);
  }
  cpu::set_simd(true);
}

TEST_CASE("cpu matmul/swiglu match the scalar reference for every dtype, T and ragged shape") {
  struct Shape {
    int64_t n, k;
  };
  // K multiple of 32 for q4; N not a multiple of the 4x4 tile or the 16-row block
  const Shape shapes[] = {{48, 64}, {37, 96}, {130, 160}};
  for (DType dt : {DType::F32, DType::Q8, DType::Q4}) {
    for (const Shape& s : shapes) {
      auto W = make_weight(randn(s.n * s.k, 11 + s.n), s.n, s.k, dt);
      auto Wd = dequantized(*W);
      auto Wu = make_weight(randn(s.n * s.k, 99 + s.n), s.n, s.k, dt);
      for (int T : {1, 3, 4, 17}) {
        CAPTURE(dtype_name(dt));
        CAPTURE(s.n);
        CAPTURE(T);
        Pool p(T, s.k);
        auto x = randn(T * s.k, 7 + T);
        auto res = randn(T * s.n, 5 + T);
        std::vector<float> want(T * s.n), deq(T * s.n), got(T * s.n);
        ref::matmul(x.data(), T, W->t, want.data(), res.data());
        ref::matmul(x.data(), T, Wd->t, deq.data(), res.data());  // dequantized-weight f32 reference
        CHECK(rel_err(want.data(), deq.data(), want.size()) < 1e-5);
        for (bool simd : {true, false}) {
          cpu::set_simd(simd);
          cpu::matmul(x.data(), T, {{&W->t, got.data(), res.data()}}, &p.pool, p.ptrs.data());
          CHECK(rel_err(got.data(), want.data(), got.size()) < 1e-5);
          std::vector<float> sw_want(T * s.n), sw_got(T * s.n);
          ref::swiglu(x.data(), T, W->t, Wu->t, sw_want.data());
          cpu::swiglu(x.data(), T, W->t, Wu->t, sw_got.data(), &p.pool, p.ptrs.data());
          CHECK(rel_err(sw_got.data(), sw_want.data(), sw_got.size()) < 1e-5);
        }
        cpu::set_simd(true);
      }
    }
  }
}

TEST_CASE("quantization error is bounded and fused multi-target matmul equals separate matmuls") {
  const int64_t n = 64, k = 256;
  auto w = randn(n * k, 3, 0.05f);
  auto q8 = make_weight(w, n, k, DType::Q8), q4 = make_weight(w, n, k, DType::Q4);
  std::vector<float> row(k);
  double e8 = 0, e4 = 0, amax = 0;
  for (int64_t r = 0; r < n; ++r) {
    dequantize_row(q8->t, r, row.data());
    for (int64_t i = 0; i < k; ++i) e8 = std::max(e8, double(std::fabs(row[i] - w[r * k + i])));
    dequantize_row(q4->t, r, row.data());
    for (int64_t i = 0; i < k; ++i) e4 = std::max(e4, double(std::fabs(row[i] - w[r * k + i])));
    for (int64_t i = 0; i < k; ++i) amax = std::max(amax, double(std::fabs(w[r * k + i])));
  }
  CHECK(e8 <= amax / 127 * 0.5 + 1e-7);  // half a step of the coarsest row scale
  CHECK(e4 <= amax / 8 + 1e-7);          // one q4 step of the coarsest group (clamp at +7)
  CHECK(e4 > e8);

  // MATMUL_N (fused QKV) is one dispatch over the concatenated rows; must equal three matmuls
  auto a = make_weight(randn(96 * 64, 1), 96, 64, DType::Q8);
  auto b = make_weight(randn(32 * 64, 2), 32, 64, DType::Q4);
  auto c = make_weight(randn(32 * 64, 3), 32, 64, DType::F32);
  for (int T : {1, 6}) {
    Pool p(T, 64);
    auto x = randn(T * 64, 9);
    std::vector<float> ya(T * 96), yb(T * 32), yc(T * 32), ra(T * 96), rb(T * 32), rc(T * 32);
    cpu::matmul(x.data(), T, {{&a->t, ya.data(), nullptr}, {&b->t, yb.data(), nullptr}, {&c->t, yc.data(), nullptr}},
                &p.pool, p.ptrs.data());
    ref::matmul(x.data(), T, a->t, ra.data());
    ref::matmul(x.data(), T, b->t, rb.data());
    ref::matmul(x.data(), T, c->t, rc.data());
    CHECK(rel_err(ya.data(), ra.data(), ya.size()) < 1e-5);
    CHECK(rel_err(yb.data(), rb.data(), yb.size()) < 1e-5);
    CHECK(rel_err(yc.data(), rc.data(), yc.size()) < 1e-5);
  }
}

TEST_CASE("int8 activations (W8A8/W4A8): quantizer matches NumPy bit-exactly, SDOT kernels match reference") {
  const GArray &xa = G("op.actq.x"), &ya = G("op.actq.y");  // (compared by value: numpy keeps -0.0)
  std::vector<int8_t> q(96);
  std::vector<float> d(3), deq(96);
  for (int t = 0; t < 3; ++t) {
    quantize_act_row(xa.f.data() + t * 96, 96, q.data(), d.data());
    for (int i = 0; i < 96; ++i) deq[i] = static_cast<float>(q[i]) * d[i / 32];
    for (int i = 0; i < 96; ++i) {
      CAPTURE(t);
      CAPTURE(i);
      CHECK(deq[i] == ya.f[t * 96 + i]);
    }
  }
  quantize_act_row(xa.f.data() + 96, 96, q.data(), d.data());
  CHECK(d[1] >= 25.f / 127.f);  // the outlier's block (row 1, block 1) gets a coarse scale ...
  CHECK(d[0] < 0.1f);           // ... its neighbours do not

  for (DType dt : {DType::Q8, DType::Q4}) {
    const int64_t n = 70, k = 160;
    auto W = make_weight(randn(n * k, 21), n, k, dt);
    auto Wu = make_weight(randn(n * k, 22), n, k, dt);
    for (int T : {1, 3, 4, 9}) {
      CAPTURE(dtype_name(dt));
      CAPTURE(T);
      Pool p(T, k);
      std::vector<int8_t> aq_q(T * k);
      std::vector<float> aq_d(T * k / 32);
      ActBuf aq{aq_q.data(), aq_d.data()};
      auto x = randn(T * k, 30 + T);
      auto res = randn(T * n, 40 + T);
      std::vector<float> want(T * n), got(T * n), f32path(T * n);
      ref::matmul(x.data(), T, W->t, want.data(), res.data(), /*act_quant=*/true);
      ref::matmul(x.data(), T, W->t, f32path.data(), res.data(), false);
      for (bool simd : {true, false}) {
        cpu::set_simd(simd);
        cpu::matmul(x.data(), T, {{&W->t, got.data(), res.data()}}, &p.pool, p.ptrs.data(), &aq);
        CHECK(rel_err(got.data(), want.data(), got.size()) < 1e-5);
        std::vector<float> sw_want(T * n), sw_got(T * n);
        ref::swiglu(x.data(), T, W->t, Wu->t, sw_want.data(), true);
        cpu::swiglu(x.data(), T, W->t, Wu->t, sw_got.data(), &p.pool, p.ptrs.data(), &aq);
        CHECK(rel_err(sw_got.data(), sw_want.data(), sw_got.size()) < 1e-5);
      }
      cpu::set_simd(true);
      // activation rounding is visible but small next to the f32-activation result
      const double e = rel_err(want.data(), f32path.data(), want.size());
      CHECK(e > 0);
      CHECK(e < 2e-2);
    }
  }
}
