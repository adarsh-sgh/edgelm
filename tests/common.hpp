// Shared test helpers: locating the generated tiny model + golden arrays (tools/make_tiny.py).
#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"
#include "edgelm/model.hpp"

namespace edgelm_test {

inline std::string testdata(const std::string& file) {
  const char* d = std::getenv("EDGELM_TESTDATA");
  return std::string(d ? d : "testdata") + "/" + file;
}

struct GArray {
  std::vector<int64_t> shape;
  std::vector<float> f;
  std::vector<int32_t> i;
};

inline const std::map<std::string, GArray>& golden() {
  static std::map<std::string, GArray> g = [] {
    std::map<std::string, GArray> m;
    std::ifstream in(testdata("golden.bin"), std::ios::binary);
    if (!in) throw std::runtime_error("missing golden.bin: run `python3 tools/make_tiny.py --out testdata`");
    char magic[4];
    uint32_t n;
    in.read(magic, 4).read(reinterpret_cast<char*>(&n), 4);
    for (uint32_t k = 0; k < n; ++k) {
      uint16_t ln;
      in.read(reinterpret_cast<char*>(&ln), 2);
      std::string name(ln, '\0');
      in.read(&name[0], ln);
      uint8_t dt, nd;
      in.read(reinterpret_cast<char*>(&dt), 1).read(reinterpret_cast<char*>(&nd), 1);
      GArray a;
      a.shape.resize(nd);
      in.read(reinterpret_cast<char*>(a.shape.data()), nd * 8);
      size_t cnt = 1;
      for (int64_t s : a.shape) cnt *= static_cast<size_t>(s);
      if (dt == 3) {
        a.i.resize(cnt);
        in.read(reinterpret_cast<char*>(a.i.data()), cnt * 4);
      } else {
        a.f.resize(cnt);
        in.read(reinterpret_cast<char*>(a.f.data()), cnt * 4);
      }
      m[name] = std::move(a);
    }
    return m;
  }();
  return g;
}

inline const GArray& G(const std::string& name) {
  auto it = golden().find(name);
  if (it == golden().end()) throw std::runtime_error("golden array missing: " + name);
  return it->second;
}

// max |a - b| relative to max(1, max |b|)
inline double rel_err(const float* a, const float* b, size_t n) {
  double d = 0, m = 1;
  for (size_t i = 0; i < n; ++i) {
    d = std::max(d, static_cast<double>(std::fabs(a[i] - b[i])));
    m = std::max(m, static_cast<double>(std::fabs(b[i])));
  }
  return d / m;
}

inline std::vector<float> randn(size_t n, uint32_t seed, float sd = 1.f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

}  // namespace edgelm_test
