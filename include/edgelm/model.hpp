// .elm loader: mmaps the file, weights point straight into the mapping (zero copy).
#pragma once
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "edgelm/graph.hpp"

namespace edgelm {

struct Config {
  int vocab = 0, dim = 0, hidden = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0, max_seq = 0;
  float rope_theta = 10000.f, norm_eps = 1e-5f;
  int bos = 0, eos = 0;
  DType weight_dtype = DType::F32;
  bool tied = false;
};

struct TokenizerData {
  std::vector<std::string> tokens;  // raw bytes
  std::vector<uint8_t> special;
  std::vector<std::array<uint32_t, 3>> merges;  // (a, b) -> result, index = rank
  int pretok = 1;                               // 1 = GPT-2 byte-level regex + split digits
};

class MappedFile;

class Model {
 public:
  static Model load(const std::string& path);

  // Load-time quantization of every f32 2-D weight (same rounding as tools/elm.py).
  void convert_weights(DType target);

  size_t weight_bytes() const;  // bytes of all weight data + scales
  size_t file_bytes() const;

  Config cfg;
  Graph graph;
  TokenizerData tok;

 private:
  std::shared_ptr<MappedFile> file_;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> owned_;
};

// Quantizers shared by convert_weights and the tests. out/scales must be presized.
void quantize_q8(const float* w, int64_t n, int64_t k, int8_t* out, float* scales);
void quantize_q4(const float* w, int64_t n, int64_t k, uint8_t* out, float* scales);  // group 32
void dequantize_row(const Tensor& w, int64_t row, float* out);

}  // namespace edgelm
