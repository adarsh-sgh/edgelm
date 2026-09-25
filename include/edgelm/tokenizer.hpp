// Byte-level BPE (GPT-2 style, as used by SmolLM2): special-token split -> individual digits ->
// GPT-2 pre-tokenizer regex (hand-written scanner) -> rank-ordered merges over raw bytes.
// Unicode classes are exact for ASCII; non-ASCII code points use a small table (Latin-1,
// punctuation, symbol and emoji blocks) and default to "letter".
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "edgelm/model.hpp"

namespace edgelm {

std::vector<std::string> pretokenize(const std::string& text);

class Tokenizer {
 public:
  explicit Tokenizer(const TokenizerData& d);
  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(int32_t id) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  int vocab_size() const { return static_cast<int>(tokens_.size()); }

 private:
  void encode_piece(const std::string& piece, std::vector<int32_t>& out) const;
  std::vector<std::string> tokens_;
  std::vector<std::pair<std::string, int32_t>> specials_;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> merges_;  // (a,b) -> (rank, result)
  int32_t byte_id_[256];
};

struct SamplerConfig {
  float temperature = 0.f;  // 0 = greedy
  int top_k = 40;
  uint64_t seed = 42;
};

int32_t argmax(const float* logits, int n);

class Sampler {
 public:
  explicit Sampler(SamplerConfig c);
  int32_t sample(const float* logits, int n);

 private:
  SamplerConfig c_;
  uint64_t state_;
  std::vector<std::pair<float, int32_t>> buf_;
};

}  // namespace edgelm
