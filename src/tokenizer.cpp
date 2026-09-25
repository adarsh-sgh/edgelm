#include "edgelm/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace edgelm {
namespace {

enum Cls { LETTER, NUMBER, SPACE, OTHER };

Cls classify(uint32_t cp) {
  if (cp < 128) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return LETTER;
    if (cp >= '0' && cp <= '9') return NUMBER;
    if (cp == ' ' || (cp >= 9 && cp <= 13)) return SPACE;
    return OTHER;
  }
  if (cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
      cp == 0x202F || cp == 0x205F || cp == 0x3000)
    return SPACE;
  if (cp < 0xC0) {
    if (cp == 0xB2 || cp == 0xB3 || cp == 0xB9 || (cp >= 0xBC && cp <= 0xBE)) return NUMBER;
    if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return LETTER;
    return OTHER;
  }
  if (cp == 0xD7 || cp == 0xF7) return OTHER;
  if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x2000 && cp <= 0x2BFF) || (cp >= 0x3000 && cp <= 0x303F) ||
      (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0x1F000 && cp <= 0x1FAFF))
    return OTHER;
  return LETTER;
}

struct Cp {
  uint32_t cp;
  size_t off, len;
};

std::vector<Cp> decode_utf8(const std::string& s) {
  std::vector<Cp> out;
  size_t i = 0;
  while (i < s.size()) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + n > s.size()) n = 1;
    uint32_t cp = n == 1 ? c : c & (0x7F >> n);
    for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<uint8_t>(s[i + k]) & 0x3F);
    out.push_back({cp, i, n});
    i += n;
  }
  return out;
}

// GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
void gpt2_split(const std::string& s, std::vector<std::string>& out) {
  const std::vector<Cp> c = decode_utf8(s);
  const size_t n = c.size();
  auto cls = [&](size_t i) { return classify(c[i].cp); };
  auto emit = [&](size_t b, size_t e) { out.push_back(s.substr(c[b].off, c[e - 1].off + c[e - 1].len - c[b].off)); };
  size_t i = 0;
  while (i < n) {
    if (c[i].cp == '\'' && i + 1 < n) {
      const uint32_t a = c[i + 1].cp, b = i + 2 < n ? c[i + 2].cp : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        emit(i, i + 2);
        i += 2;
        continue;
      }
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
        emit(i, i + 3);
        i += 3;
        continue;
      }
    }
    size_t j = i;
    if (c[j].cp == ' ' && j + 1 < n && cls(j + 1) != SPACE) ++j;
    const Cls k = cls(j);
    if (k != SPACE) {
      while (j < n && cls(j) == k) ++j;
      emit(i, j);
      i = j;
      continue;
    }
    size_t r = i;
    while (r < n && cls(r) == SPACE) ++r;
    const size_t e = (r < n && r - i >= 2) ? r - 1 : r;
    emit(i, e);
    i = e;
  }
}

}  // namespace

std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : text) {  // individual digits (ASCII) become their own pieces
    if (ch >= '0' && ch <= '9') {
      if (!cur.empty()) gpt2_split(cur, out);
      cur.clear();
      out.emplace_back(1, ch);
    } else {
      cur += ch;
    }
  }
  if (!cur.empty()) gpt2_split(cur, out);
  return out;
}

Tokenizer::Tokenizer(const TokenizerData& d) : tokens_(d.tokens) {
  std::fill(std::begin(byte_id_), std::end(byte_id_), -1);
  for (size_t i = 0; i < tokens_.size(); ++i) {
    if (d.special[i]) {
      if (!tokens_[i].empty()) specials_.push_back({tokens_[i], static_cast<int32_t>(i)});
    } else if (tokens_[i].size() == 1) {
      byte_id_[static_cast<uint8_t>(tokens_[i][0])] = static_cast<int32_t>(i);
    }
  }
  // longest first so overlapping specials resolve to the longer one
  std::sort(specials_.begin(), specials_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
  for (size_t r = 0; r < d.merges.size(); ++r) {
    const auto& m = d.merges[r];
    merges_.emplace((static_cast<uint64_t>(m[0]) << 32) | m[1], std::make_pair(static_cast<uint32_t>(r), m[2]));
  }
}

void Tokenizer::encode_piece(const std::string& piece, std::vector<int32_t>& out) const {
  std::vector<int32_t> ids;
  for (char ch : piece) {
    const int32_t id = byte_id_[static_cast<uint8_t>(ch)];
    EDGELM_CHECK(id >= 0, "tokenizer: byte missing from vocab");
    ids.push_back(id);
  }
  while (ids.size() > 1) {
    uint32_t best = std::numeric_limits<uint32_t>::max(), result = 0;
    size_t at = 0;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      auto it = merges_.find((static_cast<uint64_t>(ids[i]) << 32) | static_cast<uint32_t>(ids[i + 1]));
      if (it != merges_.end() && it->second.first < best) {
        best = it->second.first;
        result = it->second.second;
        at = i;
      }
    }
    if (best == std::numeric_limits<uint32_t>::max()) break;
    ids[at] = static_cast<int32_t>(result);
    ids.erase(ids.begin() + at + 1);
  }
  out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> out;
  size_t start = 0, i = 0;
  auto flush = [&](size_t end) {
    if (end > start)
      for (const std::string& p : pretokenize(text.substr(start, end - start))) encode_piece(p, out);
  };
  while (i < text.size()) {
    bool hit = false;
    for (const auto& [s, id] : specials_)
      if (text.compare(i, s.size(), s) == 0) {
        flush(i);
        out.push_back(id);
        i += s.size();
        start = i;
        hit = true;
        break;
      }
    if (!hit) ++i;
  }
  flush(text.size());
  return out;
}

std::string Tokenizer::decode(int32_t id) const {
  EDGELM_CHECK(id >= 0 && id < vocab_size(), "decode: id out of range");
  return tokens_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string s;
  for (int32_t id : ids) s += decode(id);
  return s;
}

int32_t argmax(const float* logits, int n) {
  return static_cast<int32_t>(std::max_element(logits, logits + n) - logits);
}

Sampler::Sampler(SamplerConfig c) : c_(c), state_(c.seed * 0x9E3779B97F4A7C15ULL + 1) {}

int32_t Sampler::sample(const float* logits, int n) {
  if (c_.temperature <= 0.f || c_.top_k == 1) return argmax(logits, n);
  const int k = c_.top_k > 0 ? std::min(c_.top_k, n) : n;
  buf_.resize(n);
  for (int i = 0; i < n; ++i) buf_[i] = {logits[i], i};
  std::partial_sort(buf_.begin(), buf_.begin() + k, buf_.end(), [](auto& a, auto& b) { return a.first > b.first; });
  double sum = 0;
  std::vector<double> p(k);
  for (int i = 0; i < k; ++i) sum += (p[i] = std::exp((buf_[i].first - buf_[0].first) / c_.temperature));
  state_ ^= state_ << 13;  // xorshift64: reproducible across platforms, unlike std distributions
  state_ ^= state_ >> 7;
  state_ ^= state_ << 17;
  double r = (state_ >> 11) * (1.0 / 9007199254740992.0) * sum;
  for (int i = 0; i < k; ++i) {
    r -= p[i];
    if (r <= 0) return buf_[i].second;
  }
  return buf_[k - 1].second;
}

}  // namespace edgelm
