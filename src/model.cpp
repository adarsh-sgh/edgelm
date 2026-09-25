#include "edgelm/model.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>

namespace edgelm {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    EDGELM_CHECK(fd >= 0, "cannot open " + path);
    struct stat st {};
    ::fstat(fd, &st);
    size_ = static_cast<size_t>(st.st_size);
    if (size_ > 0) ptr_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    EDGELM_CHECK(size_ > 0 && ptr_ != MAP_FAILED, "mmap failed: " + path);
  }
  ~MappedFile() {
    if (ptr_ && ptr_ != MAP_FAILED) ::munmap(ptr_, size_);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return static_cast<const uint8_t*>(ptr_); }
  size_t size() const { return size_; }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

namespace {

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
  template <class T>
  T get() {
    EDGELM_CHECK(p_ + sizeof(T) <= end_, "truncated model file");
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  std::string str(size_t n) {
    EDGELM_CHECK(p_ + n <= end_, "truncated model file");
    std::string s(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return s;
  }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

float round_away(float v) { return std::copysign(std::floor(std::fabs(v) + 0.5f), v); }

}  // namespace

Model Model::load(const std::string& path) {
  Model m;
  m.file_ = std::make_shared<MappedFile>(path);
  const uint8_t* base = m.file_->data();
  const size_t size = m.file_->size();
  EDGELM_CHECK(size >= 128 && std::memcmp(base, "ELM1", 4) == 0, "not an .elm file: " + path);
  Reader h(base + 4, 124);
  const uint32_t version = h.get<uint32_t>(), n_tensors = h.get<uint32_t>(), n_ops = h.get<uint32_t>();
  EDGELM_CHECK(version == 1, "unsupported .elm version");
  const uint64_t graph_off = h.get<uint64_t>(), graph_size = h.get<uint64_t>();
  const uint64_t tok_off = h.get<uint64_t>(), tok_size = h.get<uint64_t>();
  const uint64_t w_off = h.get<uint64_t>(), w_size = h.get<uint64_t>();
  EDGELM_CHECK(graph_off + graph_size <= size && tok_off + tok_size <= size && w_off + w_size <= size,
               "section out of bounds");
  Config& c = m.cfg;
  int* ints[] = {&c.vocab, &c.dim, &c.hidden, &c.n_layers, &c.n_heads, &c.n_kv_heads, &c.head_dim, &c.max_seq};
  for (int* p : ints) *p = static_cast<int>(h.get<uint32_t>());
  c.rope_theta = h.get<float>();
  c.norm_eps = h.get<float>();
  c.bos = static_cast<int>(h.get<uint32_t>());
  c.eos = static_cast<int>(h.get<uint32_t>());
  c.weight_dtype = static_cast<DType>(h.get<uint32_t>());
  h.get<uint32_t>();  // group size (also per tensor)
  c.tied = h.get<uint32_t>() & 1;

  Reader g(base + graph_off, graph_size);
  const uint8_t* wbase = base + w_off;
  for (uint32_t i = 0; i < n_tensors; ++i) {
    Tensor t;
    t.name = g.str(g.get<uint16_t>());
    t.kind = static_cast<TensorKind>(g.get<uint8_t>());
    t.dtype = static_cast<DType>(g.get<uint8_t>());
    t.rows_sym = static_cast<RowSym>(g.get<uint8_t>());
    g.get<uint8_t>();
    t.rows = g.get<int64_t>();
    t.cols = g.get<int64_t>();
    const uint64_t doff = g.get<uint64_t>(), dbytes = g.get<uint64_t>();
    const uint64_t soff = g.get<uint64_t>(), sbytes = g.get<uint64_t>();
    t.group = static_cast<int>(g.get<uint32_t>());
    if (t.is_weight()) {
      EDGELM_CHECK(doff + dbytes <= w_size && soff + sbytes <= w_size, "weight out of bounds: " + t.name);
      t.data = wbase + doff;
      t.data_bytes = dbytes;
      if (sbytes) {
        t.scales = reinterpret_cast<const float*>(wbase + soff);
        t.scale_bytes = sbytes;
      }
    }
    m.graph.tensors.push_back(std::move(t));
  }
  for (uint32_t i = 0; i < n_ops; ++i) {
    Op op;
    op.type = static_cast<OpType>(g.get<uint8_t>());
    EDGELM_CHECK(op.type < OpType::COUNT, "unknown op type");
    const int ni = g.get<uint8_t>(), no = g.get<uint8_t>();
    g.get<uint8_t>();
    for (int j = 0; j < ni; ++j) op.in.push_back(g.get<int32_t>());
    for (int j = 0; j < no; ++j) op.out.push_back(g.get<int32_t>());
    for (int& a : op.iattr) a = g.get<int32_t>();
    for (float& a : op.fattr) a = g.get<float>();
    op.name = g.str(g.get<uint16_t>());
    for (int t : op.in) EDGELM_CHECK(t >= 0 && t < static_cast<int>(n_tensors), "bad tensor ref");
    for (int t : op.out) EDGELM_CHECK(t >= 0 && t < static_cast<int>(n_tensors), "bad tensor ref");
    m.graph.ops.push_back(std::move(op));
  }
  m.graph.index_io();

  Reader tk(base + tok_off, tok_size);
  const uint32_t nv = tk.get<uint32_t>();
  m.tok.pretok = static_cast<int>(tk.get<uint32_t>());
  m.tok.tokens.resize(nv);
  m.tok.special.resize(nv);
  for (uint32_t i = 0; i < nv; ++i) {
    m.tok.tokens[i] = tk.str(tk.get<uint16_t>());
    m.tok.special[i] = tk.get<uint8_t>();
  }
  const uint32_t nm = tk.get<uint32_t>();
  m.tok.merges.resize(nm);
  for (auto& mg : m.tok.merges)
    for (auto& v : mg) v = tk.get<uint32_t>();
  return m;
}

void quantize_q8(const float* w, int64_t n, int64_t k, int8_t* out, float* scales) {
  for (int64_t r = 0; r < n; ++r) {
    const float* row = w + r * k;
    float amax = 0.f;
    for (int64_t i = 0; i < k; ++i) amax = std::max(amax, std::fabs(row[i]));
    const float s = amax / 127.0f;
    const float safe = s == 0.f ? 1.f : s;
    scales[r] = s;
    for (int64_t i = 0; i < k; ++i) {
      float q = round_away(row[i] / safe);
      out[r * k + i] = static_cast<int8_t>(std::min(127.f, std::max(-127.f, q)));
    }
  }
}

void quantize_q4(const float* w, int64_t n, int64_t k, uint8_t* out, float* scales) {
  constexpr int G = 32;
  EDGELM_CHECK(k % G == 0, "q4 needs K % 32 == 0");
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t g = 0; g < k / G; ++g) {
      const float* x = w + r * k + g * G;
      int best = 0;  // first element of max magnitude, like numpy argmax
      float amax = std::fabs(x[0]);
      for (int i = 1; i < G; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) {
          amax = a;
          best = i;
        }
      }
      const float d = x[best] / -8.0f;
      const float safe = d == 0.f ? 1.f : d;
      scales[r * (k / G) + g] = d;
      uint8_t* o = out + r * (k / 2) + g * (G / 2);
      for (int i = 0; i < G / 2; ++i) {
        auto q = [&](float v) { return static_cast<uint8_t>(std::min(7.f, std::max(-8.f, round_away(v / safe))) + 8); };
        o[i] = static_cast<uint8_t>(q(x[i]) | (q(x[i + G / 2]) << 4));
      }
    }
  }
}

void dequantize_row(const Tensor& w, int64_t row, float* out) {
  const int64_t k = w.cols;
  switch (w.dtype) {
    case DType::F32:
      std::memcpy(out, static_cast<const float*>(w.data) + row * k, k * sizeof(float));
      break;
    case DType::Q8: {
      const int8_t* q = static_cast<const int8_t*>(w.data) + row * k;
      const float s = w.scales[row];
      for (int64_t i = 0; i < k; ++i) out[i] = static_cast<float>(q[i]) * s;
      break;
    }
    case DType::Q4: {
      const uint8_t* q = static_cast<const uint8_t*>(w.data) + row * (k / 2);
      const float* s = w.scales + row * (k / 32);
      for (int64_t g = 0; g < k / 32; ++g)
        for (int i = 0; i < 16; ++i) {
          const uint8_t b = q[g * 16 + i];
          out[g * 32 + i] = static_cast<float>(static_cast<int>(b & 0x0F) - 8) * s[g];
          out[g * 32 + 16 + i] = static_cast<float>(static_cast<int>(b >> 4) - 8) * s[g];
        }
      break;
    }
    default:
      EDGELM_CHECK(false, "dequantize_row: bad dtype");
  }
}

void Model::convert_weights(DType target) {
  if (target == DType::F32) {
    EDGELM_CHECK(cfg.weight_dtype == DType::F32, "cannot dequantize a quantized file to f32");
    return;
  }
  EDGELM_CHECK(target == DType::Q8 || target == DType::Q4, "convert_weights: target must be q8/q4");
  for (Tensor& t : graph.tensors) {
    if (!t.is_weight() || t.rows <= 1 || t.dtype != DType::F32) continue;
    const float* w = static_cast<const float*>(t.data);
    auto data = std::make_shared<std::vector<uint8_t>>();
    auto sc = std::make_shared<std::vector<uint8_t>>();
    if (target == DType::Q8) {
      data->resize(t.rows * t.cols);
      sc->resize(t.rows * sizeof(float));
      quantize_q8(w, t.rows, t.cols, reinterpret_cast<int8_t*>(data->data()), reinterpret_cast<float*>(sc->data()));
      t.group = static_cast<int>(t.cols);
    } else {
      data->resize(t.rows * t.cols / 2);
      sc->resize(t.rows * (t.cols / 32) * sizeof(float));
      quantize_q4(w, t.rows, t.cols, data->data(), reinterpret_cast<float*>(sc->data()));
      t.group = 32;
    }
    t.dtype = target;
    t.data = data->data();
    t.data_bytes = data->size();
    t.scales = reinterpret_cast<const float*>(sc->data());
    t.scale_bytes = sc->size();
    owned_.push_back(data);
    owned_.push_back(sc);
  }
  cfg.weight_dtype = target;
}

size_t Model::weight_bytes() const {
  size_t n = 0;
  for (const Tensor& t : graph.tensors)
    if (t.is_weight()) n += t.data_bytes + t.scale_bytes;
  return n;
}

void Model::prefault() const {
  if (!file_) return;
  ::madvise(const_cast<uint8_t*>(file_->data()), file_->size(), MADV_WILLNEED);
  volatile uint8_t sink = 0;
  for (size_t off = 0; off < file_->size(); off += 4096) sink = sink + file_->data()[off];
  (void)sink;
}

size_t Model::file_bytes() const { return file_ ? file_->size() : 0; }

}  // namespace edgelm
