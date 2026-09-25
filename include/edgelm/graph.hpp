// Graph IR: tensors (weights mmapped from the model file, activations planned into an arena)
// and an ordered op list. Op ids must match tools/elm.py.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace edgelm {

#define EDGELM_CHECK(cond, msg)                                                                  \
  do {                                                                                           \
    if (!(cond)) throw std::runtime_error(std::string("edgelm: ") + (msg) + " [" #cond "]");     \
  } while (0)

enum class DType : uint8_t { F32 = 0, Q8 = 1, Q4 = 2, I32 = 3 };
enum class TensorKind : uint8_t { Weight = 0, Activation = 1, Input = 2, Output = 3 };
enum class RowSym : uint8_t { Const = 0, T = 1, L = 2 };  // rows = fixed, tokens in step, logit rows

enum class OpType : uint8_t {
  EMBED = 0, RMSNORM, MATMUL, ROPE, ATTENTION, ADD, SILU, MUL, LAST_ROWS,
  // produced by fusion passes
  MATMUL_N,    // one input, N weights, N outputs (fused QKV)
  FFN_SWIGLU,  // silu(x Wg^T) * (x Wu^T)
  MATMUL_ADD,  // r + x W^T (residual epilogue)
  COUNT
};

const char* op_name(OpType t);
const char* dtype_name(DType d);

struct Tensor {
  std::string name;
  TensorKind kind = TensorKind::Activation;
  DType dtype = DType::F32;
  RowSym rows_sym = RowSym::Const;
  int64_t rows = 0, cols = 0;
  // weights only
  const void* data = nullptr;
  const float* scales = nullptr;  // q8: [rows], q4: [rows * cols / group]
  size_t data_bytes = 0, scale_bytes = 0;
  int group = 0;
  bool is_weight() const { return kind == TensorKind::Weight; }
};

struct Op {
  OpType type = OpType::COUNT;
  std::string name;
  std::vector<int> in, out;
  int iattr[4] = {0, 0, 0, 0};  // ROPE: n_heads, head_dim | ATTENTION: layer, n_heads, n_kv, fused_rope
  float fattr[2] = {0, 0};      // RMSNORM: eps | ROPE/ATTENTION: theta
};

struct Graph {
  std::vector<Tensor> tensors;
  std::vector<Op> ops;
  int input = -1;   // tokens
  int output = -1;  // logits
  void index_io();
  // consumers[t] = op indices reading tensor t
  std::vector<std::vector<int>> consumers() const;
  std::string summary() const;  // op histogram
};

}  // namespace edgelm
