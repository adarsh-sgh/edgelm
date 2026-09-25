#include "edgelm/graph.hpp"

#include <map>
#include <sstream>

namespace edgelm {

const char* op_name(OpType t) {
  static const char* names[] = {"EMBED", "RMSNORM", "MATMUL",    "ROPE",       "ATTENTION",  "ADD", "SILU",
                                "MUL",   "LAST_ROWS", "MATMUL_N", "FFN_SWIGLU", "MATMUL_ADD"};
  return t < OpType::COUNT ? names[static_cast<int>(t)] : "?";
}

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::Q8: return "q8";
    case DType::Q4: return "q4";
    case DType::I32: return "i32";
  }
  return "?";
}

void Graph::index_io() {
  input = output = -1;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].kind == TensorKind::Input) input = static_cast<int>(i);
    if (tensors[i].kind == TensorKind::Output) output = static_cast<int>(i);
  }
  EDGELM_CHECK(input >= 0 && output >= 0, "graph needs one input and one output tensor");
}

std::vector<std::vector<int>> Graph::consumers() const {
  std::vector<std::vector<int>> c(tensors.size());
  for (size_t i = 0; i < ops.size(); ++i)
    for (int t : ops[i].in) c[t].push_back(static_cast<int>(i));
  return c;
}

std::string Graph::summary() const {
  std::map<std::string, int> h;
  for (const Op& op : ops) h[op_name(op.type)]++;
  std::ostringstream os;
  os << ops.size() << " ops (";
  bool first = true;
  for (auto& [k, v] : h) {
    os << (first ? "" : ", ") << k << " " << v;
    first = false;
  }
  os << ")";
  return os.str();
}

}  // namespace edgelm
