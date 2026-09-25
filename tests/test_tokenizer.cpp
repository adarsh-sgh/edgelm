// Tokenizer vs goldens from tools/make_tiny.py (Python BPE over the same merges), plus the
// GPT-2 pre-tokenizer rules. The real SmolLM2 tokenizer is checked against HF `tokenizers` by
// tools/check_real.py.
#include <algorithm>

#include "common.hpp"
#include "edgelm/tokenizer.hpp"

using namespace edgelm;
using namespace edgelm_test;

TEST_CASE("tokenizer: encode matches Python BPE goldens, decode round-trips, specials split") {
  Model m = Model::load(testdata("tiny.f32.elm"));
  Tokenizer tok(m.tok);
  for (int i = 0; i < 5; ++i) {
    const GArray& t = G("tok.text." + std::to_string(i));
    const GArray& want = G("tok.ids." + std::to_string(i));
    std::string s;
    for (int32_t b : t.i) s += static_cast<char>(b);
    CAPTURE(s);
    const std::vector<int32_t> got = tok.encode(s);
    CHECK(got == want.i);
    CHECK(tok.decode(got) == s);
  }
  // merges are applied: common words collapse to fewer tokens than bytes
  CHECK(tok.encode(" the fox").size() < 8);
  // special tokens are matched before BPE and survive round trip
  const auto ids = tok.encode("fox<|endoftext|>dog");
  CHECK(std::count(ids.begin(), ids.end(), 0) == 1);
  CHECK(tok.decode(ids) == "fox<|endoftext|>dog");
  // arbitrary bytes (invalid UTF-8 included) are byte-level encodable
  const std::string bytes = "\xff\xfe\x80 caf\xc3\xa9";
  CHECK(tok.decode(tok.encode(bytes)) == bytes);
}

TEST_CASE("pre-tokenizer: GPT-2 regex rules and individual digits") {
  using V = std::vector<std::string>;
  CHECK(pretokenize("Hello world") == V{"Hello", " world"});
  CHECK(pretokenize("it's we'll") == V{"it", "'s", " we", "'ll"});
  CHECK(pretokenize("a  b") == V{"a", " ", " b"});           // \s+(?!\S) leaves one space for the word
  CHECK(pretokenize("a\n\nb") == V{"a", "\n", "\n", "b"});   // newline is not an optional-space prefix
  CHECK(pretokenize("x 123") == V{"x", " ", "1", "2", "3"});  // digits isolated before the regex
  CHECK(pretokenize("end.  ") == V{"end", ".", "  "});
  CHECK(pretokenize("caf\xc3\xa9 \xe2\x80\x94 ok") == V{"caf\xc3\xa9", " \xe2\x80\x94", " ok"});
  CHECK(pretokenize("").empty());
}
