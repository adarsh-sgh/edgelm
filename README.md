# edgelm

A C++17 on-device LLM inference runtime with no third-party runtime dependencies (doctest is the
only vendored header, for tests). It runs [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
a Llama-architecture model with GQA (9 query / 3 KV heads), RoPE, RMSNorm, SwiGLU and tied
embeddings, from one mmapped model file. The layout follows a TFLite/LiteRT-style runtime. An
offline converter writes a flat file holding an op graph and aligned weight blobs. Graph passes
fuse ops and pack every activation into one arena. A partitioner hands op subgraphs to
delegate-style backends and lets unclaimed ops fall back to a reference backend. Execution keeps a
preallocated KV cache, with separate prefill (GEMM) and decode (GEMV) paths. Weights can be int8
per-channel or int4 group-32, optionally with dynamic int8 activations on SDOT kernels. The
profiler writes per-op Chrome traces that open in Perfetto.

Every number below was measured on one machine, with the commands shown next to it.

```
 HF safetensors + tokenizer.json
        |  tools/export.py  (NumPy only: bf16 -> f32, q8 / q4 quantization, BPE -> raw-byte merges)
        v
 model.elm  = [header | graph: tensors + ops | tokenizer | 64B-aligned weight blobs]
        |  Model::load: mmap, weights are pointers into the mapping (zero copy)
        v
+--------------------------------------------------------------------------------+
| Graph IR   484 ops: EMBED, RMSNORM, MATMUL, ROPE, ATTENTION, ADD, SILU, MUL ... |
| passes     fuse_swiglu        MATMUL(gate), MATMUL(up), SILU, MUL -> FFN_SWIGLU |
|            fuse_shared_input  MATMULs reading one tensor -> MATMUL_N (QKV)      |
|            fuse_residual      MATMUL + ADD -> MATMUL_ADD (epilogue)             |
|            fuse_rope_attention ROPE(q), ROPE(k) -> ATTENTION(rope)  => 214 ops  |
| planner    live ranges -> greedy-by-size best-fit offsets in ONE arena          |
+--------------------------------------------------------------------------------+
        |  partition_graph(preference = [cpu | matmul-only, ..., reference])
        v
+----------------------------+  +---------------------------+  +---------------------+
| cpu backend                |  | matmul-only backend       |  | reference backend   |
| NEON (+SDOT) microkernels, |  | claims MATMUL* with f32/q8|  | scalar, serial,     |
| scalar fallback, threaded  |  | weights only (NPU stand-in)| | every op and dtype  |
+----------------------------+  +---------------------------+  +---------------------+
        |
        v
 Session: KV cache [layer][kv_head][pos][64] f32, RoPE tables, per-thread scratch
   prefill(tokens): chunks of <= max_batch tokens, T > 1 -> GEMM paths, last-row logits
   decode(token):   T = 1 -> GEMV paths, attention over the cache
   Sampler: greedy | temperature + top-k (xorshift, reproducible)   Tokenizer: byte-level BPE
   Profiler: per-op spans -> Chrome trace JSON (--profile), per-op-type table (--stats)
```

## Run

```
make venv model          # python venv (numpy); download SmolLM2-135M, export f32/q8/q4/q4all .elm
make build test          # tiny random-init model + NumPy goldens, then the doctest suite
make asan tsan           # suite under ASan+UBSan, and under TSan
make bench eval          # tools/bench.sh matrix -> out/bench.jsonl; perplexity/agreement per dtype
make check               # parity vs NumPy and HF transformers (needs torch/transformers/tokenizers)

./build/edgelm run   -m models/smollm2-135m.q4.elm --act int8 -p "The three laws of thermodynamics" -n 64 --threads 4
./build/edgelm run   -m models/smollm2-135m.f32.elm --dtype q8 ...      # load-time quantization of an f32 file
./build/edgelm run   ... --temp 0.8 --top-k 40 --seed 7 --profile trace.json --stats
./build/edgelm run   ... --backend matmul-only,cpu                     # partial delegation, see `info`
./build/edgelm bench -m models/smollm2-135m.q8.elm --act int8 --threads 1,4,10 --prompt-len 128 --gen 64
./build/edgelm eval  -m models/smollm2-135m.q4.elm --ref models/smollm2-135m.f32.elm --act int8
./build/edgelm info  -m models/smollm2-135m.q8.elm [--no-fuse] [--backend matmul-only,cpu] [--ops]
```

`--act int8` turns on dynamic int8 activations for q8/q4 weights (W8A8 / W4A8). It has no effect on f32.

## Numbers

Machine: Apple M5 (4 performance + 6 efficiency cores), 16 GB, macOS, Apple clang 17, `-O3`
Release. The laptop was in use while measuring (load average around 8, with a VM running), so
treat numbers as ±10%. Model: SmolLM2-135M. Workload: a 128-token prompt, then 64 greedy decode
steps. Each figure is the median of 3 runs after one warm-up (`tools/bench.sh`); the reference
and scalar rows are single runs. TTFT is the prefill of the 128 tokens plus the first argmax.

| weights | activations | threads | prefill tok/s | TTFT ms | decode tok/s | weights MB | peak RSS MB |
|---|---|---:|---:|---:|---:|---:|---:|
| reference backend (naive scalar) | f32 | 1 | 24.7 | 5189 | 13.9 | 538.1 | 526 |
| f32, scalar path (`--no-simd`) | f32 | 1 | 28.4 | 4499 | 20.8 | 538.1 | 526 |
| q8, scalar path | f32 | 1 | 29.9 | 4284 | 23.7 | 135.4 | 144 |
| q4, scalar path | f32 | 1 | 33.5 | 3817 | 26.4 | 95.0 | 104 |
| f32 | f32 | 1 | 482.3 | 265.4 | 113.3 | 538.1 | 527 |
| f32 | f32 | 4 | 1350.7 | 94.8 | 137.2 | 538.1 | 527 |
| f32 | f32 | 10 | 2116.3 | 60.5 | 159.3 | 538.1 | 529 |
| q8 per-channel | f32 | 1 | 451.2 | 283.7 | 118.1 | 135.4 | 145 |
| q8 per-channel | f32 | 4 | 1469.6 | 87.1 | 294.0 | 135.4 | 145 |
| q8 per-channel | f32 | 10 | 2092.7 | 61.2 | 358.2 | 135.4 | 145 |
| q8 per-channel | **int8** | 1 | 678.5 | 188.7 | 290.3 | 135.4 | 145 |
| q8 per-channel | **int8** | 4 | 2041.5 | 62.7 | 417.0 | 135.4 | 146 |
| q8 per-channel | **int8** | 10 | **2509.1** | **51.0** | 427.6 | 135.4 | 147 |
| q4 g32 (+q8 embedding) | f32 | 1 | 418.8 | 305.7 | 104.3 | 95.0 | 106 |
| q4 g32 (+q8 embedding) | f32 | 4 | 1345.4 | 95.2 | 283.8 | 95.0 | 107 |
| q4 g32 (+q8 embedding) | **int8** | 1 | 579.0 | 221.1 | 263.2 | 95.0 | 106 |
| q4 g32 (+q8 embedding) | **int8** | 4 | 1803.3 | 71.0 | 466.8 | 95.0 | 107 |
| q4 g32 (+q8 embedding) | **int8** | 10 | 2302.7 | 55.6 | 473.0 | 95.0 | 108 |
| q4 g32, all weights | **int8** | 4 | 1774.0 | 72.2 | **502.9** | 84.2 | 97 |

- **Speedup over the single-thread scalar path**: f32 prefill is 17.0x with NEON on one thread
  (4x4 register-blocked tile against a scalar per-pair dot product with 4 accumulators) and 74.5x
  with 10 threads. q8 with int8 activations is 83.9x on prefill (2509 vs 29.9 tok/s) and 17.6x on
  decode at 4 threads (417 vs 23.7). Against the naive reference backend, q8+int8 prefill is 101x.
- **Decode is bandwidth-bound in f32**: 538 MB of weights per token, and it tops out at 159 tok/s,
  about 86 GB/s. q8/q4 cut the bytes. With f32 activations, though, one core turns out to be
  bound on int8-to-float widening: q8 gives 118 tok/s against 113 for f32 on one thread. Int8
  activations switch the inner loop to SDOT (16 MACs per instruction), which takes one-thread q8
  decode from 118 to 290 tok/s (2.5x) and prefill from 451 to 679 tok/s.
- **Thread scaling**: prefill keeps scaling onto the E-cores (q8+int8: 679 / 1193 / 2042 / 2018 /
  2509 tok/s at 1 / 2 / 4 / 6 / 10 threads). Decode saturates at about 4 threads (290 / 371 / 417 /
  362 / 428). The first thread pool waited for every worker to check in after each op. On this
  loaded laptop that dropped 10-thread q8 decode to 148 tok/s, because each of the ~150 dispatches
  per token waited for a descheduled worker. The current pool (below) runs at 405-428 tok/s there.
- **SDOT against the armv8.0 fallback** (`-march=armv8-a`, the widening-multiply path baseline
  Android arm64-v8a compiles to): one-thread q8+int8 prefill goes 432 -> 735 tok/s and decode
  267 -> 307 tok/s.
- **Cold start**: `run` prefaults the mapping (`madvise(WILLNEED)` plus one read per page, 39 ms
  for q4). Without it, the first 7-token prefill took 157 ms instead of 5.2 ms because of page faults.

### Accuracy (`edgelm eval`, the same run as the table above)

4115 next-token predictions over "A Scandal in Bohemia" (Project Gutenberg #1661, public domain,
`data/eval.txt`) in 256-token windows. Each quantized model is compared against f32 on the same
windows. Softmax ignores a constant shift of the logits, and q4 sometimes moves every logit by
several units: at one position the f32 mean is -8.6 and the q4 mean +2.7, which gives a raw
cosine of -0.43 on a correct prediction. So the cosine is taken on mean-centred logits, with
KL(f32 || quantized) next to it.

| weights | activations | weights MB | perplexity | top-1 agreement vs f32 | centred-logit cosine (mean) | KL nats |
|---|---|---:|---:|---:|---:|---:|
| f32 | f32 | 538.1 | 20.986 | - | - | - |
| q8 per-channel | f32 | 135.4 | 21.262 (+1.3%) | 95.12% | 0.99837 | 0.0077 |
| q8 per-channel | int8 | 135.4 | 21.421 (+2.1%) | 93.92% | 0.99773 | 0.0123 |
| q4 g32 (+q8 embedding) | f32 | 95.0 | 26.987 (+28.6%) | 71.54% | 0.96562 | 0.2389 |
| q4 g32 (+q8 embedding) | int8 | 95.0 | 26.990 (+28.6%) | 71.20% | 0.96542 | 0.2405 |
| q4 g32, all weights | f32 | 84.2 | 30.915 (+47.3%) | 60.95% | 0.94508 | 0.3686 |
| q4 g32, all weights | int8 | 84.2 | 31.061 (+48.0%) | 61.09% | 0.94480 | 0.3720 |

Int8 activations are almost free next to the weight error. Round-to-nearest 4-bit quantization is
expensive on a 135M model, and the tied embedding / LM head is the most sensitive tensor: keeping
it at q8 costs 10.8 MB and recovers 40% of the perplexity loss (+9.9 down to +6.0), so
`make model` exports q4 that way. `tools/q4_ablation.py` fake-quantizes the linears at f32 with q8 embeddings to test
other 4-bit schemes. Its symmetric g32 row reproduces the int4-kernel result exactly (26.9874):

| 4-bit scheme (linears) | perplexity | top-1 | KL |
|---|---:|---:|---:|
| symmetric g32, round-to-nearest (shipped) | 26.987 | 71.54% | 0.239 |
| asymmetric g32 (min/max, 16 levels) | 27.429 | 70.79% | 0.266 |
| symmetric g32 + per-group MSE-optimal scale search | 27.504 | 71.74% | 0.252 |
| symmetric g16 | 25.312 | 74.26% | 0.170 |

Minimizing weight MSE made perplexity worse, since weight error is not output error. The real
next steps are smaller groups or activation-aware methods (AWQ/GPTQ), neither of which is
implemented. With today's f32 scales, g16 would cost +1 bit/weight (5 -> 6); with f16 scales it
would cost +0.5.

### Numerical parity (`tools/check_real.py`, 40-token prompt)

| comparison | max abs logit diff (max abs logit 34.5) | argmax agreement |
|---|---:|---:|
| NumPy reference vs HF transformers (f32) | 9.4e-5 | 100% |
| edgelm f32 vs HF transformers | 2.4e-4 | 100% |
| edgelm f32 vs NumPy, prefill 8 + 32 decode steps through the KV cache | 2.4e-4 | 100% |
| edgelm q8 / q4 vs NumPy on the same dequantized weights | 3.2e-4 / 1.2e-4 | 100% |
| edgelm f32 / q4 vs NumPy, one full 256-token window | 4.0e-4 / 1.3e-4 | 100% |

The C++ tokenizer produces the same ids as HF `tokenizers` on 6/6 test strings: the whole
4132-token eval text, whitespace runs, digits, "é", "—", curly quotes, an emoji, and
`<|im_start|>` special tokens.

With int8 activations, edgelm and NumPy agree on argmax but differ by up to about 1.0 logit on
the 30-layer model. That gap is expected, not a bug: in NumPy alone, scaling the embeddings by
(1 + 1e-7) moves W8A8 logits by 1.19 but f32-activation logits by only 1e-4. A last-bit float
difference flips an activation's int8 rounding and the change propagates. The kernel semantics
are pinned on the 2-layer test model instead, where they match to 2e-5.

### Memory planner (`edgelm info`)

| graph | activations | planned arena (T=128) | one buffer per tensor |
|---|---:|---:|---:|
| exported, unfused (484 ops) | 485 | 2.65 MB | 183.3 MB (69x) |
| fused (214 ops) | 275 | **1.38 MB** | 83.1 MB (60x) |
| fused, decode-only plan (T=1) | 275 | 0.20 MB | 0.85 MB |

The KV cache is separate from the arena: f32, 47.2 MB at a 1024-token context (30 layers x 3 KV
heads x 64 dims x K and V). The arena is sized for `max_batch` tokens and reused by every prefill
chunk and every decode step.

### Profiling

`--profile trace.json` writes one Chrome trace event per executed op, with args {op, backend, T,
pos0}, nested inside a `prefill` or `decode` span. It opens in ui.perfetto.dev. `--stats` prints
the aggregate:

```
$ edgelm run -m models/smollm2-135m.q4.elm --act int8 -p "The three laws of thermodynamics state that" -n 64 --threads 4 --ignore-eos --stats
[q4+a8, 4 threads] load 34 ms, prompt 7 tok, TTFT 5.5 ms (prefill 1277.9 tok/s), decode 64 tok 450.2 tok/s, peak RSS 148 MB

decode: 63 run(s), 138.21 ms wall, 137.40 ms in ops
  FFN_SWIGLU (cpu)                1890 calls     42.323 ms   30.8%
  MATMUL_ADD (cpu)                3780 calls     35.407 ms   25.8%      <- o-proj and down-proj + residual
  MATMUL (cpu)                      63 calls     33.945 ms   24.7%      <- 49152 x 576 LM head (q8)
  MATMUL_N (cpu)                  1890 calls     18.006 ms   13.1%      <- fused QKV
  ATTENTION (cpu)                 1890 calls      7.128 ms    5.2%
  RMSNORM (cpu)                   3843 calls      0.559 ms    0.4%
  EMBED (reference)                 63 calls      0.025 ms    0.0%
  LAST_ROWS (reference)             63 calls      0.009 ms    0.0%
prefill: 1 run(s), 5.45 ms wall, 5.44 ms in ops
  ...
```

The LM head, one op per token, takes a quarter of decode time. It is the obvious next lever:
a vocab-sharded head, or a q4 head for anyone who accepts the accuracy cost measured above.

## Design

**Model file.** A 128-byte header (config, section offsets), then tensor records (name, kind,
dtype, symbolic rows T/L or a constant, cols, blob offsets), op records (type, inputs, outputs,
4 int + 2 float attributes, name), the tokenizer (raw-byte vocab and merge triples, so C++ needs
no JSON parser), and 64-byte-aligned weight blobs. Q8 stores int8 [N,K] plus an f32 scale per
output row (amax/127). Q4 stores 32-element groups in 16 bytes, element j in the low nibble and
element j+16 in the high one, so NEON unpacks a group with one AND and one shift. Each group
carries an f32 scale d = (signed max-magnitude value)/-8, with q in [-8, 7]. The exporter
(NumPy) and the C++ load-time quantizer (`--dtype`) use the same float32 round-half-away
arithmetic and produce identical bytes (tested).

**Fusion.** Passes pattern-match on producer/consumer maps and replace the last op of each pattern,
so the op list stays topologically ordered. That is checked in tests, along with idempotence.
MATMUL_N is one parallel dispatch over the concatenated output rows of Wq|Wk|Wv without copying
weights, and the input is int8-quantized once for all three. FFN_SWIGLU computes gate and up for
a 16-row block into per-thread scratch and writes only silu(g)*u, so the two [T, 1536] tensors
never exist. MATMUL_ADD adds the residual in the GEMM epilogue.

**Memory planner.** A tensor is live from its producing op to its last consumer (inputs from op 0,
the output to the end). Tensors are placed biggest first, each at the best-fitting gap among
already-placed tensors whose live ranges overlap, with 64-byte alignment. This is TFLite's
greedy-by-size strategy. `plan_is_valid` checks every pair for time-and-space overlap and runs on
the real graph and on 50 random DAGs with skip connections.

**Backends and partitioning.** `Backend::supports(graph, op)` plus `run(op, ctx)`. The partitioner
takes a preference list and gives each op to the first backend that claims it. Maximal runs of
consecutive ops on one backend form a partition, which is the unit a TFLite delegate kernel would
replace. The cpu backend claims everything except the data-movement ops (EMBED, LAST_ROWS), so the
real graph is 4 partitions: reference, 210 ops on cpu, reference, then 2 on cpu. `matmul-only` is
an accelerator stand-in. It claims MATMUL / MATMUL_N / MATMUL_ADD only when every weight is f32
or q8, so on the q8 model it takes 91 ops and splits the graph into 184 partitions, and on q4 it
takes none. Logits match the reference either way (tested), and greedy text is identical to the
cpu backend.

**Prefill vs decode.** One graph serves both. Tensor rows are symbolic (T = tokens this step,
L = logit rows), and LAST_ROWS makes the LM head run on only the last row unless `all_logits` is
set, which eval uses. Kernels branch on T:
- f32/q8/q4 weights with f32 activations. For T == 1, each output row is a dot product with the
  weights widened in registers: one pass over W, memory-bound. For T > 1, a 16-row weight tile is
  dequantized once into per-thread scratch and reused for every token by a 4x4 (tokens x rows)
  register-blocked NEON tile. The tile uses 16 accumulators and 8 operand registers, so each
  dequantized weight is paid for once per 128 tokens instead of once per token.
- int8 activations (`--act int8`). Each activation row is quantized per 32-element block (scale
  = amax/127, the same as GGML Q8_0). A 32-element block then costs 2 SDOTs against the q8 row,
  or against the unpacked q4 group (with a combined weight*activation scale), plus one int->float
  FMA. The weight block is loaded and unpacked once and reused for 4 tokens. Without DOTPROD the
  code uses `vmull_s8` + `vpadalq_s16`.
- Attention: K/V for the new positions are rotated and appended to the cache, then one task per
  (token, head) computes causal scores, softmax, and a NEON AXPY over V. GQA maps query head h to
  KV head h / 3.

**Thread pool.** Workers spin (`yield`) for about 200k iterations before parking on a condvar.
Each op has only microseconds of work, so the dispatch protocol matters more than the kernels at
high thread counts. A job completes when its item counter reaches the total, and the caller
never waits for workers that did not show up. Job slots are double-buffered and reference-counted:
the dispatcher marks a slot invalid, then waits for its refcount to drain before refilling it,
and a straggler that joins after the mark sees it and backs off (seq_cst on both sides). Each
thread has a fixed tid, so per-thread scratch never races. It is TSan-clean in CI.

**Tests** (15 doctest cases, 1194 assertions, all self-contained). `tools/make_tiny.py` builds a
2-layer, dim-64, GQA 4/2 random-init model with a toy BPE tokenizer and pushes it through the real
exporter as f32/q8/q4. It also writes NumPy goldens: full-sequence logits for each dtype (and each
dtype with int8 activations), per-op inputs and outputs (rmsnorm, matmul, rope, attention at a
cache offset, swiglu, activation quantizer), and tokenizer ids. The cases are:
- per-op goldens for both backends, SIMD on and off;
- cpu against reference over f32/q8/q4, T in {1, 3, 4, 17} and ragged N/K;
- quantization error bounds;
- MATMUL_N against separate matmuls;
- load-time quantization byte-identical to the exporter;
- fusion counts and topological order;
- planner validity on the real graph and random DAGs;
- partitioning with the narrow backend;
- end-to-end logits against NumPy for fused/unfused, reference/cpu/matmul-only backends, chunked
  prefill, prefill+decode, decode-only and int8 activations;
- bit-identical results across 1/2/3/8 threads;
- sampler determinism and top-k bounds;
- profiler event counts and trace JSON.

The load-time q4 byte-equality test caught a real discrepancy. The first C++ group-max loop
matched NumPy at -O0 but picked a different element at -O2, and choosing by index fixed it.

**CI** (GitHub Actions):
- build and test on Linux x86-64 with gcc (scalar kernels), Linux arm64 with gcc (NEON+SDOT) and
  macOS arm64 with clang, plus a CLI smoke run of run/bench/eval/info and a trace;
- ASan+UBSan;
- TSan;
- Android NDK arm64-v8a cross-compiles, both baseline armv8.0 (the non-SDOT fallback) and
  armv8.2-a+dotprod (build only, binaries uploaded as artifacts).

## Not done / scope

- **No NPU and no vendor SDK.** There is no QNN, LiteRT/TFLite, NNAPI, or Hexagon integration. The
  delegate interface and the `matmul-only` backend imitate how a partial-support accelerator
  gets a subgraph, but everything runs on CPU cores. Nothing has run on an Android device: the
  NDK job only builds.
- One model family: the Llama architecture, validated on SmolLM2-135M only. There is no rope
  scaling, no biases in attention, no sliding window, and no MoE.
- No f16/bf16 compute and no f16/int8 KV cache (the cache is f32, 47 MB at 1024 tokens). No
  i8mm/SMMLA kernels, although the M5 supports them. The int8 GEMM is SDOT with a 4-token x
  1-row tile, which is not a packed-panel GEMM.
- Batch size 1, with no paged or shared KV cache and no speculative decoding. The context is
  capped at `--ctx`, and running past it throws instead of sliding a window.
- q4 scales are stored as f32, so q4 costs 5 bits/weight; f16 scales would make it 4.5.
- Only round-to-nearest quantization is implemented. There is no AWQ/GPTQ/SmoothQuant or
  per-layer mixed precision. 4-bit costs +28.6% perplexity on this model (see above).
- The pre-tokenizer's Unicode classes are exact for ASCII and approximate beyond it: Latin-1,
  general punctuation, symbol and emoji blocks are classified, and every other code point counts
  as a letter. It matches HF on the strings tested, which is not a proof for all scripts.
- Benchmarks come from a single shared laptop, not an isolated device. There are no power or
  thermal measurements.
