#!/usr/bin/env bash
# Full benchmark matrix on the local machine -> out/bench.jsonl (one JSON object per row).
# 128-token prompt (prefill + TTFT), 64 greedy decode steps, median of 3 runs after a warm-up.
set -euo pipefail
cd "$(dirname "$0")/.."
B=./build/edgelm
M=models/smollm2-135m
mkdir -p out
: > out/bench.jsonl
run() { "$B" bench --json --prompt-len 128 --gen 64 "$@" | tee -a out/bench.jsonl; }

# baselines: naive scalar reference backend, then the optimized backend with NEON off
run -m $M.f32.elm --backend reference --threads 1 --reps 1 --gen 16
for d in f32 q8 q4; do run -m $M.$d.elm --no-simd --threads 1 --reps 1; done
# NEON + threads
run -m $M.f32.elm --threads 1,2,4,6,10 --reps 3
for d in q8 q4 q4all; do
  run -m $M.$d.elm --threads 1,2,4,6,10 --reps 3
  run -m $M.$d.elm --threads 1,2,4,6,10 --reps 3 --act int8
done
