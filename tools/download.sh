#!/usr/bin/env bash
# Fetch SmolLM2-135M (Apache-2.0) and export it as f32 / q8 / q4 .elm files.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
dir=models/smollm2-135m
mkdir -p "$dir"
base=https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main
for f in config.json tokenizer.json model.safetensors; do
  [ -s "$dir/$f" ] || curl -fL --retry 3 -o "$dir/$f" "$base/$f"
done
"$PY" tools/export.py --hf-dir "$dir" --out models/smollm2-135m.f32.elm --dtype f32
"$PY" tools/export.py --hf-dir "$dir" --out models/smollm2-135m.q8.elm --dtype q8
# q4 linears; the tied embedding / lm_head stays q8 (all-q4 costs ~4 ppl more, see README)
"$PY" tools/export.py --hf-dir "$dir" --out models/smollm2-135m.q4.elm --dtype q4 --embed-dtype q8
"$PY" tools/export.py --hf-dir "$dir" --out models/smollm2-135m.q4all.elm --dtype q4
