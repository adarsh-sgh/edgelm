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
for d in f32 q8 q4; do
  "$PY" tools/export.py --hf-dir "$dir" --out "models/smollm2-135m.$d.elm" --dtype "$d"
done
