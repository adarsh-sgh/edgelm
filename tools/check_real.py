#!/usr/bin/env python3
"""Numerics check on the real model: edgelm logits vs the NumPy reference (and HF transformers
when installed), for f32 and for q8/q4 against their own dequantized weights. Also checks the C++
tokenizer against HF `tokenizers` on a few strings.

  python tools/check_real.py --hf-dir models/smollm2-135m --models models/smollm2-135m.{f32,q8,q4}.elm
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import elm  # noqa: E402
import reference as ref  # noqa: E402

TEXT = ("The capital of France is Paris, and the capital of Germany is Berlin. In 1869, the "
        "transcontinental railroad was completed; it's 3,000 km long.")


def edgelm_logits(binary, model, ids, prefill, threads, backend="cpu"):
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        out = f.name
    subprocess.run([binary, "logits", "-m", model, "--tokens", ",".join(map(str, ids)), "--prefill", str(prefill),
                    "--threads", str(threads), "--backend", backend, "--out", out], check=True, capture_output=True)
    a = np.fromfile(out, dtype=np.float32)
    os.unlink(out)
    return a.reshape(len(ids), -1)


def report(name, a, b):
    d = np.abs(a - b)
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"  {name:34s} max|diff| {d.max():.2e}  (max|logit| {np.abs(b).max():.1f})  "
          f"argmax agree {np.mean(a.argmax(1) == b.argmax(1)) * 100:.0f}%  min cos {cos.min():.8f}")
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--binary", default="build/edgelm")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    hf_tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
    ids = hf_tok.encode(TEXT).ids
    print(f"{len(ids)} tokens")

    strings = [TEXT, "  multiple   spaces\n\nand newlines\t tabs  ", "caf\u00e9 na\u00efve \u2014 \u201cquotes\u201d",
               "x = f(3.14) + y[2]; // don't WE'LL 12345", "emoji \U0001F600 ok <|im_start|>user\nhi<|im_end|>",
               open(os.path.join(os.path.dirname(__file__), "..", "data", "eval.txt"), encoding="utf-8").read()]
    ok = 0
    for s in strings:
        out = subprocess.run([args.binary, "tokenize", "-m", args.models[0], "--text", s], capture_output=True,
                             text=True, check=True).stdout.split()
        want = hf_tok.encode(s).ids
        got = [int(x) for x in out]
        ok += got == want
        if got != want:
            print(f"  tokenizer MISMATCH on {s[:40]!r}: {len(got)} vs {len(want)} ids")
    print(f"tokenizer: {ok}/{len(strings)} strings match HF tokenizers exactly "
          f"(incl. data/eval.txt, {len(hf_tok.encode(strings[-1]).ids)} tokens)")

    hf_logits = None
    try:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(args.hf_dir, dtype=torch.float32)
        with torch.no_grad():
            hf_logits = m(torch.tensor([ids])).logits[0].numpy()
    except ImportError:
        print("  (transformers not installed: skipping HF comparison)")

    for path in args.models:
        cfg, W = elm.read_elm(path)
        want = ref.forward(cfg, W, ids)
        tag = os.path.basename(path)
        print(tag)
        report("numpy ref vs edgelm prefill", edgelm_logits(args.binary, path, ids, len(ids), 4), want)
        report("numpy ref vs edgelm prefill 8 + decode", edgelm_logits(args.binary, path, ids, 8, 4), want)
        if hf_logits is not None:
            if cfg["dtype"] == elm.DT_F32:
                report("HF transformers vs numpy ref", want, hf_logits)
            report("HF transformers vs edgelm", edgelm_logits(args.binary, path, ids, len(ids), 4), hf_logits)


if __name__ == "__main__":
    main()
