#!/usr/bin/env python3
"""4-bit weight quantization ablation on SmolLM2-135M. Each variant fake-quantizes every linear
weight (quantize -> dequantize) into an f32 .elm, keeps the tied embedding at q8, and runs
`edgelm eval` against the f32 model. Only the shipped scheme (symmetric, g=32, round-to-nearest)
has int4 kernels; the others exist to justify that choice.

  python tools/q4_ablation.py --hf-dir models/smollm2-135m --ref models/smollm2-135m.f32.elm
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import elm  # noqa: E402
import export  # noqa: E402

f32 = np.float32


def rtn_sym(g, scale_mult=1.0):
    idx = np.abs(g).argmax(axis=2)
    m = np.take_along_axis(g, idx[..., None], axis=2)[..., 0]
    d = (m / f32(-8.0) * f32(scale_mult)).astype(f32)
    safe = np.where(d == 0, f32(1.0), d)
    q = np.clip(elm._round_away(g / safe[..., None]), -8, 7)
    return q * d[..., None]


def sym(gs):
    return lambda w: _grouped(w, gs, rtn_sym)


def asym(gs):
    def f(g):
        lo, hi = g.min(axis=2), g.max(axis=2)
        d = ((hi - lo) / f32(15.0)).astype(f32)
        safe = np.where(d == 0, f32(1.0), d)
        q = np.clip(elm._round_away((g - lo[..., None]) / safe[..., None]), 0, 15)
        return q * d[..., None] + lo[..., None]
    return lambda w: _grouped(w, gs, f)


def mse_search(gs):
    def f(g):  # per group, pick the scale (shrunk from max/-8) that minimizes weight MSE
        best, best_err = None, None
        for s in np.linspace(1.0, 0.6, 17):
            r = rtn_sym(g, s)
            err = ((r - g) ** 2).sum(axis=2)
            if best is None:
                best, best_err = r, err
            else:
                better = (err < best_err)[..., None]
                best = np.where(better, r, best)
                best_err = np.minimum(err, best_err)
        return best
    return lambda w: _grouped(w, gs, f)


def _grouped(w, gs, fn):
    n, k = w.shape
    return fn(w.reshape(n, k // gs, gs).astype(f32)).reshape(n, k).astype(f32)


VARIANTS = {
    "sym g32 RTN (shipped q4)": sym(32),
    "asym g32 RTN (min/max)": asym(32),
    "sym g32 + MSE scale search": mse_search(32),
    "sym g16 RTN": sym(16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--binary", default="build/edgelm")
    ap.add_argument("--text", default="data/eval.txt")
    args = ap.parse_args()
    cfg, W = export.load_hf(args.hf_dir)
    tok = elm.tokenizer_from_hf(os.path.join(args.hf_dir, "tokenizer.json"))
    emb = elm.dequant_q8(*elm.quant_q8(W["model.embed_tokens.weight"]))
    print(f"{'variant':32s} {'ppl':>8s} {'top-1':>8s} {'cos':>9s} {'KL':>9s}")
    for name, fq in VARIANTS.items():
        W2 = {k: (fq(v) if v.ndim == 2 and k != "model.embed_tokens.weight" else v) for k, v in W.items()}
        W2["model.embed_tokens.weight"] = emb
        with tempfile.NamedTemporaryFile(suffix=".elm", delete=False) as f:
            path = f.name
        elm.write_elm(path, cfg, elm.build_graph(cfg, W2), tok, "f32")
        out = subprocess.run([args.binary, "eval", "-m", path, "--ref", args.ref, "--text", args.text, "--threads", "4"],
                             capture_output=True, text=True, check=True).stdout
        os.unlink(path)
        ppl = re.search(r"f32: perplexity ([\d.]+)", out).group(1)
        agree = re.search(r"top-1 agreement ([\d.]+)%", out).group(1)
        cos = re.search(r"cosine mean ([\d.]+)", out).group(1)
        kl = re.search(r"KL\(ref\|\|cand\) ([\d.]+)", out).group(1)
        print(f"{name:32s} {ppl:>8s} {agree:>7s}% {cos:>9s} {kl:>9s}")


if __name__ == "__main__":
    main()
