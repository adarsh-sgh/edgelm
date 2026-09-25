#!/usr/bin/env python3
"""Self-contained test assets for CI: a tiny random-init Llama (2 layers, dim 64, GQA 4/2) with a toy
byte-level BPE tokenizer, exported through the real exporter as f32/q8/q4, plus NumPy golden outputs.

  python tools/make_tiny.py --out testdata
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import elm  # noqa: E402
import reference as ref  # noqa: E402

CFG = dict(vocab=320, dim=64, hidden=128, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, max_seq=128,
           rope_theta=10000.0, norm_eps=1e-5, bos=0, eos=0, tied=True)

CORPUS = ("the quick brown fox jumps over the lazy dog. the dog sleeps in the sun, and the fox runs "
          "into the forest. there is nothing on the other side of the river that the fox has not seen. "
          "it's 2024 and they're testing 123 tokens; we'll see what the model does with them.")

# ASCII version of the GPT-2 pre-tokenizer regex, after splitting digits individually.
PRETOK = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def pretokenize(s):
    out = []
    for piece in re.split(r"([0-9])", s):
        if piece:
            out.extend(PRETOK.findall(piece))
    return out


def train_bpe(n_merges):
    """Toy BPE: vocab = [<|endoftext|>] + 256 bytes + merges."""
    toks = [b"<|endoftext|>"] + [bytes([i]) for i in range(256)]
    special = [1] + [0] * 256
    words = [[b + 1 for b in w.encode()] for w in pretokenize(CORPUS)]
    merges = []
    for _ in range(n_merges):
        counts = {}
        for w in words:
            for a, b in zip(w, w[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        (a, b), _ = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        new = len(toks)
        toks.append(toks[a] + toks[b])
        special.append(0)
        merges.append((a, b, new))
        for w in words:
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [new]
                i += 1
    return toks, special, merges


def bpe_encode(s, tok):
    toks, _, merges = tok
    rank = {(a, b): (r, res) for r, (a, b, res) in enumerate(merges)}
    ids = []
    for w in pretokenize(s):
        seq = [b + 1 for b in w.encode()]
        while len(seq) > 1:
            best = min(((rank[p][0], i) for i, p in enumerate(zip(seq, seq[1:])) if p in rank), default=None)
            if best is None:
                break
            i = best[1]
            seq[i:i + 2] = [rank[(seq[i], seq[i + 1])][1]]
        ids.extend(seq)
    return ids


def random_weights(cfg, rng):
    D, H, V = cfg["dim"], cfg["hidden"], cfg["vocab"]
    qd, kd = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv_heads"] * cfg["head_dim"]
    W = {"model.embed_tokens.weight": rng.normal(0, 0.5, (V, D)),
         "model.norm.weight": 1 + rng.normal(0, 0.1, D)}
    for l in range(cfg["n_layers"]):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = 1 + rng.normal(0, 0.1, D)
        W[p + "post_attention_layernorm.weight"] = 1 + rng.normal(0, 0.1, D)
        W[p + "self_attn.q_proj.weight"] = rng.normal(0, 0.15, (qd, D))
        W[p + "self_attn.k_proj.weight"] = rng.normal(0, 0.15, (kd, D))
        W[p + "self_attn.v_proj.weight"] = rng.normal(0, 0.15, (kd, D))
        W[p + "self_attn.o_proj.weight"] = rng.normal(0, 0.15, (D, qd))
        W[p + "mlp.gate_proj.weight"] = rng.normal(0, 0.15, (H, D))
        W[p + "mlp.up_proj.weight"] = rng.normal(0, 0.15, (H, D))
        W[p + "mlp.down_proj.weight"] = rng.normal(0, 0.1, (D, H))
    return {k: v.astype(np.float32) for k, v in W.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testdata")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(1234)
    cfg = dict(CFG)
    tok = train_bpe(cfg["vocab"] - 257)
    W = random_weights(cfg, rng)
    for dt in ("f32", "q8", "q4"):
        b = elm.build_graph(cfg, W)
        elm.write_elm(os.path.join(args.out, f"tiny.{dt}.elm"), cfg, b, tok, dt)

    g = {}
    S = 40
    tokens = rng.integers(0, cfg["vocab"], S)
    g["tokens"] = tokens
    for dt in ("f32", "q8", "q4"):
        c, Wd = elm.read_elm(os.path.join(args.out, f"tiny.{dt}.elm"))
        g[f"logits.{dt}"] = ref.forward(c, Wd, tokens)

    # per-op goldens
    x = rng.normal(0, 1, (5, 64)).astype(np.float32)
    w = (1 + rng.normal(0, 0.1, 64)).astype(np.float32)
    g["op.rmsnorm.x"], g["op.rmsnorm.w"], g["op.rmsnorm.y"] = x, w, ref.rmsnorm(x, w, 1e-5)
    wm = rng.normal(0, 0.2, (48, 64)).astype(np.float32)
    g["op.matmul.x"], g["op.matmul.w"], g["op.matmul.y"] = x, wm, (x @ wm.T).astype(np.float32)
    xr = rng.normal(0, 1, (5, 4 * 16)).astype(np.float32)
    g["op.rope.x"], g["op.rope.y"] = xr, ref.rope(xr, 7, 4, 16, 10000.0)  # pos0=7, 4 heads, hd 16
    q = rng.normal(0, 1, (3, 4 * 16)).astype(np.float32)
    k = rng.normal(0, 1, (8, 2 * 16)).astype(np.float32)
    v = rng.normal(0, 1, (8, 2 * 16)).astype(np.float32)
    g["op.attn.q"], g["op.attn.k"], g["op.attn.v"] = q, k, v
    g["op.attn.y"] = ref.attention(q, k, v, 4, 2, 16, pos0=5)  # queries at positions 5, 6, 7
    gt, up = rng.normal(0, 2, (5, 128)).astype(np.float32), rng.normal(0, 1, (5, 128)).astype(np.float32)
    g["op.swiglu.g"], g["op.swiglu.u"], g["op.swiglu.y"] = gt, up, ref.silu(gt) * up

    texts = ["the quick brown fox", "  hello   world\n\nthe end ", "it's 2024, they're 12 dogs!",
             "fox\tforest  \n river's", ""]
    for i, s in enumerate(texts):
        g[f"tok.text.{i}"] = np.frombuffer(s.encode(), dtype=np.uint8).astype(np.int32)
        g[f"tok.ids.{i}"] = np.asarray(bpe_encode(s, tok), dtype=np.int32)
    ref.write_golden(os.path.join(args.out, "golden.bin"), g)
    print(f"wrote tiny.{{f32,q8,q4}}.elm + golden.bin ({len(g)} arrays) to {args.out}")


if __name__ == "__main__":
    main()
