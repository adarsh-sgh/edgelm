#!/usr/bin/env python3
"""HF Llama-architecture checkpoint (safetensors + tokenizer.json) -> .elm

  python tools/export.py --hf-dir models/smollm2-135m --out models/smollm2-135m.q8.elm --dtype q8
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import elm  # noqa: E402


def read_safetensors(path):
    """Minimal safetensors reader (no dependency): -> name -> f32 ndarray."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        buf = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        raw = np.asarray(buf[base + b: base + e])
        dt = meta["dtype"]
        if dt == "BF16":
            a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            a = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            a = raw.view(np.float32).copy()
        else:
            raise ValueError(f"{name}: unsupported dtype {dt}")
        out[name] = a.reshape(meta["shape"])
    return out


def config_from_hf(hf_dir):
    c = json.load(open(os.path.join(hf_dir, "config.json")))
    assert c["model_type"] == "llama", c["model_type"]
    assert not c.get("rope_scaling"), "rope scaling not supported"
    nh = c["num_attention_heads"]
    return dict(vocab=c["vocab_size"], dim=c["hidden_size"], hidden=c["intermediate_size"],
                n_layers=c["num_hidden_layers"], n_heads=nh, n_kv_heads=c.get("num_key_value_heads", nh),
                head_dim=c.get("head_dim", c["hidden_size"] // nh), max_seq=c["max_position_embeddings"],
                rope_theta=float(c.get("rope_theta", 10000.0)), norm_eps=float(c["rms_norm_eps"]),
                bos=c.get("bos_token_id", 0) or 0, eos=c.get("eos_token_id", 0) or 0,
                tied=bool(c.get("tie_word_embeddings", False)))


def load_hf(hf_dir):
    cfg = config_from_hf(hf_dir)
    weights = {}
    for fn in sorted(os.listdir(hf_dir)):
        if fn.endswith(".safetensors"):
            weights.update(read_safetensors(os.path.join(hf_dir, fn)))
    return cfg, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "q8", "q4"], default="f32")
    ap.add_argument("--f32-embed", action="store_true", help="keep the (tied) embedding in f32")
    args = ap.parse_args()
    cfg, weights = load_hf(args.hf_dir)
    tok = elm.tokenizer_from_hf(os.path.join(args.hf_dir, "tokenizer.json"))
    b = elm.build_graph(cfg, weights)
    n = elm.write_elm(args.out, cfg, b, tok, args.dtype, quant_embed=not args.f32_embed)
    print(f"{args.out}: {args.dtype}, {len(b.tensors)} tensors, {len(b.ops)} ops, "
          f"{len(tok[0])} vocab, {len(tok[2])} merges, {n / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
