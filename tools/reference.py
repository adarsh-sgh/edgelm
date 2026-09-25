"""NumPy float32 reference for the Llama forward pass. Independent of the C++ graph/kernels:
this is the oracle the golden tests compare against, and it is itself checked against
HF transformers by tools/check_real.py."""
import struct

import numpy as np

f32 = np.float32


def rmsnorm(x, w, eps):
    x = x.astype(f32)
    ms = np.mean(x * x, axis=-1, keepdims=True, dtype=f32)
    return (x * (f32(1.0) / np.sqrt(ms + f32(eps)))).astype(f32) * w.astype(f32)


def rope_tables(positions, hd, theta):
    inv = theta ** (-(np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(f32), np.sin(ang).astype(f32)


def rope(x, pos0, n_heads, hd, theta):
    """HF rotate_half convention: pairs (i, i + hd/2)."""
    t = x.shape[0]
    cos, sin = rope_tables(np.arange(pos0, pos0 + t), hd, theta)
    x = x.reshape(t, n_heads, hd)
    h = hd // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).reshape(t, n_heads * hd).astype(f32)


def attention(q, k, v, n_heads, n_kv, hd, pos0=0):
    """q: [T, nh*hd] for positions pos0..pos0+T-1; k, v: [pos0+T, nkv*hd] (whole cache). Causal, GQA."""
    t = q.shape[0]
    s = k.shape[0]
    q = q.reshape(t, n_heads, hd)
    k = k.reshape(s, n_kv, hd)
    v = v.reshape(s, n_kv, hd)
    grp = n_heads // n_kv
    out = np.zeros((t, n_heads, hd), dtype=f32)
    scale = f32(1.0 / np.sqrt(hd))
    for h in range(n_heads):
        kh, vh = k[:, h // grp, :], v[:, h // grp, :]
        sc = (q[:, h, :] @ kh.T).astype(f32) * scale  # [t, s]
        mask = np.arange(s)[None, :] > (pos0 + np.arange(t))[:, None]
        sc[mask] = -np.inf
        sc = sc - sc.max(axis=1, keepdims=True)
        p = np.exp(sc).astype(f32)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h, :] = p @ vh
    return out.reshape(t, n_heads * hd)


def silu(x):
    return (x / (f32(1.0) + np.exp(-x))).astype(f32)


def forward(cfg, W, tokens):
    """Full-sequence forward, returns logits for every position: [S, vocab]."""
    nh, nkv, hd, eps, theta = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["norm_eps"], cfg["rope_theta"]
    emb = W["model.embed_tokens.weight"]
    x = emb[np.asarray(tokens)].astype(f32)
    for l in range(cfg["n_layers"]):
        p = f"model.layers.{l}."
        h = rmsnorm(x, W[p + "input_layernorm.weight"], eps)
        q = h @ W[p + "self_attn.q_proj.weight"].T
        k = h @ W[p + "self_attn.k_proj.weight"].T
        v = h @ W[p + "self_attn.v_proj.weight"].T
        q, k = rope(q, 0, nh, hd, theta), rope(k, 0, nkv, hd, theta)
        a = attention(q, k, v, nh, nkv, hd)
        x = x + a @ W[p + "self_attn.o_proj.weight"].T
        h2 = rmsnorm(x, W[p + "post_attention_layernorm.weight"], eps)
        m = silu(h2 @ W[p + "mlp.gate_proj.weight"].T) * (h2 @ W[p + "mlp.up_proj.weight"].T)
        x = (x + m @ W[p + "mlp.down_proj.weight"].T).astype(f32)
    x = rmsnorm(x, W["model.norm.weight"], eps)
    head = emb if cfg["tied"] else W["lm_head.weight"]
    return (x @ head.T).astype(f32)


def write_golden(path, arrays):
    """Named arrays for the C++ tests: magic EGLD, count, then (name, dtype 0=f32/3=i32, dims, data)."""
    with open(path, "wb") as f:
        f.write(b"EGLD" + struct.pack("<I", len(arrays)))
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            dt = 3 if a.dtype.kind in "iu" else 0
            a = a.astype(np.int32 if dt == 3 else np.float32)
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<BB", dt, a.ndim))
            f.write(struct.pack(f"<{a.ndim}q", *a.shape))
            f.write(a.tobytes())
