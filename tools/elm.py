"""The .elm model format: writer, reader, quantizers and the exported op graph.

Layout (little-endian):
  [0, 128)       header: magic, section offsets, model config
  graph section  tensor records, then op records
  tok section    vocab (raw bytes per token), merges (a, b -> result)
  weights        64-byte aligned blobs, one per weight tensor (+ one scale blob if quantized)

Both the C++ loader (src/model.cpp) and tools/reference.py read this file.
"""
import json
import struct

import numpy as np

MAGIC = b"ELM1"
VERSION = 1
ALIGN = 64

DT_F32, DT_Q8, DT_Q4, DT_I32 = 0, 1, 2, 3
DTYPES = {"f32": DT_F32, "q8": DT_Q8, "q4": DT_Q4}
KIND_WEIGHT, KIND_ACT, KIND_INPUT, KIND_OUTPUT = 0, 1, 2, 3
ROWS_CONST, ROWS_T, ROWS_L = 0, 1, 2

# op types; ids must match include/edgelm/graph.hpp
OP = dict(EMBED=0, RMSNORM=1, MATMUL=2, ROPE=3, ATTENTION=4, ADD=5, SILU=6, MUL=7, LAST_ROWS=8,
          MATMUL_N=9, FFN_SWIGLU=10, MATMUL_ADD=11)

Q4_GROUP = 32


# ---------------------------------------------------------------- quantization
# Rounding is floor(|v| + 0.5) * sign(v) in float32 on both sides so the C++
# load-time quantizer is bit-identical to this one (tested).

def _round_away(v):
    v = v.astype(np.float32)
    return (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.float32)


def quant_q8(w):
    """Per-output-channel symmetric int8. w: [N, K] f32 -> (int8 [N, K], f32 scale [N])."""
    w = w.astype(np.float32)
    amax = np.abs(w).max(axis=1).astype(np.float32)
    scale = (amax / np.float32(127.0)).astype(np.float32)
    safe = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    q = np.clip(_round_away(w / safe[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequant_q8(q, scale):
    return q.astype(np.float32) * scale[:, None]


def quant_q4(w, group=Q4_GROUP):
    """Group-wise symmetric int4, Q4_0 style: d = (signed max-magnitude value) / -8, q in [-8, 7].
    Packed per group of 32: byte j holds element j (low nibble) and element j+16 (high nibble),
    each stored as q + 8. Returns (uint8 [N, K/2], f32 scale [N, K/group])."""
    w = w.astype(np.float32)
    n, k = w.shape
    assert k % group == 0, f"q4 needs K % {group} == 0, got K={k}"
    g = w.reshape(n, k // group, group)
    idx = np.abs(g).argmax(axis=2)
    m = np.take_along_axis(g, idx[..., None], axis=2)[..., 0].astype(np.float32)
    d = (m / np.float32(-8.0)).astype(np.float32)
    safe = np.where(d == 0, np.float32(1.0), d).astype(np.float32)
    q = np.clip(_round_away(g / safe[..., None]), -8, 7).astype(np.int8)
    u = (q + 8).astype(np.uint8)  # [n, groups, 32]
    half = group // 2
    packed = (u[..., :half] | (u[..., half:] << 4)).astype(np.uint8)
    return packed.reshape(n, k // 2), d.reshape(n, k // group)


def dequant_q4(packed, scale, group=Q4_GROUP):
    n = packed.shape[0]
    half = group // 2
    p = packed.reshape(n, -1, half)
    lo = (p & 0x0F).astype(np.int8) - 8
    hi = (p >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=2).astype(np.float32)  # [n, groups, 32]
    return (q * scale[..., None]).reshape(n, -1)


# ---------------------------------------------------------------- graph
class GraphBuilder:
    def __init__(self):
        self.tensors = []  # dicts
        self.ops = []
        self.by_name = {}

    def tensor(self, name, kind, dtype, rows_sym, rows, cols, array=None):
        t = dict(name=name, kind=kind, dtype=dtype, rows_sym=rows_sym, rows=rows, cols=cols, array=array)
        self.by_name[name] = len(self.tensors)
        self.tensors.append(t)
        return len(self.tensors) - 1

    def act(self, name, cols, rows_sym=ROWS_T):
        return self.tensor(name, KIND_ACT, DT_F32, rows_sym, 0, cols)

    def op(self, typ, name, ins, outs, iattr=(0, 0, 0, 0), fattr=(0.0, 0.0)):
        self.ops.append(dict(type=OP[typ], name=name, ins=list(ins), outs=list(outs),
                             iattr=list(iattr) + [0] * (4 - len(iattr)), fattr=list(fattr) + [0.0] * (2 - len(fattr))))


def build_graph(cfg, weights):
    """Unfused Llama graph over T tokens. weights: name -> f32 array (HF naming).
    Fusion is left to the C++ passes on purpose."""
    b = GraphBuilder()
    D, H, nh, nkv, hd = cfg["dim"], cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    eps, theta = cfg["norm_eps"], cfg["rope_theta"]

    def w(name):
        a = weights[name]
        if a.ndim == 1:
            return b.tensor(name, KIND_WEIGHT, DT_F32, ROWS_CONST, 1, a.shape[0], a)
        return b.tensor(name, KIND_WEIGHT, DT_F32, ROWS_CONST, a.shape[0], a.shape[1], a)

    tokens = b.tensor("tokens", KIND_INPUT, DT_I32, ROWS_T, 0, 1)
    emb = w("model.embed_tokens.weight")
    x = b.act("x.embed", D)
    b.op("EMBED", "embed", [tokens, emb], [x])
    for l in range(cfg["n_layers"]):
        p = f"model.layers.{l}."
        L = f"L{l}."
        h = b.act(L + "attn_in", D)
        b.op("RMSNORM", L + "attn_norm", [x, w(p + "input_layernorm.weight")], [h], fattr=[eps])
        q, k, v = b.act(L + "q", nh * hd), b.act(L + "k", nkv * hd), b.act(L + "v", nkv * hd)
        b.op("MATMUL", L + "wq", [h, w(p + "self_attn.q_proj.weight")], [q])
        b.op("MATMUL", L + "wk", [h, w(p + "self_attn.k_proj.weight")], [k])
        b.op("MATMUL", L + "wv", [h, w(p + "self_attn.v_proj.weight")], [v])
        qr, kr = b.act(L + "q_rope", nh * hd), b.act(L + "k_rope", nkv * hd)
        b.op("ROPE", L + "rope_q", [q], [qr], iattr=[nh, hd], fattr=[theta])
        b.op("ROPE", L + "rope_k", [k], [kr], iattr=[nkv, hd], fattr=[theta])
        a = b.act(L + "attn", nh * hd)
        b.op("ATTENTION", L + "attention", [qr, kr, v], [a], iattr=[l, nh, nkv, 0], fattr=[theta])
        o = b.act(L + "attn_out", D)
        b.op("MATMUL", L + "wo", [a, w(p + "self_attn.o_proj.weight")], [o])
        x1 = b.act(L + "x_attn", D)
        b.op("ADD", L + "residual_attn", [x, o], [x1])
        h2 = b.act(L + "ffn_in", D)
        b.op("RMSNORM", L + "ffn_norm", [x1, w(p + "post_attention_layernorm.weight")], [h2], fattr=[eps])
        g, u = b.act(L + "gate", H), b.act(L + "up", H)
        b.op("MATMUL", L + "w_gate", [h2, w(p + "mlp.gate_proj.weight")], [g])
        b.op("MATMUL", L + "w_up", [h2, w(p + "mlp.up_proj.weight")], [u])
        gs = b.act(L + "gate_silu", H)
        b.op("SILU", L + "silu", [g], [gs])
        m = b.act(L + "ffn_mid", H)
        b.op("MUL", L + "gate_mul", [gs, u], [m])
        d = b.act(L + "ffn_out", D)
        b.op("MATMUL", L + "w_down", [m, w(p + "mlp.down_proj.weight")], [d])
        x2 = b.act(L + "x_out", D)
        b.op("ADD", L + "residual_ffn", [x1, d], [x2])
        x = x2
    xl = b.act("x.last", D, ROWS_L)
    b.op("LAST_ROWS", "last_rows", [x], [xl])
    xn = b.act("x.final_norm", D, ROWS_L)
    b.op("RMSNORM", "final_norm", [xl, w("model.norm.weight")], [xn], fattr=[eps])
    head = emb if cfg["tied"] else w("lm_head.weight")
    logits = b.tensor("logits", KIND_OUTPUT, DT_F32, ROWS_L, 0, cfg["vocab"])
    b.op("MATMUL", "lm_head", [xn, head], [logits])
    return b


# ---------------------------------------------------------------- tokenizer
def bytes_to_unicode():
    """GPT-2 byte <-> printable unicode map used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for c in range(256):
        if c not in bs:
            bs.append(c)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def tokenizer_from_hf(path):
    """tokenizer.json (byte-level BPE) -> (vocab bytes list, special flags, merges [(a, b, res)])."""
    tj = json.load(open(path, encoding="utf-8"))
    assert tj["model"]["type"] == "BPE"
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    vocab = tj["model"]["vocab"]
    n = max(vocab.values()) + 1
    for a in tj["added_tokens"]:
        n = max(n, a["id"] + 1)
    toks = [b""] * n
    special = [0] * n
    for s, i in vocab.items():
        toks[i] = bytes(u2b[c] for c in s)
    for a in tj["added_tokens"]:
        toks[a["id"]] = a["content"].encode("utf-8")
        special[a["id"]] = 1 if a.get("special", True) else 0
    merges = []
    for m in tj["model"]["merges"]:
        a, c = m.split(" ") if isinstance(m, str) else m
        merges.append((vocab[a], vocab[c], vocab[a + c]))
    return toks, special, merges


# ---------------------------------------------------------------- writer
def _pad(buf, align=ALIGN):
    buf.extend(b"\0" * ((-len(buf)) % align))


def write_elm(path, cfg, builder, tok, dtype="f32", quant_embed=True):
    """Serialize graph + tokenizer + weights. Linear weights (2-D) are quantized to `dtype`;
    the tied embedding too unless quant_embed=False. Norm weights stay f32."""
    toks, special, merges = tok
    blob = bytearray()
    records = []
    qdt = DTYPES[dtype]
    for t in builder.tensors:
        rec = dict(t)
        rec.update(data_off=0, data_bytes=0, scale_off=0, scale_bytes=0, group=0)
        a = t["array"]
        if t["kind"] == KIND_WEIGHT:
            is_embed = t["name"] == "model.embed_tokens.weight"
            quant = a.ndim == 2 and qdt != DT_F32 and (quant_embed or not is_embed)
            if quant and qdt == DT_Q8:
                data, scale = quant_q8(a)
                rec["dtype"], rec["group"] = DT_Q8, a.shape[1]
            elif quant and qdt == DT_Q4:
                data, scale = quant_q4(a)
                rec["dtype"], rec["group"] = DT_Q4, Q4_GROUP
            else:
                data, scale = a.astype(np.float32), None
            _pad(blob)
            rec["data_off"], rec["data_bytes"] = len(blob), data.nbytes
            blob.extend(np.ascontiguousarray(data).tobytes())
            if scale is not None:
                _pad(blob)
                rec["scale_off"], rec["scale_bytes"] = len(blob), scale.nbytes
                blob.extend(np.ascontiguousarray(scale, dtype=np.float32).tobytes())
        records.append(rec)

    g = bytearray()
    for r in records:
        name = r["name"].encode()
        g += struct.pack("<H", len(name)) + name
        g += struct.pack("<BBBBqqQQQQI", r["kind"], r["dtype"], r["rows_sym"], 0, r["rows"], r["cols"],
                         r["data_off"], r["data_bytes"], r["scale_off"], r["scale_bytes"], r["group"])
    for o in builder.ops:
        g += struct.pack("<BBBB", o["type"], len(o["ins"]), len(o["outs"]), 0)
        g += struct.pack(f"<{len(o['ins'])}i", *o["ins"]) + struct.pack(f"<{len(o['outs'])}i", *o["outs"])
        g += struct.pack("<4i2f", *o["iattr"], *o["fattr"])
        name = o["name"].encode()
        g += struct.pack("<H", len(name)) + name

    tk = bytearray(struct.pack("<II", len(toks), 1))
    for s, sp in zip(toks, special):
        tk += struct.pack("<H", len(s)) + s + struct.pack("<B", sp)
    tk += struct.pack("<I", len(merges))
    for m in merges:
        tk += struct.pack("<III", *m)

    graph_off = 128
    tok_off = graph_off + len(g)
    weights_off = tok_off + len(tk)
    weights_off += (-weights_off) % ALIGN
    hdr = MAGIC + struct.pack("<III", VERSION, len(builder.tensors), len(builder.ops))
    hdr += struct.pack("<QQQQQQ", graph_off, len(g), tok_off, len(tk), weights_off, len(blob))
    hdr += struct.pack("<8I", cfg["vocab"], cfg["dim"], cfg["hidden"], cfg["n_layers"], cfg["n_heads"],
                       cfg["n_kv_heads"], cfg["head_dim"], cfg["max_seq"])
    hdr += struct.pack("<ffII", cfg["rope_theta"], cfg["norm_eps"], cfg["bos"], cfg["eos"])
    hdr += struct.pack("<III", qdt, Q4_GROUP if qdt == DT_Q4 else 0, 1 if cfg["tied"] else 0)
    hdr += b"\0" * (128 - len(hdr))
    assert len(hdr) == 128
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(g)
        f.write(tk)
        f.write(b"\0" * (weights_off - tok_off - len(tk)))
        f.write(blob)
    return weights_off + len(blob)


# ---------------------------------------------------------------- reader
def read_elm(path):
    """Returns (cfg, weights) with every weight dequantized to f32 (HF naming)."""
    buf = open(path, "rb").read()
    assert buf[:4] == MAGIC
    _, nt, no = struct.unpack_from("<III", buf, 4)
    graph_off, _, _, _, weights_off, _ = struct.unpack_from("<QQQQQQ", buf, 16)
    c = struct.unpack_from("<8I", buf, 64)
    theta, eps, bos, eos = struct.unpack_from("<ffII", buf, 96)
    qdt, group, flags = struct.unpack_from("<III", buf, 112)
    cfg = dict(vocab=c[0], dim=c[1], hidden=c[2], n_layers=c[3], n_heads=c[4], n_kv_heads=c[5], head_dim=c[6],
               max_seq=c[7], rope_theta=theta, norm_eps=eps, bos=bos, eos=eos, tied=bool(flags & 1), dtype=qdt)
    p = graph_off
    weights = {}
    for _ in range(nt):
        (ln,) = struct.unpack_from("<H", buf, p)
        p += 2
        name = buf[p:p + ln].decode()
        p += ln
        kind, dt, _, _, rows, cols, doff, dbytes, soff, sbytes, grp = struct.unpack_from("<BBBBqqQQQQI", buf, p)
        p += struct.calcsize("<BBBBqqQQQQI")
        if kind != KIND_WEIGHT:
            continue
        data = buf[weights_off + doff: weights_off + doff + dbytes]
        if dt == DT_F32:
            a = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
            weights[name] = a[0].copy() if rows == 1 and "norm" in name else a.copy()
        elif dt == DT_Q8:
            s = np.frombuffer(buf[weights_off + soff: weights_off + soff + sbytes], dtype=np.float32)
            weights[name] = dequant_q8(np.frombuffer(data, dtype=np.int8).reshape(rows, cols), s)
        elif dt == DT_Q4:
            s = np.frombuffer(buf[weights_off + soff: weights_off + soff + sbytes], dtype=np.float32)
            weights[name] = dequant_q4(np.frombuffer(data, dtype=np.uint8).reshape(rows, cols // 2),
                                       s.reshape(rows, cols // grp), grp)
    return cfg, weights
