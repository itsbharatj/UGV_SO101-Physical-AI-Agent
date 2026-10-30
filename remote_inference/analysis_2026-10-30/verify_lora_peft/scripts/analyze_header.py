import json, re, sys, os
from collections import Counter
out = sys.argv[1]; dl = sys.argv[2]
hdr = json.load(open(os.path.join(out, "data", "finetune_header.json")))["model.safetensors"]
T = hdr["tensors"]; meta = hdr["metadata"]
print("file metadata:", meta)
print("total tensors:", len(T))
def numel(shape):
    n = 1
    for s in shape: n *= s
    return n
lora = {k: v for k, v in T.items() if re.search(r"\.lora_[AB]\.default\.weight$", k)}
nonlora = {k: v for k, v in T.items() if k not in lora}
print("LoRA tensors:", len(lora), "non-LoRA tensors:", len(nonlora))
print("LoRA dtypes:", Counter(v["dtype"] for v in lora.values()))
print("LoRA params:", sum(numel(v["shape"]) for v in lora.values()))
print("prefix check: all keys start with 'model.base_model.model.':", all(k.startswith("model.base_model.model.") for k in T))
print("keys with _orig_mod:", sum("_orig_mod" in k for k in T))
print("adapter names:", Counter(re.search(r"\.lora_[AB]\.([^.]+)\.weight$", k).group(1) for k in lora))
mods = sorted({re.sub(r"\.lora_[AB]\.default\.weight$", "", k) for k in lora})
print("distinct LoRA modules:", len(mods))
ranks = Counter(T[m + ".lora_A.default.weight"]["shape"][0] for m in mods)
ranksB = Counter(T[m + ".lora_B.default.weight"]["shape"][1] for m in mods)
print("rank from A:", ranks, "rank from B:", ranksB)
missing_base = [m for m in mods if (m + ".base_layer.weight") not in T]
print("LoRA modules without base_layer.weight:", len(missing_base), missing_base[:5])
print("LoRA leaf histogram:", Counter(m.rsplit(".", 1)[-1] for m in mods))
idx = json.load(open(os.path.join(dl, "allenai__MolmoAct2", "model.safetensors.index.json")))
base_keys = set(idx["weight_map"])
print("base index keys:", len(base_keys), "shards:", sorted(set(idx["weight_map"].values())))
P = "model.base_model.model."
stripped = {}
for k in nonlora:
    s = k[len(P):] if k.startswith(P) else k
    s = s.replace(".base_layer.", ".")
    stripped[s] = k
print("stripped non-LoRA keys:", len(stripped))
print("non-LoRA not in base index:", len(set(stripped) - base_keys), sorted(set(stripped) - base_keys)[:10])
print("base index not in non-LoRA:", len(base_keys - set(stripped)), sorted(base_keys - set(stripped))[:10])
raw_stripped = {k[len(P):] for k in nonlora}
print("without .base_layer stripping, non-LoRA not in base index:", len(raw_stripped - base_keys))
vlm = "w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding"
rx = re.compile(rf"(?:model\.(transformer|vision_backbone|image_pooling_2d|image_projector)\.(?:.*\.)?({vlm})|lm_head)$")
base_mods = {k.rsplit(".", 1)[0] for k in base_keys}
matched = sorted(m for m in base_mods if rx.fullmatch(m))
print("base modules matching LoRA regex (fullmatch):", len(matched))
print("set equality regex-matched vs checkpoint LoRA modules:", set(matched) == {m[len(P):] for m in mods})
bytes_by = Counter(); params_by = Counter()
sz = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "BOOL": 1}
for k, v in T.items():
    bytes_by[v["dtype"]] += numel(v["shape"]) * sz[v["dtype"]]
    params_by[v["dtype"]] += numel(v["shape"])
print("params by dtype:", dict(params_by), "bytes by dtype:", dict(bytes_by), "sum bytes:", sum(bytes_by.values()))
maxoff = max(v["data_offsets"][1] for v in T.values())
print("max data offset:", maxoff, "file size 12,679,625,488 -> header bytes =", 12679625488 - maxoff)
ae = {k: v for k, v in nonlora.items() if "action_expert" in k}
print("action_expert tensors:", len(ae), "params:", sum(numel(v["shape"]) for v in ae.values()), "dtypes:", Counter(v["dtype"] for v in ae.values()))
print("LoRA + action_expert params:", sum(numel(v["shape"]) for v in lora.values()) + sum(numel(v["shape"]) for v in ae.values()))
other32 = {k: v for k, v in nonlora.items() if v["dtype"] == "F32" and "action_expert" not in k}
print("other fp32 non-LoRA tensors:", len(other32), "params:", sum(numel(v["shape"]) for v in other32.values()))
print("examples:", sorted(other32)[:12])
lm = [m for m in mods if m.endswith("lm_head")]
print("lm_head LoRA modules:", lm, [ (T[m+'.lora_A.default.weight']['shape'], T[m+'.lora_B.default.weight']['shape']) for m in lm])
flops_tok = 0; flops_lm = 0; flops_vit = 0; flops_llm = 0; flops_conn = 0
for m in mods:
    r, i = T[m + ".lora_A.default.weight"]["shape"]; o = T[m + ".lora_B.default.weight"]["shape"][0]
    f = 2 * r * (i + o)
    flops_tok += f
    if m.endswith("lm_head"): flops_lm += f
    elif ".vision_backbone." in m: flops_vit += f
    elif ".transformer." in m: flops_llm += f
    else: flops_conn += f
print(f"LoRA extra FLOPs/token: total {flops_tok/1e6:.1f} MFLOP, llm {flops_llm/1e6:.1f}, lm_head {flops_lm/1e6:.1f}, vit {flops_vit/1e6:.1f}, connector {flops_conn/1e6:.1f}")
basef = 0
for m in mods:
    w = T[m + ".base_layer.weight"]["shape"]
    basef += 2 * w[0] * w[1] if len(w) == 2 else 0
print(f"base FLOPs/token through the same linears: {basef/1e9:.2f} GFLOP; LoRA/base ratio = {flops_tok/basef*100:.2f}%")
with open(os.path.join(out, "data", "lora_modules.txt"), "w") as f:
    f.write("\n".join(mods))
with open(os.path.join(out, "data", "nonlora_key_map.csv"), "w") as f:
    f.write("finetune_key,base_index_key,in_base_index\n")
    for s, k in sorted(stripped.items()):
        f.write(f"{k},{s},{s in base_keys}\n")
