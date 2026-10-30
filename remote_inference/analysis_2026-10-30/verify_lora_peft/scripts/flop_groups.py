import json, re, sys, os
out = sys.argv[1]
T = json.load(open(os.path.join(out, "data", "finetune_header.json")))["model.safetensors"]["tensors"]
mods = sorted({re.sub(r"\.lora_[AB]\.default\.weight$", "", k) for k in T if re.search(r"\.lora_[AB]\.default\.weight$", k)})
groups = {}
for m in mods:
    g = "lm_head" if m.endswith("lm_head") else "vit" if ".vision_backbone." in m else "llm" if ".transformer." in m else "connector"
    r, i = T[m + ".lora_A.default.weight"]["shape"]; o = T[m + ".lora_B.default.weight"]["shape"][0]
    w = T[m + ".base_layer.weight"]["shape"]
    d = groups.setdefault(g, {"n": 0, "lora": 0, "base": 0})
    d["n"] += 1; d["lora"] += 2 * r * (i + o); d["base"] += 2 * w[0] * w[1]
for g, d in groups.items():
    print(f"{g:9s} modules={d['n']:4d} LoRA={d['lora']/1e6:7.1f} MFLOP/token  base={d['base']/1e9:6.2f} GFLOP/token  ratio={d['lora']/d['base']*100:5.2f}%")
# whole-LLM base FLOPs/token incl. non-LoRA linears? LoRA covers all linears in llm blocks (att_proj, attn_out, ff_proj, ff_out) so base== llm linear flops
# Example per-chunk totals for a few token-count scenarios (continuous path: lm_head not executed)
for vit_tok, llm_tok in [(2*729, 600), (2*5*729, 2000), (2*7*729, 2800)]:
    tot = groups["vit"]["lora"]*vit_tok + groups["llm"]["lora"]*llm_tok + groups.get("connector", {"lora": 0})["lora"]*llm_tok
    print(f"scenario vit_tokens={vit_tok:5d} llm_tokens={llm_tok:5d}: LoRA extra = {tot/1e12:.3f} TFLOP/chunk")
