import sys, json, os
from huggingface_hub import get_safetensors_metadata, hf_hub_download
out_data, out_dl = sys.argv[1], sys.argv[2]
repo = "jainbharat/molmoact2_7k_70ep_2026-10-29"
meta = get_safetensors_metadata(repo)  # header-only range requests
hdr = {}
for fname, fmeta in meta.files_metadata.items():
    hdr[fname] = {"metadata": fmeta.metadata, "tensors": {k: {"dtype": v.dtype, "shape": v.shape, "data_offsets": v.data_offsets} for k, v in fmeta.tensors.items()}}
json.dump(hdr, open(os.path.join(out_data, "finetune_header.json"), "w"), indent=1)
print("files in metadata:", list(meta.files_metadata), "param count by dtype:", meta.parameter_count)
for repo_id, files in [
    ("allenai/MolmoAct2", ["model.safetensors.index.json", "config.json"]),
    (repo, ["config.json", "train_config.json", "logs/train_log.txt", "policy_preprocessor.json", "policy_postprocessor.json"]),
]:
    for f in files:
        p = hf_hub_download(repo_id, f, cache_dir=os.path.join(out_dl, "hf_cache"), local_dir=os.path.join(out_dl, repo_id.replace("/", "__")))
        print("downloaded", p, os.path.getsize(p))
