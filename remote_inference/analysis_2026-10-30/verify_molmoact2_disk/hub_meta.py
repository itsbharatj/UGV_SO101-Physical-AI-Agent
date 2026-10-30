import json, sys
from huggingface_hub import HfApi
api = HfApi()
out = {}
for repo in ["allenai/MolmoAct2", "jainbharat/molmoact2_7k_70ep_2026-10-29"]:
    info = api.model_info(repo, files_metadata=True)
    files = []
    for s in info.siblings:
        files.append({"path": s.rfilename, "size": s.size})
    total = sum((f["size"] or 0) for f in files)
    ign = [f for f in files if not (f["path"].endswith(".py") or f["path"].endswith(".pyc") or "__pycache__/" in f["path"])]
    total_ign = sum((f["size"] or 0) for f in ign)
    root_only = [f for f in files if "/" not in f["path"]]
    total_root = sum((f["size"] or 0) for f in root_only)
    entry = {
        "gated": info.gated,
        "sha": info.sha,
        "n_files": len(files),
        "total_bytes_all": total,
        "total_bytes_after_py_ignore": total_ign,
        "total_bytes_root_only": total_root,
        "files_over_50MB": sorted([f for f in files if (f["size"] or 0) > 50_000_000], key=lambda f: f["path"]),
        "dirs": sorted({f["path"].split("/")[0] for f in files if "/" in f["path"]}),
        "n_root_files": len(root_only),
        "root_files": sorted(root_only, key=lambda f: f["path"]),
    }
    try:
        st = api.get_safetensors_metadata(repo)
        entry["safetensors_param_count_by_dtype"] = dict(st.parameter_count)
        entry["safetensors_n_shards"] = len(st.files_metadata)
        entry["safetensors_sharded"] = st.sharded
    except Exception as e:
        entry["safetensors_meta_error"] = repr(e)
    out[repo] = entry
json.dump(out, sys.stdout, indent=1, default=str)
