import sys, json
from huggingface_hub import HfApi
api = HfApi()
for repo in ["jainbharat/molmoact2_7k_70ep_2026-10-29", "allenai/MolmoAct2"]:
    info = api.model_info(repo, files_metadata=True)
    print("=== ", repo, "sha", info.sha)
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        print(f"{(s.size or 0):>14,d}  {s.rfilename}")
