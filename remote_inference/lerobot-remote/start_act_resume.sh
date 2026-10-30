#!/bin/bash
# Resume ACT from jainbharat/act_10k_2026-10-27_20-14-01 (step 10000) to step 20000 on the 71-episode dataset,
# skipping episode 20 (6 frames). Pushes checkpoints to a NEW repo so the 10k model stays untouched.
set -e
FLAVOR=${1:-l4x1}; TIMEOUT=${2:-45m}
hf jobs run --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN --label lerobot --label act-resume \
  huggingface/lerobot-gpu:latest \
  lerobot-train --config_path=jainbharat/act_10k_2026-10-27_20-14-01 --resume=true \
    --steps=20000 --save_freq=2500 --log_freq=250 --policy.use_amp=true \
    --dataset.episodes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70]' \
    --policy.repo_id=jainbharat/act_20k_70ep_2026-10-28 --job_name=act_20k_70ep
