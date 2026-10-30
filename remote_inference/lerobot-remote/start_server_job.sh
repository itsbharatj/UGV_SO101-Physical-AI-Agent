#!/bin/bash
# Usage: ./start_server_job.sh <flavor> <device> <timeout>   e.g. ./start_server_job.sh l4x1 cuda 60m
# Starts the Pi0.5 policy server as an HF Job with SSH enabled; zenoh listens on the pod's localhost only.
set -e
FLAVOR=${1:-l4x1}; DEVICE=${2:-cuda}; TIMEOUT=${3:-60m}
CFG=$(base64 < "$(dirname "$0")/server.yaml" | tr -d '\n')
hf jobs run --detach --ssh --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN \
  -e SERVER_CFG="$CFG" -e DEVICE="$DEVICE" --label lerobot --label policy-server \
  huggingface/lerobot-gpu:latest \
  bash -c 'set -e
uv pip install --python "$(command -v python)" -q "lerobot[remote,pi] @ git+https://github.com/huggingface/lerobot.git@dd5313e305ff7aaa5b6c7b4769d1b26650ce250b"
python -c "from huggingface_hub import snapshot_download; snapshot_download(\"jainbharat/pi05_5k_2026-10-27_21-01-51\", allow_patterns=[\"checkpoints/003000/pretrained_model/*\"], local_dir=\"/tmp/dl\")"
ln -sfn /tmp/dl/checkpoints/003000/pretrained_model /tmp/pi05_3000
echo "$SERVER_CFG" | base64 -d > /tmp/server.yaml
echo "=== starting policy server on $DEVICE"
exec lerobot-policy-server --config_path=/tmp/server.yaml --model.device="$DEVICE"'
