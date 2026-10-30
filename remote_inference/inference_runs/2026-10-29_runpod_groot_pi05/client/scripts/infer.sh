#!/bin/bash
# infer.sh groot|pi05|molmoact2|act: install LeRobot, download one trained model and serve it with
# lerobot-policy-server on the pod's 127.0.0.1:7447 (reached from the Mac through an SSH tunnel).
set -uo pipefail
KIND=$1
[ -f /etc/rp_environment ] && { set -a; . /etc/rp_environment; set +a; }
W=/root/work; mkdir -p "$W"; cd "$W"; exec > >(tee -a "$W/infer.log") 2>&1
export HF_TOKEN=$(cat /root/.hf_token) PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# The container sees all host CPUs but gets ~9 vCPUs; cap torch threads to avoid oversubscription.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ACTION_FPS=${ACTION_FPS:-30}
LEROBOT=dd5313e305ff7aaa5b6c7b4769d1b26650ce250b

case $KIND in
  groot)     REPO=jainbharat/groot_n17_7k_70ep_2026-10-29;     EXTRAS=remote,groot ;;
  pi05)      REPO=jainbharat/pi05_7k_70ep_2026-10-29;          EXTRAS=remote,pi ;;
  molmoact2) REPO=jainbharat/molmoact2_7k_70ep_2026-10-29;     EXTRAS=remote,molmoact2 ;;
  act)       REPO=jainbharat/act_20k_70ep_2026-10-28;          EXTRAS=remote ;;
  *) echo "unknown model $KIND"; exit 1 ;;
esac

echo "[infer] $(date -u +%FT%TZ) installing ($KIND)"
command -v uv > /dev/null || pip install -q uv
# One venv per model: the groot and pi extras pin different transformers versions.
V=$W/venv_$KIND
[ -x "$V/bin/python" ] || uv venv -q -p 3.12 "$V"
. "$V/bin/activate"
uv pip install -q "lerobot[$EXTRAS] @ git+https://github.com/huggingface/lerobot.git@$LEROBOT" || { echo "[infer] INSTALL_FAILED"; exit 1; }

echo "[infer] $(date -u +%FT%TZ) downloading $REPO"
MODEL=/root/models/$KIND
python -c "from huggingface_hub import snapshot_download as s; s('$REPO', local_dir='$MODEL', ignore_patterns=['checkpoints/*', 'logs/*'])" \
  || { echo "[infer] DOWNLOAD_FAILED"; exit 1; }

sed "s#__MODEL__#$MODEL#; s#__KIND__#$KIND#; s#__FPS__#$ACTION_FPS#" /root/server_template.yaml > "$W/server_$KIND.yaml"
pkill -f "[l]erobot-policy-server"; sleep 2
echo "[infer] $(date -u +%FT%TZ) starting policy server for $KIND"
# Run from the model folder: pi05's processor names its bundled tokenizer by the relative path "tokenizer".
cd "$MODEL"
exec lerobot-policy-server --config_path="$W/server_$KIND.yaml"
