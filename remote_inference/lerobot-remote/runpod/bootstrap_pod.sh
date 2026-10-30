#!/bin/bash
# bootstrap_pod.sh HOST PORT KIND [AUTODELETE_MIN] [ACTION_FPS]
# Copies the HF token (as /root/.hf_token, never into pod env/settings), infer.sh, server_template.yaml and
# autodelete.sh to a fresh RunPod pod, arms the auto-delete timer and starts "infer.sh KIND" detached.
set -euo pipefail
HOST=$1; PORT=$2; KIND=$3; MIN=${4:-240}; FPS=${5:-15}
HERE=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p "$PORT" "root@$HOST")
"${SSH[@]}" 'umask 077; cat > /root/.hf_token' < ~/.cache/huggingface/token
tar czf - -C "$HERE" infer.sh server_template.yaml autodelete.sh | "${SSH[@]}" 'tar xzf - -C /root && chmod +x /root/*.sh'
"${SSH[@]}" "setsid nohup /root/autodelete.sh $MIN > /dev/null 2>&1 < /dev/null & 
             ACTION_FPS=$FPS setsid nohup bash /root/infer.sh $KIND > /dev/null 2>&1 < /dev/null &
             sleep 2; echo \"pod \$(hostname): autodelete in $MIN min, infer.sh $KIND at $FPS Hz started at \$(date -u +%T) UTC\""
