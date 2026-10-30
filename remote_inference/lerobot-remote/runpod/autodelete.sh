#!/bin/bash
# Delete this pod after $1 minutes so a forgotten inference session cannot drain credit.
set -a; . /etc/rp_environment; set +a
sleep $(($1 * 60)); echo "[infer] auto-delete after $1 min" >> /root/work/infer.log
runpodctl remove pod "$RUNPOD_POD_ID" || curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer ${RUNPOD_API_KEY:-}"
