#!/bin/bash
# Usage: ./tunnel_runpod.sh <host> <port>  — forwards this Mac's localhost:7447 to the policy server on a RunPod pod.
# <host>/<port> are the pod's direct SSH address, e.g. ./tunnel_runpod.sh 194.68.245.191 22001
exec ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519 -p "$2" -L 7447:127.0.0.1:7447 "root@$1"
