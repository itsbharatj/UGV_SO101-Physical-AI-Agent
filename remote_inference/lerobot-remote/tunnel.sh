#!/bin/bash
# Usage: ./tunnel.sh <job_id>  — forwards this Mac's localhost:7447 to the policy server inside the HF Job.
exec ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519 -L 7447:127.0.0.1:7447 "$1@ssh.hf.jobs"
