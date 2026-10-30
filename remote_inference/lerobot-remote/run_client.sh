#!/bin/bash
# Robot side of remote inference: cameras + motors on this Mac, the policy on a RunPod GPU (or HF Job).
# Requires the SSH tunnel (./tunnel_runpod.sh <host> <port>, or ./tunnel.sh <job_id>) to be open.
# Type /start to begin moving, /reset to return to the start pose, /stop to end. Ctrl+C also stops.
# Camera indices change when a camera is (un)plugged: run `.venv/bin/python check_cameras.py` first and
# pass TOP_CAM=<i> WRIST_CAM=<j> if they are not 0 and 1.
# The task text must match training: "pickup jenga and place inside the white box".
#
# Timing for a ~200 ms ping / ~10 Mbit/s upload link to the EU pod (measured 2026-10-29):
# a round trip is ~0.7 s (max ~1.7 s) but continuous aligned playback needs it under half a chunk.
# At 30 Hz a 40-action chunk lasts 1.33 s, so the arm starved; at 15 Hz it lasts 2.67 s (97% motion
# in a robot-free test). --fps must equal the server's execution.action_fps (ACTION_FPS on the pod).
# The arm follows the policy's trajectories at half the demonstration speed.
# The link also stalls for several seconds at times. With the default 5 s request timeout and
# 4 s grace, one stall ended the run; with 12 s / 10 s the arm holds still and then carries on.
# rollout_chunk_fix.py = lerobot-rollout with the cold-start chunk delay capped (see the file).
#
# Safety (added after Pi0.5 whipped the arm on 2026-10-29): every motor command may move a joint at most
# MAX_STEP degrees (gripper: percent) from where it is now, i.e. at most MAX_STEP*fps deg/s. Test a new
# policy first with  SHADOW=1 LAUNCHER="rollout_safe.py <dir>" ./run_client.sh  (nothing moves), then run
# for real with  LAUNCHER="rollout_safe.py <dir>" ./run_client.sh  with a hand on the power switch.
cd "$(dirname "$0")"
exec .venv/bin/python ${LAUNCHER:-rollout_chunk_fix.py} \
  --inference.type=remote \
  --inference.endpoint=tcp/127.0.0.1:7447 \
  --inference.deployment=so101_jenga \
  --inference.mode=chunk \
  --inference.chunk_merge=aligned \
  --inference.semantics=so101-degrees-gripper-percent-v1 \
  --inference.encoding=jpeg \
  --inference.jpeg_quality=70 \
  --inference.refill_seconds=2.5 \
  --inference.action_timeout_s=12 \
  --inference.action_starvation_grace_s=15 \
  --inference.startup_timeout_s=30 \
  --inference.max_observation_age_s=8 \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B141107591 \
  --robot.id=SO-101 \
  --robot.use_degrees=true \
  --robot.max_relative_target="{shoulder_pan: ${MAX_STEP:-5.0}, shoulder_lift: ${MAX_STEP:-5.0}, elbow_flex: ${MAX_STEP:-5.0}, wrist_flex: ${MAX_STEP:-5.0}, wrist_roll: ${MAX_STEP:-5.0}, gripper: ${MAX_STEP_GRIPPER:-10.0}}" \
  --robot.cameras="{top cam: {type: opencv, index_or_path: ${TOP_CAM:-0}, width: 640, height: 480, fps: 30}, wrist cam: {type: opencv, index_or_path: ${WRIST_CAM:-1}, width: 640, height: 480, fps: 30}}" \
  --fps=${FPS:-15} \
  --task="${TASK:-pickup jenga and place inside the white box}" \
  --interactive=true "$@"
