# Remote policy inference on the SO-101 + UGV (2026-10-29 / 30)

Running fine-tuned VLA policies on the real arm with the GPU in the cloud: the Mac drives the arm and
cameras with `lerobot-rollout` (remote engine, Zenoh over an SSH tunnel), a RunPod A40 in EU-SE-1 runs
`lerobot-policy-server`. Dataset: `jainbharat/ugv_jenga_box_20261026_214928` (70 episodes, task
"pickup jenga and place inside the white box"). Policies, all 7000 steps on that dataset:
`jainbharat/groot_n17_7k_70ep_2026-10-29`, `jainbharat/pi05_7k_70ep_2026-10-29`,
`jainbharat/molmoact2_7k_70ep_2026-10-29`.

## Outcome

| Policy | On the arm | Why |
|---|---|---|
| GR00T N1.7 | reached the block and hovered with the gripper open, never grasped (2 trials) | top camera and UGV were at a viewpoint no demo had |
| Pi0.5 | violent whipping, power had to be cut | consecutive action chunks disagreed by 33° median / 127° max at every splice, and nothing limited motion per tick |
| **MolmoAct2** | **picks the block and places it in the box, about 20 s per cycle, repeatedly** | camera re-aligned to within 1 % of the training view, 5°/tick motion cap, link timeouts sized for India to EU |

MolmoAct2 numbers (2026-10-30, 120 s run): grasp at 41 s, block in the box at ~80 s, second pick and place
by 104 s, third pick at 113 s. 187 chunks, 0 rejected, 0.61 s median round trip. A later 7-minute session by
hand produced several more cycles. With the top camera 9 % off the training view the same policy closed a
few centimetres beside the block nine times and never corrected, so camera placement decides success.

Observations for the next round: MolmoAct2 is cautious (hesitant first seconds, slow approach, repeated
closes); ACT trained for 20k steps felt faster and more confident on the same task. More demonstrations and
longer training are the obvious next steps; MolmoAct2 was also the most expensive of the three to train.

## What is in here

- `lerobot-remote/` the Mac-side client and tools. Start with `README_MOLMOACT2.md` (step by step: pod,
  tunnel, cameras, shadow test, real run). `run_client.sh` (all settings), `rollout_safe.py` (per-tick
  motion cap, range clip, shadow mode, splice gate, verified return to start, full logging),
  `rollout_chunk_fix.py` (cold-start delay cap, camera frame age), `drive_run.py` (unattended runs),
  `check_cameras.py`, `align_camera.py`, `cam_shift.py`, `arm_state.py`, `shadow_consistency.py`,
  `runpod/` (pod bootstrap, policy server config, auto-delete timer), `runs/` (the hand-driven runs:
  joints, events, summaries).
- `inference_runs/2026-10-29_runpod_groot_pi05/` logs, joint CSVs and scripts of the GR00T and Pi0.5 runs.
- `inference_runs/2026-10-30_runpod_molmoact2/` MolmoAct2 shadow and powered runs (joint CSVs, chunk
  events, contact sheets, trace plots), camera reference frames for re-aligning the top and wrist cameras,
  the pod-side server log and config, the CUDA-graph consistency check.
- `training_logs/` training logs and loss curves of the three policies.
- `analysis_2026-10-30/` the forensic analysis of the Pi0.5 incident and the GR00T hover (per-run splice
  statistics, timeout analysis, viewpoint comparison, safety design notes), with verification notes.

Per-tick camera frames of the runs (about 3600 JPEGs) were left out of the repo; they are on the Mac under
`~/Desktop/SO-101/inference_runs/` and `~/Desktop/SO-101/lerobot-remote/runs/`.

## Settings that work from here (India) to EU-SE-1

- Server `action_fps: 15`, chunk mode, aligned merge; client `--fps=15 --inference.refill_seconds=2.5
  --inference.action_timeout_s=12 --inference.action_starvation_grace_s=15 --inference.startup_timeout_s=30
  --inference.max_observation_age_s=8 --inference.jpeg_quality=70`.
- `--robot.max_relative_target` 5° per tick per joint (gripper 10): the only speed limit on these servos.
- Top camera within 3 % of the training placement (use `align_camera.py` with the reference frames).
- Shadow test (torque off) before any powered run; hand on the power switch during powered runs.
