# Running MolmoAct2 on the SO-101 (policy on a RunPod GPU, arm on this Mac)

Status 2026-10-30: `jainbharat/molmoact2_7k_70ep_2026-10-29` picked the jenga block and placed it in the box
twice in one 120 s run (about 20 s per cycle) with the settings below. Pi0.5 whipped the arm on 2026-10-29 and
GR00T hovered; the diagnosis and the per-tick cap that fixed the safety side are described in rollout_safe.py.

Two hard rules: only one process can hold the arm's serial port and only one client can be attached to the
policy server, so never start a second rollout (or arm_state.py) while one is running. And every powered run
is done with a hand on the power switch; cutting power is safe, the client faults out cleanly.

## 1. GPU pod (skip if one is running)
Create a pod on RunPod (console or `runpodctl`): A40 48 GB in EU-SE-1 was $0.59/h; 100 GB container disk (the
server downloads the 22 GB MolmoAct2 base model as well as the fine-tune); the PyTorch 2.x / CUDA 12 template;
SSH exposed. Note the direct SSH host and port (e.g. 194.68.245.209 22148). Then, from this folder:

    runpod/bootstrap_pod.sh <host> <port> molmoact2 240 15
    #                                     kind  auto-delete after 240 min   action fps (must equal FPS below)

It copies the HF token as /root/.hf_token (never into the pod's environment), installs lerobot, downloads the
model and starts `lerobot-policy-server`. Takes 10-15 min:

    ssh -p <port> root@<host> 'tail -f /root/work/infer.log'      # wait for "Policy server ready"

Extend the auto-delete timer (here: 3 more hours from now):

    ssh -p <port> root@<host> 'pkill -f "autodel[e]te.sh"; setsid nohup bash /root/autodelete.sh 180 >/dev/null 2>&1 < /dev/null &'

Delete the pod when done (console or `runpodctl remove pod <id>`); it bills while it exists, stopped or not.

## 2. Tunnel (its own terminal, leave it open)

    ./tunnel_runpod.sh <host> <port>

## 3. Cameras, arm, scene

    .venv/bin/python check_cameras.py        # prints "index i: WxH"; writes cameras_now.jpg with each index labelled
    .venv/bin/python align_camera.py <top index> ~/Desktop/SO-101/inference_runs/2026-10-30_runpod_molmoact2/camera_refs/ref_top_posB_ep17_frame0.png
    .venv/bin/python arm_state.py            # joint angles and torque state, read-only

- Brio = "top cam", USB camera = "wrist cam". Indices change when anything is (un)plugged, so check every time.
- Move the Brio until the shift in the align window's title is under 3 % (q quits, s saves a snapshot).
  At 9 % off the policy closed next to the block every time; at 1 % it succeeded.
- Start pose (torque is off, place it by hand): pan 0, shoulder_lift -100, elbow 96, wrist_flex 55,
  wrist_roll -4, gripper 1 (a few degrees either way is fine).
- Scene as in the reference frame: block on the table in front of the UGV, white box to the side, no tools or
  hands in either view. Lighting does not matter.

## 4. Shadow test (nothing moves) after any change to the server, model or scripts

    SHADOW=1 TOP_CAM=0 WRIST_CAM=1 LAUNCHER="rollout_safe.py runs/shadow" .venv/bin/python drive_run.py runs/shadow.log 45 -- ./run_client.sh
    .venv/bin/python shadow_consistency.py runs/shadow
    # want: chunks accepted, 0 rejected, 0 holds; same-state consistency median < 5 deg; in-chunk speed <= 5 deg/tick

## 5. Real run

Unattended: N seconds of policy, then it returns to the start pose on its own and stops.

    TOP_CAM=0 WRIST_CAM=1 LAUNCHER="rollout_safe.py runs/<name>" .venv/bin/python drive_run.py runs/<name>.log 120 -- ./run_client.sh

Interactive: type /start, later /reset (back to the start pose) and /stop.

    TOP_CAM=0 WRIST_CAM=1 LAUNCHER="rollout_safe.py runs/<name>" ./run_client.sh

Knobs (environment variables): MAX_STEP=5 (deg per tick per joint, keep within 4..8), MAX_STEP_GRIPPER=10,
FPS=15 (must equal the server's action fps), TASK="pickup jenga and place inside the white box".
Each run leaves runs/<name>/ with joints.csv (measured / policy target / sent), events.txt (every chunk),
a top|wrist frame every 0.5 s, and summary.txt (written at a normal exit; a killed run keeps the csv but not
the summary). A gripper reading near 11 means it is holding the block; near 1 means it closed on nothing.

## What to expect
- 15 Hz, 2 s chunks, ~0.6 s round trip from here to EU-SE-1; 0 rejected chunks in 300+ chunks today.
- The arm moves at about half the demonstration speed (15 Hz playback plus the cap).
- Typical run: a hesitant first 4 s, reach by 12 s, one or two missed closes, grasp, carry, release over the
  box, then it goes back for the block (put a new one down and it continues).
