"""lerobot-rollout with guards and a full record of the run, for the SO-101 + remote policy.

Usage (through run_client.sh):   [SHADOW=1] LAUNCHER="rollout_safe.py <out_dir>" ./run_client.sh

- SHADOW=1: everything runs for real (cameras, joint reads, remote inference, action queue, cadence) but no
  motor command is ever written and torque stays off, so the arm does not move. Use it to test a new policy
  or a new server before letting it drive the arm.
- Always: every commanded target is clipped to the range seen in the demonstrations (plus a margin), the
  driver's --robot.max_relative_target cap still applies on top (real runs), and <out_dir> gets joints.csv
  (every observed state, requested target and sent target), events.txt (engine requests/results/holds) and a
  top+wrist snapshot every 0.5 s. A summary is printed at the end: chunk turnarounds, holds, the largest
  tick-to-tick jump the policy asked for, and how often the guards had to intervene.
"""

import os
import runpy
import sys
import time

import numpy as np
from PIL import Image

out_dir = sys.argv.pop(1)
os.makedirs(out_dir, exist_ok=True)
SHADOW = os.environ.get("SHADOW") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
fix = runpy.run_path(os.path.join(HERE, "rollout_chunk_fix.py"), run_name="patch_only")

from lerobot.remote_inference.engine import RemoteInferenceEngine  # noqa: E402
from lerobot.robots.so_follower import SOFollower  # noqa: E402
from lerobot.robots.utils import ensure_safe_goal_position  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Min/max of the demonstration actions (jainbharat/ugv_jenga_box_20261026_214928, degrees, gripper 0-100)
# widened by 5 deg. A target outside this box is clipped to it.
RANGE = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-111.0, 113.0),
    "elbow_flex": (-103.0, 104.0),
    "wrist_flex": (-25.0, 112.0),
    "wrist_roll": (-93.0, 85.0),
    "gripper": (0.0, 55.0),
}
stats = {"cmds": 0, "range_clips": 0, "cap_clips": 0, "max_jump": dict.fromkeys(JOINTS, 0.0), "jumps_over_10": 0}
turnarounds, rejected, holds, splice_steps = [], 0, 0, []
_prev_goal = None
_pending_splice = [False]  # set when a chunk arrives; the next command is the first of the new chunk

joints_log = open(f"{out_dir}/joints.csv", "w")
joints_log.write("t,kind," + ",".join(JOINTS) + "\n")
events_log = open(f"{out_dir}/events.txt", "w")


def _row(kind, values):
    joints_log.write(f"{time.monotonic():.3f},{kind}," + ",".join(f"{values.get(j, float('nan')):.2f}" for j in JOINTS) + "\n")


# --- motor commands -------------------------------------------------------------------------------------
_send, _observe, _connect = SOFollower.send_action, SOFollower.get_observation, SOFollower.connect


def connect(self, calibrate: bool = True):
    _connect(self, calibrate)
    if SHADOW:
        self.bus.disable_torque()
        print("[rollout_safe] SHADOW mode: torque disabled, motor commands are logged but not sent")


def send_action(self, action):
    global _prev_goal
    goal = {j: float(action[f"{j}.pos"]) for j in JOINTS if f"{j}.pos" in action}
    stats["cmds"] += 1
    if _prev_goal is not None:
        jump = {j: abs(goal[j] - _prev_goal[j]) for j in goal if j in _prev_goal}
        for j, v in jump.items():
            stats["max_jump"][j] = max(stats["max_jump"][j], v)
        if max(jump.values(), default=0.0) > 10.0:
            stats["jumps_over_10"] += 1
        if _pending_splice[0]:  # step from the old chunk's last command to the new chunk's first
            splice_steps.append(max(jump.values(), default=0.0))
            _pending_splice[0] = False
    _prev_goal = goal
    _row("cmd", goal)
    clipped = {j: min(max(v, RANGE[j][0]), RANGE[j][1]) for j, v in goal.items()}
    stats["range_clips"] += sum(abs(clipped[j] - goal[j]) > 1e-6 for j in goal)
    action = {**action, **{f"{j}.pos": v for j, v in clipped.items()}}
    if SHADOW:
        # Apply the same cap the driver would, then stop short of the bus write.
        capped = clipped
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
            capped = ensure_safe_goal_position({j: (v, present[j]) for j, v in clipped.items()}, self.config.max_relative_target)
        sent = {f"{j}.pos": v for j, v in capped.items()}
    else:
        sent = _send(self, action)
    sent_pos = {k.removesuffix(".pos"): float(v) for k, v in sent.items()}
    stats["cap_clips"] += sum(abs(sent_pos[j] - clipped[j]) > 1e-4 for j in clipped if j in sent_pos)
    _row("sent", sent_pos)
    return sent


def get_observation(self):
    obs = _observe(self)
    _row("obs", {j: float(obs[f"{j}.pos"]) for j in JOINTS})
    return obs


SOFollower.connect, SOFollower.send_action, SOFollower.get_observation = connect, send_action, get_observation

# --- engine events and camera snapshots -------------------------------------------------------------------
_event, _notify = RemoteInferenceEngine._event, RemoteInferenceEngine.notify_observation
_last_frame = [0.0]


def _event_hook(self, name, **values):
    global rejected, holds
    events_log.write(f"{time.monotonic():.3f} {name} {values}\n")
    events_log.flush()
    if name == "result":
        if values.get("accepted"):
            turnarounds.append(float(values.get("turnaround_s", 0.0)))
            _pending_splice[0] = True
        else:
            rejected += 1
    elif name == "planned_hold":
        holds += 1
    return _event(self, name, **values)


def notify_observation(self, obs):
    now = time.monotonic()
    if now - _last_frame[0] >= 0.5 and self.runtime.active:
        _last_frame[0] = now
        try:
            views = [obs[k] for k in ("top cam", "wrist cam") if isinstance(obs.get(k), np.ndarray)]
            tile = np.hstack([np.asarray(Image.fromarray(v).resize((320, 240))) for v in views])
            Image.fromarray(tile).save(f"{out_dir}/{time.strftime('%H%M%S')}_{int(now * 10) % 10}.jpg", quality=80)
        except Exception:
            pass
    return _notify(self, obs)


RemoteInferenceEngine._event, RemoteInferenceEngine.notify_observation = _event_hook, notify_observation

# --- return-to-start that actually arrives -------------------------------------------------------------------
# lerobot interpolates back to the start pose over a fixed 3 s. With the per-command cap the servos follow at
# ~40 deg/s, so a 190 deg shoulder move ends 75 deg short and torque is then cut with the arm half raised (run 1).
# Move at <= 20 deg/s instead and repeat until every joint is within 5 deg of the start pose.
from lerobot.rollout.strategies import core as _core  # noqa: E402

_strategy_cls = next(c for c in vars(_core).values() if isinstance(c, type) and "return_to_initial_position" in vars(c))
_orig_return = _strategy_cls.return_to_initial_position


def _return_to_initial_position(hw, duration_s: float = 3.0, fps: int = 50) -> bool:
    target = hw.initial_position
    for attempt in range(3):
        if target is None:
            break
        obs = hw.robot_wrapper.get_observation()
        delta = max(abs(float(obs[k]) - float(target[k])) for k in target)
        if delta < 5.0:
            print(f"[rollout_safe] back at the start pose (within {delta:.1f} deg)")
            return True
        print(f"[rollout_safe] returning to start: {delta:.0f} deg to go, pass {attempt + 1}")
        if not _orig_return(hw, duration_s=max(duration_s, delta / 20.0), fps=fps):
            return False
    return _orig_return(hw, duration_s=duration_s, fps=fps)


_strategy_cls.return_to_initial_position = staticmethod(_return_to_initial_position)


def summary():
    t = np.array(turnarounds) if turnarounds else np.zeros(0)
    lines = [
        f"mode: {'SHADOW (no motor writes)' if SHADOW else 'REAL'}; commands: {stats['cmds']}",
        f"chunks accepted: {len(t)}, rejected: {rejected}, starvation holds: {holds}",
        (f"turnaround median {np.median(t):.2f} s, p90 {np.quantile(t, 0.9):.2f} s, max {t.max():.2f} s" if len(t) else "no chunks received"),
        "largest tick-to-tick jump requested by the policy (deg): " + ", ".join(f"{j} {v:.1f}" for j, v in stats["max_jump"].items()),
        f"commands with a jump over 10 deg: {stats['jumps_over_10']}; range clips: {stats['range_clips']}; cap clips: {stats['cap_clips']}",
        (f"chunk-splice step (new chunk's first target vs previous target), n={len(splice_steps)}: median {np.median(splice_steps):.1f} deg, "
         f"p90 {np.quantile(splice_steps, 0.9):.1f}, max {max(splice_steps):.1f}  [go/no-go gate: median < 10, max < 20; Pi0.5 was 33 / 127, GR00T 9 / 20]"
         if splice_steps else "no chunk splices observed"),
    ]
    text = "[rollout_safe] summary\n  " + "\n  ".join(lines)
    print(text)
    with open(f"{out_dir}/summary.txt", "w") as f:
        f.write(text + "\n")


try:
    sys.exit(fix["main"]())
finally:
    joints_log.close()
    events_log.close()
    summary()
