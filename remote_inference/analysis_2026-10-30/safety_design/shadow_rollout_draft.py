"""DRAFT shadow launcher for lerobot-rollout on the SO-101 (NOT yet run; review before use).

Shadow mode = the full rollout loop runs (cameras, Present_Position reads, remote requests, action
queue, hold logic, 15 Hz cadence) but NO motion register is ever written and torque is NEVER enabled,
so the arm cannot move whatever the policy returns. Preconditions: arm resting in its gravity-stable
rest pose on the table; the policy server reachable; a device configured for "wrist cam" (the server
descriptor requires both image features, lerobot/rollout/context.py:758-762).

Usage (same flags as run_client.sh, plus the ones below):
  LAUNCHER=shadow_rollout_draft.py ./run_client.sh --robot.max_relative_target=5 --duration=30 \
      --return_to_initial_position=false
Outputs: $SHADOW_DIR/shadow_cmds.csv with rows t,kind(raw|clamped|obs),6 joints.
"""
import csv
import logging
import os
import runpy
import sys
import time

# 1) Existing fixes from the project launcher: CHUNK delay cap + 1.5 s camera-frame tolerance.
runpy.run_path("/Users/bharatjain/Desktop/SO-101/lerobot-remote/rollout_chunk_fix.py", run_name="patch_only")

from lerobot.motors.motors_bus import SerialMotorsBus  # noqa: E402
from lerobot.robots.so_follower.so_follower import SOFollower  # noqa: E402
from lerobot.rollout.strategies.core import RolloutStrategy  # noqa: E402

log = logging.getLogger("shadow")
OUT = os.environ.get("SHADOW_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), time.strftime("shadow_%Y%m%d_%H%M%S"))
os.makedirs(OUT, exist_ok=True)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# ---- Patch A: block every motion-register write at the bus. ---------------------------------------
# SOFollower.send_action (so_follower.py:212) is the only Goal_Position writer on the rollout path;
# blocking here also covers hold() (robot_wrapper.py:130) and return_to_initial_position (core.py:336).
BLOCKED = {"goal_position", "goal_speed", "goal_velocity", "goal_time", "torque_enable"}
_sync_write, _write = SerialMotorsBus.sync_write, SerialMotorsBus.write
blocked_writes = [0]


def sync_write(self, data_name, values, **kw):
    if data_name.lower() in BLOCKED:
        blocked_writes[0] += 1
        return None
    return _sync_write(self, data_name, values, **kw)


def write(self, data_name, motor, value, **kw):
    if data_name.lower() in BLOCKED:
        blocked_writes[0] += 1
        return None
    return _write(self, data_name, motor, value, **kw)


SerialMotorsBus.sync_write, SerialMotorsBus.write = sync_write, write


# ---- Patch B: never enable torque. --------------------------------------------------------------
# SOFollower.configure() runs inside bus.torque_disabled() whose exit calls enable_torque
# (motors_bus.py:701-705); suppressing it leaves every motor torque-free after connect().
def enable_torque(self, motors=None, num_retry=0):
    log.warning("shadow: enable_torque suppressed for %s", motors or "all motors")


SerialMotorsBus.enable_torque = enable_torque

# ---- Patch C: verify torque is off right after connect, then log raw vs clamped goals. ----------
_connect = SOFollower.connect


def connect(self, calibrate=True):
    _connect(self, calibrate)
    torque = self.bus.sync_read("Torque_Enable", normalize=False)
    if any(int(v) for v in torque.values()):
        self.bus.disable_torque(num_retry=5)
        torque = self.bus.sync_read("Torque_Enable", normalize=False)
    if any(int(v) for v in torque.values()):
        self.disconnect()
        raise SystemExit(f"shadow: torque still enabled after connect: {torque}")
    print(f"shadow: Torque_Enable={torque} (all off); motion writes blocked; log dir {OUT}", flush=True)


SOFollower.connect = connect

fh = open(os.path.join(OUT, "shadow_cmds.csv"), "w", newline="")
csvw = csv.writer(fh)
csvw.writerow(["t", "kind"] + JOINTS)
_send, _observe = SOFollower.send_action, SOFollower.get_observation


def send_action(self, action):
    csvw.writerow([f"{time.monotonic():.3f}", "raw"] + [f"{float(action.get(j + '.pos', float('nan'))):.2f}" for j in JOINTS])
    # Original driver code: Present_Position read + max_relative_target clamp; its Goal_Position write is blocked by Patch A.
    applied = _send(self, action)
    csvw.writerow([f"{time.monotonic():.3f}", "clamped"] + [f"{float(applied[j + '.pos']):.2f}" for j in JOINTS])
    fh.flush()
    return applied


def get_observation(self):
    obs = _observe(self)
    csvw.writerow([f"{time.monotonic():.3f}", "obs"] + [f"{float(obs[j + '.pos']):.2f}" for j in JOINTS])
    return obs


SOFollower.send_action, SOFollower.get_observation = send_action, get_observation

# ---- Patch D (optional): skip the 3 s return-to-initial interpolation (/reset and teardown). ------
# Harmless either way in shadow mode (writes are blocked); this only saves time.
RolloutStrategy.return_to_initial_position = staticmethod(lambda hw, duration_s=3.0, fps=50: True)

from lerobot.scripts.lerobot_rollout import main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        fh.close()
        print(f"shadow: blocked motion-register writes={blocked_writes[0]}; log {OUT}/shadow_cmds.csv", flush=True)
