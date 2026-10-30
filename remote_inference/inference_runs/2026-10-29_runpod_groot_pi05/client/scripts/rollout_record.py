"""Run rollout_chunk_fix.py and save a top+wrist snapshot every 0.5 s to <frames_dir> for review.

Usage: rollout_record.py <frames_dir> <rollout args...>
"""

import os
import runpy
import sys
import time

import numpy as np
from PIL import Image

frames_dir = sys.argv.pop(1)
os.makedirs(frames_dir, exist_ok=True)
fix = runpy.run_path("/Users/bharatjain/Desktop/SO-101/lerobot-remote/rollout_chunk_fix.py", run_name="patch_only")

from lerobot.remote_inference.engine import RemoteInferenceEngine  # noqa: E402

_notify = RemoteInferenceEngine.notify_observation
_last = [0.0]


def notify_observation(self, obs):
    now = time.monotonic()
    if now - _last[0] >= 0.5 and self.runtime.active:
        _last[0] = now
        try:
            views = [obs[k] for k in ("top cam", "wrist cam") if isinstance(obs.get(k), np.ndarray)]
            tile = np.hstack([np.asarray(Image.fromarray(v).resize((320, 240))) for v in views])
            Image.fromarray(tile).save(f"{frames_dir}/{time.strftime('%H%M%S')}_{int(now * 10) % 10}.jpg", quality=80)
        except Exception:
            pass
    return _notify(self, obs)


RemoteInferenceEngine.notify_observation = notify_observation

# Engine events (requests, results with accept/reject reason, cancellations), unaffected by log muting.
events = open(f"{frames_dir}/events.txt", "w")
_event = RemoteInferenceEngine._event


def _event_hook(self, name, **values):
    events.write(f"{time.monotonic():.3f} {name} {values}\n")
    events.flush()
    return _event(self, name, **values)


RemoteInferenceEngine._event = _event_hook

# Joint log: every observed state and every commanded goal, for comparing command vs motion.
from lerobot.robots.so_follower import SOFollower  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
log = open(f"{frames_dir}/joints.csv", "w")
log.write("t,kind," + ",".join(JOINTS) + "\n")
_send, _observe = SOFollower.send_action, SOFollower.get_observation


def send_action(self, action):
    log.write(f"{time.monotonic():.3f},cmd," + ",".join(f"{float(action.get(j + '.pos', float('nan'))):.2f}" for j in JOINTS) + "\n")
    return _send(self, action)


def get_observation(self):
    obs = _observe(self)
    log.write(f"{time.monotonic():.3f},obs," + ",".join(f"{float(obs[j + '.pos']):.2f}" for j in JOINTS) + "\n")
    return obs


SOFollower.send_action, SOFollower.get_observation = send_action, get_observation
try:
    sys.exit(fix["main"]())
finally:
    log.close()
