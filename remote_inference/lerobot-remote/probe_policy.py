"""Read the arm's joint state (torque untouched) and current camera frames, send them to the policy
server once, and print the predicted 40-step action chunk next to the current state.
Also prints first-frame state/action of a few training episodes for comparison.
"""

import runpy
import time

import cv2
import numpy as np
import pandas as pd

runpy.run_path("/Users/bharatjain/Desktop/SO-101/lerobot-remote/rollout_chunk_fix.py", run_name="patch_only")

from lerobot.inference.factory import RemoteInferenceConfig  # noqa: E402
from lerobot.remote_inference.client import RemoteClient  # noqa: E402
from lerobot.remote_inference.engine import RemoteInferenceEngine  # noqa: E402
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: E402

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
np.set_printoptions(precision=1, suppress=True, linewidth=150)

robot = SOFollower(SOFollowerRobotConfig(port="/dev/tty.usbmodem5B141107591", id="SO-101", use_degrees=True))
robot.bus.connect(handshake=False)
pos = robot.bus.sync_read("Present_Position")
robot.bus.disconnect(disable_torque=False)
state = np.array([pos[n] for n in NAMES], dtype=np.float32)
print("current state     ", state)

frames = {}
for idx, key in ((0, "observation.images.top cam"), (1, "observation.images.wrist cam")):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):
        ok, f = cap.read(); time.sleep(0.03)
    cap.release()
    frames[key] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

cfg = RemoteInferenceConfig(
    endpoint="tcp/127.0.0.1:7447", deployment="so101_jenga", mode="chunk", chunk_merge="aligned",
    semantics="so101-degrees-gripper-percent-v1", encoding="jpeg", jpeg_quality=70,
    refill_seconds=2.5, action_starvation_grace_s=4, max_observation_age_s=8,
)
client = RemoteClient.connect(cfg)
caps = client.capabilities
client.admit(features=caps.features, action_feature=caps.action_feature, semantics=cfg.semantics,
             action_interval=caps.action_interval, mode="chunk")
engine = RemoteInferenceEngine(client=client, config=cfg, task="pickup jenga and place inside the white box")
engine.start(); engine.resume()
engine.submit_observation({"observation.state": state, **frames})
actions = []
deadline = time.monotonic() + 10
while time.monotonic() < deadline and len(actions) < caps.execution_steps:
    engine.submit_observation({"observation.state": state, **frames})
    if engine.dispatch_allowed():
        a = engine.get_action(None)
        if a is not None:
            actions.append(a.numpy())
            continue
    time.sleep(0.01)
engine.stop()
actions = np.array(actions)
print(f"predicted chunk ({len(actions)} actions); rows = steps 0,5,10,...:")
for i in range(0, len(actions), 5):
    print(f"  step {i:2d}        ", actions[i], " delta", actions[i] - state)

root = "/Users/bharatjain/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000"
for ep in (0, 10, 35, 55, 69):
    df = pd.read_parquet(f"{root}/episode_{ep:06d}.parquet")
    s, a = np.stack(df["observation.state"]), np.stack(df["action"])
    print(f"train ep {ep:2d} state0", s[0], " action@1s", a[min(30, len(a) - 1)], f" len {len(a) / 30:.0f}s")
