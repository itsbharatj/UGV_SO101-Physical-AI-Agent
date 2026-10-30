"""Robot-free remote-inference test: drive the real RemoteInferenceEngine at 30 Hz with camera frames.

Usage: remote_bench.py <seconds> <jpeg_quality> <refill_s> <starvation_grace_s>
Prints per-request payload size, turnaround and server durations, then whether playback starved.
"""

import logging
import runpy
import sys
import time

import numpy as np
from PIL import Image

# Same delay cap as run_client.sh.
runpy.run_path("/Users/bharatjain/Desktop/SO-101/lerobot-remote/rollout_chunk_fix.py", run_name="patch_only")

from lerobot.inference.factory import RemoteInferenceConfig  # noqa: E402
from lerobot.remote_inference.client import RemoteClient  # noqa: E402
from lerobot.remote_inference.engine import RemoteInferenceEngine  # noqa: E402

SCRATCH = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
seconds, quality, refill, grace = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
max_age = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

cfg = RemoteInferenceConfig(
    endpoint="tcp/127.0.0.1:7447", deployment="so101_jenga", mode="chunk", chunk_merge="aligned",
    semantics="so101-degrees-gripper-percent-v1", encoding="jpeg", jpeg_quality=quality,
    refill_seconds=refill, action_starvation_grace_s=grace, max_observation_age_s=max_age,
)
client = RemoteClient.connect(cfg)

results = []
_infer = client.infer
def infer(request, **kw):
    t = time.monotonic()
    try:
        chunk = _infer(request, **kw)
        results.append((t, time.monotonic() - t, round(chunk.server_durations.get("worker", 0) * 1000), None))
        return chunk
    except Exception as exc:
        results.append((t, time.monotonic() - t, None, type(exc).__name__))
        raise
client.infer = infer

caps = client.capabilities
client.admit(features=caps.features, action_feature=caps.action_feature, semantics=cfg.semantics,
             action_interval=caps.action_interval, mode="chunk")
engine = RemoteInferenceEngine(client=client, config=cfg, task="pickup jenga and place inside the white box")
engine.start()
engine.resume()

frames = {
    "observation.images.top cam": np.asarray(Image.open(f"{SCRATCH}/cam0.jpg").convert("RGB")),
    "observation.images.wrist cam": np.asarray(Image.open(f"{SCRATCH}/cam1.jpg").convert("RGB")),
}
state = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0], dtype=np.float32)
t0 = time.monotonic(); tick = 0; actions = 0; empty_ticks = 0; holds = 0
while time.monotonic() - t0 < seconds and not engine.failed:
    engine.submit_observation({"observation.state": state, **frames})
    if engine.dispatch_allowed():
        if engine.get_action(None) is not None:
            actions += 1
        elif actions:
            empty_ticks += 1
    else:
        holds += 1
        engine.acknowledge_hold()
    tick += 1
    time.sleep(max(0.0, t0 + tick * caps.action_interval - time.monotonic()))
elapsed = time.monotonic() - t0
failure = engine.runtime.failure
engine.stop()

ok = [r[1] for r in results if r[3] is None]
print(f"=== q{quality} refill {refill}s grace {grace}s: ran {elapsed:.1f}s, motion {actions * caps.action_interval:.1f}s at {1 / caps.action_interval:.0f} Hz "
      f"({100 * actions / max(tick, 1):.0f}% of ticks), paused ticks {empty_ticks + holds}, failure={failure!r}")
print(f"    {len(ok)} chunks, turnaround median {1000 * np.median(ok):.0f} ms, max {1000 * max(ok):.0f} ms; "
      f"server worker median {np.median([r[2] for r in results if r[2] is not None]):.0f} ms; "
      f"errors {[r[3] for r in results if r[3]]}")
