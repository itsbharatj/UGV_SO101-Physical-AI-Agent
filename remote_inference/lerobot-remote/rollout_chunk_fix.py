"""lerobot-rollout with two fixes for remote inference over a slow link.

1. Plain-chunk request delay capped at the chunk length. In chunk mode the client sends
   delay = ceil(turnaround / action interval), using the cold first turnaround for request two.
   One slow cold request (2.1 s against GR00T's 1.33 s chunk) gives a delay above the chunk length,
   which the server rejects as "Invalid RTC delay" and the rollout aborts. Plain chunks never use
   the delay (only RTC does), so capping it changes nothing else.
2. Camera frames up to 1.5 s old are accepted (LeRobot aborts at 0.5 s). The wrist camera stalled
   for ~1 s mid-run on 2026-10-29 and recovered; a longer stall still stops the rollout.
"""

import sys

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.inference import execution
from lerobot.inference.contracts import ExecutionMode

_estimate_delay = execution.estimate_delay


def estimate_delay(turnaround, interval, mode, training_max_delay, available):
    delay = _estimate_delay(turnaround, interval, mode, training_max_delay, available)
    return min(delay, available) if mode is ExecutionMode.CHUNK else delay


execution.estimate_delay = estimate_delay

MAX_FRAME_AGE_MS = 1500
_read_latest = OpenCVCamera.read_latest


def read_latest(self, max_age_ms: int = MAX_FRAME_AGE_MS):
    return _read_latest(self, max(max_age_ms, MAX_FRAME_AGE_MS))


OpenCVCamera.read_latest = read_latest

from lerobot.scripts.lerobot_rollout import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
