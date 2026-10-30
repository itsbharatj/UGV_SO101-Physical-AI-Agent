"""Read-only probe of the SO-101 follower: joint positions (degrees, gripper %) and torque state. Never writes."""
import sys
from lerobot.robots.so_follower import SOFollower, SO101FollowerConfig

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem5B141107591"
r = SOFollower(SO101FollowerConfig(port=port, id="SO-101", use_degrees=True))
r.bus.connect()
try:
    pos = r.bus.sync_read("Present_Position", num_retry=3)
    torque = r.bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
    for j in pos:
        print(f"{j:14s} {pos[j]:8.2f}   torque {'ON' if torque[j] else 'off'}")
finally:
    r.bus.disconnect(disable_torque=False)
