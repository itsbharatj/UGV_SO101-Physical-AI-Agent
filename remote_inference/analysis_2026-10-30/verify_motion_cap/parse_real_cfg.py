# Parse the real so101_follower robot config through lerobot's draccus registry (no hardware is touched: config only).
import draccus
from dataclasses import dataclass, field
from lerobot.robots import RobotConfig
import lerobot.robots.so_follower  # registers so101_follower
@dataclass
class Wrap:
    robot: RobotConfig = field(default_factory=lambda: None)
for v in ("10", "10.0", "{shoulder_pan: 10.0, shoulder_lift: 10.0, elbow_flex: 10.0, wrist_flex: 10.0, wrist_roll: 10.0, gripper: 30.0}", "{gripper: 30.0}"):
    try:
        c = draccus.parse(config_class=Wrap, args=["--robot.type=so101_follower","--robot.port=/dev/null","--robot.use_degrees=true",f"--robot.max_relative_target={v}"])
        print(repr(v), "->", repr(c.robot.max_relative_target), type(c.robot.max_relative_target).__name__)
    except Exception as e:
        print(repr(v), "-> ERROR", type(e).__name__, str(e)[:160])
