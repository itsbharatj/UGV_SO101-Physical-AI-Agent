import draccus, dataclasses, sys
from dataclasses import dataclass
@dataclass
class C:
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = True
for arg in ("--max_relative_target=10", "--max_relative_target=10.0", "--max_relative_target=5.0", "--max_relative_target={shoulder_pan: 10.0}"):
    try:
        c = draccus.parse(config_class=C, args=[arg])
        print(arg, "->", repr(c.max_relative_target), type(c.max_relative_target).__name__)
    except Exception as e:
        print(arg, "-> ERROR", type(e).__name__, e)
print("draccus", getattr(draccus, "__version__", "?"))
