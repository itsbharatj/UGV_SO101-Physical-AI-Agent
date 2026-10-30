"""Parse LeRobot training logs into CSVs and plot loss curves.

Run: ~/Desktop/SO-101/so101-inverse-kinematics/src/.venv/bin/python plot_losses.py
Reads the *.log files next to this script, writes <run>.csv and loss_curves.png.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# run name -> (log file, step offset for resumed runs, marker that starts the attempt to keep)
RUNS = {
    "ACT 0-10k (HF Jobs)": ("hfjob_6ae1060c3fb22a235dd329da.log", 0, None),
    "ACT 10k-16k (HF Jobs, 70 eps)": ("hfjob_6ae2218441e90535df777155.log", 10_000, None),
    "Pi0.5 7k (RunPod)": ("pi05_run.log", 0, "08:18:48Z installing"),
    "MolmoAct2 (RunPod)": ("molmoact2_run.log", 0, "LAST_INSTALL"),
}
LINE = re.compile(r"(\d+)/\d+ \[[^\]]*\]INFO .*? step:\S+ .*?loss:([\d.]+) grdn:([\d.]+) lr:([\d.e+-]+)")


def parse(path, offset, marker):
    text = path.read_text(errors="ignore").replace("\r", "\n")
    if marker == "LAST_INSTALL":
        text = text[text.rfind("installing ("):]
    elif marker:
        text = text[text.find(marker):]
    rows = []
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            rows.append({"step": int(m[1]) + offset, "loss": float(m[2]), "grad_norm": float(m[3]), "lr": float(m[4])})
    return rows


fig, axes = plt.subplots(1, len(RUNS), figsize=(5 * len(RUNS), 4))
for ax, (name, (log, offset, marker)) in zip(axes, RUNS.items()):
    path = HERE / log
    rows = parse(path, offset, marker) if path.exists() else []
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    with (HERE / f"{slug}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "grad_norm", "lr"])
        w.writeheader()
        w.writerows(rows)
    if rows:
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], lw=1.5)
        ax.set_yscale("log")
        ax.set_title(f"{name}\nlast loss {rows[-1]['loss']:.3f} at step {rows[-1]['step']}", fontsize=9)
    else:
        ax.set_title(f"{name}\n(no data)", fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (log scale)")
    ax.grid(alpha=0.3)
    print(f"{name}: {len(rows)} points -> {slug}.csv")
fig.tight_layout()
fig.savefig(HERE / "loss_curves.png", dpi=110)
print("wrote", HERE / "loss_curves.png")
