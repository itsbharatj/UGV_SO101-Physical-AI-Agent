"""What a SHADOW run (arm not moving, so every chunk is predicted from the same state) says about a policy.

Usage: shadow_consistency.py <run_dir written by rollout_safe.py> [fps=15]

Reconstructs each chunk's played segment from joints.csv (cmd rows) and events.txt (request capture_time, result
arrival), indexes every command by its position in its chunk (k = round((t - capture_time) * fps)), then reports:
  1. same-state consistency: |chunk_{i+1}[k] - chunk_i[k]| over the steps both chunks cover. Two chunks predicted
     from the same observation should agree; this is the spread that becomes a splice jump on a moving arm.
  2. in-chunk speed: per-tick delta inside a chunk (what the 5 deg/tick cap will see), and the total excursion
     per joint from the start pose over the horizon (where the policy wants to go first).
The splice numbers printed by rollout_safe.py are NOT meaningful in shadow mode: the arm never follows the old
chunk, so a fast policy shows a big step from the old chunk's cursor to the new chunk's start even if it is
perfectly consistent.
"""
import ast
import sys

import numpy as np

run_dir = sys.argv[1]
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

requests, chunks = {}, []  # chunks: (arrival_t, capture_t)
for line in open(f"{run_dir}/events.txt"):
    t, name, rest = line.split(" ", 2)
    v = ast.literal_eval(rest.strip())
    if name == "request":
        requests[v["request_id"]] = v["capture_time"]
    elif name == "result" and v.get("accepted") and v["request_id"] in requests:
        chunks.append((float(t), requests[v["request_id"]]))
chunks.sort()

cmds, obs0 = [], None
for line in open(f"{run_dir}/joints.csv"):
    p = line.strip().split(",")
    if p[1] == "cmd":
        cmds.append((float(p[0]), np.array(p[2:], dtype=float)))
    elif p[1] == "obs" and obs0 is None:
        obs0 = np.array(p[2:], dtype=float)

# assign commands to chunks
per_chunk = [dict() for _ in chunks]
ci = -1
for t, a in cmds:
    while ci + 1 < len(chunks) and chunks[ci + 1][0] <= t:
        ci += 1
    if ci < 0:
        continue
    k = int(round((t - chunks[ci][1]) * fps))
    per_chunk[ci].setdefault(k, a)

print(f"{len(chunks)} chunks, {len(cmds)} commands, start pose {np.round(obs0, 1).tolist() if obs0 is not None else '?'}")
cons_all, cons_first, speeds, excursions = [], [], [], []
for i in range(len(per_chunk) - 1):
    A, B = per_chunk[i], per_chunk[i + 1]
    ks = sorted(set(A) & set(B))
    if ks:
        d = np.array([np.abs(B[k] - A[k]) for k in ks])  # (n_k, 6)
        cons_all.append(d.max(axis=1))
        cons_first.append(d[0].max())
for seg in per_chunk:
    ks = sorted(seg)
    if len(ks) >= 2:
        arr = np.array([seg[k] for k in ks])
        speeds.append(np.abs(np.diff(arr, axis=0)).max(axis=1))
        if obs0 is not None:
            excursions.append(arr[-1] - obs0)
if cons_all:
    c = np.concatenate(cons_all)
    print(f"1. same-state consistency |next chunk - this chunk| at equal step (worst joint, deg): "
          f"median {np.median(c):.1f}, p90 {np.quantile(c, 0.9):.1f}, max {c.max():.1f}; "
          f"at the first shared step: median {np.median(cons_first):.1f}, max {max(cons_first):.1f}, n={len(cons_first)}")
    print("   [Pi0.5-style multimodality would show up here as tens of degrees; <5 deg = chunks agree]")
if speeds:
    s = np.concatenate(speeds)
    print(f"2. in-chunk speed (deg per tick, worst joint): median {np.median(s):.1f}, p90 {np.quantile(s, 0.9):.1f}, max {s.max():.1f}; "
          f"ticks over 5 deg: {(s > 5).sum()} of {len(s)} (the cap will slow those)")
if excursions:
    e = np.array(excursions)
    print("   where the policy heads from the start pose over the played part of each chunk (mean delta, deg):")
    print("   " + ", ".join(f"{j} {m:+.1f}" for j, m in zip(J, e.mean(axis=0))))
    print("   gripper max commanded: %.1f" % max(seg[k][5] for seg in per_chunk for k in seg))
