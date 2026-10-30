"""Steady-state |cmd - obs| while the commanded target is constant (hold/starvation periods): the servo's static error incl. gravity sag."""
import json, sys
import numpy as np, pandas as pd
S = '/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
OUT = sys.argv[1]
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
out = {}
for name in ["run3_frames", "run4_frames", "run5_frames", "run6_frames", "pi05_run1_frames", "pi05_run2_frames"]:
    import os
    if not os.path.exists(f"{S}/{name}/joints.csv"): continue
    df = pd.read_csv(f"{S}/{name}/joints.csv"); df.columns = [c.strip() for c in df.columns]
    cmd = df[df.kind == "cmd"].reset_index(drop=True); obs = df[df.kind == "obs"].reset_index(drop=True)
    c = cmd[J].to_numpy(float); tc = cmd.t.to_numpy(float); o = obs[J].to_numpy(float); to = obs.t.to_numpy(float)
    # find runs of >= 8 consecutive cmds (>=0.5 s) with identical targets (|d|<0.05 all joints), take obs in the last 40% of the run
    same = (np.abs(np.diff(c, axis=0)) < 0.05).all(axis=1)
    segs = []; start = None
    for i, s in enumerate(same):
        if s and start is None: start = i
        if (not s or i == len(same) - 1) and start is not None:
            end = i if not s else i + 1
            if end - start >= 8 and tc[end] - tc[start] >= 0.5: segs.append((start, end))
            start = None
    rows = []
    for a, b in segs:
        t0 = tc[a] + 0.6 * (tc[b] - tc[a]); t1 = tc[b]
        m = (to >= t0) & (to <= t1)
        if m.sum() == 0: continue
        err = o[m].mean(axis=0) - c[a]
        rows.append({"t_rel": round(float(tc[a] - tc[0]), 1), "dur_s": round(float(tc[b] - tc[a]), 2), "n_obs": int(m.sum()),
                     "pose_cmd": [round(float(x), 1) for x in c[a]], "obs_minus_cmd": [round(float(x), 2) for x in err]})
    out[name] = rows
json.dump(out, open(f"{OUT}/static_error_holds.json", "w"), indent=1)
allerr = []
for name, rows in out.items():
    print("==", name, "hold segments:", len(rows))
    for r in rows:
        print("  ", r)
        allerr.append(r["obs_minus_cmd"])
allerr = np.array(allerr)
if len(allerr):
    print("ALL holds: |obs-cmd| per joint  p50", np.round(np.percentile(np.abs(allerr), 50, axis=0), 2), " p90", np.round(np.percentile(np.abs(allerr), 90, axis=0), 2), " max", np.round(np.abs(allerr).max(axis=0), 2))
    print("ALL holds: signed mean obs-cmd", np.round(allerr.mean(axis=0), 2))
