"""Independent recomputation of steady-state obs-cmd during constant-target holds, with settling check."""
import json, os
import numpy as np, pandas as pd
S = '/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
out = {}
for name in ["run4_frames", "run5_frames", "run6_frames", "pi05_run1_frames", "pi05_run2_frames"]:
    df = pd.read_csv(f"{S}/{name}/joints.csv"); df.columns = [c.strip() for c in df.columns]
    cmd = df[df.kind == "cmd"].reset_index(drop=True); obs = df[df.kind == "obs"].reset_index(drop=True)
    c = cmd[J].to_numpy(float); tc = cmd.t.to_numpy(float); o = obs[J].to_numpy(float); to = obs.t.to_numpy(float)
    print(f"\n== {name}: {len(c)} cmds over {tc[-1]-tc[0]:.1f}s, {len(o)} obs; cmd dt median {np.median(np.diff(tc)):.3f}s")
    # segments of identical consecutive commands (exact equality on all 6 joints)
    same = (c[1:] == c[:-1]).all(axis=1)
    segs = []; start = None
    for i, s in enumerate(same):
        if s and start is None: start = i
        if (not s or i == len(same) - 1) and start is not None:
            end = i if not s else i + 1
            if end - start >= 8 and tc[end] - tc[start] >= 0.5: segs.append((start, end))
            start = None
    rows = []
    for a, b in segs:
        T0, T1 = tc[a], tc[b]
        m_all = (to >= T0) & (to <= T1)
        m_last = (to >= T0 + 0.6 * (T1 - T0)) & (to <= T1)
        m_first = (to >= T0) & (to < T0 + 0.3 * (T1 - T0))
        # the obs just before the hold's first command (approach state)
        if m_last.sum() == 0: continue
        e_last = o[m_last].mean(axis=0) - c[a]
        e_first = o[m_first].mean(axis=0) - c[a] if m_first.sum() else [np.nan]*6
        e_final = o[m_all][-1] - c[a]
        drift = o[m_all][-1] - o[m_all][0]  # motion of the arm during the hold
        rows.append({"t_rel": round(float(T0 - tc[0]), 1), "dur_s": round(float(T1 - T0), 2), "n_obs_last40": int(m_last.sum()),
                     "pose_cmd": [round(float(x), 1) for x in c[a]],
                     "obs_minus_cmd_last40": [round(float(x), 2) for x in e_last],
                     "obs_minus_cmd_first30": [round(float(x), 2) for x in e_first],
                     "obs_minus_cmd_final_sample": [round(float(x), 2) for x in e_final],
                     "obs_drift_during_hold": [round(float(x), 2) for x in drift]})
    out[name] = rows
    for r in rows: print("  ", r)
json.dump(out, open("holds_recompute.json", "w"), indent=1)
# Summary for extended (non-rest) holds: rest = shoulder_lift < -90
ext = [(n, r) for n, rs in out.items() for r in rs if r["pose_cmd"][1] > -90]
print("\nExtended-pose holds:", len(ext))
E = np.array([r["obs_minus_cmd_last40"] for _, r in ext])
for j, jn in enumerate(J):
    print(f"  {jn:14s} obs-cmd range [{E[:, j].min():+.2f}, {E[:, j].max():+.2f}]  mean {E[:, j].mean():+.2f}")
rest = [(n, r) for n, rs in out.items() for r in rs if r["pose_cmd"][1] <= -90]
R = np.array([r["obs_minus_cmd_last40"] for _, r in rest])
print("Rest-pose holds:", len(rest))
for j, jn in enumerate(J):
    print(f"  {jn:14s} obs-cmd range [{R[:, j].min():+.2f}, {R[:, j].max():+.2f}]")
