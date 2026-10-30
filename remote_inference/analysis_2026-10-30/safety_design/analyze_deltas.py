"""Per-step joint deltas in the training demos (30 fps) and in the recorded rollouts; clamp simulation."""
import glob, os, sys, json
import numpy as np, pandas as pd

OUT = sys.argv[1]
DS = os.path.expanduser('~/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000')
S = '/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PCT = [50, 90, 99, 99.9, 100]
TH = [1, 1.5, 2, 3, 4, 5, 8, 10, 15]

def pct_table(x, name):
    rows = []
    for j, jn in enumerate(J):
        v = np.abs(x[:, j])
        r = {"series": name, "joint": jn, "n": int(len(v))}
        for p in PCT:
            r[f"p{p}"] = round(float(np.percentile(v, p)), 3)
        for t in TH:
            r[f"frac_gt_{t}"] = round(float((v > t).mean()), 6)
        rows.append(r)
    return rows

report = {}
rows = []
dA, dS, A, Sx, AmS, per_ep = [], [], [], [], [], []
files = sorted(glob.glob(f'{DS}/episode_*.parquet'))
for f in files:
    ep = int(os.path.basename(f)[8:14])
    if ep == 20:
        continue
    df = pd.read_parquet(f, columns=['action', 'observation.state', 'timestamp', 'frame_index'])
    order = np.argsort(df['frame_index'].to_numpy())
    a = np.stack(df['action'].to_numpy())[order].astype(float)
    s = np.stack(df['observation.state'].to_numpy())[order].astype(float)
    ts = df['timestamp'].to_numpy()[order]
    dt = np.diff(ts)
    dA.append(np.diff(a, axis=0)); dS.append(np.diff(s, axis=0)); A.append(a); Sx.append(s); AmS.append(a - s)
    per_ep.append({"ep": ep, "frames": int(len(a)), "dur_s": round(float(len(a) / 30), 2),
                   "dt_med": round(float(np.median(dt)), 4), "dt_max": round(float(dt.max()), 4),
                   "max_abs_dA_per_joint": [round(float(np.abs(np.diff(a, axis=0))[:, j].max()), 2) for j in range(6)]})
dA = np.vstack(dA); dS = np.vstack(dS); A = np.vstack(A); Sx = np.vstack(Sx); AmS = np.vstack(AmS)
report["dataset"] = {"episodes": len(per_ep), "frames": int(len(A)), "fps": 30}
rows += pct_table(dA, "demo |action[t+1]-action[t]| per 30fps step")
rows += pct_table(dS, "demo |state[t+1]-state[t]| per 30fps step")
rows += pct_table(AmS, "demo |action-state| same frame (tracking/offset)")
# 2-step deltas (what one 15 Hz tick would be if the chunk were played at real demo speed)
rows += pct_table(A[2:] - A[:-2], "demo |action[t+2]-action[t]| (2 steps = 66ms demo time)")
ranges = []
for j, jn in enumerate(J):
    ranges.append({"joint": jn,
                   "action_min": round(float(A[:, j].min()), 2), "action_p0.5": round(float(np.percentile(A[:, j], 0.5)), 2),
                   "action_p99.5": round(float(np.percentile(A[:, j], 99.5)), 2), "action_max": round(float(A[:, j].max()), 2),
                   "state_min": round(float(Sx[:, j].min()), 2), "state_max": round(float(Sx[:, j].max()), 2),
                   "state_first_frame_mean": round(float(np.mean([e[0, j] for e in [np.stack(pd.read_parquet(f, columns=['observation.state'])['observation.state'].to_numpy()) for f in files[:1]]])), 2)})
report["joint_ranges"] = ranges
report["per_episode"] = per_ep

# Recorded rollouts: joints.csv rows "t,kind,6 joints"
runs = {}
for name in ["run5_frames", "run3_frames", "run4_frames", "run6_frames", "pi05_run1_frames", "pi05_run2_frames"]:
    p = f"{S}/{name}/joints.csv"
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    cmd = df[df["kind"] == "cmd"].reset_index(drop=True)
    obs = df[df["kind"] == "obs"].reset_index(drop=True)
    if len(cmd) < 2:
        runs[name] = {"cmds": int(len(cmd)), "obs": int(len(obs))}
        continue
    c = cmd[J].to_numpy(float); tc = cmd["t"].to_numpy(float)
    o = obs[J].to_numpy(float); to = obs["t"].to_numpy(float)
    dc = np.diff(c, axis=0); dtc = np.diff(tc)
    # command minus latest observation before it
    idx = np.searchsorted(to, tc, side="right") - 1
    ok = idx >= 0
    cmo = c[ok] - o[idx[ok]]
    r = {"cmds": int(len(c)), "obs": int(len(o)), "span_s": round(float(tc[-1] - tc[0]), 1),
         "cmd_dt_med": round(float(np.median(dtc)), 4), "cmd_dt_p99": round(float(np.percentile(dtc, 99)), 3),
         "cmd_to_cmd": pct_table(dc, name + " |cmd[k+1]-cmd[k]|"),
         "cmd_minus_obs": pct_table(cmo, name + " |cmd-latest obs|")}
    # clamp simulation relative to latest observed position (proxy for Present_Position read in send_action)
    sim = {}
    for X in [1, 1.5, 2, 3, 5]:
        clipped = (np.abs(cmo) > X)
        sim[str(X)] = {"frac_cmds_clamped_any_joint": round(float(clipped.any(axis=1).mean()), 4),
                       "per_joint": {J[j]: round(float(clipped[:, j].mean()), 4) for j in range(6)}}
    r["clamp_sim_vs_obs"] = sim
    # joints that left the training box
    box = {jn: (rg["action_min"], rg["action_max"]) for jn, rg in zip(J, ranges)}
    r["cmds_outside_train_box"] = {jn: int(((c[:, j] < box[jn][0]) | (c[:, j] > box[jn][1])).sum()) for j, jn in enumerate(J)}
    r["cmds_outside_train_box_margin10"] = {jn: int(((c[:, j] < box[jn][0] - 10) | (c[:, j] > box[jn][1] + 10)).sum()) for j, jn in enumerate(J)}
    runs[name] = r
report["runs"] = runs

pd.DataFrame(rows).to_csv(f"{OUT}/delta_percentiles.csv", index=False)
pd.DataFrame(ranges).to_csv(f"{OUT}/joint_ranges.csv", index=False)
with open(f"{OUT}/analysis.json", "w") as fh:
    json.dump(report, fh, indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows)[["series", "joint", "p50", "p90", "p99", "p99.9", "p100", "frac_gt_2", "frac_gt_3", "frac_gt_5"]].to_string(index=False))
print(pd.DataFrame(ranges).to_string(index=False))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("cmd_to_cmd", "cmd_minus_obs")} for k, v in runs.items()}, indent=1))
for k, v in runs.items():
    if "cmd_to_cmd" in v:
        print(pd.DataFrame(v["cmd_to_cmd"] + v["cmd_minus_obs"])[["series", "joint", "p50", "p90", "p99", "p100", "frac_gt_3", "frac_gt_5", "frac_gt_15"]].to_string(index=False))
print("episodes:", report["dataset"], "max per-ep |dA| overall:", max(max(e["max_abs_dA_per_joint"]) for e in per_ep))
