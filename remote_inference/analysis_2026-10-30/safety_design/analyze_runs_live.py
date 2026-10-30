"""Rollout jumps restricted to live 15 Hz ticks (an obs within 0.1 s before the cmd), with chunk-boundary attribution."""
import json, os, re, sys
import numpy as np, pandas as pd
S = '/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
OUT = sys.argv[1]
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
res = {}
lines_out = []
for name in ["run5_frames", "run6_frames", "pi05_run2_frames"]:
    df = pd.read_csv(f"{S}/{name}/joints.csv"); df.columns = [c.strip() for c in df.columns]
    cmd = df[df.kind == "cmd"].reset_index(drop=True); obs = df[df.kind == "obs"].reset_index(drop=True)
    c = cmd[J].to_numpy(float); tc = cmd.t.to_numpy(float); o = obs[J].to_numpy(float); to = obs.t.to_numpy(float)
    idx = np.searchsorted(to, tc, side="right") - 1
    live = (idx >= 0) & ((tc - np.where(idx >= 0, to[np.clip(idx, 0, None)], -1e9)) <= 0.1)
    # chunk acceptance times from events.txt
    ev = open(f"{S}/{name}/events.txt").read().splitlines()
    acc = [float(l.split()[0]) for l in ev if " result " in l and "'accepted': True" in l]
    resets = [float(l.split()[0]) for l in ev if l.split()[1] == "reset"]
    cl = c[live]; tl = tc[live]; ol = o[idx[live]]
    dcl = np.diff(cl, axis=0); dtl = np.diff(tl)
    same_seg = dtl < 0.2  # consecutive live ticks only
    dcl = dcl[same_seg]; tj = tl[1:][same_seg]
    cmo = cl - ol
    r = {"live_cmds": int(live.sum()), "all_cmds": int(len(c)), "chunks_accepted": len(acc), "resets": len(resets)}
    for lab, arr in [("live |cmd[k+1]-cmd[k]|", dcl), ("live |cmd-obs|", cmo)]:
        r[lab] = {J[j]: {"p50": round(float(np.percentile(np.abs(arr[:, j]), 50)), 2), "p90": round(float(np.percentile(np.abs(arr[:, j]), 90)), 2),
                         "p99": round(float(np.percentile(np.abs(arr[:, j]), 99)), 2), "max": round(float(np.abs(arr[:, j]).max()), 2),
                         "n_gt3": int((np.abs(arr[:, j]) > 3).sum()), "n_gt5": int((np.abs(arr[:, j]) > 5).sum()), "n_gt15": int((np.abs(arr[:, j]) > 15).sum())} for j in range(6)}
    # clamp simulation on live ticks vs latest obs
    r["clamp_sim_live"] = {str(X): round(float((np.abs(cmo) > X).any(axis=1).mean()), 3) for X in [1, 2, 3, 5]}
    # attribute big cmd-to-cmd jumps (>5 deg any joint) to chunk boundaries: within 0.15 s after an accepted result
    big = np.where((np.abs(dcl) > 5).any(axis=1))[0]
    near_chunk = 0
    for k in big:
        t = tj[k]
        if any(0 <= t - a <= 0.15 for a in acc):
            near_chunk += 1
    r["big_jumps_gt5_any_joint"] = int(len(big)); r["big_jumps_within_150ms_after_chunk_accept"] = near_chunk
    # top 8 jumps
    mags = np.abs(dcl).max(axis=1); top = np.argsort(-mags)[:8]
    r["top_jumps"] = [{"t_rel_s": round(float(tj[k] - tl[0]), 2), "max_joint_jump": round(float(mags[k]), 1),
                       "per_joint": [round(float(x), 1) for x in dcl[k]],
                       "ms_after_last_chunk": (round(float(min((tj[k] - a) for a in acc if a <= tj[k]) * 1000)) if any(a <= tj[k] for a in acc) else None)} for k in top]
    res[name] = r
json.dump(res, open(f"{OUT}/live_tick_analysis.json", "w"), indent=1)
for name, r in res.items():
    print("==", name, {k: r[k] for k in ["live_cmds", "all_cmds", "chunks_accepted", "resets", "clamp_sim_live", "big_jumps_gt5_any_joint", "big_jumps_within_150ms_after_chunk_accept"]})
    for lab in ["live |cmd[k+1]-cmd[k]|", "live |cmd-obs|"]:
        print("  ", lab); print(pd.DataFrame(r[lab]).T.to_string())
    print("   top jumps:", json.dumps(r["top_jumps"]))
