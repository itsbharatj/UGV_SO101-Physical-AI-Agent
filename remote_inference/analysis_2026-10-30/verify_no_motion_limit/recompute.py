import sys, json, numpy as np, pandas as pd
S = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
J = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
out = {}
for name in ["pi05_run2_frames","pi05_run1_frames","run5_frames","run6_frames","run4_frames","run3_frames"]:
    try:
        df = pd.read_csv(f"{S}/{name}/joints.csv"); df.columns=[c.strip() for c in df.columns]
    except Exception as e:
        print(name, "skip:", e); continue
    cmd = df[df.kind=="cmd"].reset_index(drop=True); obs = df[df.kind=="obs"].reset_index(drop=True)
    c = cmd[J].to_numpy(float); tc = cmd.t.to_numpy(float); o = obs[J].to_numpy(float); to = obs.t.to_numpy(float)
    # --- cmd-to-cmd jumps, consecutive cmds with dt<=0.1 s (one 15 Hz tick) ---
    dt = np.diff(tc); d = np.diff(c, axis=0); tick = dt <= 0.1
    mag = np.abs(d).max(axis=1)
    r = {"n_cmd": len(c), "n_obs": len(o), "span_s": round(float(tc[-1]-tc[0]),1),
         "cmd_dt_median_ms": round(float(np.median(dt)*1000),1),
         "cmd_dt_p90_ms": round(float(np.percentile(dt,90)*1000),1),
         "n_consecutive_cmd_pairs_dt_le_0.1": int(tick.sum()),
         "jumps_gt5_any_joint_(dt<=0.1)": int(((mag>5)&tick).sum()),
         "jumps_gt10_any_joint_(dt<=0.1)": int(((mag>10)&tick).sum()),
         "jumps_gt15_any_joint_(dt<=0.1)": int(((mag>15)&tick).sum()),
         "jumps_gt5_any_joint_(all pairs)": int((mag>5).sum()),
         "max_cmd_jump_deg": round(float(mag[tick].max()),2)}
    k = int(np.argmax(np.where(tick, mag, -1)))
    r["max_jump"] = {"t_abs": round(float(tc[k+1]),3), "t_rel_first_cmd_s": round(float(tc[k+1]-tc[0]),2),
                     "dt_ms": round(float(dt[k]*1000),1), "joint": J[int(np.argmax(np.abs(d[k])))],
                     "per_joint_delta": {J[j]: round(float(d[k][j]),1) for j in range(6)},
                     "cmd_before": {J[j]: round(float(c[k][j]),1) for j in range(6)},
                     "cmd_after": {J[j]: round(float(c[k+1][j]),1) for j in range(6)}}
    # first 'live' cmd per investigator: an obs within 0.1 s before the cmd
    idx = np.searchsorted(to, tc, side="right")-1
    live = (idx>=0) & ((tc - np.where(idx>=0, to[np.clip(idx,0,None)], -1e9)) <= 0.1)
    tl0 = tc[live][0] if live.any() else tc[0]
    r["max_jump"]["t_rel_first_live_cmd_s"] = round(float(tc[k+1]-tl0),2)
    # --- observed per-tick motion (empirical slew), consecutive obs with dt<=0.1 ---
    dto = np.diff(to); do = np.diff(o, axis=0); ticko = dto <= 0.1
    mago = np.abs(do).max(axis=1)
    r["obs_dt_median_ms"] = round(float(np.median(dto)*1000),1)
    r["obs_max_step_deg_per_tick"] = {J[j]: round(float(np.abs(do[ticko][:,j]).max()),2) for j in range(6)}
    r["obs_steps_gt15_any_joint"] = int(((mago>15)&ticko).sum())
    r["obs_steps_gt10_any_joint"] = int(((mago>10)&ticko).sum())
    ko = int(np.argmax(np.where(ticko, mago, -1)))
    r["obs_max_step"] = {"t_abs": round(float(to[ko+1]),3), "dt_ms": round(float(dto[ko]*1000),1),
                         "joint": J[int(np.argmax(np.abs(do[ko])))], "deg": round(float(mago[ko]),2),
                         "deg_per_s": round(float(mago[ko]/dto[ko]),0)}
    # --- trace after the max cmd jump: how fast did obs follow? ---
    t0 = tc[k+1]; sel = (to >= t0-0.07) & (to <= t0+0.6)
    jj = [J.index("shoulder_lift"), J.index("elbow_flex")]
    r["trace_after_max_jump"] = [{"t_rel_ms": round(float((t-t0)*1000)), "shoulder_lift": round(float(p[jj[0]]),1), "elbow_flex": round(float(p[jj[1]]),1)} for t,p in zip(to[sel], o[sel])]
    out[name] = r
json.dump(out, open(f"{S}/wf_out/verify_no_motion_limit/recompute.json","w"), indent=1)
for n, r in out.items():
    print("==", n); 
    for kk, v in r.items():
        if kk != "trace_after_max_jump": print("  ", kk, "=", v)
    if n == "pi05_run2_frames":
        print("   trace after max jump (obs, ms rel to the jump cmd):")
        for row in r["trace_after_max_jump"]: print("     ", row)
