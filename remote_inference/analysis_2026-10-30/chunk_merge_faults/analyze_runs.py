"""Read-only analysis of recorded remote-inference runs: turnarounds, aligned-merge trims,
splice discontinuities (command jump at the first tick after an accepted chunk) and pauses.
Writes CSV/TXT into the directory given as argv[2]. Inputs: scratch dir argv[1]."""
import ast, csv, os, statistics as st, sys
import numpy as np

SCRATCH, OUT = sys.argv[1], sys.argv[2]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RUNS = ["run3_frames", "run4_frames", "run5_frames", "run6_frames", "pi05_run1_frames", "pi05_run2_frames"]

def parse_events(path):
    ev = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        t, rest = line.split(" ", 1)
        if " " in rest:
            name, d = rest.split(" ", 1)
            try:
                d = ast.literal_eval(d)
            except Exception:
                d = {}
        else:
            name, d = rest, {}
        ev.append((float(t), name, d))
    return ev

def parse_joints(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["t"]), row["kind"], np.array([float(row[j]) for j in JOINTS])))
    return rows

report = []
turn_rows = []
splice_rows = []
net_turnarounds_15hz = []
per_run_summary = []
for run in RUNS:
    d = os.path.join(SCRATCH, run)
    if not os.path.isdir(d):
        continue
    if not os.path.exists(os.path.join(d, "events.txt")):
        continue
    ev = parse_events(os.path.join(d, "events.txt"))
    sched = [e for e in ev if e[1] == "scheduling"]
    horizon = sched[0][2].get("execution_horizon_s") if sched else None
    steps = 50 if run.startswith("pi05") else 40
    fps = round(steps / horizon, 1) if horizon else None
    results = [e for e in ev if e[1] == "result"]
    requests = {e[2]["request_id"]: e for e in ev if e[1] == "request"}
    accepted = [e for e in results if e[2].get("accepted")]
    rejected = [e for e in results if not e[2].get("accepted")]
    waits = [e for e in ev if e[1] == "waiting"]
    resumes = [e for e in ev if e[1] == "resumed"]
    inval = [e for e in ev if e[1] == "invalidated"]
    cancelled = [e for e in ev if e[1] == "request_cancelled"]
    faults = [e for e in ev if e[1] == "fault"]
    turn = [e[2]["turnaround_s"] for e in results]
    worker = [e[2].get("server_durations", {}).get("worker") for e in results]
    net = [t - w for t, w in zip(turn, worker) if w is not None]
    trims = [e[2]["merge"].get("trimmed_actions") for e in accepted if "merge" in e[2]]
    usable = [e[2]["merge"].get("usable_actions") for e in accepted if "merge" in e[2]]
    t0 = ev[0][0]
    for e in results:
        req = requests.get(e[2]["request_id"])
        turn_rows.append({"run": run, "fps": fps, "t_rel": round(e[0] - t0, 3), "turnaround_s": round(e[2]["turnaround_s"], 3),
                          "server_worker_s": e[2].get("server_durations", {}).get("worker"),
                          "accepted": e[2].get("accepted"), "trimmed": e[2].get("merge", {}).get("trimmed_actions"),
                          "usable": e[2].get("merge", {}).get("usable_actions"),
                          "playback_at_submission_s": req[2].get("playback_at_submission_s") if req else None,
                          "delay_sent": req[2].get("delay") if req else None})
    if fps == 15.0:
        net_turnarounds_15hz.extend(net)
    # splice analysis
    jpath = os.path.join(d, "joints.csv")
    splice_jumps, resume_jumps = [], []
    all_jumps = []
    big_at_splice = 0
    big_total = 0
    if os.path.exists(jpath):
        rows = parse_joints(jpath)
        cmds = [(t, v) for t, k, v in rows if k == "cmd"]
        for (t1, v1), (t2, v2) in zip(cmds, cmds[1:]):
            if t2 - t1 <= 0.15:
                all_jumps.append((t2, float(np.max(np.abs(v2[:5] - v1[:5])))))
        accept_times = [e[0] for e in accepted]
        resume_times = [e[0] for e in resumes]
        for ta in accept_times:
            before = [c for c in cmds if c[0] <= ta]
            after = [c for c in cmds if c[0] > ta and c[0] - ta <= 0.3]
            if before and after:
                jump = np.abs(after[0][1] - before[-1][1])
                splice_jumps.append(float(np.max(jump[:5])))
                splice_rows.append({"run": run, "t_rel": round(ta - t0, 3), "max_joint_jump_deg": round(float(np.max(jump[:5])), 2),
                                    "gripper_jump": round(float(jump[5]), 2), "dt_cmd_s": round(after[0][0] - before[-1][0], 3)})
        for tr in resume_times:
            before = [c for c in cmds if c[0] <= tr]
            after = [c for c in cmds if c[0] > tr and c[0] - tr <= 0.5]
            if before and after:
                resume_jumps.append(float(np.max(np.abs(after[0][1][:5] - before[-1][1][:5]))))
        big = [(t, j) for t, j in all_jumps if j > 15]
        big_total = len(big)
        for t, j in big:
            if any(-0.07 <= t - ta <= 0.25 for ta in accept_times) or any(-0.07 <= t - tr <= 0.5 for tr in resume_times):
                big_at_splice += 1
    nonsplice = [j for t, j in all_jumps if not any(-0.07 <= t - e[0] <= 0.25 for e in accepted)]
    s = {
        "run": run, "fps": fps, "horizon_s": horizon, "n_results": len(results), "n_accepted": len(accepted), "n_rejected": len(rejected),
        "turnaround_median": round(st.median(turn), 3) if turn else None, "turnaround_p90": round(float(np.percentile(turn, 90)), 3) if turn else None,
        "turnaround_max": round(max(turn), 3) if turn else None,
        "net_only_median": round(st.median(net), 3) if net else None, "net_only_max": round(max(net), 3) if net else None,
        "server_worker_median": round(st.median([w for w in worker if w is not None]), 3) if any(w is not None for w in worker) else None,
        "trim_median": st.median(trims) if trims else None, "trim_max": max(trims) if trims else None,
        "usable_median": st.median(usable) if usable else None, "usable_min": min(usable) if usable else None,
        "n_waits": len(waits), "n_resumes": len(resumes), "resume_wait_s": [round(e[2]["wait_s"], 2) for e in resumes],
        "invalidation_wait_s": [round(e[2]["wait_s"], 2) for e in inval], "n_cancelled": len(cancelled),
        "faults": [e[2].get("reason") for e in faults],
        "splice_jump_median_deg": round(st.median(splice_jumps), 2) if splice_jumps else None,
        "splice_jump_max_deg": round(max(splice_jumps), 2) if splice_jumps else None,
        "splice_jumps_gt15": sum(j > 15 for j in splice_jumps), "n_splices_measured": len(splice_jumps),
        "resume_jump_max_deg": round(max(resume_jumps), 2) if resume_jumps else None,
        "nonsplice_jump_p99_deg": round(float(np.percentile(nonsplice, 99)), 2) if nonsplice else None,
        "nonsplice_jump_max_deg": round(max(nonsplice), 2) if nonsplice else None,
        "big_jumps_gt15_total": big_total, "big_jumps_gt15_within_splice_or_resume_window": big_at_splice,
    }
    per_run_summary.append(s)
    report.append(str(s))

with open(os.path.join(OUT, "turnarounds.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(turn_rows[0].keys()))
    w.writeheader(); w.writerows(turn_rows)
with open(os.path.join(OUT, "splice_jumps.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(splice_rows[0].keys()))
    w.writeheader(); w.writerows(splice_rows)

# Pause simulation for MolmoAct2 (30 steps): back-to-back aligned requests when refill >= H.
net = np.array(net_turnarounds_15hz)
lines = ["Empirical network-only turnaround (turnaround_s - server worker time) from 15 Hz runs: n=%d median=%.3f p90=%.3f max=%.3f" % (len(net), np.median(net), np.percentile(net, 90), net.max())]
for compute in (0.3, 0.6, 1.0):
    T = net + compute
    for fps, H in ((30, 1.0), (15, 2.0), (10, 3.0)):
        pairs = np.array([T[i] + T[i + 1] for i in range(len(T) - 1)])
        frac_pause = float(np.mean(pairs >= H))
        no_suffix = float(np.mean(T >= H))
        used_frac = float(np.mean(np.clip((H - T) / H, 0, 1)))
        per_min = 60.0 / float(np.mean(T)) * frac_pause
        lines.append("compute=%.1fs fps=%d H=%.1fs: P(T_k+T_k+1>=H)=%.2f  P(T>=H, no usable suffix)=%.2f  mean usable fraction of each chunk=%.2f  est. pauses/min=%.1f  mean T=%.2f" % (compute, fps, H, frac_pause, no_suffix, used_frac, per_min, T.mean()))
with open(os.path.join(OUT, "pause_simulation.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "per_run_summary.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\n".join(report))
print()
print("\n".join(lines))
