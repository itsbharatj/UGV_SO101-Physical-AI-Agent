"""Independent recomputation of the pause-rate claim from the recorded events.txt files (read-only)."""
import ast, csv, math, os, re
import numpy as np

S = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
OUT = os.path.join(S, "wf_out", "verify_pause_rate")
RUNS = [("run5", f"{S}/run5_frames/events.txt", 40), ("run6", f"{S}/run6_frames/events.txt", 40),
        ("pi05_run1", f"{S}/pi05_run1_frames/events.txt", 50), ("pi05_run2", f"{S}/pi05_run2_frames/events.txt", 50),
        ("molmo_shadow1", f"{S}/molmo_shadow1/events.txt", 30)]
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
def parse(path):
    ev = []
    for line in open(path):
        m = re.match(r"(\S+) (\S+) ?(.*)$", line.rstrip("\n"))
        if not m: continue
        t, name, rest = m.groups()
        try: d = ast.literal_eval(rest) if rest else {}
        except Exception: d = {"raw": rest}
        ev.append((float(t), name, d))
    return ev
def q(x, p): return float(np.percentile(x, p))
data, allrows = {}, []
for run, path, steps in RUNS:
    ev = parse(path)
    H = [d["execution_horizon_s"] for t, n, d in ev if n == "scheduling"][0]
    dt = H / steps
    reqs = {d["request_id"]: (t, d) for t, n, d in ev if n == "request"}
    res = [(t, d) for t, n, d in ev if n == "result"]
    rows = []
    for i, (t, d) in enumerate(res):
        tq, rq = reqs[d["request_id"]]
        sd = d.get("server_durations", {}); m = d.get("merge", {})
        rows.append(dict(run=run, i=i, t_req=round(tq, 3), t_res=round(t, 3), tau=d["turnaround_s"], worker=sd.get("worker"), policy=sd.get("policy"),
                         pre=sd.get("preprocessing"), post=sd.get("postprocessing"), server_queue=sd.get("queue"), net=d["turnaround_s"] - sd["worker"],
                         accepted=d["accepted"], trim=m.get("trimmed_actions"), usable=m.get("usable_actions"), pb_sub=rq["playback_at_submission_s"],
                         margin=d.get("timing_margin_s"), delay=rq["delay"], obs_cursor=rq["observation_cursor"], sub_cursor=rq["submission_cursor"],
                         cold=(i == 0), restart=(i > 0 and rq["playback_at_submission_s"] == 0.0), qpb_after=d["queue_playback_s"]))
    data[run] = dict(H=H, dt=dt, steps=steps, rows=rows, n_req=len(reqs),
                     waits=[t for t, n, d in ev if n == "waiting"], cancels=[(t, d) for t, n, d in ev if n == "request_cancelled"],
                     resumed=[(t, d) for t, n, d in ev if n == "resumed"], inval=[(t, d) for t, n, d in ev if n == "invalidated"],
                     faults=[(t, d) for t, n, d in ev if n == "fault"], holds=[(t, d) for t, n, d in ev if n == "planned_hold"])
    allrows += rows
with open(os.path.join(OUT, "all_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)

P("=== 1. Per-run summary (from events.txt) ===")
for run in data:
    D = data[run]; R = D["rows"]
    tau = np.array([r["tau"] for r in R]); wk = np.array([r["worker"] for r in R]); net = tau - wk
    ss = np.array([r["tau"] for r in R if not r["cold"]]); ssn = np.array([r["net"] for r in R if not r["cold"]])
    P(f"{run}: H={D['H']:.3f}s dt={D['dt']*1000:.1f}ms steps={D['steps']} requests={D['n_req']} results={len(R)} accepted={sum(r['accepted'] for r in R)} "
      f"waiting={len(D['waits'])} cancelled={len(D['cancels'])} resumed={[round(d['wait_s'],2) for t,d in D['resumed']]} faults={[d.get('reason') for t,d in D['faults']]}")
    P(f"   cold tau={tau[0]:.3f} (net {net[0]:.3f}); restarts(pb_sub=0, i>0)={[ (r['i'], round(r['tau'],3)) for r in R if r['restart']]}")
    if len(ss):
        P(f"   steady tau: n={len(ss)} median={np.median(ss):.3f} p90={q(ss,90):.3f} max={ss.max():.3f} | net: median={np.median(ssn):.3f} p90={q(ssn,90):.3f} max={ssn.max():.3f}")
    P(f"   worker: median={np.median(wk):.3f} p90={q(wk,90):.3f} max={wk.max():.3f} | policy median={np.median([r['policy'] for r in R]):.3f} post median={np.median([r['post'] for r in R]):.3f} pre median={np.median([r['pre'] for r in R]):.3f} server_queue max={max(r['server_queue'] for r in R):.4f}")
    acc = [r for r in R if r["accepted"] and not r["cold"] and not r["restart"]]
    if acc:
        diff = np.array([r["trim"] - r["tau"] / D["dt"] for r in acc])
        mg = np.array([r["margin"] for r in acc if r["margin"] is not None])
        P(f"   trim - tau/dt (ticks): mean={diff.mean():.2f} min={diff.min():.2f} max={diff.max():.2f}; timing margin: min={mg.min():.3f} median={np.median(mg):.3f}; usable min={min(r['usable'] for r in acc)}")

P("\n=== 2. Reproduce the investigator's network-only sample (all results of the 15 Hz runs, incl. cold starts) ===")
inv_order = ["run5", "run6", "pi05_run1", "pi05_run2"]
net_all = np.concatenate([[r["net"] for r in data[run]["rows"]] for run in inv_order])
P(f"n={len(net_all)} median={np.median(net_all):.3f} p90={q(net_all,90):.3f} max={net_all.max():.3f}")
def pair_prob(T, H):
    s = T[:-1] + T[1:]; return float(np.mean(s >= H)), int(np.sum(s >= H)), len(s)
for c in (0.3, 0.6, 1.0):
    T = net_all + c
    P(f"compute={c}: " + "  ".join(f"H={H}: P={pair_prob(T,H)[0]:.2f}" for H in (1.0, 2.0, 3.0)) + f"  meanT={T.mean():.2f} pauses/min(inv formula)={60/T.mean()*pair_prob(T,2.0)[0]:.1f}")
cold_vals = [(run, round(data[run]["rows"][0]["net"], 3)) for run in inv_order]
P(f"cold-start net values included in that sample: {cold_vals}")
P(f"sample composition: " + ", ".join(f"{run}={len(data[run]['rows'])}" for run in inv_order))

P("\n=== 3. Variants: exclude cold starts / restarts; within-run pairs only; measured MolmoAct2 worker time ===")
molmo_wk = np.array([r["worker"] for r in data["molmo_shadow1"]["rows"]])
wk_med, wk_p90 = float(np.median(molmo_wk)), q(molmo_wk, 90)
P(f"MolmoAct2 worker (molmo_shadow1, n={len(molmo_wk)}): median={wk_med:.3f} p90={wk_p90:.3f} max={molmo_wk.max():.3f}  policy median={np.median([r['policy'] for r in data['molmo_shadow1']['rows']]):.3f} post median={np.median([r['post'] for r in data['molmo_shadow1']['rows']]):.3f}")
def within_run_pairs(runs, drop_cold, drop_restart, compute, H, extra=0.0):
    hits = tot = 0
    for run in runs:
        R = [r for r in data[run]["rows"] if not (drop_cold and r["cold"]) and not (drop_restart and r["restart"])]
        T = np.array([r["net"] for r in R]) + compute
        s = T[:-1] + T[1:]; hits += int(np.sum(s + extra >= H)); tot += len(s)
    return hits, tot
for label, dc, dr in (("all incl. cold (investigator)", False, False), ("drop first result of each run", True, False), ("drop first + post-starvation restarts", True, True)):
    for c in (0.3, wk_med, wk_p90):
        parts = []
        for H in (1.0, 2.0, 3.0):
            h, n = within_run_pairs(inv_order, dc, dr, c, H); parts.append(f"H={H}: {h}/{n}={h/n:.3f}")
        h2, n2 = within_run_pairs(inv_order, dc, dr, c, 2.0, extra=2/15)
        P(f"[{label}] compute={c:.3f}: " + "  ".join(parts) + f"  | H=2.0 with +2 ticks discretization: {h2}/{n2}={h2/n2:.3f}")

P("\n=== 4. MolmoAct2 shadow run 2026-10-30 (15 Hz, H=2.0 s): direct empirical test ===")
R = data["molmo_shadow1"]["rows"]; tau = np.array([r["tau"] for r in R]); net = np.array([r["net"] for r in R])
s = tau[:-1] + tau[1:]
P(f"results={len(R)} accepted={sum(r['accepted'] for r in R)} waiting events={len(data['molmo_shadow1']['waits'])} cancelled={len(data['molmo_shadow1']['cancels'])}")
P(f"tau all: median={np.median(tau):.3f} p90={q(tau,90):.3f} max={tau.max():.3f}; tau steady: median={np.median(tau[1:]):.3f} p90={q(tau[1:],90):.3f} max={tau[1:].max():.3f}")
P(f"net all: median={np.median(net):.3f} p90={q(net,90):.3f} max={net.max():.3f}")
P(f"consecutive pair sums: n={len(s)} max={s.max():.3f} count>=2.0: {int(np.sum(s>=2.0))} count>=1.867 (H-2dt): {int(np.sum(s>=2.0-2/15))} count>=1.5: {int(np.sum(s>=1.5))}")
mg = np.array([r["margin"] for r in R if r["margin"] is not None]); P(f"timing margin: min={mg.min():.3f} median={np.median(mg):.3f}; usable min={min(r['usable'] for r in R)} median={np.median([r['usable'] for r in R])}")
dur = R[-1]["t_res"] - R[0]["t_req"]; P(f"session span={dur:.1f}s; splices/min={len(s)/dur*60:.1f}; expected pauses at the finding's 14/min over this span={14*dur/60:.1f}, observed=0; P(0 of {len(s)} | p=0.20)={0.8**len(s):.1e}")
P(f"shadow link vs 10-29 link, steady-state net: shadow median={np.median(net[1:]):.3f} p90={q(net[1:],90):.3f} max={net[1:].max():.3f}")
# what would the shadow-run net distribution give at 30 Hz / 10 Hz
for H, fps in ((1.0, 30), (2.0, 15), (3.0, 10)):
    T = net + wk_med; s2 = T[:-1] + T[1:]; P(f"   shadow-link projection fps={fps} H={H}: P(pair>=H)={np.mean(s2>=H):.2f} mean usable fraction={np.mean(np.clip((H-T)/H,0,1)):.2f}")

P("\n=== 5. Calibration of the pair-sum rule on the 2026-10-29 runs (actual turnarounds vs observed starvation holds) ===")
for run in ("run5", "run6", "pi05_run2"):
    D = data[run]; R = D["rows"]; H = D["H"]
    T = np.array([r["tau"] for r in R]); s = T[:-1] + T[1:]
    idx = np.where(s >= H)[0]
    # requests that never produced a result (cancelled) sit between results; show them
    P(f"{run}: H={H:.3f} consecutive result pairs n={len(s)} with sum>=H: {len(idx)} {[(int(i), round(float(s[i]),3)) for i in idx]}  sum>=H-2dt: {int(np.sum(s>=H-2*D['dt']))}; observed waiting={len(D['waits'])} cancelled={[(round(t,3), round(d['wait_s'],3), d.get('playback_at_submission_s')) for t,d in D['cancels']]}")
    # for each cancelled request: the previous accepted result's usable and the time to the hold
    for t, d in D["cancels"]:
        prev = [r for r in R if r["t_res"] <= t]
        if prev:
            pr = prev[-1]; P(f"   cancelled request {d['request_id'][:8]} at +{t - pr['t_res']:.3f}s after result i={pr['i']} (tau={pr['tau']:.3f}, trim={pr['trim']}, usable={pr['usable']} -> {pr['qpb_after']:.3f}s of playback)")

with open(os.path.join(OUT, "recompute_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
