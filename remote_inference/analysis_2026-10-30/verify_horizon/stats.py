import ast, re, itertools
import numpy as np
S = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
paths = {"run5": f"{S}/run5_frames/events.txt", "run6": f"{S}/run6_frames/events.txt", "pi05_run2": f"{S}/pi05_run2_frames/events.txt", "molmo_shadow1": f"{S}/molmo_shadow1/events.txt"}
def parse(path):
    ev = []
    for line in open(path):
        m = re.match(r"(\S+) (\S+) (.*)$", line.rstrip("\n"))
        t, name, rest = m.groups()
        try: d = ast.literal_eval(rest)
        except Exception: d = {"raw": rest}
        ev.append((float(t), name, d))
    return ev
data = {}
for run, p in paths.items():
    ev = parse(p)
    H = [d["execution_horizon_s"] for t, n, d in ev if n == "scheduling"][0]
    reqs = {d["request_id"]: (t, d) for t, n, d in ev if n == "request"}
    res = [(t, d) for t, n, d in ev if n == "result"]
    taus = np.array([d["turnaround_s"] for t, d in res])
    margins = np.array([np.nan if d["timing_margin_s"] is None else d["timing_margin_s"] for t, d in res])
    pb = np.array([reqs[d["request_id"]][1]["playback_at_submission_s"] for t, d in res])
    pol = np.array([d["server_durations"]["policy"] for t, d in res])
    worker = np.array([d["server_durations"].get("worker", np.nan) for t, d in res])
    gaps = np.array([reqs[res[i+1][1]["request_id"]][0] - res[i][0] for i in range(len(res)-1)])  # next request submit - this result
    data[run] = dict(H=H, taus=taus, margins=margins, pb=pb, pol=pol, worker=worker, gaps=gaps, n_req=len(reqs), n_res=len(res),
                     waits=sum(1 for t,n,d in ev if n=="waiting"), resumed=[d for t,n,d in ev if n=="resumed"], cancelled=[d for t,n,d in ev if n=="request_cancelled"], invalidated=[d for t,n,d in ev if n=="invalidated"])
    print(f"{run}: H={H:.3f} requests={len(reqs)} results={len(res)} waiting={data[run]['waits']} resumed={data[run]['resumed']} cancelled={[round(c['wait_s'],3) for c in data[run]['cancelled']]} invalidated={[round(c['wait_s'],3) for c in data[run]['invalidated']]}")
    print(f"   first(cold) tau={taus[0]:.3f}; policy duration median {np.median(pol):.3f} max {pol.max():.3f}; worker median {np.nanmedian(worker):.3f}")
    print(f"   gap result->next request: median {np.median(gaps)*1000:.0f} ms, max {gaps.max()*1000:.0f} ms (n={len(gaps)})")
    # check the continuity identity: pb_sub_i ≈ H - trim_{i-1}*dt, and margin_i = pb_sub_i - tau_i
    print(f"   min timing margin {np.nanmin(margins):.3f} s; margins<0: {int(np.nansum(margins<0))}")

def pstats(x, label):
    q = lambda p: np.percentile(x, p)
    print(f"{label}: n={len(x)} median={np.median(x):.3f} mean={x.mean():.3f} p90={q(90):.3f} p95={q(95):.3f} max={x.max():.3f}  (nearest: p90={np.percentile(x,90,method='nearest'):.3f} p95={np.percentile(x,95,method='nearest'):.3f})")

def pair_probs(x, Hs, label):
    x = np.asarray(x)
    sums = x[:, None] + x[None, :]
    n = len(x)
    iu = np.triu_indices(n, k=1)
    out = []
    for H in Hs:
        p_all = (sums > H).mean()
        p_distinct = (sums[iu] > H).mean()
        out.append((H, p_all, p_distinct))
    print(f"{label}: " + "  ".join(f"P(sum>{H:.3f})={100*p_all:.1f}% (distinct {100*p_d:.1f}%)" for H, p_all, p_d in out))

def consec_probs(runs, skip, Hs, label):
    tot = {H: 0 for H in Hs}; cnt = 0
    for run in runs:
        t = data[run]["taus"][skip:]
        for a, b in zip(t[:-1], t[1:]):
            cnt += 1
            for H in Hs:
                tot[H] += (a + b > H)
    print(f"{label}: consecutive pairs n={cnt}: " + "  ".join(f"P(sum>{H:.3f})={100*tot[H]/cnt:.1f}% ({tot[H]})" for H in Hs))

Hs = [2.0, 2.6667, 3.0, 3.3333, 4.0]
pooled_runs = ["run5", "run6", "pi05_run2"]
print("\n--- pooled 10-29 steady-state turnarounds, variants ---")
for skip, name in [(1, "skip first (cold) per run"), (2, "skip first two per run"), (0, "all results incl. cold")]:
    x = np.concatenate([data[r]["taus"][skip:] for r in pooled_runs])
    pstats(x, name)
    pair_probs(x, Hs, "   independent pairs")
    consec_probs(pooled_runs, skip, Hs, "   empirical")
# variant: skip 1 per run but drop pi05_run2 result 1 (2.957) -> see if that is the investigator's 112
x = np.concatenate([data["run5"]["taus"][1:], data["run6"]["taus"][1:], np.delete(data["pi05_run2"]["taus"], [0, 1])])
pstats(x, "skip cold + drop pi05_run2 #1 (2.957 s) only")
print("\n--- GR00T-only (run5+run6, skip cold) and Pi0.5-only ---")
pstats(np.concatenate([data["run5"]["taus"][1:], data["run6"]["taus"][1:]]), "GR00T runs")
pstats(data["pi05_run2"]["taus"][1:], "pi05_run2 (skip cold)")
pstats(data["pi05_run2"]["taus"][3:], "pi05_run2 after the initial stall (results 3+)")
print("\n--- MolmoAct2 shadow run 2026-10-30 (H=2.000 s, 15 Hz) ---")
x = data["molmo_shadow1"]["taus"][1:]
pstats(x, "molmo_shadow1 skip cold")
pstats(data["molmo_shadow1"]["taus"], "molmo_shadow1 all")
pair_probs(x, Hs, "   independent pairs")
consec_probs(["molmo_shadow1"], 1, Hs, "   empirical")
m = data["molmo_shadow1"]["margins"]
print(f"   timing margins: min {np.nanmin(m):.3f} median {np.nanmedian(m):.3f}; pb_sub median {np.median(data['molmo_shadow1']['pb'][1:]):.3f}")
print(f"   policy duration: median {np.median(data['molmo_shadow1']['pol']):.3f} p90 {np.percentile(data['molmo_shadow1']['pol'],90):.3f} max {data['molmo_shadow1']['pol'].max():.3f}; worker median {np.nanmedian(data['molmo_shadow1']['worker']):.3f}")
# what the 10-29 pooled distribution predicts for the horizons actually run on 10-29, vs observed
print("\n--- method check: predicted vs observed starvations on 10-29 ---")
x = np.concatenate([data[r]["taus"][1:] for r in pooled_runs])
sums = x[:, None] + x[None, :]
for run in pooled_runs:
    H = data[run]["H"]; n_spl = data[run]["n_req"] - 1
    print(f"{run}: H={H:.3f}, splices={n_spl}, observed starvations={data[run]['waits']}, predicted (indep pairs) {100*(sums>H).mean():.1f}% -> {n_spl*(sums>H).mean():.1f} expected")
# consecutive-lag correlation
print("\n--- lag-1 correlation of consecutive turnarounds (10-29 pooled, skip cold) ---")
a = np.concatenate([data[r]["taus"][1:-1] for r in pooled_runs]); b = np.concatenate([data[r]["taus"][2:] for r in pooled_runs])
print(f"corr={np.corrcoef(a,b)[0,1]:.3f} n={len(a)}")
np.savetxt(f"{S}/wf_out/verify_horizon/pooled_1029_taus_skipcold.csv", np.concatenate([data[r]["taus"][1:] for r in pooled_runs]), fmt="%.4f", header="turnaround_s (run5,run6,pi05_run2; first result of each run excluded)")
np.savetxt(f"{S}/wf_out/verify_horizon/molmo_shadow1_taus.csv", data["molmo_shadow1"]["taus"], fmt="%.4f", header="turnaround_s molmo_shadow1 all results (first is cold)")
