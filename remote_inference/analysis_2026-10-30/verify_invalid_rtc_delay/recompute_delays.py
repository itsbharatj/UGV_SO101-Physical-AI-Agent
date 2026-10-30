"""Recompute request delays from recorded engine events (read-only analysis)."""
import ast, csv, math, sys, os
SCRATCH = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
RUNS = {
    "groot_run5": (f"{SCRATCH}/run5_frames/events.txt", 40),
    "groot_run6": (f"{SCRATCH}/run6_frames/events.txt", 40),
    "pi05_run1": (f"{SCRATCH}/pi05_run1_frames/events.txt", 50),
    "pi05_run2": (f"{SCRATCH}/pi05_run2_frames/events.txt", 50),
    "molmo_shadow1": (f"{SCRATCH}/molmo_shadow1/events.txt", 30),
}
FPS = 15.0
INTERVAL = 1.0 / FPS
rows = []
summary = []
for run, (path, pred_steps) in RUNS.items():
    if not os.path.exists(path):
        summary.append((run, "missing events file"))
        continue
    events = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            t, name, rest = line.split(" ", 2)
            try:
                body = ast.literal_eval(rest)
            except Exception:
                body = {}
            events.append((float(t), name, body))
    horizon = None
    for _, name, body in events:
        if name == "scheduling":
            horizon = body.get("execution_horizon_s")
    n_req = 0
    cold = None
    max_steady_accepted = 0.0
    max_over_playback = -1e9
    n_uncapped_invalid = 0
    n_capped_mismatch = 0
    n_uncapped_mismatch = 0
    pending = {}
    accepted_turnarounds = []
    for t, name, body in events:
        if name == "request":
            n_req += 1
            delay = body["delay"]
            rt = body["recent_turnaround_s"]
            pb = body["playback_at_submission_s"]
            available = int(round(pb / INTERVAL))
            uncapped = math.ceil(rt / INTERVAL) if rt else 0
            uncapped = uncapped if available else 0
            capped = min(uncapped, available)
            if delay != capped:
                n_capped_mismatch += 1
            if delay != uncapped:
                n_uncapped_mismatch += 1
            if uncapped > pred_steps:
                n_uncapped_invalid += 1
            pending[body["request_id"]] = (t, pb, n_req)
            rows.append(dict(run=run, t=t, req_no=n_req, request_id=body["request_id"][:8], delay_sent=delay,
                             recent_turnaround_s=round(rt, 3), playback_at_submission_s=round(pb, 3),
                             available_steps=available, uncapped_delay=uncapped, capped_delay=capped,
                             prediction_steps=pred_steps, uncapped_would_be_rejected=uncapped > pred_steps))
        elif name == "result":
            rid = body["request_id"]
            ta = body["turnaround_s"]
            if body.get("accepted"):
                accepted_turnarounds.append(ta)
                if rid in pending:
                    t0, pb, no = pending[rid]
                    if no == 1:
                        cold = ta
                    else:
                        max_steady_accepted = max(max_steady_accepted, ta)
                        if pb > 0:
                            max_over_playback = max(max_over_playback, ta - pb)
    summary.append((run, dict(prediction_steps=pred_steps, horizon_s=horizon, requests=n_req,
                               cold_turnaround_s=None if cold is None else round(cold, 3),
                               cold_uncapped_delay=None if cold is None else math.ceil(cold / INTERVAL),
                               cold_exceeds_horizon=None if cold is None else cold > pred_steps * INTERVAL,
                               max_steady_accepted_turnaround_s=round(max_steady_accepted, 3),
                               max_accepted_minus_playback_s=None if max_over_playback < -1e8 else round(max_over_playback, 3),
                               delay_matches_capped_rule=n_capped_mismatch == 0,
                               delay_matches_uncapped_rule=n_uncapped_mismatch == 0,
                               n_requests_uncapped_delay_gt_prediction_steps=n_uncapped_invalid)))
out = sys.argv[1]
with open(os.path.join(out, "request_delays.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(out, "summary.txt"), "w") as f:
    for run, s in summary:
        f.write(f"{run}: {s}\n")
        print(run, s)
