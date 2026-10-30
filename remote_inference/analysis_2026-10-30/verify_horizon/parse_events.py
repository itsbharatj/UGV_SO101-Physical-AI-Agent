import ast, re, sys, json
import numpy as np
S = "/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
runs = {"run5": f"{S}/run5_frames/events.txt", "run6": f"{S}/run6_frames/events.txt", "pi05_run2": f"{S}/pi05_run2_frames/events.txt", "pi05_run1": f"{S}/pi05_run1_frames/events.txt"}
def parse(path):
    ev = []
    for line in open(path):
        m = re.match(r"(\S+) (\S+) (.*)$", line.rstrip("\n"))
        if not m: continue
        t, name, rest = m.groups()
        try: d = ast.literal_eval(rest)
        except Exception: d = {"raw": rest}
        ev.append((float(t), name, d))
    return ev
out = {}
for run, path in runs.items():
    ev = parse(path)
    H = [d["execution_horizon_s"] for t, n, d in ev if n == "scheduling"][0]
    reqs = {d["request_id"]: (t, d) for t, n, d in ev if n == "request"}
    res = [(t, d) for t, n, d in ev if n == "result"]
    print(f"\n=== {run}: horizon {H:.4f} s, {len(reqs)} requests, {len(res)} results")
    rows = []
    for i, (t, d) in enumerate(res):
        rq_t, rq = reqs[d["request_id"]]
        rows.append(dict(i=i, t_req=rq_t, t_res=t, tau=d["turnaround_s"], margin=d["timing_margin_s"], pb_sub=rq["playback_at_submission_s"], qpb=d["queue_playback_s"], acc=d["accepted"], pol=d.get("server_durations", {}).get("policy"), trim=d.get("merge", {}).get("trimmed_actions"), usable=d.get("merge", {}).get("usable_actions"), age=rq["observation_age_at_submission_s"], gen=None))
    for r in rows:
        print(f"{r['i']:3d} req {r['t_req']:.3f} res {r['t_res']:.3f} tau {r['tau']:.3f} pb_sub {r['pb_sub']:.3f} margin {str(None) if r['margin'] is None else format(r['margin'], '.3f'):>6} qpb_after {r['qpb']:.3f} trim {r['trim']} usable {r['usable']} policy {r['pol']:.3f} acc {r['acc']}")
    out[run] = dict(H=H, rows=rows)
json.dump(out, open(f"{S}/wf_out/verify_horizon/parsed_events.json", "w"), indent=1)
