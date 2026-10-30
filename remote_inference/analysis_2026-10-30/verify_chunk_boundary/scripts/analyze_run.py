import ast, sys, csv, math
import numpy as np

frames_dir = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
window = float(sys.argv[3]) if len(sys.argv) > 3 else 0.065
J = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]

# ---- events
events = []
with open(f"{frames_dir}/events.txt") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line: continue
        t, name, rest = line.split(" ", 2)
        try:
            d = ast.literal_eval(rest)
        except Exception as e:
            d = {"_raw": rest}
        events.append((float(t), name, d))

results = [(t, d) for t, n, d in events if n == "result"]
acc = [(t, d) for t, d in results if d.get("accepted")]
acc_merge = [(t, d) for t, d in acc if (d.get("merge") or {}).get("accepted")]
print(f"events: {len(events)}  requests: {sum(1 for _,n,_ in events if n=='request')}  results: {len(results)}  accepted: {len(acc)}  merge-accepted: {len(acc_merge)}")
rej = [(t, d) for t, d in results if not d.get("accepted")]
for t, d in rej: print("  rejected result:", t, {k: d[k] for k in d if k in ('reason','merge','turnaround_s')})
acc_t = np.array([t for t, _ in acc])
sp = np.diff(acc_t)
print(f"accepted-result spacing: n={len(sp)} median={np.median(sp):.3f} mean={sp.mean():.3f} min={sp.min():.3f} max={sp.max():.3f}")
ta = np.array([d['turnaround_s'] for _, d in acc])
print(f"turnaround: median={np.median(ta):.3f} p90={np.percentile(ta,90):.3f} max={ta.max():.3f}")
usable = np.array([d['merge']['usable_actions'] for _, d in acc])
overlap = np.array([d['merge']['overlap_steps'] for _, d in acc])
trimmed = np.array([d['merge']['trimmed_actions'] for _, d in acc])
blended = np.array([d['merge']['blended_steps'] for _, d in acc])
print(f"usable_actions: median={np.median(usable):.0f} min={usable.min()} max={usable.max()}; overlap_steps median={np.median(overlap):.0f} min={overlap.min()} max={overlap.max()}; trimmed median={np.median(trimmed):.0f}; blended sum={blended.sum()}")
# scheduling event
for t, n, d in events:
    if n == 'scheduling': print("scheduling:", d)

# ---- joints
rows = []
with open(f"{frames_dir}/joints.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append((float(row['t']), row['kind'], np.array([float(row[j]) for j in J])))
cmd = [(t, v) for t, k, v in rows if k == 'cmd']
obs = [(t, v) for t, k, v in rows if k == 'obs']
print(f"rows: {len(rows)} cmd: {len(cmd)} obs: {len(obs)}")
ct = np.array([t for t, _ in cmd]); cv = np.array([v for _, v in cmd])
t0 = ct[0]
print(f"first cmd t={t0:.3f}, last cmd t={ct[-1]:.3f}, duration {ct[-1]-t0:.1f}s; first request t={[t for t,n,_ in events if n=='request'][0]:.3f}; first accepted result t={acc_t[0]:.3f}")
dt = np.diff(ct)
print(f"cmd tick interval: median={np.median(dt)*1000:.1f} ms  p90={np.percentile(dt,90)*1000:.1f} ms  max={dt.max()*1000:.1f} ms; n(dt>0.1s)={int((dt>0.1).sum())}")
dv = np.abs(np.diff(cv, axis=0))           # per joint
step = dv.max(axis=1)                       # max over joints per tick
nan_mask = np.isnan(step)
print(f"NaN steps: {int(nan_mask.sum())}")
print(f"per-tick step (max over joints): median={np.nanmedian(step):.2f} deg, p90={np.nanpercentile(step,90):.2f}, p99={np.nanpercentile(step,99):.2f}, max={np.nanmax(step):.2f}")
jumps = np.where(step > thresh)[0]          # index i means cmd[i] -> cmd[i+1]
print(f"jumps >{thresh} deg: {len(jumps)}")
# per joint count of > thresh
for jn, col in zip(J, dv.T):
    print(f"   {jn}: n(>{thresh})={int((col>thresh).sum())} max={np.nanmax(col):.1f}")
jump_sizes = step[jumps]
if len(jumps):
    print(f"jump size: median={np.median(jump_sizes):.1f} max={jump_sizes.max():.1f} min={jump_sizes.min():.1f}")
# lag from preceding accepted result to the time of the second cmd of the jump
def lag_to_prev(t, times):
    i = np.searchsorted(times, t, side='right') - 1
    return (t - times[i]) if i >= 0 else np.inf
lags = np.array([lag_to_prev(ct[i+1], acc_t) for i in jumps])
dts = dt[jumps]
print(f"lag of jump cmd (second cmd) after preceding accepted result: median={np.median(lags):.3f} p90={np.percentile(lags,90):.3f} max={lags.max():.3f}; n(lag<={window})={int((lags<=window).sum())}/{len(lags)}")
print(f"tick interval at jumps: median={np.median(dts)*1000:.1f} ms max={dts.max()*1000:.1f} ms")
# also lag with respect to the first cmd of the pair
lags1 = np.array([lag_to_prev(ct[i], acc_t) for i in jumps])
print(f"lag of first cmd of the pair after preceding accepted result: median={np.median(lags1):.3f}, n(first cmd BEFORE result, i.e. result between the two cmds)={int(((lags1>lags) | np.isinf(lags1)).sum())}")
# chance baseline: fraction of all cmd ticks (second cmd) within window of a preceding accepted result
all_lags = np.array([lag_to_prev(ct[i+1], acc_t) for i in range(len(ct)-1)])
print(f"chance baseline: fraction of all ticks within {window}s after an accepted result = {(all_lags<=window).mean():.3f} ({int((all_lags<=window).sum())}/{len(all_lags)})")
# how many accepted results are followed by a jump within window
hits = 0
for t in acc_t:
    i = np.searchsorted(ct, t, side='left')  # first cmd at or after result
    if i < len(ct) and i >= 1 and ct[i]-t <= window and step[i-1] > thresh: hits += 1
print(f"accepted results followed (within {window}s) by a >{thresh} deg jump: {hits}/{len(acc_t)}")
# results NOT followed by a jump: how big is the first step after them
first_steps = []
for t in acc_t:
    i = np.searchsorted(ct, t, side='left')
    if 1 <= i < len(ct) and ct[i]-t <= window: first_steps.append(step[i-1])
first_steps = np.array(first_steps)
print(f"first step after each accepted result (n={len(first_steps)}): median={np.median(first_steps):.1f} deg, n>5={int((first_steps>5).sum())}, n>15={int((first_steps>15).sum())}")
# steps not at a boundary
mask_b = all_lags <= window
print(f"steps NOT within {window}s after a result: n={int((~mask_b).sum())} median={np.nanmedian(step[~mask_b]):.2f} max={np.nanmax(step[~mask_b]):.2f} n>{thresh}={int((step[~mask_b]>thresh).sum())} n>5={int((step[~mask_b]>5).sum())}")
print(f"steps within {window}s after a result: n={int(mask_b.sum())} median={np.nanmedian(step[mask_b]):.2f}")
# list jumps
print("jumps (t_rel to first cmd, lag after result, step, per-joint deltas):")
for i, l, s in zip(jumps, lags, jump_sizes):
    print(f"  +{ct[i+1]-t0:6.2f}s lag={l:.3f}s dt={dt[i]*1000:.0f}ms step={s:6.1f}  d=" + " ".join(f"{x:6.1f}" for x in np.diff(cv[i:i+2],axis=0)[0]))
# ---- range check vs training range (passed as file if given)
import os
rng_file = os.environ.get("RANGE_FILE")
if rng_file:
    rng = np.loadtxt(rng_file, delimiter=',', skiprows=1)  # rows: joint, cols: min,max
    lo, hi = rng[:,0], rng[:,1]
    out = (cv < lo) | (cv > hi)
    print("out-of-training-range commands:")
    for k, jn in enumerate(J):
        print(f"   {jn}: {out[:,k].mean()*100:.2f}% ({int(out[:,k].sum())}/{len(cv)})  cmd min={np.nanmin(cv[:,k]):.2f} max={np.nanmax(cv[:,k]):.2f}  train [{lo[k]:.2f},{hi[k]:.2f}]")
    print(f"   any joint: {out.any(axis=1).mean()*100:.2f}%")
