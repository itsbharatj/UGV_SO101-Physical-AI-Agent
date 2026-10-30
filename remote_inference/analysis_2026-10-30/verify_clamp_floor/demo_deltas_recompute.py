"""Independent recomputation of per-step |delta action| in the training demos (1-step = 33 ms at 30 fps, 2-step = 66 ms)."""
import glob, os
import numpy as np, pandas as pd
DS = os.path.expanduser('~/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000')
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
d1, d2, ams, n_ep, frames, dts = [], [], [], 0, 0, []
for f in sorted(glob.glob(f'{DS}/episode_*.parquet')):
    ep = int(os.path.basename(f)[8:14])
    if ep == 20: continue
    df = pd.read_parquet(f, columns=['action', 'observation.state', 'timestamp', 'frame_index'])
    df = df.sort_values('frame_index')
    a = np.stack(df['action'].to_numpy()).astype(float); s = np.stack(df['observation.state'].to_numpy()).astype(float)
    ts = df['timestamp'].to_numpy().astype(float); dts.append(np.diff(ts))
    d1.append(np.diff(a, axis=0)); d2.append(a[2:] - a[:-2]); ams.append(a - s); n_ep += 1; frames += len(a)
d1 = np.abs(np.vstack(d1)); d2 = np.abs(np.vstack(d2)); ams = np.abs(np.vstack(ams)); dts = np.concatenate(dts)
print(f"episodes {n_ep}, frames {frames}, 1-step rows {len(d1)}, 2-step rows {len(d2)}; timestamp dt median {np.median(dts):.4f} s, max {dts.max():.4f} s")
rows = []
for label, X in [("1-step (33 ms)", d1), ("2-step (66 ms)", d2), ("|action-state| same frame", ams)]:
    print(f"\n{label}:")
    print(f"  {'joint':14s} {'p50':>6s} {'p90':>6s} {'p99':>6s} {'p99.9':>6s} {'max':>6s} | frac>3 frac>4 frac>5 frac>8")
    for j, jn in enumerate(J):
        v = X[:, j]; p = np.percentile(v, [50, 90, 99, 99.9, 100])
        fr = [(v > t).mean() for t in (3, 4, 5, 8)]
        print(f"  {jn:14s} {p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f} {p[3]:6.2f} {p[4]:6.2f} | {fr[0]:6.3%} {fr[1]:6.3%} {fr[2]:6.3%} {fr[3]:6.3%}")
        rows.append(dict(series=label, joint=jn, p50=p[0], p90=p[1], p99=p[2], p99_9=p[3], max=p[4], frac_gt3=fr[0], frac_gt4=fr[1], frac_gt5=fr[2], frac_gt8=fr[3]))
pd.DataFrame(rows).to_csv("demo_deltas_recompute.csv", index=False)
