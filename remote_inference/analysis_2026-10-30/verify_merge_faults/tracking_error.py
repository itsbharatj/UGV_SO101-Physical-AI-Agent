"""Tracking error |cmd - latest obs| per tick (joints 0-4), per run: how often would a goal-present cap of 5/8 deg clip?"""
import csv, os, sys
import numpy as np
SCRATCH=sys.argv[1]
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for run in ["run3_frames","run5_frames","run6_frames","pi05_run2_frames"]:
    p=os.path.join(SCRATCH,run,"joints.csv")
    if not os.path.exists(p): continue
    rows=[(float(r["t"]),r["kind"],np.array([float(r[j]) for j in J])) for r in csv.DictReader(open(p))]
    ct=np.array([t for t,k,_ in rows if k=="cmd"]); cv=np.array([v for t,k,v in rows if k=="cmd"])
    ot=np.array([t for t,k,_ in rows if k=="obs"]); ov=np.array([v for t,k,v in rows if k=="obs"])
    idx=np.searchsorted(ot,ct,side="right")-1; ok=idx>=0
    err=np.abs(cv[ok,:5]-ov[idx[ok],:5]); m=err.max(axis=1)
    # only consider ticks where the arm is actually moving: obs changed >0.5 deg over the last 0.5 s
    print(run,"n_cmd",len(ct),"tracking |cmd-obs| any-joint max per tick: median %.2f p90 %.2f p99 %.2f max %.2f; frac ticks >5: %.3f  >8: %.3f  >10: %.3f"%(np.median(m),np.percentile(m,90),np.percentile(m,99),m.max(),np.mean(m>5),np.mean(m>8),np.mean(m>10)))
    print("   per-joint p90:",np.round(np.percentile(err,90,axis=0),2),"p99:",np.round(np.percentile(err,99,axis=0),2))
