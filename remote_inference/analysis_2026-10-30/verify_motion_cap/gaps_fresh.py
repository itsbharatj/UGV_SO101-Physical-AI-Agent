import numpy as np, pandas as pd
S="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
runs={"pi05_run1":"pi05_run1_frames","pi05_run2":"pi05_run2_frames","groot_run4":"run4_frames","groot_run5":"run5_frames","groot_run6":"run6_frames"}
rows=[]
for name,d in runs.items():
    df=pd.read_csv(f"{S}/{d}/joints.csv")
    cmd=df[df.kind=="cmd"].reset_index(drop=True); obs=df[df.kind=="obs"].reset_index(drop=True)
    idx=np.searchsorted(obs.t.values, cmd.t.values, side="right")-1
    ok=idx>=0; idx=np.clip(idx,0,None)
    age=cmd.t.values-obs.t.values[idx]
    fresh=ok&(age<=0.05)   # obs read in the same 66 ms tick (policy loop), excludes the 50 Hz reset interpolation
    gap=np.abs(cmd[J].values-obs[J].values[idx])[fresh]
    mg=gap.max(axis=1)
    arm=gap[:,:5].max(axis=1)  # joints in degrees only
    r=dict(run=name,policy_ticks=int(fresh.sum()),excluded_cmds=int((~fresh).sum()),
           ticks_gap_gt5=int((mg>5).sum()),ticks_gap_gt10=int((mg>10).sum()),ticks_gap_gt15=int((mg>15).sum()),
           frac_gt5=round((mg>5).mean(),3),frac_gt10=round((mg>10).mean(),3),frac_gt15=round((mg>15).mean(),3),
           arm_frac_gt5=round((arm>5).mean(),3),arm_frac_gt10=round((arm>10).mean(),3),
           gap_p50=round(np.percentile(mg,50),2),gap_p90=round(np.percentile(mg,90),2),gap_p99=round(np.percentile(mg,99),2),gap_max=round(mg.max(),2))
    for k,j in enumerate(J):
        r[f"{j}_gt5"]=int((gap[:,k]>5).sum()); r[f"{j}_gt10"]=int((gap[:,k]>10).sum()); r[f"{j}_p99"]=round(np.percentile(gap[:,k],99),1); r[f"{j}_max"]=round(gap[:,k].max(),1)
    rows.append(r)
out=pd.DataFrame(rows); out.to_csv(f"{S}/wf_out/verify_motion_cap/policy_tick_gap_summary.csv",index=False)
pd.set_option("display.width",250); print(out.T.to_string())
# speed check: obs-to-obs per-tick movement in policy loop vs training per-step delta
print("\n--- obs-to-obs per-tick movement (deg per 66 ms tick) in policy-loop ticks ---")
for name,d in runs.items():
    df=pd.read_csv(f"{S}/{d}/joints.csv"); obs=df[df.kind=="obs"].reset_index(drop=True)
    do=np.abs(np.diff(obs[J[:5]].values,axis=0)).max(axis=1); dt=np.diff(obs.t.values); m=dt<=0.1
    print(f"{name}: n={m.sum()} p50={np.percentile(do[m],50):.2f} p90={np.percentile(do[m],90):.2f} p99={np.percentile(do[m],99):.2f} max={do[m].max():.2f}  (training per-step |delta action| any arm joint p50/p90/p99: see train_stats)")
