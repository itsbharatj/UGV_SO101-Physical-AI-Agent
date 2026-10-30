import sys, numpy as np, pandas as pd
S="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
runs={"pi05_run1":"pi05_run1_frames","pi05_run2":"pi05_run2_frames","groot_run4":"run4_frames","groot_run5":"run5_frames","groot_run6":"run6_frames"}
rows=[]; jump_rows=[]
for name,d in runs.items():
    df=pd.read_csv(f"{S}/{d}/joints.csv")
    cmd=df[df.kind=="cmd"].reset_index(drop=True); obs=df[df.kind=="obs"].reset_index(drop=True)
    # preceding obs for each cmd (same tick: obs read before send)
    idx=np.searchsorted(obs.t.values, cmd.t.values, side="right")-1
    ok=idx>=0
    gap=cmd[J].values-obs[J].values[np.clip(idx,0,None)]
    gap[~ok]=np.nan
    obs_age=cmd.t.values-obs.t.values[np.clip(idx,0,None)]
    absgap=np.abs(gap); maxgap=np.nanmax(absgap,axis=1)
    # cmd-to-cmd deltas
    dc=np.diff(cmd[J].values,axis=0); dt=np.diff(cmd.t.values)
    maxd=np.abs(dc).max(axis=1)
    one_tick=dt<=0.1
    j10=(maxd>10)&one_tick; j15=(maxd>15)&one_tick; j5=(maxd>5)&one_tick
    j10_any=(maxd>10); j15_any=(maxd>15)
    # for jump ticks, the gap of the NEW command vs present (index i+1)
    g_at_j10=maxgap[1:][j10]
    clipped10=np.sum(g_at_j10>10); clipped5=np.sum(g_at_j10>5)
    # per-joint: does the jumping joint itself exceed 10 vs present?
    jj=np.abs(dc).argmax(axis=1)
    gj=np.array([absgap[i+1,jj[i]] for i in range(len(dc))])
    jointclipped10=np.sum(gj[j10]>10)
    # obs-to-obs per-tick movement (physical evidence of no cap)
    do=np.diff(obs[J].values,axis=0); dto=np.diff(obs.t.values); maxo=np.abs(do).max(axis=1); oo=dto<=0.1
    rows.append(dict(run=name,n_cmd=len(cmd),n_obs=len(obs),dur_s=round(cmd.t.max()-cmd.t.min(),2),
        cmd_dt_median_ms=round(np.median(dt)*1000,1),
        jumps_gt5=int(j5.sum()),jumps_gt10=int(j10.sum()),jumps_gt15=int(j15.sum()),jumps_gt10_anydt=int(j10_any.sum()),jumps_gt15_anydt=int(j15_any.sum()),
        max_cmd_jump=round(maxd.max(),2),
        jumps_gt10_with_gap_gt10=int(clipped10),jumps_gt10_with_gap_gt5=int(clipped5),jumps_gt10_jointgap_gt10=int(jointclipped10),
        min_gap_at_jump10=round(float(np.min(g_at_j10)) if len(g_at_j10) else float('nan'),2),
        ticks_gap_gt5=int(np.nansum(maxgap>5)),ticks_gap_gt10=int(np.nansum(maxgap>10)),ticks_gap_gt15=int(np.nansum(maxgap>15)),
        frac_gap_gt5=round(float(np.nanmean(maxgap>5)),3),frac_gap_gt10=round(float(np.nanmean(maxgap>10)),3),
        gap_p50=round(float(np.nanpercentile(maxgap,50)),2),gap_p90=round(float(np.nanpercentile(maxgap,90)),2),gap_p99=round(float(np.nanpercentile(maxgap,99)),2),gap_max=round(float(np.nanmax(maxgap)),2),
        obs_age_median_ms=round(float(np.nanmedian(obs_age))*1000,1),obs_age_max_ms=round(float(np.nanmax(obs_age))*1000,1),
        obs_step_gt10=int(np.sum((maxo>10)&oo)),obs_step_gt15=int(np.sum((maxo>15)&oo)),obs_step_max=round(float(maxo[oo].max()),2),
        obs_speed_max_degps=round(float((maxo[oo]/dto[oo]).max()),0)))
    for i in np.where(j10)[0]:
        r=dict(run=name,t=round(cmd.t[i+1],3),dt_ms=round(dt[i]*1000,1),joint=J[jj[i]],cmd_jump=round(abs(dc[i,jj[i]]),2),
               gap_joint_vs_present=round(gj[i],2),max_gap_any_joint=round(maxgap[i+1],2),
               prev_cmd=round(cmd[J[jj[i]]][i],2),new_cmd=round(cmd[J[jj[i]]][i+1],2),present=round(obs[J[jj[i]]].values[idx[i+1]],2))
        jump_rows.append(r)
out=pd.DataFrame(rows); out.to_csv(f"{S}/wf_out/verify_motion_cap/run_jump_gap_summary.csv",index=False)
pd.DataFrame(jump_rows).to_csv(f"{S}/wf_out/verify_motion_cap/jumps_gt10_with_gaps.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",60)
print(out.T.to_string())
jr=pd.DataFrame(jump_rows)
print("\n--- pi05_run2 jumps >10 deg (cmd-to-cmd, one tick) with goal-vs-present gap ---")
print(jr[jr.run=="pi05_run2"].to_string(index=False))
print("\n--- other runs' jumps >10 ---")
print(jr[jr.run!="pi05_run2"].to_string(index=False))
