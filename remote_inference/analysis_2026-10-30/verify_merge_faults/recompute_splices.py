"""Independent recomputation of splice/tick statistics from joints.csv + events.txt (read-only)."""
import ast, csv, os, sys
import numpy as np

SCRATCH = sys.argv[1]; OUT = sys.argv[2]
J = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
RUNS = ["run3_frames","run4_frames","run5_frames","run6_frames","pi05_run1_frames","pi05_run2_frames"]

def events(path):
    ev=[]
    for line in open(path):
        line=line.rstrip("\n")
        if not line: continue
        t,rest=line.split(" ",1)
        name,_,d=rest.partition(" ")
        try: d=ast.literal_eval(d) if d else {}
        except Exception: d={}
        ev.append((float(t),name,d))
    return ev

def joints(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]),r["kind"],np.array([float(r[j]) for j in J])))
    return rows

out_rows=[]; summary=[]
for run in RUNS:
    d=os.path.join(SCRATCH,run)
    if not (os.path.exists(os.path.join(d,"joints.csv")) and os.path.exists(os.path.join(d,"events.txt"))): continue
    ev=events(os.path.join(d,"events.txt")); rows=joints(os.path.join(d,"joints.csv"))
    t0=ev[0][0]
    acc=[e for e in ev if e[1]=="result" and e[2].get("accepted")]
    res=[e for e in ev if e[1]=="resumed"]
    cmds=[(t,v) for t,k,v in rows if k=="cmd"]; obs=[(t,v) for t,k,v in rows if k=="obs"]
    ct=np.array([t for t,_ in cmds]); cv=np.array([v for _,v in cmds])
    ot=np.array([t for t,_ in obs]); ov=np.array([v for _,v in obs])
    # tick-to-tick command deltas
    dt=np.diff(ct); dv=np.abs(np.diff(cv,axis=0))
    jmax5=dv[:,:5].max(axis=1); jmax6=dv.max(axis=1)
    # splice ticks: index of first cmd strictly after each accept time
    splice_idx=set(); splice_info=[]
    for ta,_,e in acc:
        i=np.searchsorted(ct,ta,side="right")
        if 0<i<len(ct):
            splice_idx.add(i)
            splice_info.append((i,ta,e))
    resume_idx=set()
    for tr,_,e in res:
        i=np.searchsorted(ct,tr,side="right")
        if 0<i<len(ct): resume_idx.add(i)
    # per transition i-1 -> i  (delta index i-1)
    is_splice=np.zeros(len(dv),bool); is_resume=np.zeros(len(dv),bool)
    for i in splice_idx: is_splice[i-1]=True
    for i in resume_idx: is_resume[i-1]=True
    valid=dt<=0.15
    big=(jmax5>15)&valid
    big_idx=np.where(big)[0]
    # neighbours: within +-1 tick of a splice
    near=np.zeros(len(dv),bool)
    for i in np.where(is_splice|is_resume)[0]:
        near[max(0,i-1):i+2]=True
    n_big=int(big.sum()); n_big_exact=int((big&(is_splice|is_resume)).sum()); n_big_near=int((big&near).sum())
    sp=jmax5[is_splice&valid]; nonsp=jmax5[valid&~is_splice&~is_resume]
    # include-gripper variants
    big6=(jmax6>15)&valid
    # physical: obs tick deltas
    odt=np.diff(ot); odv=np.abs(np.diff(ov,axis=0)); omax5=odv[:,:5].max(axis=1)
    ovalid=odt<=0.15
    # at each splice: new cmd vs last obs before it, old cmd vs last obs; and obs movement in next 0.5 s
    phys=[]
    for i,ta,e in splice_info:
        tnew=ct[i]; k=np.searchsorted(ot,tnew,side="right")-1
        if k<0: continue
        last_obs=ov[k]
        new_vs_obs=np.abs(cv[i,:5]-last_obs[:5]).max(); old_vs_obs=np.abs(cv[i-1,:5]-last_obs[:5]).max()
        # obs after 0.2s and 0.5s
        k2=np.searchsorted(ot,tnew+0.2,side="right")-1; k5=np.searchsorted(ot,tnew+0.5,side="right")-1
        moved02=np.abs(ov[k2,:5]-last_obs[:5]).max() if k2>k else np.nan
        moved05=np.abs(ov[k5,:5]-last_obs[:5]).max() if k5>k else np.nan
        # which joint had the max cmd jump
        jj=int(np.argmax(np.abs(cv[i,:5]-cv[i-1,:5])))
        phys.append(dict(run=run,t_rel=round(ta-t0,3),cmd_jump=round(float(jmax5[i-1]),2),joint=J[jj],
                         dt_cmd=round(float(dt[i-1]),3),new_cmd_vs_obs=round(float(new_vs_obs),2),old_cmd_vs_obs=round(float(old_vs_obs),2),
                         obs_moved_0p2s=round(float(moved02),2),obs_moved_0p5s=round(float(moved05),2),
                         trim=e["merge"].get("trimmed_actions"),usable=e["merge"].get("usable_actions"),turnaround=round(e["turnaround_s"],3)))
    out_rows.extend(phys)
    s=dict(run=run,n_cmd=len(cmds),n_obs=len(obs),dur_s=round(ct[-1]-ct[0],1),median_tick_s=round(float(np.median(dt)),4),
           n_accepted=len(acc),n_splice_ticks=int(is_splice.sum()),n_resume_ticks=int(is_resume.sum()),
           big_gt15_joints=n_big,big_exact_at_splice_or_resume=n_big_exact,big_within_1tick=n_big_near,
           big_gt15_incl_gripper=int(big6.sum()),
           big_gt15_any_dt=int((jmax5>15).sum()),
           splice_median=round(float(np.median(sp)),2) if len(sp) else None,splice_max=round(float(sp.max()),2) if len(sp) else None,
           splice_gt15=int((sp>15).sum()),splice_dt_min=round(float(dt[is_splice].min()),3) if is_splice.any() else None,
           splice_dt_max=round(float(dt[is_splice].max()),3) if is_splice.any() else None,
           nonsplice_p99=round(float(np.percentile(nonsp,99)),2) if len(nonsp) else None,nonsplice_max=round(float(nonsp.max()),2) if len(nonsp) else None,
           obs_tick_max=round(float(omax5[ovalid].max()),2) if ovalid.any() else None,
           obs_tick_p99=round(float(np.percentile(omax5[ovalid],99)),2) if ovalid.any() else None,
           obs_ticks_gt5=int((omax5[ovalid]>5).sum()),obs_ticks_gt15=int((omax5[ovalid]>15).sum()),
           )
    summary.append(s)
    # list the big jumps not exactly at a splice tick
    for i in big_idx:
        if not (is_splice[i] or is_resume[i]):
            # distance to nearest splice tick
            sidx=np.array(sorted(splice_idx|resume_idx)); dist=int(np.min(np.abs(sidx-(i+1)))) if len(sidx) else None
            print(run,"BIG not exactly at splice: delta idx",i,"t_rel",round(ct[i+1]-t0,3),"jump",round(float(jmax5[i]),2),"nearest splice tick dist",dist)
with open(os.path.join(OUT,"splice_physical.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
with open(os.path.join(OUT,"recomputed_summary.txt"),"w") as f:
    for s in summary: f.write(str(s)+"\n")
for s in summary: print(s)
