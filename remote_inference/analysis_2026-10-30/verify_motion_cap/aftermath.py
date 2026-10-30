import re, numpy as np, pandas as pd
S="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
for name,d in {"groot_run5":"run5_frames","groot_run6":"run6_frames","groot_run4":"run4_frames","pi05_run2":"pi05_run2_frames"}.items():
    df=pd.read_csv(f"{S}/{d}/joints.csv"); cmd=df[df.kind=="cmd"].reset_index(drop=True); obs=df[df.kind=="obs"].reset_index(drop=True)
    idx=np.searchsorted(obs.t.values,cmd.t.values,side="right")-1; ok=idx>=0; idx=np.clip(idx,0,None); age=cmd.t.values-obs.t.values[idx]
    fresh=ok&(age<=0.05)
    gap=np.abs(cmd[J[:5]].values-obs[J[:5]].values[idx]).max(axis=1)
    dc=np.abs(np.diff(cmd[J[:5]].values,axis=0)).max(axis=1); dt=np.diff(cmd.t.values)
    jump_t=cmd.t.values[1:][(dc>10)&(dt<=0.1)]
    jump5_t=cmd.t.values[1:][(dc>5)&(dt<=0.1)]
    arrivals=[]
    try:
        for line in open(f"{S}/{d}/events.txt"):
            if " result " in line or line.split()[1].startswith("result"):
                arrivals.append(float(line.split()[0]))
    except FileNotFoundError: pass
    arrivals=np.array(arrivals)
    def since(ts, ref):
        if len(ref)==0: return np.full(len(ts),np.nan)
        i=np.searchsorted(ref,ts,side="right")-1
        out=ts-ref[np.clip(i,0,None)]; out[i<0]=np.nan; return out
    for thr in (10,5):
        sel=fresh&(gap>thr); ts=cmd.t.values[sel]
        s_j=since(ts,jump_t); s_j5=since(ts,jump5_t); s_a=since(ts,arrivals)
        print(f"{name} gap>{thr}: {sel.sum()} ticks of {fresh.sum()} | within 0.3 s after a >10 deg cmd jump: {np.sum(s_j<=0.3)} | within 0.3 s after a >5 deg cmd jump: {np.sum(s_j5<=0.3)} | within 0.3 s after a chunk arrival: {np.sum(s_a<=0.3) if len(arrivals) else 'n/a'} | n_arrivals={len(arrivals)}")
        far=ts[(s_j5>0.3)|np.isnan(s_j5)]
        if len(far): 
            g=gap[sel][(s_j5>0.3)|np.isnan(s_j5)]
            print(f"    not splice-aftermath: {len(far)} ticks, gap values: {np.round(np.sort(g)[::-1][:12],1)}")
