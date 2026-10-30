import glob, numpy as np, pandas as pd
D="/Users/bharatjain/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000"
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
files=sorted(glob.glob(f"{D}/episode_*.parquet"))
def load(exclude20):
    steps=[];gaps=[];steps2=[];vel_gap=[]
    for f in files:
        ep=int(f.split("_")[-1].split(".")[0])
        if exclude20 and ep==20: continue
        df=pd.read_parquet(f)
        a=np.stack(df["action"].values).astype(np.float64); s=np.stack(df["observation.state"].values).astype(np.float64)
        steps.append(np.abs(np.diff(a,axis=0))); steps2.append(np.abs(a[2:]-a[:-2])); gaps.append(a-s)
        v=np.abs(np.diff(a,axis=0)); vel_gap.append(np.column_stack([v, np.abs(a-s)[1:]]))
    return np.concatenate(steps),np.concatenate(steps2),np.concatenate(gaps),vel_gap
for ex in (True,False):
    st,st2,g,vg=load(ex)
    print(f"\n=== exclude ep20={ex}: {len(files)-(1 if ex else 0)} episodes, {len(g)} frames ===")
    rows=[]
    for k,j in enumerate(J):
        ag=np.abs(g[:,k])
        rows.append(dict(joint=j,step_p99=np.percentile(st[:,k],99),step_p999=np.percentile(st[:,k],99.9),step_max=st[:,k].max(),step_gt10=int((st[:,k]>10).sum()),
            step2_p99=np.percentile(st2[:,k],99),step2_max=st2[:,k].max(),
            gap_p50=np.percentile(ag,50),gap_p90=np.percentile(ag,90),gap_p99=np.percentile(ag,99),gap_p999=np.percentile(ag,99.9),gap_max=ag.max(),
            frac_gap_gt10=(ag>10).mean(),frac_gap_gt5=(ag>5).mean(),frac_halfgap_gt10=(ag/2>10).mean(),frac_halfgap_gt5=(ag/2>5).mean()))
    df=pd.DataFrame(rows); pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
    print(df.round(3).to_string(index=False))
    if ex: df.round(4).to_csv("/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad/wf_out/verify_motion_cap/train_stats_verify.csv",index=False)
    anyg=np.abs(g).max(axis=1); print("any-joint gap: p99=%.2f p999=%.2f max=%.2f frac>10=%.4f frac>5=%.4f frac(gap/2)>10=%.4f frac(gap/2)>5=%.4f"%(np.percentile(anyg,99),np.percentile(anyg,99.9),anyg.max(),(anyg>10).mean(),(anyg>5).mean(),(anyg/2>10).mean(),(anyg/2>5).mean()))
    anys=st.max(axis=1); print("any-joint per-step delta: p99=%.2f p999=%.2f max=%.2f"%(np.percentile(anys,99),np.percentile(anys,99.9),anys.max()))
# velocity dependence of the gap (is lag ~ proportional to speed?) for shoulder_lift & elbow_flex
st,st2,g,vg=load(True); vg=np.concatenate(vg)
print("\n=== |gap| vs per-step |delta action| (velocity proxy), median |gap| per velocity bin ===")
bins=[0,0.5,1,2,3,4,5,6,7,8,10]
for k,j in enumerate(J):
    v=vg[:,k]; ga=vg[:,6+k]; out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(v>=lo)&(v<hi)
        if m.sum()>=30: out.append(f"[{lo},{hi}):n={m.sum()} med={np.median(ga[m]):.1f} p90={np.percentile(ga[m],90):.1f}")
    print(j, " | ".join(out))
