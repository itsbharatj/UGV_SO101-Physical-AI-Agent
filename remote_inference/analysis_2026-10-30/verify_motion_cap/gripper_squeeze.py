import glob, numpy as np, pandas as pd
D="/Users/bharatjain/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000"
A=[];Sx=[];E=[]
for f in sorted(glob.glob(f"{D}/episode_*.parquet")):
    ep=int(f.split("_")[-1].split(".")[0])
    if ep==20: continue
    df=pd.read_parquet(f); a=np.stack(df["action"].values); s=np.stack(df["observation.state"].values)
    A.append(a[:,5]); Sx.append(s[:,5]); E.append(np.full(len(a),ep))
a=np.concatenate(A); s=np.concatenate(Sx); e=np.concatenate(E)
gap=s-a  # positive = follower gripper more open than commanded (blocked by object / closing lag)
print("gripper action: min %.2f p5 %.2f p50 %.2f max %.2f ; state: min %.2f p5 %.2f max %.2f"%(a.min(),np.percentile(a,5),np.percentile(a,50),a.max(),s.min(),np.percentile(s,5),s.max()))
# 'closed command' frames: action within 2 units of its minimum, and gripper nearly static (|da|<0.5) -> squeeze, not motion
da=np.abs(np.diff(a,prepend=a[0]))
closed=(a<=a.min()+2.0)&(da<0.5)
g=gap[closed]
print("closed-command static frames: n=%d (%.1f%% of frames); state-action gap p10 %.2f p50 %.2f p90 %.2f p99 %.2f max %.2f"%(closed.sum(),100*closed.mean(),*np.percentile(g,[10,50,90,99]),g.max()))
print("fraction of closed-command frames with gap>5: %.3f  >10: %.3f  >15: %.3f"%((g>5).mean(),(g>10).mean(),(g>15).mean()))
# per-episode: median squeeze gap while closed (does the follower stall at ~10 on the block?)
med=[np.median(gap[closed&(e==k)]) for k in np.unique(e) if (closed&(e==k)).sum()>10]
print("per-episode median squeeze gap: min %.2f p50 %.2f max %.2f over %d episodes"%(np.min(med),np.median(med),np.max(med),len(med)))
# how long are the squeezes (frames per episode with closed&gap>5)?
cnt=[(closed&(e==k)&(gap>5)).sum() for k in np.unique(e)]
print("frames per episode with closed command and gap>5: p50 %d max %d (30 fps)"%(np.median(cnt),np.max(cnt)))
