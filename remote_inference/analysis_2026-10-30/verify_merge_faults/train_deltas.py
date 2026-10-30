"""Training-set per-frame action deltas (30 fps) and 2-frame (15 Hz-equivalent) deltas; clip fractions."""
import glob, os, sys
import numpy as np, pandas as pd
D="/Users/bharatjain/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000"
files=sorted(glob.glob(os.path.join(D,"episode_*.parquet")))
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
d1=[];d2=[];lead=[];n=0;eps=[]
for f in files:
    df=pd.read_parquet(f,columns=["action","observation.state","frame_index","episode_index"])
    a=np.stack(df["action"].to_numpy()); s=np.stack(df["observation.state"].to_numpy())
    eps.append(int(df["episode_index"].iloc[0])); n+=len(a)
    d1.append(np.abs(np.diff(a,axis=0))); d2.append(np.abs(a[2:]-a[:-2])); lead.append(np.abs(a-s))
d1=np.concatenate(d1);d2=np.concatenate(d2);lead=np.concatenate(lead)
print("episodes",len(files),"ep20 present:",20 in eps,"frames",n,"min/max ep idx",min(eps),max(eps))
def rep(name,x):
    print(name,"per joint p99:",np.round(np.percentile(x,99,axis=0),2),"max:",np.round(x.max(axis=0),2))
rep("per-frame |dA| (30fps)",d1); rep("2-frame |dA| (15Hz-equiv)",d2); rep("|action-state| lead",lead)
m5=d1[:,:5].max(axis=1); m5b=d2[:,:5].max(axis=1)
for cap in (5.0,8.0):
    print(f"cap {cap}: frac of 30fps frames with any-joint(0-4) |dA|>{cap}: {np.mean(m5>cap):.4f}; incl gripper {np.mean(d1.max(axis=1)>cap):.4f}; 15Hz-equiv frames any joint(0-4): {np.mean(m5b>cap):.4f}; frac of (frame,joint0-4) pairs 30fps: {np.mean(d1[:,:5]>cap):.4f}")
    print(f"   lead |a-s|>{cap} frac frames any joint0-4: {np.mean(lead[:,:5].max(axis=1)>cap):.4f}")
print("overall p99 over joints0-4 of 30fps per-frame |dA| (pooled):",round(float(np.percentile(d1[:,:5],99)),2),"max",round(float(d1[:,:5].max()),2),"incl gripper max",round(float(d1.max()),2))
