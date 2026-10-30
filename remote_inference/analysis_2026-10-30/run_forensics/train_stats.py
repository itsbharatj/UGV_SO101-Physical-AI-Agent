import glob, os, sys
import numpy as np, pandas as pd
D='/Users/bharatjain/.cache/huggingface/lerobot/jainbharat/ugv_jenga_box_20261026_214928_v21_70ep/data/chunk-000'
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
files=sorted(glob.glob(D+'/episode_*.parquet'))
rows=[]; per_ep=[]
for f in files:
    df=pd.read_parquet(f)
    ep=int(os.path.basename(f)[8:14])
    a=np.stack(df['action'].to_numpy()); s=np.stack(df['observation.state'].to_numpy()); ts=df['timestamp'].to_numpy()
    d=np.abs(np.diff(a,axis=0))
    # 1 s displacement at 30 fps = 30 steps
    w=30
    disp=np.abs(a[w:]-a[:-w]) if len(a)>w else np.zeros((0,6))
    per_ep.append(dict(ep=ep,n=len(a),dur=float(ts[-1]-ts[0]),
        **{f'min_{j}':float(a[:,i].min()) for i,j in enumerate(J)},
        **{f'max_{j}':float(a[:,i].max()) for i,j in enumerate(J)},
        **{f'maxstep_{j}':float(d[:,i].max()) for i,j in enumerate(J)},
        **{f'max1s_{j}':float(disp[:,i].max()) if len(disp) else np.nan for i,j in enumerate(J)},
        **{f'first_{j}':float(a[0,i]) for i,j in enumerate(J)},
        **{f'last_{j}':float(a[-1,i]) for i,j in enumerate(J)},
        **{f'smin_{j}':float(s[:,i].min()) for i,j in enumerate(J)},
        **{f'smax_{j}':float(s[:,i].max()) for i,j in enumerate(J)},
        **{f'cmd_minus_state_absmax_{j}':float(np.abs(a[:,i]-s[:,i]).max()) for i,j in enumerate(J)},
    ))
    rows.append((ep,a,s,d,disp))
pe=pd.DataFrame(per_ep); pe.to_csv('train_per_episode.csv',index=False)
A=np.concatenate([r[1] for r in rows]); S=np.concatenate([r[2] for r in rows]); Dd=np.concatenate([r[3] for r in rows]); Disp=np.concatenate([r[4] for r in rows])
print('episodes',len(files),'frames',len(A),'episode idx present:',sorted(pe.ep)[:3],'...',sorted(pe.ep)[-3:], 'missing 20?', 20 not in set(pe.ep))
print('duration per episode: median %.1f s, min %.1f, max %.1f'%(pe.dur.median(),pe.dur.min(),pe.dur.max()))
out=[]
for i,j in enumerate(J):
    x=Dd[:,i]
    out.append(dict(joint=j, action_min=A[:,i].min(), action_max=A[:,i].max(), state_min=S[:,i].min(), state_max=S[:,i].max(),
        step_median=np.median(x), step_p90=np.percentile(x,90), step_p99=np.percentile(x,99), step_p999=np.percentile(x,99.9), step_max=x.max(),
        step_gt10_count=int((x>10).sum()), step_gt15_count=int((x>15).sum()), step_gt5_count=int((x>5).sum()),
        disp1s_median=np.median(Disp[:,i]), disp1s_p99=np.percentile(Disp[:,i],99), disp1s_max=Disp[:,i].max(),
        cmd_minus_state_p99=np.percentile(np.abs(A[:,i]-S[:,i]),99), cmd_minus_state_max=np.abs(A[:,i]-S[:,i]).max()))
st=pd.DataFrame(out); st.to_csv('train_delta_stats.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(st.round(2).to_string())
# equivalent per-tick at 15 Hz: 2-step displacement at 30 fps
D2=np.concatenate([np.abs(r[1][2:]-r[1][:-2]) for r in rows])
out2=[]
for i,j in enumerate(J):
    x=D2[:,i]; out2.append(dict(joint=j, step15_median=np.median(x), step15_p99=np.percentile(x,99), step15_p999=np.percentile(x,99.9), step15_max=x.max(), gt10=int((x>10).sum()), gt15=int((x>15).sum())))
st2=pd.DataFrame(out2); st2.to_csv('train_delta_stats_15hz_equiv.csv',index=False); print('\n2-step (15 Hz-equivalent) deltas:'); print(st2.round(2).to_string())
# where do the biggest steps happen?
big=[]
for ep,a,s,d,disp in rows:
    idx=np.argwhere(d>10)
    for k,i in idx: big.append((ep,int(k),J[i],float(d[k,i]),float(a[k,i]),float(a[k+1,i])))
bd=pd.DataFrame(big,columns=['ep','step','joint','delta','from','to']); bd.to_csv('train_big_steps.csv',index=False)
print('\nsteps >10 deg:',len(bd)); print(bd.sort_values('delta',ascending=False).head(25).to_string())
# per-episode ranges to check bimodality in shoulder_lift / elbow_flex
print('\nper-episode shoulder_lift min/max and elbow_flex min/max:')
print(pe[['ep','n','min_shoulder_lift','max_shoulder_lift','min_elbow_flex','max_elbow_flex','first_shoulder_lift','first_elbow_flex','last_shoulder_lift','last_elbow_flex']].round(1).to_string())
