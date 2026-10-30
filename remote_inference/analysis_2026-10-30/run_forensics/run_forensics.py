import ast, os, sys, json
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

SCR='/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
OUT=SCR+'/wf_out/run_forensics'
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
RUNS=[('pi05_run2','pi05_run2_frames'),('pi05_run1','pi05_run1_frames'),('groot_run5','run5_frames'),('groot_run6','run6_frames'),('groot_run4','run4_frames')]
train=pd.read_csv(OUT+'/train_delta_stats.csv').set_index('joint')
WIN=0.15

def load_events(path):
    ev=[]
    if not os.path.exists(path): return ev
    for line in open(path):
        t,name,rest=line.rstrip('\n').split(' ',2)
        ev.append((float(t),name,ast.literal_eval(rest)))
    return ev

def lag_estimate(tc,xc,to,xo,maxlag=2.0,step=1/30):
    # find lag L minimizing mean |cmd(t) - obs(t+L)|  (obs reaches the command L seconds later)
    best=None
    for L in np.arange(0,maxlag+1e-9,step):
        xo_i=np.interp(tc+L,to,xo,left=np.nan,right=np.nan)
        m=np.isfinite(xo_i)
        if m.sum()<20: continue
        e=np.mean(np.abs(xc[m]-xo_i[m]))
        if best is None or e<best[1]: best=(L,e)
    return best

summary=[]; jump_rows=[]; splice_rows=[]
for key,d in RUNS:
    df=pd.read_csv(f'{SCR}/{d}/joints.csv')
    cmd=df[df.kind=='cmd'].reset_index(drop=True); obs=df[df.kind=='obs'].reset_index(drop=True)
    ev=load_events(f'{SCR}/{d}/events.txt')
    arr=np.array([t for t,n,v in ev if n=='result' and v.get('accepted') and (v.get('merge') or {}).get('accepted',True)])
    holds=[(t,n,v) for t,n,v in ev if n in ('planned_hold','waiting','resumed','invalidated','fault','request_cancelled')]
    t=cmd.t.values; X=cmd[J].values
    dt=np.diff(t); dX=np.diff(X,axis=0); aX=np.abs(dX)
    tick_t=t[1:]            # time of the command that completes the delta
    maxj=aX.max(axis=1)
    # does a chunk arrival fall within WIN of this tick (either side) or between the two commands?
    near=np.zeros(len(tick_t),bool)
    if len(arr):
        for i in range(len(tick_t)):
            lo=t[i]-WIN; hi=t[i+1]+WIN
            near[i]=np.any((arr>=lo)&(arr<=hi))
    gap=dt>0.2   # delta spans a pause (hold / startup), not a single 66 ms tick
    one_tick=~gap
    n10=int(((maxj>10)&one_tick).sum()); n15=int(((maxj>15)&one_tick).sum()); n5=int(((maxj>5)&one_tick).sum())
    n15_near=int(((maxj>15)&one_tick&near).sum()); n10_near=int(((maxj>10)&one_tick&near).sum())
    n15_gap=int(((maxj>15)&gap).sum())
    mid=one_tick&~near
    run_len=t[-1]-t[0]
    rec=dict(run=key, n_cmd=len(t), duration_s=round(run_len,2), n_arrivals=len(arr),
             cmd_ticks_dt_median_ms=round(float(np.median(dt))*1000,1), cmd_ticks_dt_p99_ms=round(float(np.percentile(dt,99))*1000,1),
             jumps_gt5=n5, jumps_gt10=n10, jumps_gt15=n15, jumps_gt10_near_arrival=n10_near, jumps_gt15_near_arrival=n15_near,
             jumps_gt15_across_pause=n15_gap,
             frac_gt15_near_arrival=round(n15_near/n15,3) if n15 else None, frac_gt10_near_arrival=round(n10_near/n10,3) if n10 else None,
             ticks_near_arrival=int((one_tick&near).sum()), ticks_mid_chunk=int(mid.sum()),
             midchunk_maxjoint_delta_median=round(float(np.median(maxj[mid])),2) if mid.any() else None,
             midchunk_maxjoint_delta_p90=round(float(np.percentile(maxj[mid],90)),2) if mid.any() else None,
             midchunk_maxjoint_delta_p99=round(float(np.percentile(maxj[mid],99)),2) if mid.any() else None,
             midchunk_maxjoint_delta_max=round(float(maxj[mid].max()),2) if mid.any() else None,
             neararrival_maxjoint_delta_median=round(float(np.median(maxj[one_tick&near])),2) if (one_tick&near).any() else None,
             neararrival_maxjoint_delta_p90=round(float(np.percentile(maxj[one_tick&near],90)),2) if (one_tick&near).any() else None,
             neararrival_maxjoint_delta_max=round(float(maxj[one_tick&near].max()),2) if (one_tick&near).any() else None)
    # per-joint stats
    for i,j in enumerate(J):
        x=aX[:,i]
        rec[f'{j}_gt10']=int(((x>10)&one_tick).sum()); rec[f'{j}_gt15']=int(((x>15)&one_tick).sum())
        rec[f'{j}_mid_p99']=round(float(np.percentile(x[mid],99)),2) if mid.any() else None
        rec[f'{j}_mid_max']=round(float(x[mid].max()),2) if mid.any() else None
        rec[f'{j}_cmd_min']=round(float(X[:,i].min()),2); rec[f'{j}_cmd_max']=round(float(X[:,i].max()),2)
        lo,hi=train.loc[j,'action_min'],train.loc[j,'action_max']
        below=np.clip(lo-X[:,i],0,None); above=np.clip(X[:,i]-hi,0,None)
        rec[f'{j}_cmd_outside_train_frac']=round(float(((below>0)|(above>0)).mean()),4)
        rec[f'{j}_cmd_outside_train_maxdeg']=round(float(max(below.max(),above.max())),2)
        # tracking: obs at same tick (obs row precedes cmd row); align by nearest previous obs
        oi=np.searchsorted(obs.t.values,t,side='right')-1; ok=oi>=0
        err=np.abs(X[ok,i]-obs[J[i]].values[oi[ok]])
        rec[f'{j}_track_err_mean']=round(float(err.mean()),2); rec[f'{j}_track_err_p99']=round(float(np.percentile(err,99)),2); rec[f'{j}_track_err_max']=round(float(err.max()),2)
        if len(obs)>30:
            be=lag_estimate(t,X[:,i],obs.t.values,obs[j].values)
            if be: rec[f'{j}_lag_s']=round(be[0],3); rec[f'{j}_lag_resid']=round(be[1],2)
    # jump list
    for i in np.where((maxj>10)&one_tick)[0]:
        k=int(np.argmax(aX[i])); 
        nearest=float(np.min(np.abs(arr-t[i+1]))) if len(arr) else None
        jump_rows.append(dict(run=key,t=t[i+1],t_rel=round(t[i+1]-t[0],3),dt_ms=round(dt[i]*1000,1),joint=J[k],delta=round(aX[i,k],2),frm=round(X[i,k],2),to=round(X[i+1,k],2),near_arrival=bool(near[i]),nearest_arrival_s=round(nearest,3) if nearest is not None else None, all_deltas=' '.join(f'{v:.1f}' for v in aX[i])))
    # splice table: for every arrival, delta between the last cmd before and first cmd after
    for a in arr:
        b=np.searchsorted(t,a)  # first cmd index at/after arrival
        if 0<b<len(t):
            dd=np.abs(X[b]-X[b-1]); 
            # also the delta over the next 2 ticks
            dd2=np.abs(X[min(b+1,len(t)-1)]-X[b-1])
            splice_rows.append(dict(run=key,t_arrival=a,t_rel=round(a-t[0],3),gap_to_next_cmd_ms=round((t[b]-a)*1000,1),max_joint_delta=round(float(dd.max()),2),joint=J[int(np.argmax(dd))],max_joint_delta_2ticks=round(float(dd2.max()),2),**{f'd_{j}':round(float(dd[i]),2) for i,j in enumerate(J)}))
    summary.append(rec)
    # ---- plot with PIL ----
    W,H=1800,1500; img=Image.new('RGB',(W,H),'white'); dr=ImageDraw.Draw(img)
    try: font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',14); fontb=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',18)
    except Exception: font=ImageFont.load_default(); fontb=font
    t0=t[0]; T=run_len if run_len>0 else 1; L,R,TOP,BOT=90,20,50,30; ph=(H-TOP-BOT)//6
    dr.text((L,10),f'{key}: commanded (red) vs observed (blue) joint positions; green ticks = accepted chunk arrivals; orange = |delta cmd|>15 deg in one tick; grey bands = hold/waiting', fill='black', font=fontb)
    for i,j in enumerate(J):
        y0=TOP+i*ph; y1=y0+ph-25
        xs=np.concatenate([X[:,i],obs[j].values]); lo,hi=np.nanmin(xs),np.nanmax(xs); 
        if hi-lo<1: hi=lo+1
        def px(tt): return L+(tt-t0)/T*(W-L-R)
        def py(v): return y1-(v-lo)/(hi-lo)*(y1-y0)
        dr.rectangle([L,y0,W-R,y1],outline='#cccccc')
        # hold bands
        hs=None
        for tt,n,v in holds:
            if n in ('planned_hold','waiting') and hs is None: hs=tt
            if n in ('resumed','fault') and hs is not None: dr.rectangle([px(hs),y0,px(tt),y1],fill='#e6e6e6'); hs=None
        for a in arr: dr.line([px(a),y0,px(a),y0+10],fill='green',width=2)
        pts=[(px(tt),py(v)) for tt,v in zip(obs.t.values,obs[j].values)]
        if len(pts)>1: dr.line(pts,fill='#1f77b4',width=2)
        pts=[(px(tt),py(v)) for tt,v in zip(t,X[:,i])]
        if len(pts)>1: dr.line(pts,fill='#d62728',width=1)
        for k in np.where((aX[:,i]>15)&one_tick)[0]: dr.ellipse([px(t[k+1])-4,py(X[k+1,i])-4,px(t[k+1])+4,py(X[k+1,i])+4],outline='orange',width=2)
        dr.text((5,y0),f'{j}\n{lo:.0f}..{hi:.0f}',fill='black',font=font)
        for s in range(0,int(T)+1,5): dr.text((px(t0+s)-6,y1+3),str(s),fill='black',font=font)
    img.save(f'{OUT}/{key}_joints.png')
    # delta plot: per tick max |delta| across joints, with arrivals
    W2,H2=1800,420; img2=Image.new('RGB',(W2,H2),'white'); dr2=ImageDraw.Draw(img2)
    dr2.text((L,10),f'{key}: per-tick max |delta cmd| over joints (deg) — black bars; green ticks = chunk arrivals; dashed = 10 and 15 deg',fill='black',font=fontb)
    y0,y1=50,H2-40; mx=max(20,float(maxj.max())*1.05)
    def px2(tt): return L+(tt-t0)/T*(W2-L-R)
    def py2(v): return y1-v/mx*(y1-y0)
    dr2.rectangle([L,y0,W2-R,y1],outline='#cccccc')
    for v in (10,15):
        for xx in range(L,W2-R,12): dr2.line([xx,py2(v),xx+6,py2(v)],fill='#999999')
        dr2.text((L-40,py2(v)-7),str(v),fill='black',font=font)
    for a in arr: dr2.line([px2(a),y0,px2(a),y0+12],fill='green',width=2)
    for k in range(len(maxj)):
        if one_tick[k]: dr2.line([px2(t[k+1]),y1,px2(t[k+1]),py2(maxj[k])],fill='black',width=1)
    for s in range(0,int(T)+1,5): dr2.text((px2(t0+s)-6,y1+3),str(s),fill='black',font=font)
    dr2.text((L-40,py2(mx)),f'{mx:.0f}',fill='black',font=font)
    img2.save(f'{OUT}/{key}_tick_deltas.png')

pd.DataFrame(summary).to_csv(OUT+'/run_summary.csv',index=False)
pd.DataFrame(jump_rows).to_csv(OUT+'/jumps_gt10.csv',index=False)
pd.DataFrame(splice_rows).to_csv(OUT+'/splice_deltas.csv',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_columns',200); pd.set_option('display.max_rows',500)
S=pd.DataFrame(summary).set_index('run')
core=['n_cmd','duration_s','n_arrivals','cmd_ticks_dt_median_ms','cmd_ticks_dt_p99_ms','jumps_gt5','jumps_gt10','jumps_gt15','jumps_gt10_near_arrival','jumps_gt15_near_arrival','jumps_gt15_across_pause','frac_gt15_near_arrival','frac_gt10_near_arrival','ticks_near_arrival','ticks_mid_chunk','midchunk_maxjoint_delta_median','midchunk_maxjoint_delta_p90','midchunk_maxjoint_delta_p99','midchunk_maxjoint_delta_max','neararrival_maxjoint_delta_median','neararrival_maxjoint_delta_p90','neararrival_maxjoint_delta_max']
print('=== CORE SUMMARY'); print(S[core].T.to_string())
print('\n=== PER-JOINT jumps / mid-chunk p99,max'); print(S[[c for c in S.columns if c.endswith('_gt10') or c.endswith('_gt15') or c.endswith('_mid_p99') or c.endswith('_mid_max')]].T.to_string())
print('\n=== PER-JOINT cmd range, outside-train, tracking, lag'); print(S[[c for c in S.columns if 'cmd_min' in c or 'cmd_max' in c or 'outside' in c or 'track' in c or 'lag' in c]].T.to_string())
sp=pd.DataFrame(splice_rows)
print('\n=== SPLICE deltas per run (max joint delta at the first command after each accepted arrival)')
print(sp.groupby('run').max_joint_delta.describe(percentiles=[.5,.9]).round(2).to_string())
print('\n=== splices with >15 deg at first command after arrival, per run'); print(sp.groupby('run').apply(lambda g: pd.Series({'n_splices':len(g),'gt15':(g.max_joint_delta>15).sum(),'gt10':(g.max_joint_delta>10).sum(),'gt15_within2ticks':(g.max_joint_delta_2ticks>15).sum()})).to_string())
jp=pd.DataFrame(jump_rows)
print('\n=== pi05_run2 jumps >10 (first 60)'); print(jp[jp.run=='pi05_run2'].head(60).to_string())
print('\n=== other runs jumps >10'); print(jp[jp.run!='pi05_run2'].to_string())
