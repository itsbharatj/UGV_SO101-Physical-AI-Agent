import ast, os, sys
import numpy as np, pandas as pd
SCR='/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
OUT=SCR+'/wf_out/verify_splice'
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)

def load_events(path):
    ev=[]
    for line in open(path):
        t,name,rest=line.rstrip('\n').split(' ',2); ev.append((float(t),name,ast.literal_eval(rest)))
    return ev

def analyse(key, d, violent_from=None):
    print('\n'+'='*100+f'\n{key}  ({d})')
    df=pd.read_csv(f'{SCR}/{d}/joints.csv')
    cmd=df[df.kind=='cmd'].reset_index(drop=True); obs=df[df.kind=='obs'].reset_index(drop=True)
    ev=load_events(f'{SCR}/{d}/events.txt')
    req={v['request_id']:(t,v) for t,n,v in ev if n=='request'}
    res=[(t,v) for t,n,v in ev if n=='result']
    n_acc=sum(1 for t,v in res if v.get('accepted')); n_macc=sum(1 for t,v in res if v.get('accepted') and v.get('merge',{}).get('accepted',True))
    print(f'results={len(res)} accepted={n_acc} merge_accepted={n_macc} requests={len(req)} cancelled={sum(1 for t,n,v in ev if n=="request_cancelled")}')
    arr=np.array([t for t,v in res if v.get('accepted') and v.get('merge',{}).get('accepted',True)])
    t=cmd.t.values; X=cmd[J].values; to=obs.t.values; XO=obs[J].values; t0=t[0]
    dt=np.diff(t); aX=np.abs(np.diff(X,axis=0)); maxj=aX.max(axis=1); one=dt<=0.2
    print(f'cmd rows={len(t)} obs rows={len(to)} run length {t[-1]-t0:.2f}s; cmd dt median {np.median(dt)*1000:.0f} ms, p99 {np.percentile(dt,99)*1000:.0f} ms; obs dt median {np.median(np.diff(to))*1000:.0f} ms')
    # ---- arrivals / turnaround / request cadence
    ta=np.array([v['turnaround_s'] for tt,v in res if v.get('accepted')])
    print('turnaround_s all: '+' '.join(f'{x:.2f}' for x in ta))
    ia=np.diff(arr); print(f'inter-arrival (s): n={len(ia)} median {np.median(ia):.2f} p10 {np.percentile(ia,10):.2f} p90 {np.percentile(ia,90):.2f} min {ia.min():.2f} max {ia.max():.2f}')
    # time from each accepted result to the next request submission
    rq=sorted((tt,v) for tt,v in [(tt,v) for tt,n,v in ev if n=='request'])
    lag=[]; pbs=[]; eff=[]
    for a in arr:
        nxt=[(tt,v) for tt,v in rq if tt>=a-1e-6]
        if nxt: lag.append(nxt[0][0]-a); pbs.append(nxt[0][1]['playback_at_submission_s']); eff.append(nxt[0][1]['effective_refill_s'])
    lag=np.array(lag); print(f'result->next request delay (s): median {np.median(lag):.3f} p90 {np.percentile(lag,90):.3f} max {lag.max():.3f}; playback_at_submission median {np.median(pbs):.2f} s; effective_refill_s values: {sorted(set(round(e,3) for e in eff))}')
    tr=[v['merge']['trimmed_actions'] for tt,v in res if v.get('accepted')]; us=[v['merge']['usable_actions'] for tt,v in res if v.get('accepted')]
    print(f'merge.trimmed_actions: {tr}\nmerge.usable_actions median {np.median(us):.0f} min {min(us)} max {max(us)}')
    # ---- splice table
    rows=[]
    for k,(a,(tt,v)) in enumerate(zip(arr,[(tt,v) for tt,v in res if v.get('accepted') and v.get('merge',{}).get('accepted',True)])):
        b=np.searchsorted(t,a)
        if not (0<b<len(t)): continue
        d=np.abs(X[b]-X[b-1]); r=dict(k=k,t_rel=a-t0,arr_to_cmd_ms=(t[b]-a)*1000,prev_cmd_gap_ms=(t[b]-t[b-1])*1000,splice=d.max(),joint=J[int(np.argmax(d))],trim=v['merge']['trimmed_actions'])
        rid=v['request_id']
        if rid in req:
            cap=req[rid][1]['capture_time']; oc=np.searchsorted(to,cap,side='right')-1; cc=np.searchsorted(t,cap,side='right')-1; oa=np.searchsorted(to,a,side='right')-1
            r['cap_age_s']=a-cap
            if oc>=0 and cc>=0 and oa>=0:
                e_cap=np.abs(X[cc]-XO[oc]); r['track_err_at_capture']=e_cap.max(); r['track_err_at_capture_samejoint']=e_cap[int(np.argmax(d))]
                r['new_minus_obscap']=np.abs(X[b]-XO[oc]).max(); r['old_minus_obsarr']=np.abs(X[b-1]-XO[oa]).max()
                # does the new command move back toward the observed state (relative to old cmd)? on the max joint
                jj=int(np.argmax(d)); r['toward_obs']=bool(np.sign(X[b,jj]-X[b-1,jj])==np.sign(XO[oc,jj]-X[b-1,jj]))
                r['between']=bool(min(X[b-1,jj],XO[oc,jj])-1<=X[b,jj]<=max(X[b-1,jj],XO[oc,jj])+1)
        jj=int(np.argmax(d))
        if b>=4 and (t[b-1]-t[b-4])<0.4:
            prev_dir=X[b-1,jj]-X[b-4,jj]; r['prev_motion_3ticks']=prev_dir; r['reverses']=bool(np.sign(X[b,jj]-X[b-1,jj])==-np.sign(prev_dir)) if abs(prev_dir)>1 else None
            r['old_before']=X[b-1,jj]; r['new_after']=X[b,jj]
        for i,j in enumerate(J): r['d_'+j]=d[i]
        rows.append(r)
    S=pd.DataFrame(rows); S.to_csv(OUT+f'/{key}_splices.csv',index=False)
    print(f'splices={len(S)}; splice step max-joint: median {S.splice.median():.1f} p90 {S.splice.quantile(.9):.1f} max {S.splice.max():.1f}; >15: {(S.splice>15).sum()} >10: {(S.splice>10).sum()} >30: {(S.splice>30).sum()}')
    print(f'  arrival->first cmd gap ms: median {S.arr_to_cmd_ms.median():.0f} max {S.arr_to_cmd_ms.max():.0f}; prev cmd gap ms max {S.prev_cmd_gap_ms.max():.0f}')
    print('  splice steps in arrival order: '+' '.join(f'{x:.0f}' for x in S.splice))
    if 'track_err_at_capture' in S:
        T=S.dropna(subset=['track_err_at_capture'])
        print(f'  capture age at arrival: median {T.cap_age_s.median():.2f}s; tracking err at capture (max joint) median {T.track_err_at_capture.median():.1f} p90 {T.track_err_at_capture.quantile(.9):.1f}; |new-obs_cap| median {T.new_minus_obscap.median():.1f}; |old-obs_arr| median {T.old_minus_obsarr.median():.1f}')
        def spear(a,b):
            ra=pd.Series(a).rank().values; rb=pd.Series(b).rank().values; return np.corrcoef(ra,rb)[0,1]
        print(f'  corr(splice, track_err_at_capture): pearson {T.splice.corr(T.track_err_at_capture):.2f} spearman {spear(T.splice.values,T.track_err_at_capture.values):.2f}; same-joint pearson {T.splice.corr(T.track_err_at_capture_samejoint):.2f}')
        big=T[T.splice>15]; print(f'  splices >15: n={len(big)} median tracking err at capture {big.track_err_at_capture.median():.1f} deg (same joint {big.track_err_at_capture_samejoint.median():.1f}); splice/track ratio median {(big.splice/big.track_err_at_capture.clip(lower=0.5)).median():.1f}')
        print(f'  new cmd moves toward obs (max joint): {T.toward_obs.mean()*100:.0f}% ; new cmd between old cmd and obs_cap: {T.between.mean()*100:.0f}%')
    if 'reverses' in S:
        R=S[(S.splice>15)].dropna(subset=['reverses']); print(f'  splices >15 with arm moving >1 deg over previous 3 ticks: n={len(R)}, new chunk reverses that direction in {R.reverses.mean()*100:.0f}%')
        print('  lift/elbow before->after at >30 deg splices (t_rel, joint, old->new):'); print('   '+'; '.join(f'{r.t_rel:.1f}s {r.joint} {r.old_before:.0f}->{r.new_after:.0f}' for r in S[S.splice>30].itertuples() if hasattr(r,'old_before') and r.old_before==r.old_before))
    # windows
    S['win']=(S.t_rel//10*10).astype(int); print('  median splice per 10 s window:'); print(S.groupby('win').splice.agg(['count','median','max']).round(1).to_string())
    if len(S)>=12: print(f'  first 10 after index 2: median {S.splice.iloc[2:12].median():.1f}; last 10: median {S.splice.iloc[-10:].median():.1f}; excluding last 2: last 10 median {S.splice.iloc[-12:-2].median():.1f}')
    # ---- jumps and their relation to arrivals (strict: arrival in (t_i, t_{i+1}])
    strict=np.zeros(len(dt),bool); tol=np.zeros(len(dt),bool); near15=np.zeros(len(dt),bool)
    for i in range(len(dt)):
        strict[i]=np.any((arr>t[i])&(arr<=t[i+1])); tol[i]=np.any((arr>t[i]-0.07)&(arr<=t[i+1]+0.07)); near15[i]=np.any((arr>=t[i]-0.15)&(arr<=t[i+1]+0.15))
    for thr in (15,10,5):
        m=(maxj>thr)&one; print(f'one-tick jumps >{thr}: {m.sum()}  strict-splice {np.sum(m&strict)}  within+-70ms {np.sum(m&tol)}  within+-150ms {np.sum(m&near15)}')
    print(f'ticks: one-tick {one.sum()}; strict splice ticks {np.sum(one&strict)} ({np.mean(strict[one])*100:.1f}%), +-70ms {np.sum(one&tol)} ({np.mean(tol[one])*100:.1f}%), +-150ms {np.sum(one&near15)} ({np.mean(near15[one])*100:.1f}%)')
    for name,mask in (('mid (not within +-150ms)',one&~near15),('mid (not within +-70ms)',one&~tol),('mid (not strict splice)',one&~strict)):
        v=maxj[mask]; print(f'  {name}: n={len(v)} median {np.median(v):.2f} p90 {np.percentile(v,90):.2f} p99 {np.percentile(v,99):.2f} max {v.max():.2f}')
    nm=(maxj>10)&one&~strict
    if nm.any(): print('  >10 jumps NOT at a strict splice tick: '+'; '.join(f't_rel {t[i+1]-t0:.2f}s {maxj[i]:.1f}deg nearest arrival {np.min(np.abs(arr-t[i+1])):.3f}s' for i in np.where(nm)[0]))
    # ---- observed speeds
    dto=np.diff(to); ok=(dto>0.02)&(dto<0.3); sp=np.abs(np.diff(XO,axis=0))/dto[:,None]; spm=sp[ok].max(axis=1)
    print(f'observed speed (deg/s, max joint per obs step, steps 20-300ms): max {spm.max():.0f} p99 {np.percentile(spm,99):.0f} p90 {np.percentile(spm,90):.0f}; per-joint max '+' '.join(f'{j}={sp[ok][:,i].max():.0f}' for i,j in enumerate(J)))
    dpo=np.abs(np.diff(XO,axis=0))[ok].max(axis=1); print(f'observed per-step displacement (deg per obs step ~{np.median(dto[ok])*1000:.0f} ms): max {dpo.max():.1f} p99 {np.percentile(dpo,99):.1f}; steps >5 deg: {(dpo>5).sum()} of {len(dpo)}; steps >10 deg: {(dpo>10).sum()}')
    # commanded speeds for comparison
    cs=(aX/dt[:,None])[one]; print(f'commanded speed (deg/s): max {cs.max():.0f} p99 {np.percentile(cs.max(axis=1),99):.0f}')
    # ---- tracking |cmd-obs| same tick, whole run and phases
    oi=np.searchsorted(to,t,side='right')-1; okk=oi>=0; E=np.abs(X[okk]-XO[oi[okk]]); tt=t[okk]-t0
    def rep(mask,label):
        if mask.sum()<5: return
        print(f'  {label}: n={mask.sum()} '+' '.join(f'{j} p99={np.percentile(E[mask,i],99):.0f}/max={E[mask,i].max():.0f}' for i,j in enumerate(J)))
    print('tracking error |cmd-obs| (deg) same tick:'); rep(np.ones(len(tt),bool),'whole run')
    for w in range(0,int(tt.max())+1,10): rep((tt>=w)&(tt<w+10),f'{w:2d}-{w+10:2d}s')
    if violent_from is not None: rep(tt>=violent_from,f'>= {violent_from}s (violent phase)')
    # lag estimate
    lags=[]
    for i,j in enumerate(J):
        best=None
        for L in np.arange(0,1.0+1e-9,1/30):
            xo=np.interp(t+L,to,XO[:,i],left=np.nan,right=np.nan); m=np.isfinite(xo)
            e=np.mean(np.abs(X[m,i]-xo[m])); best=(L,e) if best is None or e<best[1] else best
        lags.append(best[0])
    print('best lag per joint (s): '+' '.join(f'{j}={l:.3f}' for j,l in zip(J,lags)))
    return S

S2=analyse('pi05_run2','pi05_run2_frames',violent_from=25)
S5=analyse('groot_run5','run5_frames')
S6=analyse('groot_run6','run6_frames')
