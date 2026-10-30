import ast, os, numpy as np, pandas as pd
SCR='/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
OUT=SCR+'/wf_out/run_forensics'
J=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
def load_events(path):
    ev=[]
    if not os.path.exists(path): return ev
    for line in open(path):
        t,name,rest=line.rstrip('\n').split(' ',2); ev.append((float(t),name,ast.literal_eval(rest)))
    return ev
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',200)
rows=[]
for key,d in [('pi05_run2','pi05_run2_frames'),('pi05_run1','pi05_run1_frames'),('groot_run5','run5_frames'),('groot_run6','run6_frames'),('groot_run4','run4_frames')]:
    df=pd.read_csv(f'{SCR}/{d}/joints.csv'); ev=load_events(f'{SCR}/{d}/events.txt')
    cmd=df[df.kind=='cmd'].reset_index(drop=True); obs=df[df.kind=='obs'].reset_index(drop=True)
    arr=[t for t,n,v in ev if n=='result' and v.get('accepted')]
    inval=[t for t,n,v in ev if n in ('invalidate','planned_hold')]
    t0=arr[0] if arr else cmd.t.iloc[0]; t1=min([x for x in inval if x>t0], default=cmd.t.iloc[-1])
    # policy-active ticks: cmd rows preceded by an obs row within 0.1 s (one real control tick)
    oi=np.searchsorted(obs.t.values,cmd.t.values,side='right')-1
    ok=(oi>=0); dto=np.full(len(cmd),np.inf); dto[ok]=cmd.t.values[ok]-obs.t.values[oi[ok]]
    act=ok&(dto<0.1)&(cmd.t.values>=t0)&(cmd.t.values<=t1)
    r=dict(run=key,policy_window_s=round(t1-t0,2),active_ticks=int(act.sum()))
    for i,j in enumerate(J):
        e=np.abs(cmd[j].values[act]-obs[j].values[oi[act]])
        r[f'{j}_err_mean']=round(e.mean(),2); r[f'{j}_err_p90']=round(np.percentile(e,90),2); r[f'{j}_err_p99']=round(np.percentile(e,99),2); r[f'{j}_err_max']=round(e.max(),2)
    rows.append(r)
    if key=='groot_run4':
        # timeline: when did commands stop changing, when did the run end
        X=cmd[J].values; ch=np.where(np.abs(np.diff(X,axis=0)).max(axis=1)>0.01)[0]
        print(f'groot_run4: first cmd {cmd.t.iloc[0]:.3f}, last changing cmd at {cmd.t.iloc[ch[-1]+1]:.3f} (+{cmd.t.iloc[ch[-1]+1]-cmd.t.iloc[0]:.2f}s), last cmd {cmd.t.iloc[-1]:.3f} (+{cmd.t.iloc[-1]-cmd.t.iloc[0]:.2f}s), last obs {obs.t.iloc[-1]:.3f}; n cmd {len(cmd)}')
        gaps=np.diff(cmd.t.values); big=np.where(gaps>0.3)[0]
        print('  cmd gaps >0.3s at:',[(round(cmd.t.iloc[i]-cmd.t.iloc[0],2),round(gaps[i],2)) for i in big])
    if key=='pi05_run2':
        # splice growth & flip pattern for shoulder_lift / elbow_flex
        sp=pd.read_csv(OUT+'/splice_deltas.csv'); sp=sp[sp.run=='pi05_run2'].reset_index(drop=True)
        print('\npi05_run2 splice max-joint deltas in arrival order (deg):'); print(' '.join(f'{v:.0f}' for v in sp.max_joint_delta))
        print('first 10 after resume: median %.1f; last 10: median %.1f'%(sp.max_joint_delta.iloc[2:12].median(), sp.max_joint_delta.iloc[-10:].median()))
        print('\nshoulder_lift / elbow_flex command just before -> just after each splice (t_rel):')
        X=cmd[J].values; t=cmd.t.values
        for a in arr:
            b=np.searchsorted(t,a)
            if 0<b<len(t): print(f'  {a-t[0]:6.2f}s  lift {X[b-1,1]:7.1f} -> {X[b,1]:7.1f}   elbow {X[b-1,2]:7.1f} -> {X[b,2]:7.1f}   wrist_flex {X[b-1,3]:6.1f} -> {X[b,3]:6.1f}   obs lift {obs[J[1]].values[oi[b]]:7.1f} elbow {obs[J[2]].values[oi[b]]:7.1f}')
        # hold detail: commands during planned hold 87000.769-87003.974
        h=cmd[(cmd.t>=87000.6)&(cmd.t<=87004.3)]
        print('\ncommands around the starvation hold (87000.769 hold -> 87003.974 resumed):'); print(h.to_string(index=False))
T=pd.DataFrame(rows); T.to_csv(OUT+'/tracking_error_policy_window.csv',index=False)
print('\n=== tracking error |cmd-obs| during the policy-active window'); print(T.set_index('run').T.to_string())
# pooled steady-state turnarounds and horizon feasibility
ta=[]
for d in ['run5_frames','run6_frames','pi05_run2_frames']:
    ev=load_events(f'{SCR}/{d}/events.txt'); res=[v['turnaround_s'] for t,n,v in ev if n=='result' and v.get('accepted')]
    ta+=res[2:]  # drop the two cold results of each run
ta=np.array(ta); print(f'\npooled steady-state turnarounds n={len(ta)}: median {np.median(ta):.2f} p90 {np.percentile(ta,90):.2f} p95 {np.percentile(ta,95):.2f} max {ta.max():.2f}')
pairs=ta[:-1]+ta[1:]
for N,f in [(30,15),(30,10),(40,15),(50,15),(30,7.5)]:
    H=N/f; print(f'chunk {N} @ {f} Hz: horizon {H:.2f}s; fraction of consecutive turnaround pairs with tau1+tau2 > horizon (=> starvation): {np.mean(pairs>H):.3f}; single tau > horizon: {np.mean(ta>H):.3f}; usable after median tau: {H-np.median(ta):.2f}s')
