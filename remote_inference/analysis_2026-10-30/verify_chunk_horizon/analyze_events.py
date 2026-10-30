import ast, sys, statistics as st, csv, os
S='/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
runs={'groot_run3':'run3_frames','groot_run4':'run4_frames','groot_run5':'run5_frames','groot_run6':'run6_frames','pi05_run1':'pi05_run1_frames','pi05_run2':'pi05_run2_frames'}
out=open('turnaround_stats.csv','w',newline=''); w=csv.writer(out)
w.writerow(['run','horizon_s','n_requests','n_results','n_accepted','n_rejected','n_cancelled','ta_median','ta_p90','ta_max','n_ta_ge_2.0','n_ta_ge_2.667','n_ta_ge_3.333','median_gap_result_to_next_request_s','n_gaps','n_gap_le_0.1s','usable_median_steps_actual','usable_if_30step_median','n_unusable_if_30step','buffered_on_arrival_if_30step_median_s','max_effective_refill_s'])
for name,d in runs.items():
    p=os.path.join(S,d,'events.txt')
    if not os.path.exists(p): print(name,'no events'); continue
    ev=[]
    for line in open(p):
        line=line.strip()
        if not line: continue
        t,rest=line.split(' ',1); kind,_,payload=rest.partition(' ')
        try: data=ast.literal_eval(payload) if payload else {}
        except Exception: data={}
        ev.append((float(t),kind,data))
    horizon=next((e[2].get('execution_horizon_s') for e in ev if e[1]=='scheduling'),None)
    reqs=[e for e in ev if e[1]=='request']; res=[e for e in ev if e[1]=='result']
    canc=[e for e in ev if e[1]=='request_cancelled']
    tas=[e[2]['turnaround_s'] for e in res]
    acc=sum(1 for e in res if e[2].get('accepted')); rej=len(res)-acc
    # gap from each result to the next request submission
    gaps=[]
    for i,e in enumerate(res):
        nxt=[r for r in reqs if r[0]>=e[0]]
        if nxt: gaps.append(nxt[0][0]-e[0])
    trims=[e[2]['merge'].get('trimmed_actions') for e in res if e[2].get('merge')]
    usable_actual=[e[2]['merge'].get('usable_actions') for e in res if e[2].get('merge') and e[2]['merge'].get('usable_actions') is not None]
    # hypothetical 30-step chunk at same 15 Hz: usable = 30 - trimmed (if trimmed<30)
    hyp=[30-t for t in trims if t is not None]
    unusable=sum(1 for u in hyp if u<=0)
    hyp_pos=[u for u in hyp if u>0]
    eff=[e[2].get('effective_refill_s') for e in reqs]
    q=lambda xs,f: (sorted(xs)[min(len(xs)-1,int(f*len(xs)))] if xs else None)
    w.writerow([name,horizon,len(reqs),len(res),acc,rej,len(canc),
        round(st.median(tas),3) if tas else None, round(q(tas,0.9),3) if tas else None, round(max(tas),3) if tas else None,
        sum(1 for x in tas if x>=2.0), sum(1 for x in tas if x>=2.667), sum(1 for x in tas if x>=3.333),
        round(st.median(gaps),3) if gaps else None, len(gaps), sum(1 for g in gaps if g<=0.1),
        st.median(usable_actual) if usable_actual else None, st.median(hyp_pos) if hyp_pos else None, unusable,
        round(st.median(hyp_pos)/15,3) if hyp_pos else None, round(max(eff),3) if eff else None])
    print(f'{name}: horizon={horizon} req={len(reqs)} res={len(res)} acc={acc} rej={rej} canc={len(canc)} ta med/p90/max={st.median(tas):.3f}/{q(tas,0.9):.3f}/{max(tas):.3f} n>=2.0s={sum(1 for x in tas if x>=2.0)} gaps med={st.median(gaps) if gaps else None:.3f} n_gap<=0.1={sum(1 for g in gaps if g<=0.1)}/{len(gaps)} trims={trims}')
    print('   turnarounds:', [round(x,2) for x in tas])
out.close()
print(open('turnaround_stats.csv').read())
