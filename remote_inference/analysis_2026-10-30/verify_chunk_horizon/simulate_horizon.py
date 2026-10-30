"""Replay recorded request turnarounds through a model of lerobot's aligned chunk scheduler
(inference/execution.py: should_request :352-379, effective_refill :224-226, _accept_aligned :531-574,
_wait_for_actions :265-276) for different execution horizons / rates.
Starvation cost model: in-flight request cancelled, server invalidation round trip (INVALIDATE_S, from the
recorded 'invalidated' wait_s), then a fresh request whose chunk is fully usable (cursor reset to 0)."""
import ast, os, csv, statistics as st
S='/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad'
INVALIDATE_S=1.21  # pi05_run2 events: 'invalidated' wait_s=1.211
def load(d):
    tas=[]
    for line in open(os.path.join(S,d,'events.txt')):
        t,rest=line.strip().split(' ',1); kind,_,payload=rest.partition(' ')
        if kind=='result': tas.append(ast.literal_eval(payload)['turnaround_s'])
    return tas
def simulate(tas, H, fps, refill, sim_s=120.0):
    dt=1.0/fps; t=0.0; cursor=0; queue=0; pending=None; i=0; n_req=0
    window=[]; init_ta=0.0; completed=0; last_anchor=None
    holds=[]; hold_until=None; has_executed=False
    on_arrival=[]; at_submit=[]; unusable=0; results=0
    def eff(): return max(refill, (max(window) if window else init_ta)+dt)
    while t<sim_s:
        # arrival check (worker thread) before this tick's pop
        if pending is not None and t>=pending['arrive']:
            ta=pending['ta']; completed+=1
            if completed>1: window.append(ta); window=window[-100:]
            else: init_ta=ta
            trim=cursor-pending['anchor']; usable=H-trim; results+=1
            if usable>0: queue=usable; on_arrival.append(usable*dt)
            else: unusable+=1
            pending=None
        if hold_until is not None:
            if t>=hold_until:
                hold_until=None; cursor=0; queue=0; last_anchor=None
                # fresh request after invalidation
                ta=tas[i%len(tas)]; i+=1; n_req+=1
                pending={'arrive':t+ta,'ta':ta,'anchor':cursor}; at_submit.append(0.0)
            t+=dt; continue
        # control tick: pop
        if queue>0:
            queue-=1; cursor+=1; has_executed=True
        elif has_executed:
            # starvation: cancel in-flight request, hold for invalidation RTT
            holds.append(t); pending=None; hold_until=t+INVALIDATE_S; has_executed=False
            t+=dt; continue
        # worker: should_request
        if pending is None and queue*dt<=eff()+1e-9 and (last_anchor is None or cursor>last_anchor):
            ta=tas[i%len(tas)]; i+=1; n_req+=1
            pending={'arrive':t+ta,'ta':ta,'anchor':cursor}; last_anchor=cursor; at_submit.append(queue*dt)
        t+=dt
    held_s=len(holds)*INVALIDATE_S+sum(0 for _ in holds)
    return dict(H=H,fps=fps,refill=refill,horizon_s=round(H/fps,3),requests=n_req,results=results,unusable=unusable,
        starvation_holds=len(holds),holds_per_min=round(len(holds)/(sim_s/60),2),
        buffered_on_arrival_median_s=round(st.median(on_arrival),3) if on_arrival else None,
        buffered_on_arrival_p10_s=round(sorted(on_arrival)[max(0,int(0.1*len(on_arrival)))],3) if on_arrival else None,
        queue_at_submit_median_s=round(st.median(at_submit),3) if at_submit else None,
        queue_at_submit_p10_s=round(sorted(at_submit)[max(0,int(0.1*len(at_submit)))],3) if at_submit else None)
rows=[]
for name,d in [('groot_run5','run5_frames'),('groot_run6','run6_frames'),('pi05_run2','pi05_run2_frames')]:
    tas=load(d)
    tas_steady=tas[1:]  # drop cold start so the replay reflects steady-state link latency
    for label,trace in [('as_recorded',tas),('steady_state',tas_steady)]:
        for H,fps,refill in [(30,15,2.5),(30,15,1.2),(40,15,2.5),(50,15,2.5),(30,10,2.5),(30,10,1.5)]:
            r=simulate(trace,H,fps,refill); r.update(trace=name,variant=label,n_turnarounds=len(trace)); rows.append(r)
with open('horizon_simulation.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['trace']:10s} {r['variant']:12s} H={r['H']:2d}@{r['fps']:2d}Hz refill={r['refill']:.1f} horizon={r['horizon_s']:.2f}s "
          f"req={r['requests']:3d} unusable={r['unusable']:2d} holds/min={r['holds_per_min']:5.2f} "
          f"buf_arrival med/p10={r['buffered_on_arrival_median_s']}/{r['buffered_on_arrival_p10_s']} "
          f"queue_at_submit med/p10={r['queue_at_submit_median_s']}/{r['queue_at_submit_p10_s']}")
