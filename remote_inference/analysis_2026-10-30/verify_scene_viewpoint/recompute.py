import cv2, numpy as np, os, csv, itertools, json
SC="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
OUT=SC+"/wf_out/verify_scene_viewpoint"
W,H=640,480
def load_train(ep): return cv2.imread(f"{OUT}/train_frame0/top_ep{ep:02d}_f0.png")
def up(im): return cv2.resize(im,(W,H),interpolation=cv2.INTER_CUBIC)
def first_frame(d):
    fs=sorted(f for f in os.listdir(SC+"/"+d) if f.endswith(".jpg")); return cv2.imread(f"{SC}/{d}/{fs[0]}")
tests={}
tests["run5 first 172120"]=up(first_frame("run5_frames")[:, :320])
tests["run3 first"]=up(first_frame("run3_frames")[:, :320])
tests["pi05_run2 first"]=up(first_frame("pi05_run2_frames")[:, :320])
tests["pi05_scene_before 17:32"]=up(cv2.imread(SC+"/pi05_scene_before.jpg")[:, :480])
tests["scene_before 17:08"]=up(cv2.imread(SC+"/scene_before.jpg")[:, :480])
tests["cams.jpg 16:43 idx0"]=up(cv2.imread(SC+"/cams.jpg")[:360, :480])
tests["cams_after 17:11 idx0"]=up(cv2.imread(SC+"/cams_after.jpg")[:360, :480])
for k,v in tests.items(): cv2.imwrite(OUT+"/test_"+k.replace(" ","_").replace(":","").replace(".","")+".png", v)
sift=cv2.SIFT_create(nfeatures=4000)
def feats(im):
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); return sift.detectAndCompute(g,None)
bf=cv2.BFMatcher(cv2.NORM_L2)
def match(f1,f2,mask1=None):
    k1,d1=f1;k2,d2=f2
    if d1 is None or d2 is None: return None
    m=bf.knnMatch(d1,d2,k=2); good=[]
    for pr in m:
        if len(pr)<2: continue
        a,b=pr
        if a.distance<0.75*b.distance:
            if mask1 is not None:
                x,y=k1[a.queryIdx].pt
                if not mask1(x,y): continue
            good.append(a)
    return good
def homog(f1,f2,mask1=None):
    good=match(f1,f2,mask1)
    if good is None or len(good)<8: return None
    k1,_=f1;k2,_=f2
    p1=np.float32([k1[g.queryIdx].pt for g in good]); p2=np.float32([k2[g.trainIdx].pt for g in good])
    Hm,inl=cv2.findHomography(p1,p2,cv2.RANSAC,4.0)
    if Hm is None: return None
    n=int(inl.sum())
    def ap(pt):
        v=Hm@np.array([pt[0],pt[1],1.0]); return v[:2]/v[2]
    c=np.array([W/2,H/2]); cs=ap(c)-c
    corners=[np.array(p,float) for p in [(0,0),(W,0),(0,H),(W,H)]]
    cshift=[float(np.linalg.norm(ap(p)-p)/W*100) for p in corners]
    # jacobian at center
    e=1.0; J=np.array([(ap(c+[e,0])-ap(c-[e,0]))/(2*e),(ap(c+[0,e])-ap(c-[0,e]))/(2*e)]).T
    rot=np.degrees(np.arctan2(J[1,0]-J[0,1],J[0,0]+J[1,1])); sc=np.sqrt(abs(np.linalg.det(J)))
    # pure translation estimate on inliers (median)
    inl=inl.ravel().astype(bool); dv=p2[inl]-p1[inl]; med=np.median(dv,axis=0)
    return dict(good=len(good),inliers=n,dx_pct=float(cs[0]/W*100),dy_pct=float(cs[1]/W*100),center_pct_w=float(np.linalg.norm(cs)/W*100),corner_pct_w=[round(x,1) for x in cshift],rot_deg=float(rot),scale=float(sc),med_dx_pct=float(med[0]/W*100),med_dy_pct=float(med[1]/W*100))
trainF={ep:feats(load_train(ep)) for ep in range(70)}
testF={k:feats(v) for k,v in tests.items()}
groups={"A":list(range(0,9)),"B":list(range(9,39)),"C":list(range(39,70))}
rows=[]
for tk in tests:
    for ep in range(70):
        r=homog(testF[tk],trainF[ep])
        if r is None: continue
        r.update(test=tk,ep=ep,grp=[g for g,v in groups.items() if ep in v][0]); rows.append(r)
with open(OUT+"/homography_test_vs_train_full.csv","w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# summaries
def summ(tk,grp):
    rs=[r for r in rows if r["test"]==tk and r["grp"]==grp and r["inliers"]>=30]
    if not rs: return None
    arr=np.array([[r["center_pct_w"],r["rot_deg"],r["scale"],r["dx_pct"],r["dy_pct"],r["inliers"]] for r in rs])
    return dict(n=len(rs),center_med=float(np.median(arr[:,0])),center_min=float(arr[:,0].min()),center_max=float(arr[:,0].max()),rot_med=float(np.median(arr[:,1])),rot_min=float(arr[:,1].min()),rot_max=float(arr[:,1].max()),scale_med=float(np.median(arr[:,2])),dx_med=float(np.median(arr[:,3])),dy_med=float(np.median(arr[:,4])),inl_med=float(np.median(arr[:,5])))
print("== TEST vs TRAIN group summaries (center shift % of W, rot deg; inliers>=30) ==")
for tk in tests:
    for g in groups:
        s=summ(tk,g); print(f"{tk:28s} vs {g}: {s}")
print("\n== run5 vs ep17/ep12/ep34/ep30/ep36/ep09 full rows ==")
for r in rows:
    if r["test"]=="run5 first 172120" and r["ep"] in (17,12,34,30,36,9,37,38): print({k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()})
# within-placement baselines
print("\n== TRAIN within-placement baselines ==")
pairs=[(17,37),(12,30),(9,38),(34,35),(0,8),(39,47),(58,69),(39,69),(17,30),(9,17),(13,36),(20,25),(40,60),(48,65)]
base=[]
for a,b in pairs:
    r=homog(trainF[a],trainF[b]); r.update(a=a,b=b); base.append(r); print(a,b,{k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()})
# all pairs within B (30 eps -> 435 pairs) center shift distribution
cb=[]
for a,b in itertools.combinations(groups["B"],2):
    r=homog(trainF[a],trainF[b])
    if r and r["inliers"]>=30: cb.append(r["center_pct_w"])
cb=np.array(cb); print(f"within-B all pairs: n={len(cb)} median={np.median(cb):.2f} p90={np.percentile(cb,90):.2f} max={cb.max():.2f}")
cc=[]
for a,b in itertools.combinations(groups["C"],2):
    r=homog(trainF[a],trainF[b])
    if r and r["inliers"]>=30: cc.append(r["center_pct_w"])
cc=np.array(cc); print(f"within-C all pairs: n={len(cc)} median={np.median(cc):.2f} p90={np.percentile(cc,90):.2f} max={cc.max():.2f}")
# test-vs-test (camera stability through the session)
print("\n== TEST vs TEST (session camera stability) ==")
keys=list(tests)
for a,b in itertools.combinations(keys,2):
    r=homog(testF[a],testF[b]); print(f"{a:26s} -> {b:26s}: {None if r is None else {k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items() if k in ('inliers','center_pct_w','dx_pct','dy_pct','rot_deg','scale')}}")
