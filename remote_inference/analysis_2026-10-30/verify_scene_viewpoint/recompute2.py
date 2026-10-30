import cv2, numpy as np, os, itertools, csv
SC="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
OUT=SC+"/wf_out/verify_scene_viewpoint"
W,H=640,480
def T(ep): return cv2.imread(f"{OUT}/train_frame0/top_ep{ep:02d}_f0.png")
tests={k:cv2.imread(f"{OUT}/test_{k}.png") for k in ["run5_first_172120","pi05_scene_before_1732","camsjpg_1643_idx0","scene_before_1708","cams_after_1711_idx0"]}
sift=cv2.SIFT_create(nfeatures=4000); bf=cv2.BFMatcher(cv2.NORM_L2)
def feats(im): return sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
def pairs(f1,f2):
    k1,d1=f1;k2,d2=f2; good=[a for a,b in bf.knnMatch(d1,d2,k=2) if a.distance<0.75*b.distance]
    return np.float32([k1[g.queryIdx].pt for g in good]), np.float32([k2[g.trainIdx].pt for g in good])
def rot_estimates(p1,p2):
    Hm,inl=cv2.findHomography(p1,p2,cv2.RANSAC,4.0); inl=inl.ravel().astype(bool)
    Hn=Hm/Hm[2,2]
    r_xaxis=np.degrees(np.arctan2(Hn[1,0],Hn[0,0])); r_yaxis=np.degrees(np.arctan2(-Hn[0,1],Hn[1,1]))
    A,_=cv2.estimateAffinePartial2D(p1[inl],p2[inl],method=cv2.LMEDS); r_sim=np.degrees(np.arctan2(A[1,0],A[0,0])); s_sim=np.hypot(A[0,0],A[1,0])
    Af,_=cv2.estimateAffine2D(p1[inl],p2[inl],method=cv2.LMEDS); 
    U,S,Vt=np.linalg.svd(Af[:,:2]); R=U@Vt; r_aff=np.degrees(np.arctan2(R[1,0],R[0,0]))
    return dict(inl=int(inl.sum()),rot_H_xaxis=round(float(r_xaxis),2),rot_H_yaxis=round(float(r_yaxis),2),rot_similarity=round(float(r_sim),2),scale_sim=round(float(s_sim),3),rot_affine_polar=round(float(r_aff),2),persp=[float(f"{Hn[2,0]:.2e}"),float(f"{Hn[2,1]:.2e}")])
F={ep:feats(T(ep)) for ep in range(70)}; FT={k:feats(v) for k,v in tests.items()}
print("== rotation estimators: run5 first vs B episodes ==")
for ep in [9,12,17,30,34,36,37]:
    p1,p2=pairs(FT["run5_first_172120"],F[ep]); print("run5 vs ep",ep,rot_estimates(p1,p2))
for ep in [39,48]:
    p1,p2=pairs(FT["run5_first_172120"],F[ep]); print("run5 vs ep",ep,"(C)",rot_estimates(p1,p2))
for a,b in [(17,37),(9,38)]:
    p1,p2=pairs(F[a],F[b]); print("baseline ep",a,"vs",b,rot_estimates(p1,p2))
print("\n== cams.jpg 16:43 idx0 vs C episodes, at 640 and at 320 ==")
for ep in [39,48,58]:
    p1,p2=pairs(FT["camsjpg_1643_idx0"],F[ep]); Hm,inl=cv2.findHomography(p1,p2,cv2.RANSAC,4.0)
    c=np.array([320,240,1.0]); v=Hm@c; v=v[:2]/v[2]-c[:2]; print("640px: ep",ep,"center shift px",v.round(1),"pct_w",round(float(np.linalg.norm(v)/640*100),2),"inl",int(inl.sum()))
    a=cv2.resize(tests["camsjpg_1643_idx0"],(320,240),interpolation=cv2.INTER_AREA); b=cv2.resize(T(ep),(320,240),interpolation=cv2.INTER_AREA)
    p1,p2=pairs(feats(a),feats(b)); Hm,inl=cv2.findHomography(p1,p2,cv2.RANSAC,3.0); c=np.array([160,120,1.0]); v=Hm@c; v=v[:2]/v[2]-c[:2]; print("320px: ep",ep,"center shift px",v.round(1),"pct_w",round(float(np.linalg.norm(v)/320*100),2),"inl",int(inl.sum()))
# ---- pixel correlation ----
print("\n== pixel correlation (grayscale Pearson) ==")
def gray(im,sz): return cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),sz,interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
for sz in [(160,120),(320,240)]:
    tr={ep:gray(T(ep),sz) for ep in range(70)}
    def corr(a,b): return float(np.corrcoef(a,b)[0,1])
    for k in ["run5_first_172120","pi05_scene_before_1732","camsjpg_1643_idx0"]:
        g=gray(tests[k],sz); cs=np.array([corr(g,tr[ep]) for ep in range(70)])
        print(sz,k,"max",cs.max().round(3),"argmax ep",int(cs.argmax()),"median",np.median(cs).round(3),"maxB",cs[9:39].max().round(3),"maxC",cs[39:70].max().round(3))
    for name,rng in [("A",range(0,9)),("B",range(9,39)),("C",range(39,70))]:
        v=np.array([corr(tr[a],tr[b]) for a,b in itertools.combinations(rng,2)]); print(sz,"within",name,"median",np.median(v).round(3),"min",v.min().round(3),"p10",np.percentile(v,10).round(3))
    vAB=np.array([corr(tr[a],tr[b]) for a in range(0,9) for b in range(9,39)]); vBC=np.array([corr(tr[a],tr[b]) for a in range(9,39) for b in range(39,70)])
    print(sz,"between A-B median",np.median(vAB).round(3),"max",vAB.max().round(3),"| B-C median",np.median(vBC).round(3),"max",vBC.max().round(3))
