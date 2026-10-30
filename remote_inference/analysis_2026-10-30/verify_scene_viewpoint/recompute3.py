import cv2, numpy as np, os, itertools, csv
SC="/private/tmp/claude-501/-Users-bharatjain/018dccab-326f-457e-8e7d-081b76548077/scratchpad"
OUT=SC+"/wf_out/verify_scene_viewpoint"; W,H=640,480
def T(ep): return cv2.imread(f"{OUT}/train_frame0/top_ep{ep:02d}_f0.png")
tests={k:cv2.imread(f"{OUT}/test_{k}.png") for k in ["run5_first_172120","run3_first","pi05_run2_first","pi05_scene_before_1732"]}
sift=cv2.SIFT_create(nfeatures=4000); bf=cv2.BFMatcher(cv2.NORM_L2)
def feats(im): return sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
def pairs(f1,f2):
    k1,d1=f1;k2,d2=f2; good=[a for a,b in bf.knnMatch(d1,d2,k=2) if a.distance<0.75*b.distance]
    return np.float32([k1[g.queryIdx].pt for g in good]), np.float32([k2[g.trainIdx].pt for g in good])
F={ep:feats(T(ep)) for ep in range(9,39)}; FT={k:feats(v) for k,v in tests.items()}
# ---- (2) rover-region SIFT shift: keypoints in TRAIN frame inside rover bbox (x<0.5W, 0.08H<y<0.62H), matched to test ----
print("== rover-region SIFT median shift (train rover bbox -> test), % of W ==")
for k in ["run5_first_172120","pi05_scene_before_1732"]:
    for ep in [17,30,36,12,34]:
        p_test,p_train=pairs(FT[k],F[ep]); m=(p_train[:,0]<0.5*W)&(p_train[:,1]>0.08*H)&(p_train[:,1]<0.62*H)
        d=(p_test[m]-p_train[m])
        if m.sum()>=4:
            A,inl=cv2.estimateAffinePartial2D(p_train[m],p_test[m],method=cv2.RANSAC,ransacReprojThreshold=6.0)
            ninl=int(inl.sum()) if inl is not None else 0
            dd=d[inl.ravel().astype(bool)] if ninl>=4 else d
            print(f"{k:24s} ep{ep}: n_matches_in_rover_bbox={int(m.sum())} ransac_inl={ninl} median dx={np.median(dd[:,0])/W*100:.1f}% dy={np.median(dd[:,1])/W*100:.1f}%  (all-match median dx={np.median(d[:,0])/W*100:.1f}%)")
        else: print(k,ep,"too few",int(m.sum()))
# ---- (3) parallax test: fit H on all matches (dominant plane=table); residuals of far-background matches (train x>0.70W, y<0.45H) and of table matches (train y>0.55H) ----
print("\n== parallax test: residual (test_actual - H(train)) for far-background vs table matches, % of W ==")
def resid(fa,fb,label):
    pa,pb=pairs(fa,fb)  # pa in frame a, pb in frame b
    Hm,inl=cv2.findHomography(pb,pa,cv2.RANSAC,4.0)   # maps b->a
    pr=cv2.perspectiveTransform(pb.reshape(-1,1,2),Hm).reshape(-1,2); r=pa-pr
    bg=(pb[:,0]>0.70*W)&(pb[:,1]<0.45*H); tb=(pb[:,1]>0.55*H)&(pb[:,0]>0.35*W)
    rv=(pb[:,0]<0.5*W)&(pb[:,1]>0.08*H)&(pb[:,1]<0.62*H)
    def med(mask):
        v=r[mask]; v=v[np.linalg.norm(v,axis=1)<0.15*W] if len(v) else v
        return (len(v), np.median(v[:,0])/W*100 if len(v) else None, np.median(v[:,1])/W*100 if len(v) else None)
    print(f"{label:34s} inl={int(inl.sum())} far-bg(n,dx,dy)={tuple(round(x,2) if isinstance(x,float) else x for x in med(bg))} table={tuple(round(x,2) if isinstance(x,float) else x for x in med(tb))} rover={tuple(round(x,2) if isinstance(x,float) else x for x in med(rv))}")
for ep in [17,30,36,12,34,9,38]: resid(FT["run5_first_172120"],F[ep],f"run5 vs ep{ep}")
resid(FT["pi05_scene_before_1732"],F[17],"pi05_scene_before vs ep17")
resid(F[37],F[17],"baseline ep37 vs ep17"); resid(F[38],F[9],"baseline ep38 vs ep9"); resid(F[30],F[12],"baseline ep30 vs ep12")
# ---- (1) rover right-end via template matching: template = right part of rover from ep17 ----
print("\n== rover right-end x (template from ep17, NCC) ==")
tpl_src=T(17); x0,x1,y0,y1=205,310,225,345; tpl=cv2.cvtColor(tpl_src[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
cv2.imwrite(OUT+"/rover_template_ep17.png",tpl_src[y0:y1,x0:x1])
rows=[]
def locate(im,label,tag):
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); band=g[120:420,:]
    best=None
    for s in [0.9,0.95,1.0,1.05,1.1]:
        t=cv2.resize(tpl,None,fx=s,fy=s); r=cv2.matchTemplate(band,t,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
        if best is None or mx>best[0]: best=(mx,loc,s,t.shape)
    mx,(lx,ly),s,(th,tw)=best; right_end=(lx+tw*(x1-x0-(x1-x0))/ (x1-x0) + tw)  # right edge of template box
    # the rover's right edge inside the template is at ~ (300-205)/105 of template width
    rover_right=lx+tw*(300-205)/(x1-x0)
    rows.append(dict(src=label,kind=tag,ncc=round(float(mx),3),scale=s,x_box=lx,y_box=ly+120,rover_right_x_frac=round(float(rover_right)/W,3)))
    vis=im.copy(); cv2.rectangle(vis,(lx,ly+120),(lx+tw,ly+120+th),(0,255,255),2); cv2.line(vis,(int(rover_right),0),(int(rover_right),H),(0,0,255),1)
    return vis
vis=[]
for ep in range(9,39): v=locate(T(ep),f"ep{ep:02d}","train B"); 
for k in tests: v=locate(tests[k],k,"test"); vis.append(cv2.resize(v,(320,240)))
for ep in [17,30,36,12]: vis.append(cv2.resize(locate(T(ep),f"ep{ep:02d}","train B"),(320,240)))
with open(OUT+"/rover_right_end_x.csv","w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
tb=[r for r in rows if r["kind"]=="train B"][:30]; xs=np.array([r["rover_right_x_frac"] for r in tb]); nc=np.array([r["ncc"] for r in tb])
print("train B rover_right x frac: min",xs.min(),"median",np.median(xs),"max",xs.max(),"| ncc min",nc.min(),"median",np.median(nc))
for r in rows:
    if r["kind"]=="test": print(r)
print("low-ncc train frames:",[(r["src"],r["ncc"],r["rover_right_x_frac"]) for r in tb if r["ncc"]<0.5])
cv2.imwrite(OUT+"/rover_locate_vis.jpg",np.vstack([np.hstack(vis[0:4]),np.hstack(vis[4:8])]),[cv2.IMWRITE_JPEG_QUALITY,85])
# ---- (4) gridded comparison ----
def grid(im,label):
    v=im.copy()
    for i in range(1,10):
        cv2.line(v,(int(W*i/10),0),(int(W*i/10),H),(255,255,0),1); cv2.line(v,(0,int(H*i/10)),(W,int(H*i/10)),(255,255,0),1); cv2.putText(v,f"{i/10:.1f}",(int(W*i/10)+2,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
    cv2.putText(v,label,(5,H-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2); return v
g=np.vstack([np.hstack([grid(T(17),"TRAIN ep17 f0 (B)"),grid(T(30),"TRAIN ep30 f0 (B)"),grid(T(36),"TRAIN ep36 f0 (B)")]),np.hstack([grid(tests["run5_first_172120"],"TEST run5 first"),grid(tests["pi05_scene_before_1732"],"TEST pi05_scene_before"),grid(T(39),"TRAIN ep39 f0 (C)")])])
cv2.imwrite(OUT+"/grid_compare_mine.jpg",cv2.resize(g,(1440,720)),[cv2.IMWRITE_JPEG_QUALITY,85])
