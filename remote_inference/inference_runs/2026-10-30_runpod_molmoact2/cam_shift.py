"""Headless camera-vs-reference check: grabs one 640x480 frame from <index>, feature-matches it to <reference image>,
prints the centre shift (% of width) and saves shift_<index>.jpg = reference | live | 50/50 blend with reference edges."""
import sys, time
import cv2, numpy as np

idx, ref_path = int(sys.argv[1]), sys.argv[2]
ref = cv2.resize(cv2.imread(ref_path), (640, 480))
cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame = None
for _ in range(15):  # let exposure settle, keep the last good frame
    ok, f = cap.read()
    if ok: frame = f
    time.sleep(0.05)
cap.release()
if frame is None: sys.exit(f"index {idx}: no frame")
frame = cv2.resize(frame, (640, 480))
sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
k2, d2 = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
msg = "not enough matches (camera probably looks at a different scene)"
if d1 is not None and d2 is not None and len(k2) > 10:
    good = [m for m, nn in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * nn.distance]
    if len(good) >= 12:
        src = np.float32([k1[m.queryIdx].pt for m in good]); dst = np.float32([k2[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None and mask.sum() >= 10:
            c = cv2.perspectiveTransform(np.float32([[[320, 240]]]), H)[0, 0]
            dx, dy = (c[0] - 320) / 640 * 100, (c[1] - 240) / 640 * 100
            scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
            msg = f"shift {np.hypot(dx, dy):.1f}% of width (dx {dx:+.1f}%, dy {dy:+.1f}%), scale x{scale:.2f}, {int(mask.sum())} inlier matches"
blend = cv2.addWeighted(ref, 0.5, frame, 0.5, 0)
blend[cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 80, 160) > 0] = (0, 255, 255)
cv2.imwrite(f"shift_{idx}.jpg", np.hstack([ref, frame, blend]))
print(f"index {idx} vs {ref_path.split('/')[-1]}: {msg}; wrote shift_{idx}.jpg")
