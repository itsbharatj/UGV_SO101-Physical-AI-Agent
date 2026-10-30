"""Live camera view blended with a training reference frame, to put a camera back where it was during recording.

Usage: .venv/bin/python align_camera.py <camera index> <reference image>
Keys: q quits, s saves align_<index>.jpg. Move the camera until the table edge, the desk legs and the robot's
tracks and turret line up (double images merge into one). The title shows how far the live frame still is from
the reference (feature-matched shift as % of the frame width; recording sessions agreed to within ~2 %).
"""

import sys

import cv2
import numpy as np

idx, ref_path = int(sys.argv[1]), sys.argv[2]
ref = cv2.imread(ref_path)
ref = cv2.resize(ref, (640, 480))
cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
sift = cv2.SIFT_create()
k_ref, d_ref = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
matcher = cv2.BFMatcher()
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.resize(frame, (640, 480))
    n += 1
    shift_txt = ""
    if n % 10 == 0 and d_ref is not None:  # feature-matched shift, every 10th frame
        k, d = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        if d is not None and len(k) > 10:
            good = [m for m, nn in matcher.knnMatch(d_ref, d, k=2) if m.distance < 0.75 * nn.distance]
            if len(good) >= 12:
                src = np.float32([k_ref[m.queryIdx].pt for m in good])
                dst = np.float32([k[m.trainIdx].pt for m in good])
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    c = cv2.perspectiveTransform(np.float32([[[320, 240]]]), H)[0, 0]
                    shift = float(np.hypot(c[0] - 320, c[1] - 240)) / 640 * 100
                    shift_txt = f"  shift {shift:.1f}% of width ({int(mask.sum())} matches)"
                    last = shift_txt
    blend = cv2.addWeighted(ref, 0.5, frame, 0.5, 0)
    edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 80, 160)
    blend[edges > 0] = (0, 255, 255)  # reference edges in yellow
    cv2.putText(blend, f"cam {idx} vs {ref_path.split('/')[-1]}" + (locals().get("last") or ""), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("align camera (q quits, s saves)", blend)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite(f"align_{idx}.jpg", np.hstack([ref, frame, blend]))
        print(f"saved align_{idx}.jpg")
cap.release()
cv2.destroyAllWindows()
