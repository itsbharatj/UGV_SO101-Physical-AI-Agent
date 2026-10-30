"""Show which OpenCV index is which camera before a run (macOS renumbers them when a camera is plugged in).

Usage: .venv/bin/python check_cameras.py   -> prints one line per index and writes cameras_now.jpg (tiles labelled
by index). Pick the indices and run:  TOP_CAM=<i> WRIST_CAM=<j> ./run_client.sh ...
"""

import time

import cv2
import numpy as np

tiles = []
for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(12):  # let exposure settle
        ok, frame = cap.read()
        time.sleep(0.03)
    cap.release()
    if frame is None:
        print(f"index {idx}: opened but gave no frame")
        continue
    print(f"index {idx}: {frame.shape[1]}x{frame.shape[0]}  mean brightness {frame.mean():.0f}")
    tile = cv2.resize(frame, (480, 360))
    cv2.putText(tile, f"index {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    tiles.append(tile)
if tiles:
    cv2.imwrite("cameras_now.jpg", np.hstack(tiles))
    print(f"wrote cameras_now.jpg ({len(tiles)} cameras)")
