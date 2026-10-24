#!/usr/bin/env python3
"""Make the wrist camera mount for a 25 x 24 mm Raspberry-Pi-style camera board (Waveshare 160 deg fisheye, the camera
from the UGV Rover's pan-tilt).  Starts from TheRobotStudio's official SO-101 wrist camera mount
(source_models/wrist_camera_mount_so101_v1.stl, Apache-2.0, built for a 32 x 32 mm UVC board with holes on 27 x 27 mm)
and redraws only the camera plate:
  * fills the old 27 x 27 mm holes and the 25 x 16 mm window
  * new 13.5 x 17 mm lens window on the same optical axis as the official camera (so the view of the gripper is unchanged)
  * 4 slotted M2 holes: 20-21 mm apart one way, 12.5-13.5 mm the other (fits genuine and clone boards)
Output: meshes/mount/wrist_cam_mount_pi_board.stl  (mm, same frame as the official mount, so it drops into the URDF)
    python3 scripts/make_wrist_mount.py
"""
import os
import numpy as np
import trimesh
from shapely.geometry import box as sbox

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PKG, "source_models", "wrist_camera_mount_so101_v1.stl")
OUT = os.path.join(PKG, "meshes", "mount", "wrist_cam_mount_pi_board.stl")

# camera plate of the official mount (mm, measured from the mesh)
N = np.array([-0.423, -0.906, 0.0]); N /= np.linalg.norm(N)        # outward normal of the plate's outer face
E1 = np.array([-0.906, 0.423, 0.0]); E1 /= np.linalg.norm(E1)      # in-plane, along a row of the old holes
E2 = np.array([0.0, 0.0, 1.0])
OUTER_T, PLATE_T = 55.45, 4.2                                       # outer face at N.x = 55.45, plate 4.2 mm thick
C_OUT = np.array([-49.22, -37.84, 17.5]) + N * (OUTER_T - 55.1)     # window centre on the outer face = optical axis
FILL = (15.5, 15.5)                    # half sizes of the region that is filled solid (covers the old holes)
WINDOW = (6.75, 8.5)                   # half sizes of the new lens window: 13.5 (along E1) x 17 mm, clears the M12 lens barrel
SLOT_X, SLOT_Y, HOLE_R = (10.0, 10.5), (6.25, 6.75), 1.15            # hole-centre ranges (20-21 x 12.5-13.5 mm) and M2 clearance radius


def local_to_mount():
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = E1, E2, N, C_OUT
    return T


def prism(poly, z0, z1):
    m = trimesh.creation.extrude_polygon(poly, z1 - z0)
    m.apply_translation([0, 0, z0]); m.apply_transform(local_to_mount())
    return m


def main():
    base = trimesh.load(SRC, force="mesh")
    fill = prism(sbox(-FILL[0], -FILL[1], FILL[0], FILL[1]), -PLATE_T + 0.05, 0.0)
    part = trimesh.boolean.union([base, fill], engine="manifold")
    cuts = [prism(sbox(-WINDOW[0], -WINDOW[1], WINDOW[0], WINDOW[1]), -PLATE_T - 1, 1)]
    for sx in (1, -1):
        for sy in (1, -1):
            xs, ys = sorted(sx * np.array(SLOT_X)), sorted(sy * np.array(SLOT_Y))
            cuts.append(prism(sbox(xs[0], ys[0], xs[1], ys[1]).buffer(HOLE_R, resolution=24), -PLATE_T - 1, 1))
    part = trimesh.boolean.difference([part] + cuts, engine="manifold")
    assert part.is_watertight, "result is not watertight"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    part.export(OUT)
    # report the material left around the holes
    inner_edge = SLOT_X[0] - HOLE_R - WINDOW[0]
    print(f"wrote {OUT}  ({len(part.faces)} faces, {part.volume / 1000:.1f} cm3)")
    print(f"web between window and slots: {inner_edge:.2f} mm; plate thickness {PLATE_T} mm -> use M2 x 8 screws + M2 nuts")


if __name__ == "__main__":
    main()
