#!/usr/bin/env python3
"""Wrist camera mounts for a 25 x 24 mm Raspberry-Pi-style camera board (the Waveshare 160 deg camera from the UGV
Rover's pan-tilt), derived from TheRobotStudio's official SO-101 wrist camera mount
(source_models/wrist_camera_mount_so101_v1.stl, Apache-2.0; built for a 32 x 32 mm UVC board with holes on 27 x 27 mm).

The wrist attachment and the arm bar are kept.  The camera plate is rebuilt for the Pi board:
  * 13.5 x 17 mm lens window on the official optical axis
  * 4 slotted M2 holes: 20-21 mm apart one way, 12.5-13.5 mm the other (genuine and clone boards)
and can be moved further out along the arm (--extend, mm), shifted sideways away from the gripper body (--offset, mm,
the plate is widened toward the bar) and yawed about the bar's edge (--tilt, deg; negative turns the camera toward the
gripper, "auto" centres the fingertip in the camera's horizontal field).

Every variant writes meshes/mount/wrist_cam_mount_pi_board<suffix>.stl plus an entry in meshes/mount/wrist_mounts.json
holding the camera pose for scripts/build_urdf.py.

    python3 scripts/make_wrist_mount.py                       # the three standard variants
    python3 scripts/make_wrist_mount.py --extend 30 --tilt 10 --suffix _custom
"""
import argparse, json, math, os
import numpy as np
import trimesh
from shapely.geometry import box as sbox

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PKG, "source_models", "wrist_camera_mount_so101_v1.stl")
OUT_DIR = os.path.join(PKG, "meshes", "mount")
REGISTRY = os.path.join(OUT_DIR, "wrist_mounts.json")

# ---- official mount geometry (mm, mesh frame), measured from the STL
N = np.array([-0.423, -0.906, 0.0]); N /= np.linalg.norm(N)        # camera plate outward normal; camera looks along -N
E1 = np.array([-0.906, 0.423, 0.0]); E1 /= np.linalg.norm(E1)      # across the plate, away from the arm bar
E2 = np.array([0.0, 0.0, 1.0])                                     # along the plate (0..35 mm)
PLATE_T0, PLATE_T1 = 51.3, 55.45       # plate inner / outer face, as N-coordinate
PLATE_U = (9.0, 45.2)                  # plate extent along E1
BAR_U = (7.4, 11.7)                    # arm bar extent along E1 (bar runs along N into the plate)
Z = (0.0, 35.0)
PLATE_T = PLATE_T1 - PLATE_T0
CAM_U, CAM_Z = 28.59, 17.5             # optical axis crosses the plate here (official camera origin)
CAM_ORIGIN = np.array([-49.41, -38.13, 17.5])                      # official wrist_camera_joint origin (mount frame, mm)
# ---- camera plate for the Pi board
WINDOW = (6.75, 8.5)                                    # half sizes: 13.5 (E1) x 17 (E2) mm, clears the M12 lens barrel
SLOT_U, SLOT_Z, HOLE_R = (10.0, 10.5), (6.25, 6.75), 1.15   # slot centre ranges (20-21 x 12.5-13.5 mm), M2 clearance


def frame(origin, x, y, z):
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def plate_solid(T0, T1, offset=0.0):
    """Camera plate (with window and slots) between N = T0 and N = T1; the camera axis shifted by `offset` along E1
    and the plate widened back to the arm bar so it stays connected."""
    cu = CAM_U + offset
    def prism(poly, lo, hi):
        p = trimesh.creation.extrude_polygon(poly, hi - lo); p.apply_translation([0, 0, lo])
        p.apply_transform(frame(N * T0 + E1 * cu + E2 * CAM_Z, E1, E2, N))
        return p
    th = T1 - T0
    plate = prism(sbox(min(PLATE_U[0], BAR_U[0]) - cu, Z[0] - CAM_Z, PLATE_U[1] + offset - cu, Z[1] - CAM_Z), 0, th)
    cuts = [prism(sbox(-WINDOW[0], -WINDOW[1], WINDOW[0], WINDOW[1]), -1, th + 1)]
    for su in (1, -1):
        for sz in (1, -1):
            us, zs = sorted(su * np.array(SLOT_U)), sorted(sz * np.array(SLOT_Z))
            cuts.append(prism(sbox(us[0], zs[0], us[1], zs[1]).buffer(HOLE_R, resolution=24), -1, th + 1))
    return trimesh.boolean.difference([plate] + cuts, engine="manifold")


TIP_FROM_CAM = (126.1, -22.3)          # fingertip from the official camera: along -N, along E1 (mm, from the URDF at zero pose)


def make(extend=0.0, tilt_deg=0.0, suffix="", offset=0.0):
    if tilt_deg == "auto":   # yaw that puts the fingertip on the camera's E1-N centre line
        tilt_deg = math.degrees(math.atan2(TIP_FROM_CAM[1] - offset, TIP_FROM_CAM[0] + extend))
    base = trimesh.load(SRC, force="mesh")
    CUT = PLATE_T0 - 0.2                                 # cutting exactly on the plate face leaves an open mesh
    body = base.slice_plane(plane_origin=N * CUT, plane_normal=-N, cap=True)      # wrist side only
    body.merge_vertices(); body.fix_normals()
    assert body.is_volume, "cut body is not a closed volume"
    parts = [body]
    T0 = PLATE_T0 + extend
    plate_start = (CUT if extend == 0 else T0)
    if extend > 0:   # extend the arm bar along N
        bar = trimesh.creation.box((BAR_U[1] - BAR_U[0], Z[1] - Z[0], extend + 0.3))
        bar.apply_transform(frame(N * (CUT - 0.05 + (extend + 0.3) / 2) + E1 * (BAR_U[0] + BAR_U[1]) / 2 + E2 * (Z[0] + Z[1]) / 2, E1, E2, N))
        parts.append(bar)
    plate = plate_solid(plate_start, T0 + PLATE_T, offset)
    pivot = N * T0 + E1 * BAR_U[1]                       # tilt about the plate's bar-side edge, along E2
    Rt = trimesh.transformations.rotation_matrix(math.radians(tilt_deg), E2, pivot)
    plate.apply_transform(Rt)
    if tilt_deg != 0:     # fill the wedge the tilt opens between bar and plate
        wedge = trimesh.creation.box((BAR_U[1] - BAR_U[0] + 0.1, Z[1] - Z[0], PLATE_T + 2))
        wedge.apply_transform(frame(N * (T0 + PLATE_T / 2) + E1 * (BAR_U[0] + BAR_U[1]) / 2 + E2 * 17.5, E1, E2, N))
        parts.append(wedge)
    part = trimesh.boolean.union(parts + [plate], engine="manifold")
    assert part.is_watertight, "variant is not watertight"
    name = f"wrist_cam_mount_pi_board{suffix}.stl"
    part.export(os.path.join(OUT_DIR, name))
    Tcam = np.eye(4); Tcam[:3, 3] = CAM_ORIGIN + N * extend + E1 * offset
    Tcam = Rt @ Tcam                                     # camera follows the plate
    reg = json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}
    reg[suffix.lstrip("_") or "default"] = {"mesh": name, "extend_mm": extend, "offset_mm": offset, "tilt_deg": round(tilt_deg, 1),
                                           "camera_xyz_mm": Tcam[:3, 3].round(3).tolist(),
                                           "camera_rot_rows": Tcam[:3, :3].round(6).tolist(),
                                           "volume_cm3": round(part.volume / 1000, 1)}
    json.dump(reg, open(REGISTRY, "w"), indent=2)
    print(f"wrote {name}: extend {extend} mm, offset {offset} mm, tilt {tilt_deg:.1f} deg, {part.volume / 1000:.1f} cm3, "
          f"width from wrist plate {np.ptp(part.vertices @ E1):.0f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", type=float); ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--tilt", default="auto", help="degrees, or auto"); ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.extend is None:
        make(0.0, 0.0, "")                                  # official geometry, Pi-board plate
        make(0.0, "auto", "_side20", offset=20.0)           # camera 20 mm further from the gripper body, aimed at the tips
        make(10.0, "auto", "_side30_ext10", offset=30.0)    # 30 mm sideways and 10 mm back: more table, body at the edge
    else:
        make(a.extend, a.tilt if a.tilt == "auto" else float(a.tilt), a.suffix, offset=a.offset)
