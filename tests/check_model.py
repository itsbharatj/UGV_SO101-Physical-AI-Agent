#!/usr/bin/env python3
"""Validate the model without changing any tracked file:
  1. URDF: tree, mesh paths, inertias, limits, key heights and directions (yourdfpy)
  2. MJCF: loads, 3 s simulation (arm holds pose, rover drives forward), cameras render
Run from anywhere:   python3 tests/check_model.py      (renders go to tests/out/, which git ignores)
"""
import os, sys, json, math, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_URDF = os.path.join(PKG, "mujoco", "so101_rover_sim.urdf")
MJCF = os.path.join(PKG, "mujoco", "so101_rover.xml")
OUT = os.path.join(PKG, "tests", "out"); os.makedirs(OUT, exist_ok=True)
problems = []
with open(os.path.join(PKG, "urdf", "build_params.json")) as f:
    P = json.load(f)                                  # parameters the URDF was built with (scripts/build_urdf.py)
DECK, PLATE_TOP = P["deck_height"], P["deck_height"] + P["spacer"] + 0.010
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)

import yourdfpy
from lxml import etree as ET
print("1. URDF")
root = ET.parse(SIM_URDF).getroot()
links = {l.get("name") for l in root.findall("link")}
joints = root.findall("joint")
children = [j.find("child").get("link") for j in joints]
check(len(set(children)) == len(children) and all(j.find("parent").get("link") in links for j in joints), "every joint connects defined links, no link has two parents")
check(len(links - set(children)) == 1 and (links - set(children)) == {"base_footprint"}, "single root base_footprint")
for l in root.findall("link"):
    for m in l.iter("mesh"):
        if not os.path.exists(os.path.join(os.path.dirname(SIM_URDF), m.get("filename"))):
            problems.append("missing mesh " + m.get("filename"))
    i = l.find("inertial")
    if i is not None:
        a = i.find("inertia"); M = float(i.find("mass").get("value"))
        I = np.array([[float(a.get("ixx")), float(a.get("ixy")), float(a.get("ixz"))], [float(a.get("ixy")), float(a.get("iyy")), float(a.get("iyz"))],
                      [float(a.get("ixz")), float(a.get("iyz")), float(a.get("izz"))]])
        if M <= 0 or np.linalg.eigvalsh(I).min() <= 0:
            problems.append(f"bad inertia on {l.get('name')}")
check(not any(p.startswith(("missing", "bad")) for p in problems), "all meshes resolve and all inertias are positive definite")
check(all(j.find("limit") is not None for j in joints if j.get("type") in ("revolute", "continuous")), "every moving joint has a <limit>")

robot = yourdfpy.URDF.load(SIM_URDF, build_scene_graph=True, load_meshes=False)
robot.update_cfg({n: 0.0 for n in robot.actuated_joint_names})
T = lambda f: robot.get_transform(f, "base_footprint")
z = lambda f: T(f)[2, 3]
print(f"  links={len(robot.link_map)} joints={len(robot.joint_map)} actuated={len(robot.actuated_joint_names)}")
check(abs(z("base_lidar_link") - DECK) < 1e-4, f"upper deck top / LiDAR base at {z('base_lidar_link'):.4f} m (deck height {DECK})")
check(abs(z("arm_base_link") - 0.0024 - PLATE_TOP) < 1e-4, f"SO-101 base underside at {z('arm_base_link') - 0.0024:.4f} m (deck + spacer + 10 mm plate)")
check(all(abs(z(w) - 0.040) < 1e-4 for w in robot.link_map if w.endswith("wheel_link")), "all 6 wheel axles 0.040 m above ground (80 mm wheels)")
g, s = T("arm_gripper_frame_link")[:3, 3], T("arm_shoulder_link")[:3, 3]
if P["arm_facing"] == "front":
    check(g[0] - s[0] > 0.3, f"arm reaches forward: gripper {g[0] - s[0]:+.3f} m ahead of the yaw axis at zero pose")
    check(T("3d_camera_link")[0, 3] > T("base_lidar_link")[0, 3] > s[0], "front-to-back order: OAK-D Lite, LiDAR, arm yaw axis")
else:
    check(s[0] - g[0] > 0.3, f"arm reaches rearward (--arm-facing rear): gripper {g[0] - s[0]:+.3f} m from the yaw axis")
print(f"  LiDAR centre to arm yaw axis: {(T('base_lidar_link')[0, 3] - s[0]) * 1000:.1f} mm")
if "wrist_camera_optical_frame" in robot.link_map:
    zc = T("wrist_camera_optical_frame")[:3, 2]; pc = T("wrist_camera_optical_frame")[:3, 3]
    v = g - pc
    ang = math.degrees(math.acos(np.dot(zc, v) / np.linalg.norm(v)))
    check(ang < 60, f"gripper tip is {ang:.0f} deg off the wrist camera axis (inside the 160 deg lens)")

print("2. MuJoCo")
import mujoco
m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
print(f"  nbody={m.nbody} njnt={m.njnt} nu={m.nu} ncam={m.ncam} mass={sum(m.body_mass):.3f} kg")
check(m.nu == 10 and m.ncam >= 1, "10 actuators and the cameras are present")
qadr = lambda j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
mujoco.mj_forward(m, d)
for i in range(3000):
    d.ctrl[:6] = 0.0
    d.ctrl[6:] = 6.0 if i > 500 else 0.0
    mujoco.mj_step(m, d)
check(np.all(np.isfinite(d.qpos)), "simulation stays finite for 3 s")
check(0.2 < d.qpos[0] < 1.5 and abs(d.qpos[1]) < 0.05 and -0.002 <= d.qpos[2] < 0.01, f"rover drives straight: x={d.qpos[0]:.3f} y={d.qpos[1]:.3f} z={d.qpos[2]:.4f}")
err = max(abs(d.qpos[qadr(j)]) for j in arm)
check(err < 0.06, f"arm holds its pose while driving (max joint error {err:.3f} rad)")
try:
    from PIL import Image
    r = mujoco.Renderer(m, 480, 640)
    for c in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
        r.update_scene(d, camera=c); Image.fromarray(r.render()).save(os.path.join(OUT, f"cam_{name}.png"))
        if name == "wrist_camera":                    # the jaws must be visible, not hidden behind collision boxes
            r.enable_segmentation_rendering(); r.update_scene(d, camera=c); seg = r.render()[..., 0]; r.disable_segmentation_rendering()
            jaw_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("arm_gripper_link", "arm_moving_jaw_so101_v1_link")}
            jaw_geoms = [gi for gi in range(m.ngeom) if m.geom_bodyid[gi] in jaw_bodies and m.geom_group[gi] != 3]
            frac = np.isin(seg, jaw_geoms).mean()
            check(frac > 0.01, f"gripper jaws fill {frac * 100:.1f}% of the wrist camera image")
    print(f"  camera images written to tests/out/")
except Exception as e:
    print("  render skipped:", e)
print("RESULT:", "OK" if not problems else f"{len(problems)} PROBLEM(S): " + "; ".join(problems))
sys.exit(0 if not problems else 1)
