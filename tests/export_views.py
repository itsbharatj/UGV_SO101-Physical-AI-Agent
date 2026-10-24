#!/usr/bin/env python3
"""Export the assembled robot as STL/GLB and render reference views (poses chosen to match the build photos).
    python3 tests/export_views.py
"""
import os, sys, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PKG)
import yourdfpy, trimesh, mujoco
from PIL import Image

# arm pose like the photos: folded forward over the LiDAR, gripper pointing forward-down
POSE = {"shoulder_pan": 0.0, "shoulder_lift": -1.0, "elbow_flex": 1.0, "wrist_flex": 0.9, "wrist_roll": 0.0, "gripper": 0.3}

r = yourdfpy.URDF.load("mujoco/so101_rover_sim.urdf", build_scene_graph=True, load_meshes=True, load_collision_meshes=False)
cfg = {n: 0.0 for n in r.actuated_joint_names}; cfg.update(POSE); r.update_cfg(cfg)
sc = r.scene
T = lambda f: r.get_transform(f, "base_footprint")
tip = T("arm_gripper_frame_link")[:3, 3]
cam_axis = T("wrist_camera_optical_frame")[:3, 2] if "wrist_camera_optical_frame" in r.link_map else None
m = mujoco.MjModel.from_xml_path("mujoco/so101_rover.xml")          # load everything before touching export/
os.makedirs("export", exist_ok=True)
for f in os.listdir("export"):
    if f.startswith("view_"):
        os.remove(os.path.join("export", f))
sc.export("export/so101_rover_assembly.glb")
merged = trimesh.util.concatenate([sc.geometry[sc.graph[nd][1]].copy().apply_transform(sc.graph.get(nd)[0]) for nd in sc.graph.nodes_geometry])
merged.export("export/so101_rover_assembly.stl")
merged.copy().apply_scale(1000).export("export/so101_rover_assembly_mm.stl")
print("STL faces", len(merged.faces), "bounds (m)", np.round(merged.bounds, 3).tolist())
print("gripper tip in this pose (x fwd, z up):", np.round(tip, 3), " wrist cam looks along", None if cam_axis is None else np.round(cam_axis, 2))
d = mujoco.MjData(m)
for j, v in POSE.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
mujoco.mj_forward(m, d)
rr = mujoco.Renderer(m, 800, 1280); opt = mujoco.MjvOption()          # default groups 0-2: visuals + floor; collision is group 3
camv = mujoco.MjvCamera(); camv.type = mujoco.mjtCamera.mjCAMERA_FREE; camv.lookat[:] = [0.0, 0, 0.22]; camv.distance = 0.62
# MuJoCo free camera sits at lookat - distance * forward(azimuth): azimuth 180 looks back at the robot's front.
# Names and angles match reference_photos/: side_right.jpg, front.jpg, front_3q.jpg
views = [("side_right", 90, -5), ("front_3q", 145, -22), ("rear_3q", 320, -25), ("top", 0, -89), ("front", 180, -8), ("rear", 0, -8)]
imgs = []
for name, az, elv in views:
    camv.azimuth = az; camv.elevation = elv; rr.update_scene(d, camv, opt)
    im = Image.fromarray(rr.render()); im.save(f"export/view_{name}.png"); imgs.append(im)
w, h = imgs[0].size; sheet = Image.new("RGB", (3 * w, 2 * h))
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % 3) * w, (i // 3) * h))
sheet.save("export/so101_rover_views.png")
print("exports done:", sorted(os.listdir("export")))
