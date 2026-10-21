#!/usr/bin/env python3
"""Validate the generated model: URDF parse + frame checks (yourdfpy), MuJoCo load + 3 s simulation,
MJCF export with actuators, and a rendered picture.  Run from the package root:
    python3 tests/check_model.py
"""
import os, sys, math, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_URDF = os.path.join(PKG, "mujoco", "so101_rover_sim.urdf")
ROS_URDF = os.path.join(PKG, "urdf", "so101_rover.urdf")
ok = True

# ---------------------------------------------------------------- 1. URDF structure + forward kinematics
import yourdfpy
robot = yourdfpy.URDF.load(SIM_URDF, build_scene_graph=True, load_meshes=True, load_collision_meshes=False)
print(f"links={len(robot.link_map)} joints={len(robot.joint_map)} actuated={robot.actuated_joint_names}")
missing = [j for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"] if j not in robot.joint_map]
assert not missing, f"missing arm joints {missing}"
cfg = {n: 0.0 for n in robot.actuated_joint_names}
robot.update_cfg(cfg)
T = lambda frame: robot.get_transform(frame, "base_footprint")
for f in ["base_link", "base_lidar_link", "3d_camera_link", "arm_mount_plate_link", "arm_base_link", "arm_shoulder_link",
          "arm_gripper_frame_link", "wrist_camera_optical_frame"]:
    if f in robot.link_map:
        p = T(f)[:3, 3]
        print(f"  {f:28s} x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f}  (m, from ground, rover frame)")
z_base = T("arm_base_link")[2, 3] + (-0.0024)
print(f"  SO-101 base underside height above ground: {z_base:.4f} m  (tray 0.120 + spacer + 0.010 plate)")
lidar_to_yaw = T("base_lidar_link")[0, 3] - T("arm_shoulder_link")[0, 3]
print(f"  LiDAR centre to arm yaw axis: {lidar_to_yaw*1000:.1f} mm  <- measure this on the real rover to confirm placement")
# the camera should look forward (+x rover) and down at tilt 45 deg when the arm is in its zero pose
if "wrist_camera_optical_frame" in robot.link_map:
    zc = T("wrist_camera_optical_frame")[:3, 2]
    print(f"  wrist camera optical axis (world): {np.round(zc, 3)}")
g = T("arm_gripper_frame_link")[:3, 3]; s = T("arm_shoulder_link")[:3, 3]
print(f"  arm zero-pose reach direction: {'rover FRONT (+x)' if g[0] > s[0] else 'rover REAR (-x)'}  (gripper {g[0]-s[0]:+.3f} m from the yaw axis)")

# mesh files exist?
for l in robot.link_map.values():
    for v in l.visuals + l.collisions:
        if v.geometry.mesh is not None:
            fn = os.path.join(os.path.dirname(SIM_URDF), v.geometry.mesh.filename)
            if not os.path.exists(fn):
                print("MISSING MESH", fn); ok = False

# ---------------------------------------------------------------- 2. MuJoCo: load, add actuators, simulate, export MJCF
import mujoco
spec = mujoco.MjSpec.from_file(SIM_URDF)
spec.modelname = "so101_rover"
arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
wheel_joints = ["left_up_wheel_link_joint", "left_down_wheel_link_joint", "right_up_wheel_link_joint", "right_down_wheel_link_joint"]
for jn in arm_joints:
    a = spec.add_actuator(); a.name = jn; a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.gainprm = [20.0] + [0.0] * 9; a.biasprm = [0.0, -20.0, -1.0] + [0.0] * 7   # position servo kp=20, kv=1
    a.ctrlrange = [-3.2, 3.2]; a.ctrllimited = True; a.forcerange = [-3.0, 3.0]; a.forcelimited = True
for jn in wheel_joints:
    a = spec.add_actuator(); a.name = jn.replace("_link_joint", "_motor"); a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE; a.gainprm = [0.1] + [0.0] * 9; a.biasprm = [0.0, 0.0, -0.1] + [0.0] * 7      # velocity servo kv=0.1 (max ~0.6 Nm)
    a.ctrlrange = [-20, 20]; a.ctrllimited = True
for j in spec.joints:
    if j.name in arm_joints:
        j.damping[0] = 0.6; j.armature = 0.028; j.frictionloss = 0.05
    elif j.name in wheel_joints:
        j.damping[0] = 0.02; j.armature = 0.002
# free-floating rover on a ground plane
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], name="floor")
root_body = spec.body("base_footprint")
root_body.add_freejoint()
root_body.pos = [0, 0, 0.002]
spec.option.timestep = 0.001
spec.visual.global_.offwidth = 1280; spec.visual.global_.offheight = 800
for (pos, d) in [([1.0, -1.5, 2.0], [-0.4, 0.6, -0.8]), ([-1.5, 1.0, 1.5], [0.6, -0.4, -0.6])]:
    L = spec.worldbody.add_light(); L.pos = pos; L.dir = d; L.diffuse = [0.7, 0.7, 0.7]; L.specular = [0.2, 0.2, 0.2]; L.castshadow = False
model = spec.compile()
data = mujoco.MjData(model)
print(f"MuJoCo: nbody={model.nbody} njnt={model.njnt} nu={model.nu} ngeom={model.ngeom} mass_total={sum(model.body_mass):.3f} kg")
mujoco.mj_forward(model, data)
# hold the arm at zero, drive forward at 6 rad/s wheels for 3 s
for i in range(3000):
    data.ctrl[:6] = 0.0
    data.ctrl[6:] = 6.0 if i > 500 else 0.0
    mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        print("NaN in simulation"); ok = False; break
base_pos = data.qpos[:3]
arm_q = np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in arm_joints])
print(f"  after 3 s: rover position {np.round(base_pos, 3)}, arm joint error max {np.abs(arm_q).max():.3f} rad")
print(f"  contacts at end: {[ (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)) for c in data.contact[:data.ncon]][:8]}")
if base_pos[2] < 0.0 or base_pos[2] > 0.1:
    print("  rover height looks wrong"); ok = False
if abs(base_pos[0]) < 0.05 or abs(base_pos[0]) > 1.5:
    print("  rover did not drive forward"); ok = False
# export MJCF next to the URDF (mesh paths relative to mujoco/)
mjcf_path = os.path.join(PKG, "mujoco", "so101_rover.xml")
with open(mjcf_path, "w") as f:
    f.write(spec.to_xml())
print("wrote", mjcf_path)
m2 = mujoco.MjModel.from_xml_path(mjcf_path); print("  MJCF reloads: nbody", m2.nbody)

# ---------------------------------------------------------------- 3. picture
try:
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 720, 1080)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.05, 0.0, 0.18]; cam.distance = 0.85; cam.azimuth = 140; cam.elevation = -20
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[2] = 1; opt.geomgroup[1] = 1
    r.update_scene(data, cam, opt)
    from PIL import Image
    Image.fromarray(r.render()).save(os.path.join(PKG, "tests", "render_mujoco.png"))
    cam.azimuth = 40; cam.elevation = -25; r.update_scene(data, cam, opt)
    Image.fromarray(r.render()).save(os.path.join(PKG, "tests", "render_mujoco_front.png"))
    print("wrote tests/render_mujoco.png")
except Exception as e:
    print("render skipped:", e)

print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
sys.exit(0 if ok else 1)
