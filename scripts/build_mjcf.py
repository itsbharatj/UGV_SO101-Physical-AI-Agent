#!/usr/bin/env python3
"""Build mujoco/so101_rover.xml from mujoco/so101_rover_sim.urdf: free-floating rover on a floor, actuators,
both cameras and a gripper site.  Run after scripts/build_urdf.py:
    python3 scripts/build_mjcf.py
Actuators (ctrl vector order): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper  (position
targets in rad, LeRobot order), then left_up, left_down, right_up, right_down wheel speeds (rad/s). Middle wheels are passive.
"""
import os
import numpy as np
import mujoco
import yourdfpy
from scipy.spatial.transform import Rotation as R

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_URDF = os.path.join(PKG, "mujoco", "so101_rover_sim.urdf")
OUT = os.path.join(PKG, "mujoco", "so101_rover.xml")
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
WHEEL_JOINTS = ["left_up_wheel_link_joint", "left_down_wheel_link_joint", "right_up_wheel_link_joint", "right_down_wheel_link_joint"]
# (name, URDF optical frame, MuJoCo body it is fused into, vertical field of view in degrees)
CAMERAS = [("front_camera", "front_camera_optical_frame", "base_footprint", 54.0),     # OAK-D Lite RGB 69 x 54 deg
           ("wrist_camera", "wrist_camera_optical_frame", "arm_gripper_link", 105.0)]  # 160 deg fisheye, pinhole approx (120 deg H)


def main():
    urdf = yourdfpy.URDF.load(SIM_URDF, build_scene_graph=True, load_meshes=False)
    urdf.update_cfg({n: 0.0 for n in urdf.actuated_joint_names})

    spec = mujoco.MjSpec.from_file(SIM_URDF)
    spec.modelname = "so101_rover"
    spec.option.timestep = 0.001
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 800
    for jn in ARM_JOINTS:
        j = spec.joint(jn)
        j.damping[0] = 0.6; j.armature = 0.028; j.frictionloss = 0.05       # STS3215 values from the SO-101 MJCF
        a = spec.add_actuator(); a.name = jn; a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm = [20.0] + [0.0] * 9; a.biasprm = [0.0, -20.0, -1.0] + [0.0] * 7   # position servo kp=20, kv=1
        a.ctrlrange = list(j.range); a.ctrllimited = True
        a.forcerange = [-3.0, 3.0]; a.forcelimited = True                   # STS3215 stall ~2.9 Nm at 12 V
    for jn in WHEEL_JOINTS:
        j = spec.joint(jn); j.armature = 0.002
        a = spec.add_actuator(); a.name = jn.replace("_link_joint", "_motor"); a.target = jn; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm = [0.1] + [0.0] * 9; a.biasprm = [0.0, 0.0, -0.1] + [0.0] * 7      # velocity servo kv=0.1
        a.ctrlrange = [-32.0, 32.0]; a.ctrllimited = True
    # cameras: MuJoCo cameras look along -z with +y up, ROS optical frames along +z with +y down -> flip about x
    for name, frame, body, fovy in CAMERAS:
        if frame not in urdf.link_map:
            continue
        T = np.linalg.inv(urdf.get_transform(body, "base_footprint")) @ urdf.get_transform(frame, "base_footprint")
        Rm = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        q = R.from_matrix(Rm).as_quat()                                      # x y z w
        c = spec.body(body).add_camera(); c.name = name; c.pos = T[:3, 3]; c.quat = [q[3], q[0], q[1], q[2]]; c.fovy = fovy
    T = np.linalg.inv(urdf.get_transform("arm_gripper_link", "base_footprint")) @ urdf.get_transform("arm_gripper_frame_link", "base_footprint")
    q = R.from_matrix(T[:3, :3]).as_quat()
    s = spec.body("arm_gripper_link").add_site(); s.name = "gripper_frame"; s.pos = T[:3, 3]; s.quat = [q[3], q[0], q[1], q[2]]
    s.size = [0.004, 0, 0]; s.group = 3
    # world: floor, lights, free-floating rover
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], name="floor", rgba=[0.55, 0.55, 0.58, 1])
    for pos, d in [([1.0, -1.5, 2.0], [-0.4, 0.6, -0.8]), ([-1.5, 1.0, 1.5], [0.6, -0.4, -0.6])]:
        L = spec.worldbody.add_light(); L.pos = pos; L.dir = d; L.diffuse = [0.7, 0.7, 0.7]; L.specular = [0.2, 0.2, 0.2]; L.castshadow = False
    root = spec.body("base_footprint"); root.add_freejoint(); root.pos = [0, 0, 0.0005]
    spec.compile()
    with open(OUT, "w") as f:
        f.write(spec.to_xml())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
