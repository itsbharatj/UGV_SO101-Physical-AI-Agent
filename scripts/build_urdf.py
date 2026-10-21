#!/usr/bin/env python3
"""Build the combined Waveshare UGV Rover + SO-101 follower URDF.

Sources (unchanged geometry, only renamed/re-pathed):
  * Waveshare ugv_ws  ->  src/ugv_main/ugv_description/urdf/ugv_rover.urdf   (rover, wheels, LiDAR, OAK)
  * TheRobotStudio SO-ARM100 -> Simulation/SO101/so101_new_calib.urdf        (arm, "new calibration")
Added here:
  * printed adapter plate + 4 spacers (SO101_UGV_Rover_Mount/FINAL_FILES)
  * SO-101 base placed on the plate
  * OAK-D Lite wrist tilt mount (SO101_OAK_Tilt_Mount_v3) with a settable tilt
  * optical frames for both cameras, <mujoco> extension block

Usage:
  python3 scripts/build_urdf.py                 # writes urdf/so101_rover.urdf (package://) and mujoco/so101_rover_sim.urdf (relative)
  python3 scripts/build_urdf.py --spacer 0.0 --tilt 45 --arm-battery
"""
import argparse, copy, math, os, sys
from lxml import etree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "source_models")  # original upstream files kept for reference

# ----------------------------------------------------------------------------- geometry constants (metres)
TRAY_TOP_Z = 0.040            # top sheet of the rover, in rover base_link frame (Waveshare URDF)
PLATE_T = 0.010               # printed adapter thickness
SO101_BASE_BOTTOM = -0.0024   # z of the SO-101 base underside in its own base_link frame (official URDF)
# Adapter plate position: Codex's rover-CAD X axis runs rearward, origin such that the plate spans X=-8..94 mm.
# Anchored on the LiDAR: plate front edge sits 4.815 mm behind the LiDAR housing (centre x=0.03981, r=0.018).
LIDAR_X, LIDAR_R, GAP = 0.0398145505519817, 0.018, 0.004815
PLATE_FRONT_X = LIDAR_X - LIDAR_R - GAP       # rover x of the plate's front edge (Codex X = -8 mm)
CODEX_X0 = PLATE_FRONT_X + 0.008              # rover x where Codex X = 0   ->  x_rover = CODEX_X0 - X_codex
# SO-101 base holes in its base_link frame: wide pair (front, under the yaw motor) at x=-0.0137, y=+-0.0317
SO101_WIDE_PAIR_X = -0.0137
# Codex adapter: wide pair at X=7.5 mm, narrow pair at X=77.275 mm; chassis bolts/spacers at X=4,40 Y=+-43
ARM_WIDE_PAIR_X_ROVER = CODEX_X0 - 0.0075
SPACER_XY = [(CODEX_X0 - 0.004, 0.043), (CODEX_X0 - 0.004, -0.043), (CODEX_X0 - 0.040, 0.043), (CODEX_X0 - 0.040, -0.043)]
# Wrist camera mount (SO101_OAK_Tilt_Mount_v3 design_meta.json), in SO-101 gripper_link frame, metres
HINGE_P = np.array([0.0, 0.09391541144431395, 0.010560720250484465])
THETA0_DEG = 37.5
LENS_P = np.array([0.0, 0.07601436790187365, -0.021080332830784126])


def fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def rpy_from_matrix(m):
    return R.from_matrix(m).as_euler("xyz")


def el(parent, tag, **attrs):
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    return e


def add_origin(parent, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    el(parent, "origin", xyz=fmt(xyz), rpy=fmt(rpy))


def add_mesh_visual(link, filename, xyz=(0, 0, 0), rpy=(0, 0, 0), scale=None, material=None, rgba=None, collision=True):
    for tag in (["visual", "collision"] if collision else ["visual"]):
        g = el(link, tag)
        add_origin(g, xyz, rpy)
        geo = el(g, "geometry")
        m = el(geo, "mesh", filename=filename)
        if scale:
            m.set("scale", fmt([scale] * 3))
        if tag == "visual" and (material or rgba):
            mat = el(g, "material", name=material or "")
            if rgba:
                el(mat, "color", rgba=fmt(rgba))


def add_box_visual(link, size, xyz=(0, 0, 0), rpy=(0, 0, 0), rgba=(0.3, 0.3, 0.3, 1), collision=True, collision_only=False):
    for tag in (["collision"] if collision_only else (["visual", "collision"] if collision else ["visual"])):
        g = el(link, tag)
        add_origin(g, xyz, rpy)
        geo = el(g, "geometry")
        el(geo, "box", size=fmt(size))
        if tag == "visual":
            el(el(g, "material", name=""), "color", rgba=fmt(rgba))


def add_inertial(link, mass, xyz, ixx, iyy, izz, ixy=0, ixz=0, iyz=0):
    i = el(link, "inertial")
    add_origin(i, xyz)
    el(i, "mass", value=f"{mass:.6g}")
    el(i, "inertia", ixx=f"{ixx:.6g}", ixy=f"{ixy:.6g}", ixz=f"{ixz:.6g}", iyy=f"{iyy:.6g}", iyz=f"{iyz:.6g}", izz=f"{izz:.6g}")


def box_inertia(m, sx, sy, sz):
    return (m / 12 * (sy * sy + sz * sz), m / 12 * (sx * sx + sz * sz), m / 12 * (sx * sx + sy * sy))


def add_fixed_joint(robot, name, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    j = el(robot, "joint", name=name, type="fixed")
    add_origin(j, xyz, rpy)
    el(j, "parent", link=parent)
    el(j, "child", link=child)
    return j


def build(mesh_prefix, spacer, tilt_deg, arm_battery, wrist_camera, out_path, arm_facing="rear"):
    rover = ET.parse(os.path.join(SRC, "ugv_rover.urdf")).getroot()
    arm = ET.parse(os.path.join(SRC, "so101_new_calib.urdf")).getroot()

    robot = ET.Element("robot", name="so101_rover")
    robot.append(ET.Comment(" Generated by scripts/build_urdf.py. Rover: Waveshare ugv_description (ugv_rover). "
                            "Arm: TheRobotStudio SO-ARM100 so101_new_calib. Mount, spacers, wrist camera: this package. "))
    # materials
    for name, rgba in [("rover_grey", (0.55, 0.57, 0.6, 1)), ("printed_black", (0.12, 0.12, 0.12, 1)),
                       ("printed_white", (0.9, 0.9, 0.9, 1)), ("so101_print", (0.85, 0.85, 0.88, 1)),
                       ("sts3215", (0.1, 0.1, 0.1, 1)), ("camera_black", (0.15, 0.16, 0.18, 1)),
                       ("battery_black", (0.05, 0.05, 0.05, 1))]:
        el(el(robot, "material", name=name), "color", rgba=fmt(rgba))

    # ---------------- rover: copy links/joints except the pan-tilt subtree
    drop = {"pt_base_link", "pt_link1", "pt_link2", "pt_camera_link"}
    for node in rover:
        if node.tag not in ("link", "joint"):
            continue
        if node.tag == "link" and node.get("name") in drop:
            continue
        if node.tag == "joint" and (node.find("child").get("link") in drop or node.find("parent").get("link") in drop):
            continue
        n = copy.deepcopy(node)
        for m in n.iter("mesh"):
            m.set("filename", f"{mesh_prefix}/rover/{os.path.basename(m.get('filename'))}")
        for mat in n.iter("material"):
            if mat.get("name") == "":
                mat.set("name", "rover_grey")
                for c in list(mat):
                    mat.remove(c)
        if n.tag == "link":
            name = n.get("name")
            if name == "base_link":
                for c in n.findall("collision"):
                    n.remove(c)
                add_box_visual(n, (0.245, 0.230, 0.085), xyz=(0, 0, -0.0025), collision_only=True)   # bottom 0.035 m above ground
            elif name.endswith("wheel_link"):
                for c in n.findall("collision"):
                    n.remove(c)
                c = el(n, "collision"); add_origin(c, (0, 0, 0), (0, 0, 0))
                el(el(c, "geometry"), "cylinder", radius="0.04", length="0.0426")
        robot.append(n)
    # front camera optical frame (ROS convention: z forward, x right, y down)
    cam_opt = el(robot, "link", name="front_camera_optical_frame")
    add_fixed_joint(robot, "front_camera_optical_joint", "3d_camera_link", "front_camera_optical_frame",
                    xyz=(0.017, 0, 0), rpy=(-math.pi / 2, 0, -math.pi / 2))

    # ---------------- adapter plate + spacers (one fixed link)
    plate = el(robot, "link", name="arm_mount_plate_link")
    plate_mass = 0.105
    ixx, iyy, izz = box_inertia(plate_mass, 0.102, 0.128, 0.010)
    add_inertial(plate, plate_mass, (0, 0, 0.005), ixx, iyy, izz)
    # Codex frame -> rover frame: rotate 180 deg about z, shift so Codex X=0 lands on CODEX_X0 (relative to the plate link
    # which we place at x=CODEX_X0, z = tray top + spacer).  Plate STL is in mm.
    add_mesh_visual(plate, f"{mesh_prefix}/mount/rover_adapter_plate.stl", xyz=(0, 0, 0), rpy=(0, 0, math.pi),
                    scale=0.001, material="printed_black", collision=False)
    add_box_visual(plate, (0.102, 0.128, 0.010), xyz=(-0.043, 0, 0.005), collision_only=True)
    for (x, y) in SPACER_XY:
        if spacer > 0:
            add_mesh_visual(plate, f"{mesh_prefix}/mount/rover_adapter_spacer.stl", xyz=(x - CODEX_X0, y, -spacer),
                            scale=0.001, material="printed_black", collision=False)
    add_fixed_joint(robot, "arm_mount_plate_joint", "base_link", "arm_mount_plate_link",
                    xyz=(CODEX_X0, 0, TRAY_TOP_Z + spacer))

    # ---------------- SO-101 arm, links prefixed arm_, joint names kept (LeRobot names)
    def arm_name(n):
        return "arm_" + n
    for node in arm:
        if node.tag == "material":
            continue
        if node.tag not in ("link", "joint"):
            continue
        n = copy.deepcopy(node)
        if n.tag == "link":
            n.set("name", arm_name(n.get("name")))
        else:
            n.find("parent").set("link", arm_name(n.find("parent").get("link")))
            n.find("child").set("link", arm_name(n.find("child").get("link")))
        for m in n.iter("mesh"):
            m.set("filename", f"{mesh_prefix}/so101/{os.path.basename(m.get('filename'))}")
        for mat in n.iter("material"):
            if mat.get("name") == "3d_printed":
                mat.set("name", "so101_print")
        robot.append(n)
    # base of the arm: wide hole pair over the plate's wide pair, arm reaches toward rover +x (yaw 180 deg)
    arm_z = TRAY_TOP_Z + spacer + PLATE_T - SO101_BASE_BOTTOM
    if arm_facing == "rear":
        # AS BUILT with the revision-A plate: the wide hole pair (motor-holder end of the SO-101 base) sits toward the
        # LiDAR, so the arm's zero-yaw reach direction points to the rover's REAR.  x_rover = arm_x - x_so101 (180 deg yaw)
        arm_x = ARM_WIDE_PAIR_X_ROVER + SO101_WIDE_PAIR_X
        add_fixed_joint(robot, "arm_mount_joint", "base_link", "arm_base_link", xyz=(arm_x, 0, arm_z), rpy=(0, 0, math.pi))
    else:
        # MIRRORED plate (wide pair at Codex X=77.275, narrow pair at X=7.5): arm reaches toward the rover FRONT.
        # NOTE: the plate mesh shown is still revision A; print a mirrored plate before building this variant.
        arm_x = (CODEX_X0 - 0.077275) - SO101_WIDE_PAIR_X
        add_fixed_joint(robot, "arm_mount_joint", "base_link", "arm_base_link", xyz=(arm_x, 0, arm_z), rpy=(0, 0, 0))

    # ---------------- wrist camera mount (OAK-D Lite tilt mount v3), in arm_gripper_link frame
    if wrist_camera:
        base = el(robot, "link", name="wrist_cam_mount_base_link")
        ixx, iyy, izz = box_inertia(0.020, 0.035, 0.087, 0.054)
        add_inertial(base, 0.020, (0.0, 0.065, 0.0), ixx, iyy, izz)
        add_mesh_visual(base, f"{mesh_prefix}/mount/wrist_cam_mount_base.stl", scale=0.001, material="printed_white", collision=False)
        add_box_visual(base, (0.012, 0.025, 0.054), xyz=(-0.006, 0.0395, 0.0), collision_only=True)
        add_fixed_joint(robot, "wrist_cam_mount_base_joint", "arm_gripper_link", "wrist_cam_mount_base_link")

        # cradle + camera rotate about the hinge axis (x) through HINGE_P by -(tilt - 37.5) deg
        a = -math.radians(tilt_deg - THETA0_DEG)
        Rm = R.from_euler("x", a).as_matrix()
        t = HINGE_P - Rm @ HINGE_P
        cradle = el(robot, "link", name="wrist_cam_cradle_link")
        m_cradle, m_cam = 0.015, 0.061
        ixx, iyy, izz = box_inertia(m_cradle + m_cam, 0.091, 0.030, 0.040)
        add_inertial(cradle, m_cradle + m_cam, (0.0, 0.082, -0.005), ixx, iyy, izz)
        add_mesh_visual(cradle, f"{mesh_prefix}/mount/wrist_cam_mount_cradle.stl", scale=0.001, material="printed_white", collision=False)
        add_mesh_visual(cradle, f"{mesh_prefix}/mount/oak_d_lite_wrist.stl", scale=0.001, material="camera_black", collision=True)
        add_fixed_joint(robot, "wrist_cam_tilt_joint", "wrist_cam_mount_base_link", "wrist_cam_cradle_link", xyz=t, rpy=(a, 0, 0))
        # optical frame at the RGB lens: Xc=-X, Yc=(0,cos,-sin), Zc=(0,-sin,-cos) at theta0 (cradle frame)
        th = math.radians(THETA0_DEG)
        Rc = np.column_stack([[-1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, -math.sin(th), -math.cos(th)]])
        el(robot, "link", name="wrist_camera_optical_frame")
        add_fixed_joint(robot, "wrist_camera_optical_joint", "wrist_cam_cradle_link", "wrist_camera_optical_frame",
                        xyz=LENS_P, rpy=rpy_from_matrix(Rc))

    # ---------------- optional arm battery (3S 26700 pack, 80x27x72 mm, 0.3 kg) strapped beside the plate
    if arm_battery:
        b = el(robot, "link", name="arm_battery_link")
        ixx, iyy, izz = box_inertia(0.30, 0.080, 0.027, 0.072)
        add_inertial(b, 0.30, (0, 0, 0.036), ixx, iyy, izz)
        add_box_visual(b, (0.080, 0.027, 0.072), xyz=(0, 0, 0.036), rgba=(0.05, 0.05, 0.05, 1))
        add_fixed_joint(robot, "arm_battery_joint", "base_link", "arm_battery_link", xyz=(CODEX_X0 - 0.045, 0.082, TRAY_TOP_Z))

    # ---------------- MuJoCo extension (ignored by ROS / Isaac)
    mj = el(robot, "mujoco")
    el(mj, "compiler", meshdir=".", balanceinertia="true", discardvisual="false", strippath="false")

    tree = ET.ElementTree(robot)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacer", type=float, default=0.006, help="spacer height under the plate, m (0 if bolted flat)")
    ap.add_argument("--tilt", type=float, default=45.0, help="wrist camera tilt, degrees (7.5 deg steps, 37.5 is the design reference)")
    ap.add_argument("--arm-battery", action="store_true", help="add the 3S Li-ion arm battery pack beside the plate")
    ap.add_argument("--no-wrist-camera", action="store_true")
    ap.add_argument("--arm-facing", choices=["rear", "front"], default="rear",
                    help="rear = as built with the revision-A plate (default); front = mirrored plate, arm reaches forward")
    a = ap.parse_args()
    ros = build("package://so101_rover_description/meshes", a.spacer, a.tilt, a.arm_battery, not a.no_wrist_camera,
                os.path.join(PKG, "urdf", "so101_rover.urdf"), a.arm_facing)
    sim = build("../meshes", a.spacer, a.tilt, a.arm_battery, not a.no_wrist_camera,
                os.path.join(PKG, "mujoco", "so101_rover_sim.urdf"), a.arm_facing)
    print("wrote", ros); print("wrote", sim)
