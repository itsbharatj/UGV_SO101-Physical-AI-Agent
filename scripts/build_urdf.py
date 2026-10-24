#!/usr/bin/env python3
"""Build the combined Waveshare UGV Rover + SO-101 follower URDF, matched to the robot as built.

Sources (geometry and inertia unchanged, only renamed / re-pathed):
  * Waveshare ugv_ws  ->  src/ugv_main/ugv_description/urdf/ugv_rover.urdf   (lower body, wheels, LiDAR, OAK-D Lite)
  * TheRobotStudio SO-ARM100 -> Simulation/SO101/so101_new_calib.urdf        (arm, "new calibration" zero = mid range)
Added here, from the physical build:
  * Jetson-kit upper deck (metal plate on 4 pillars) and the Jetson under it (schematic boxes, measured heights)
  * printed adapter plate (SO101_UGV_Rover_Mount/FINAL_FILES) bolted flat on the deck, triangle toward the REAR
  * SO-101 base on the plate, arm reaching FORWARD over the LiDAR
  * revision-2 fixed wrist bracket (SO101_OAK_RGB_v2) carrying the Waveshare 160 deg USB camera module
  * optical frames for both cameras, <mujoco> extension block

Measured on the robot (24 Oct 2026):  ground -> top of upper metal deck = 0.175 m,
                                      top of lower body -> underside of SO-101 base = 0.085 m.
Usage:
  python3 scripts/build_urdf.py                      # as built
  python3 scripts/build_urdf.py --deck-height 0.175 --spacer 0 --lidar-to-yaw 0.055 --arm-battery
"""
import argparse, copy, math, os
from lxml import etree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "source_models")

# ----------------------------------------------------------------------------- fixed geometry (metres)
BASE_LINK_Z = 0.08             # Waveshare: base_link is 0.08 above base_footprint (ground)
LOWER_BODY_TOP_MESH = 0.040    # top of Waveshare's lower-body mesh, base_link frame (0.12 above ground)
PLATE_T = 0.010                # printed adapter thickness
SO101_BASE_BOTTOM = -0.0024    # z of the SO-101 base underside in its own base_link frame (official URDF)
SO101_YAW_X = 0.0388353        # shoulder_pan axis x in the SO-101 base_link frame (official URDF)
SO101_WIDE_PAIR_X = -0.0137    # wide (63.5 mm) hole pair x in the SO-101 base_link frame; narrow pair at +0.0561
LIDAR_X = 0.0398145505519817   # LiDAR centre x, base_link frame (Waveshare URDF)
CAM_BELOW_DECK = 0.018         # front OAK-D Lite centre sits 18 mm below the deck (Waveshare URDF)
# printed plate, "Codex" frame: X=-8..94 (triangle at X=-8), Y=+-64; arm holes wide pair X=7.5, narrow X=77.275;
# chassis bolts X=4,40 Y=+-43.  As built the triangle points to the REAR, so Codex +X = rover +x (forward).
PLATE_X_MIN, PLATE_X_MAX = -0.008, 0.094
PLATE_WIDE_PAIR_X = 0.0075
PLATE_BOLTS = [(0.004, 0.043), (0.004, -0.043), (0.040, 0.043), (0.040, -0.043)]
# revision-2 wrist bracket (SO101_OAK_RGB_v2/pose.json), in the official Wrist_Roll_Follower STL frame (mm -> m)
V2_LENS = np.array([0.035, -0.060, 0.015])
V2_VIEW_DIR = np.array([-0.5414372919953874, 0.6725929093110402, 0.5044446819832802])   # toward the fixed-finger pad
V2_RIGHT = np.array([0.7789652826707469, 0.6270670525499513, 0.0])


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


def add_mesh(link, filename, xyz=(0, 0, 0), rpy=(0, 0, 0), scale=None, material=None, collision=False):
    for tag in (["visual", "collision"] if collision else ["visual"]):
        g = el(link, tag)
        add_origin(g, xyz, rpy)
        m = el(el(g, "geometry"), "mesh", filename=filename)
        if scale:
            m.set("scale", fmt([scale] * 3))
        if tag == "visual" and material:
            el(g, "material", name=material)


def add_box(link, size, xyz=(0, 0, 0), rpy=(0, 0, 0), material=None, visual=True, collision=False):
    for tag in (["visual"] if visual else []) + (["collision"] if collision else []):
        g = el(link, tag)
        add_origin(g, xyz, rpy)
        el(el(g, "geometry"), "box", size=fmt(size))
        if tag == "visual" and material:
            el(g, "material", name=material)


def add_cylinder(link, radius, length, xyz=(0, 0, 0), rpy=(0, 0, 0), material=None, visual=True, collision=False):
    for tag in (["visual"] if visual else []) + (["collision"] if collision else []):
        g = el(link, tag)
        add_origin(g, xyz, rpy)
        el(el(g, "geometry"), "cylinder", radius=f"{radius:.6g}", length=f"{length:.6g}")
        if tag == "visual" and material:
            el(g, "material", name=material)


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


def build(mesh_prefix, out_path, deck_height=0.175, spacer=0.0, lidar_to_yaw=0.055, arm_facing="front",
          wrist_mount="v2", arm_battery=False):
    deck_z = deck_height - BASE_LINK_Z                      # top of the upper metal deck, base_link frame
    plate_z = deck_z + spacer                               # underside of the printed plate
    arm_base_z = plate_z + PLATE_T - SO101_BASE_BOTTOM      # SO-101 base_link origin
    rover = ET.parse(os.path.join(SRC, "ugv_rover.urdf")).getroot()
    arm = ET.parse(os.path.join(SRC, "so101_new_calib.urdf")).getroot()

    robot = ET.Element("robot", name="so101_rover")
    robot.append(ET.Comment(" Generated by scripts/build_urdf.py - do not edit by hand. Rover: Waveshare ugv_description "
                            "(ugv_rover). Arm: TheRobotStudio SO-ARM100 so101_new_calib. Deck, plate, wrist camera: this package. "))
    for name, rgba in [("rover_grey", (0.16, 0.17, 0.19, 1)), ("deck_black", (0.12, 0.12, 0.13, 1)),
                       ("brass", (0.72, 0.58, 0.3, 1)), ("pcb_green", (0.1, 0.35, 0.2, 1)),
                       ("printed_black", (0.08, 0.08, 0.08, 1)), ("printed_white", (0.92, 0.92, 0.92, 1)),
                       ("so101_print", (0.93, 0.93, 0.95, 1)), ("sts3215", (0.1, 0.1, 0.1, 1)),
                       ("camera_black", (0.15, 0.16, 0.18, 1)), ("lidar_black", (0.1, 0.1, 0.12, 1)),
                       ("battery_black", (0.05, 0.05, 0.05, 1)), ("tyre_black", (0.08, 0.08, 0.08, 1))]:
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
            for c in list(mat):
                mat.remove(c)
            mat.set("name", "rover_grey")
        if n.tag == "joint" and n.get("name") in ("base_lidar_link_joint", "3d_camera_link_joint"):
            o = n.find("origin"); xyz = [float(v) for v in o.get("xyz").split()]
            xyz[2] = deck_z if n.get("name") == "base_lidar_link_joint" else deck_z - CAM_BELOW_DECK
            o.set("xyz", fmt(xyz))
        if n.tag == "link":
            name = n.get("name")
            if name == "base_link":
                for c in n.findall("collision"):
                    n.remove(c)
                add_box(n, (0.245, 0.230, 0.085), xyz=(0, 0, -0.0025), visual=False, collision=True)   # bottom 0.035 m above ground
                # Jetson-kit upper deck: metal plate on 4 brass pillars, Jetson Orin Nano dev kit under it (schematic)
                rise = deck_z - LOWER_BODY_TOP_MESH
                deck_len, deck_wid, deck_t = 0.205, 0.190, 0.003
                deck_cx = -0.0175                                        # x from -0.120 (rear lip) to +0.085 (over the OAK)
                add_box(n, (deck_len, deck_wid, deck_t), xyz=(deck_cx, 0, deck_z - deck_t / 2), material="deck_black")
                add_box(n, (deck_len, deck_wid, deck_t), xyz=(deck_cx, 0, deck_z - deck_t / 2), visual=False, collision=True)
                add_box(n, (0.004, deck_wid, 0.018), xyz=(-0.120 + 0.002, 0, deck_z - 0.009), material="deck_black")   # rear lip
                for (px, py) in [(0.060, 0.070), (0.060, -0.070), (-0.090, 0.070), (-0.090, -0.070)]:
                    add_cylinder(n, 0.0035, rise, xyz=(px, py, LOWER_BODY_TOP_MESH + rise / 2), material="brass")
                add_box(n, (0.103, 0.090, 0.012), xyz=(-0.015, 0, LOWER_BODY_TOP_MESH + 0.016), material="pcb_green")      # Jetson carrier
                add_box(n, (0.070, 0.045, 0.020), xyz=(-0.015, 0, LOWER_BODY_TOP_MESH + 0.032), material="rover_grey")     # Jetson module + heatsink
            elif name.endswith("wheel_link"):
                for c in n.findall("collision"):
                    n.remove(c)
                add_cylinder(n, 0.04, 0.0426, visual=False, collision=True)
                for mat in n.iter("material"):
                    mat.set("name", "tyre_black")
            elif name == "base_lidar_link":
                for mat in n.iter("material"):
                    mat.set("name", "lidar_black")
            elif name == "3d_camera_link":
                for mat in n.iter("material"):
                    mat.set("name", "camera_black")
        robot.append(n)
    # front camera optical frame (ROS convention: z forward, x right, y down), at the OAK's front face
    el(robot, "link", name="front_camera_optical_frame")
    add_fixed_joint(robot, "front_camera_optical_joint", "3d_camera_link", "front_camera_optical_frame",
                    xyz=(0.017, 0, 0), rpy=(-math.pi / 2, 0, -math.pi / 2))

    # ---------------- arm position: yaw axis a measured distance behind the LiDAR centre, on the rover centreline
    yaw_x = LIDAR_X - lidar_to_yaw
    if arm_facing == "front":
        # AS BUILT: plate triangle toward the rear, SO-101 motor-holder end (wide pair) at the rear, arm reaches forward
        arm_yaw = 0.0
        arm_x = yaw_x - SO101_YAW_X
        wide_pair_x = arm_x + SO101_WIDE_PAIR_X
        plate_x0 = wide_pair_x - PLATE_WIDE_PAIR_X          # rover x of Codex X=0; Codex +X = rover +x
        plate_sign = 1.0
    else:
        # Codex's drawing orientation: triangle toward the LiDAR; the arm then reaches toward the rear
        arm_yaw = math.pi
        arm_x = yaw_x + SO101_YAW_X
        wide_pair_x = arm_x - SO101_WIDE_PAIR_X
        plate_x0 = wide_pair_x + PLATE_WIDE_PAIR_X          # Codex +X = rover -x
        plate_sign = -1.0

    # ---------------- printed adapter plate (+ spacers if any), one fixed link at Codex X=0 on the plate underside
    plate = el(robot, "link", name="arm_mount_plate_link")
    plate_mass = 0.105
    ixx, iyy, izz = box_inertia(plate_mass, 0.102, 0.128, PLATE_T)
    add_inertial(plate, plate_mass, (plate_sign * 0.043, 0, PLATE_T / 2), ixx, iyy, izz)
    add_mesh(plate, f"{mesh_prefix}/mount/rover_adapter_plate.stl", rpy=(0, 0, 0 if plate_sign > 0 else math.pi),
             scale=0.001, material="printed_black")
    add_box(plate, (0.102, 0.128, PLATE_T), xyz=(plate_sign * 0.043, 0, PLATE_T / 2), visual=False, collision=True)
    if spacer > 0:
        for (bx, by) in PLATE_BOLTS:
            add_mesh(plate, f"{mesh_prefix}/mount/rover_adapter_spacer.stl", xyz=(plate_sign * bx, by, -spacer),
                     scale=0.001, material="printed_black")
    add_fixed_joint(robot, "arm_mount_plate_joint", "base_link", "arm_mount_plate_link", xyz=(plate_x0, 0, plate_z))

    # ---------------- SO-101 arm, links prefixed arm_, joint names kept (LeRobot names)
    def arm_name(n):
        return "arm_" + n
    for node in arm:
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
    add_fixed_joint(robot, "arm_mount_joint", "base_link", "arm_base_link", xyz=(arm_x, 0, arm_base_z), rpy=(0, 0, arm_yaw))

    # ---------------- wrist camera
    if wrist_mount == "v2":
        # bracket STL is in the official Wrist_Roll_Follower STL frame (mm); that frame -> gripper_link is Rx(pi) + (0,-0.218,0.95) mm
        br = el(robot, "link", name="wrist_cam_bracket_link")
        ixx, iyy, izz = box_inertia(0.016 + 0.015, 0.06, 0.05, 0.03)
        add_inertial(br, 0.031, (0.035, -0.060, 0.010), ixx, iyy, izz)
        add_mesh(br, f"{mesh_prefix}/mount/wrist_cam_bracket_v2.stl", scale=0.001, material="printed_white")
        add_box(br, (0.060, 0.030, 0.030), xyz=(0.035, -0.060, 0.012), visual=False, collision=True)
        add_fixed_joint(robot, "wrist_cam_bracket_joint", "arm_gripper_link", "wrist_cam_bracket_link",
                        xyz=(0, -0.000218, 0.000950), rpy=(math.pi, 0, 0))
        z_opt = V2_VIEW_DIR / np.linalg.norm(V2_VIEW_DIR)
        x_opt = V2_RIGHT - z_opt * np.dot(V2_RIGHT, z_opt); x_opt /= np.linalg.norm(x_opt); y_opt = np.cross(z_opt, x_opt)
        Rc = np.column_stack([x_opt, y_opt, z_opt])                      # optical frame: x right, y down, z forward
        cam = el(robot, "link", name="wrist_camera_link")
        add_inertial(cam, 0.015, (0, 0, 0), 2e-6, 2e-6, 2e-6)
        # Waveshare USB camera module (board 25 x 24 mm, lens on +x of its mesh); 160 deg lens
        Rm = np.column_stack([z_opt, -x_opt, -y_opt])
        add_mesh(cam, f"{mesh_prefix}/rover/usb_camera.stl", xyz=V2_LENS - Rm @ np.array([0.017, 0, 0]),
                 rpy=rpy_from_matrix(Rm), material="camera_black")
        add_fixed_joint(robot, "wrist_camera_joint", "wrist_cam_bracket_link", "wrist_camera_link")
        el(robot, "link", name="wrist_camera_optical_frame")
        add_fixed_joint(robot, "wrist_camera_optical_joint", "wrist_camera_link", "wrist_camera_optical_frame",
                        xyz=V2_LENS, rpy=rpy_from_matrix(Rc))

    # ---------------- optional arm battery (3S 26700 Li-ion pack, 80x27x72 mm, 0.3 kg) strapped on the deck beside the plate
    if arm_battery:
        b = el(robot, "link", name="arm_battery_link")
        ixx, iyy, izz = box_inertia(0.30, 0.080, 0.027, 0.072)
        add_inertial(b, 0.30, (0, 0, 0.036), ixx, iyy, izz)
        add_box(b, (0.080, 0.027, 0.072), xyz=(0, 0, 0.036), material="battery_black", collision=True)
        add_fixed_joint(robot, "arm_battery_joint", "base_link", "arm_battery_link", xyz=(plate_x0 + plate_sign * 0.043, 0.080, deck_z))

    mj = el(robot, "mujoco")
    el(mj, "compiler", meshdir=".", balanceinertia="true", discardvisual="false", strippath="false")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(robot).write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-height", type=float, default=0.175, help="ground to top of the upper metal deck, m (measured 0.175)")
    ap.add_argument("--spacer", type=float, default=0.0, help="spacer height under the printed plate, m (0 = bolted flat, as built)")
    ap.add_argument("--lidar-to-yaw", type=float, default=0.055, help="LiDAR centre to arm yaw axis, m (estimated from photos)")
    ap.add_argument("--arm-facing", choices=["front", "rear"], default="front",
                    help="front = as built (plate triangle to the rear); rear = Codex's drawing orientation")
    ap.add_argument("--wrist-mount", choices=["v2", "none"], default="v2",
                    help="v2 = fixed bracket + Waveshare USB camera (as built); none = no wrist camera")
    ap.add_argument("--arm-battery", action="store_true", help="add the 3S Li-ion arm battery pack on the deck")
    a = ap.parse_args()
    kw = dict(deck_height=a.deck_height, spacer=a.spacer, lidar_to_yaw=a.lidar_to_yaw, arm_facing=a.arm_facing,
              wrist_mount=a.wrist_mount, arm_battery=a.arm_battery)
    print("wrote", build("package://so101_rover_description/meshes", os.path.join(PKG, "urdf", "so101_rover.urdf"), **kw))
    print("wrote", build("../meshes", os.path.join(PKG, "mujoco", "so101_rover_sim.urdf"), **kw))
