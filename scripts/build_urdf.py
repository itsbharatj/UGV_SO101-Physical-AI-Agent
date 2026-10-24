#!/usr/bin/env python3
"""Build the combined Waveshare UGV Rover + SO-101 follower URDF, matched to the robot as built.

Sources (geometry and inertia unchanged, only renamed / re-pathed):
  * Waveshare ugv_ws  ->  src/ugv_main/ugv_description/urdf/ugv_rover.urdf   (lower body, wheels, LiDAR, OAK-D Lite)
  * TheRobotStudio SO-ARM100 -> Simulation/SO101/so101_new_calib.urdf        (arm, "new calibration" zero = mid range)
Added here, from the physical build (photos in reference_photos/):
  * lower body cut at its real flat top (the Waveshare mesh carries a 2.8 cm cover the Jetson kit does not have)
  * Jetson-kit upper deck: folded metal tray on brass pillars, Jetson Orin Nano stack under it (schematic boxes)
  * printed adapter plate (SO101_UGV_Rover_Mount/FINAL_FILES) bolted flat on the deck, triangle toward the REAR
  * SO-101 base on the plate, arm reaching FORWARD over the LiDAR
  * revision-2 wrist bracket (SO101_OAK_RGB_v2) with the Waveshare 160 deg USB camera tied at the rod / frame corner
  * passive middle wheels, camera and LiDAR frames, Gazebo sensors, <mujoco> compiler block

Measured on the robot (24 Oct 2026): ground -> top of the upper metal deck = 0.175 m.
Usage:
  python3 scripts/build_urdf.py                       # as built, then: python3 scripts/build_mjcf.py
  python3 scripts/build_urdf.py --lidar-to-yaw 0.060 --spacer 0.006 --arm-battery && python3 scripts/build_mjcf.py
"""
import argparse, copy, json, math, os
from lxml import etree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "source_models")

# ----------------------------------------------------------------------------- geometry (metres)
BASE_LINK_Z = 0.081212         # base_link above ground: puts the wheel axles (base_link z -0.041212) exactly 0.040 up
LOWER_BODY_TOP = 0.0115        # flat top of the lower body, base_link frame (0.093 m above ground; photos ~0.093)
PLATE_T = 0.010                # printed adapter thickness
SO101_BASE_BOTTOM = -0.0024    # z of the SO-101 base underside in its own base_link frame (official URDF)
SO101_YAW_X = 0.0388353        # shoulder_pan axis x in the SO-101 base_link frame (official URDF)
SO101_WIDE_PAIR_X = -0.0137    # wide (63.5 mm) hole pair x in the SO-101 base_link frame
LIDAR_X = 0.0398145505519817   # LiDAR centre x, base_link frame (Waveshare URDF)
CAM_BELOW_DECK = 0.018         # front OAK-D Lite centre sits 18 mm below the deck (Waveshare URDF)
WHEEL_X, WHEEL_Y, WHEEL_Z = 0.0855, 0.08726, -0.041212
MID_WHEEL_MASS = 0.207085933743005
# upper deck (folded tray), base_link frame, from the photos
# folded tray: 125 mm flat top, side walls slope out to 160 mm 18 mm down, then back in to 127 mm 28 mm down;
# the printed plate overhangs its rear edge by ~27 mm (side photo)
DECK_X0, DECK_X1, DECK_TOP_W, DECK_T = -0.066, 0.090, 0.125, 0.003
DECK_WALL = [(0.0625, 0.0), (0.080, -0.018), (0.0635, -0.028)]     # (y, z below deck top) corner points of one side wall
# Jetson Orin Nano carrier: 79 x 100 mm board on a 58 x 86 mm hole pattern; heights above ground from the photos
JETSON_X = 0.0105
STANDOFFS = [(JETSON_X + 0.029, 0.043), (JETSON_X + 0.029, -0.043), (JETSON_X - 0.029, 0.043), (JETSON_X - 0.029, -0.043)]
JETSON_BOARD_Z = (0.131, 0.134)        # carrier board bottom / top, m above ground
SPACER_Z, LOWER_BRASS_Z = (0.094, 0.112), (0.112, 0.131)
# printed plate, "Codex" frame: X=-8..94 (triangle at X=-8), Y=+-64; arm holes wide pair X=7.5; chassis bolts X=4,40 Y=+-43
PLATE_WIDE_PAIR_X = 0.0075
PLATE_BOLTS = [(0.004, 0.043), (0.004, -0.043), (0.040, 0.043), (0.040, -0.043)]
# wrist bracket (SO101_OAK_RGB_v2), frame = official Wrist_Roll_Follower STL frame, metres
USB_CAM_LENS = np.array([0.020, -0.090, 0.008])      # where the USB board is tied: corner of the rods and the U-frame
AIM_TARGET = np.array([-0.0133, 0.0, 0.060])         # fixed-finger contact pad (SO101_OAK_RGB_v2_pose.json)


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


def tidy(node):
    """Drop schema-invalid leftovers from the upstream exporters."""
    if node.tag == "link":
        for o in node.findall("origin"):
            node.remove(o)
    if node.tag == "joint" and node.get("type") == "fixed":
        for a in node.findall("axis"):
            node.remove(a)
        for lim in node.findall("limit"):
            node.remove(lim)


def gazebo_camera(robot, link, name, hfov_rad, width, height, near, far, rate=30):
    g = el(robot, "gazebo", reference=link)
    s = el(g, "sensor", name=name, type="camera")
    el(s, "update_rate").text = str(rate)
    el(s, "always_on").text = "1"
    el(s, "topic").text = f"{name}/image"
    c = el(s, "camera")
    el(c, "horizontal_fov").text = f"{hfov_rad:.4f}"
    im = el(c, "image"); el(im, "width").text = str(width); el(im, "height").text = str(height); el(im, "format").text = "R8G8B8"
    cl = el(c, "clip"); el(cl, "near").text = str(near); el(cl, "far").text = str(far)
    el(c, "optical_frame_id").text = f"{name}_optical_frame"


def build(mesh_prefix, out_path, deck_height=0.175, spacer=0.0, lidar_to_yaw=0.065, arm_facing="front",
          wrist_mount="v2", arm_battery=False):
    deck_z = deck_height - BASE_LINK_Z                      # top of the upper metal deck, base_link frame
    plate_z = deck_z + spacer                               # underside of the printed plate
    arm_base_z = plate_z + PLATE_T - SO101_BASE_BOTTOM      # SO-101 base_link origin
    rover = ET.parse(os.path.join(SRC, "ugv_rover.urdf")).getroot()
    arm = ET.parse(os.path.join(SRC, "so101_new_calib.urdf")).getroot()

    robot = ET.Element("robot", name="so101_rover")
    robot.append(ET.Comment(" Generated by scripts/build_urdf.py - do not edit by hand. Rover: Waveshare ugv_description "
                            "(ugv_rover). Arm: TheRobotStudio SO-ARM100 so101_new_calib. Deck, plate, wrist camera: this package. "))
    mats = [("body_black", (0.09, 0.09, 0.10, 1)), ("deck_metal", (0.26, 0.27, 0.30, 1)), ("brass", (0.72, 0.58, 0.3, 1)),
            ("pcb_green", (0.1, 0.35, 0.2, 1)), ("heatsink", (0.45, 0.46, 0.48, 1)), ("printed_black", (0.03, 0.03, 0.03, 1)),
            ("printed_white", (0.92, 0.92, 0.92, 1)), ("so101_print", (0.93, 0.93, 0.95, 1)), ("sts3215", (0.1, 0.1, 0.1, 1)),
            ("oak_grey", (0.36, 0.37, 0.40, 1)), ("lidar_black", (0.14, 0.14, 0.17, 1)), ("tyre_black", (0.06, 0.06, 0.06, 1)),
            ("camera_board", (0.12, 0.25, 0.18, 1))]
    if arm_battery:
        mats.append(("battery_black", (0.05, 0.05, 0.05, 1)))
    for name, rgba in mats:
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
        tidy(n)
        for m in n.iter("mesh"):
            fn = os.path.basename(m.get("filename"))
            if fn == "base_link.stl":
                fn = "base_link_jetson_kit.stl"
            m.set("filename", f"{mesh_prefix}/rover/{fn}")
        for mat in n.iter("material"):
            for c in list(mat):
                mat.remove(c)
            mat.set("name", "body_black")
        if n.tag == "joint":
            name = n.get("name"); o = n.find("origin")
            if name == "base_joint":
                xyz = [float(v) for v in o.get("xyz").split()]; xyz[2] = BASE_LINK_Z; o.set("xyz", fmt(xyz))
            elif name in ("base_lidar_link_joint", "3d_camera_link_joint"):
                xyz = [float(v) for v in o.get("xyz").split()]
                xyz[2] = deck_z if name == "base_lidar_link_joint" else deck_z - CAM_BELOW_DECK
                o.set("xyz", fmt(xyz))
            elif name.endswith("wheel_link_joint"):
                el(n, "limit", effort="2.0", velocity="32")       # 1.3 m/s max speed on 80 mm wheels
                el(n, "dynamics", damping="0.02", friction="0.0")
        if n.tag == "link":
            name = n.get("name")
            if name == "base_link":
                mass = n.find("inertial/mass")                    # upstream body mass includes the middle wheels,
                mass.set("value", f"{float(mass.get('value')) - 2 * MID_WHEEL_MASS:.6g}")   # which are now separate links
                for c in n.findall("collision"):
                    n.remove(c)
                body_bot = -0.045                                 # body underside 0.036 m above ground
                add_box(n, (0.245, 0.230, LOWER_BODY_TOP - body_bot), xyz=(0, 0, (LOWER_BODY_TOP + body_bot) / 2),
                        visual=False, collision=True)
                # Jetson-kit upper deck: folded tray on brass pillars, Jetson Orin Nano stack under it (schematic)
                gz = lambda h: h - BASE_LINK_Z                    # height above ground -> base_link z
                cx, L = (DECK_X0 + DECK_X1) / 2, DECK_X1 - DECK_X0
                add_box(n, (L, DECK_TOP_W, DECK_T), xyz=(cx, 0, deck_z - DECK_T / 2), material="deck_metal", collision=True)
                for sy in (1, -1):
                    for (y0, z0), (y1, z1) in zip(DECK_WALL[:-1], DECK_WALL[1:]):
                        dy, dz = sy * (y1 - y0), z1 - z0
                        add_box(n, (L, math.hypot(dy, dz), 0.002), xyz=(cx, sy * (y0 + y1) / 2, deck_z + (z0 + z1) / 2),
                                rpy=(math.atan2(dz, dy), 0, 0), material="deck_metal")
                b0, b1 = JETSON_BOARD_Z
                for (px, py) in STANDOFFS:
                    for (z0, z1, mat) in [(gz(SPACER_Z[0]), gz(SPACER_Z[1]), "heatsink"),
                                          (gz(LOWER_BRASS_Z[0]), gz(LOWER_BRASS_Z[1]), "brass"),
                                          (gz(b1), deck_z - DECK_T, "brass")]:          # pillars from the board up to the deck
                        add_cylinder(n, 0.0028, z1 - z0, xyz=(px, py, (z0 + z1) / 2), material=mat)
                add_box(n, (0.079, 0.100, b1 - b0), xyz=(JETSON_X, 0, gz((b0 + b1) / 2)), material="pcb_green")   # carrier board
                add_box(n, (0.065, 0.065, 0.024), xyz=(JETSON_X, 0, gz(b1) + 0.012), material="heatsink")       # module, heatsink, fan
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
                    mat.set("name", "oak_grey")
        robot.append(n)

    # passive middle wheels (the rover has 6 wheels, 4 driven; Waveshare bakes the middle pair into the body mesh)
    for side, sy in (("left", 1), ("right", -1)):
        w = el(robot, "link", name=f"{side}_mid_wheel_link")
        add_inertial(w, MID_WHEEL_MASS, (0, 0, 0), 0.000116541923504163, 0.000116541923504163, 0.000170742352371108)
        add_cylinder(w, 0.04, 0.0426, visual=False, collision=True)
        j = el(robot, "joint", name=f"{side}_mid_wheel_link_joint", type="continuous")
        add_origin(j, (0, sy * WHEEL_Y, WHEEL_Z), (math.pi / 2, 0, 0))
        el(j, "parent", link="base_link"); el(j, "child", link=f"{side}_mid_wheel_link"); el(j, "axis", xyz="0 0 -1")
        el(j, "limit", effort="0", velocity="32"); el(j, "dynamics", damping="0.002", friction="0.0")

    # front camera optical frame (ROS: z forward, x right, y down), at the OAK's front face
    el(robot, "link", name="front_camera_optical_frame")
    add_fixed_joint(robot, "front_camera_optical_joint", "3d_camera_link", "front_camera_optical_frame",
                    xyz=(0.017, 0, 0), rpy=(-math.pi / 2, 0, -math.pi / 2))

    # ---------------- arm position: yaw axis lidar_to_yaw behind the LiDAR centre, on the centreline
    yaw_x = LIDAR_X - lidar_to_yaw
    if arm_facing == "rear" and lidar_to_yaw < 0.09:
        raise SystemExit("--arm-facing rear puts the SO-101 base into the LiDAR unless --lidar-to-yaw is at least 0.09 m")
    if arm_facing == "front":
        # AS BUILT: plate triangle to the rear, SO-101 motor-holder end (wide pair) at the rear, arm reaches forward
        arm_yaw = 0.0; arm_x = yaw_x - SO101_YAW_X
        plate_x0 = arm_x + SO101_WIDE_PAIR_X - PLATE_WIDE_PAIR_X; plate_sign = 1.0
    else:
        # Codex's drawing orientation: triangle toward the LiDAR, arm reaches toward the rear
        arm_yaw = math.pi; arm_x = yaw_x + SO101_YAW_X
        plate_x0 = arm_x - SO101_WIDE_PAIR_X + PLATE_WIDE_PAIR_X; plate_sign = -1.0

    plate = el(robot, "link", name="arm_mount_plate_link")
    ixx, iyy, izz = box_inertia(0.105, 0.102, 0.128, PLATE_T)
    add_inertial(plate, 0.105, (plate_sign * 0.043, 0, PLATE_T / 2), ixx, iyy, izz)
    add_mesh(plate, f"{mesh_prefix}/mount/rover_adapter_plate.stl", rpy=(0, 0, 0 if plate_sign > 0 else math.pi),
             scale=0.001, material="printed_black")
    add_box(plate, (0.102, 0.128, PLATE_T), xyz=(plate_sign * 0.043, 0, PLATE_T / 2), visual=False, collision=True)
    if spacer > 0:
        for (bx, by) in PLATE_BOLTS:
            add_mesh(plate, f"{mesh_prefix}/mount/rover_adapter_spacer.stl", xyz=(plate_sign * bx, by, -spacer),
                     scale=0.001, material="printed_black")
    add_fixed_joint(robot, "arm_mount_plate_joint", "base_link", "arm_mount_plate_link", xyz=(plate_x0, 0, plate_z))

    # ---------------- SO-101 arm, links prefixed arm_, joint names kept (LeRobot names)
    for node in arm:
        if node.tag not in ("link", "joint"):
            continue
        n = copy.deepcopy(node)
        tidy(n)
        if n.tag == "link":
            n.set("name", "arm_" + n.get("name"))
            if n.get("name") == "arm_gripper_frame_link":        # pure frame: upstream gives it a 1e-9 kg, zero-inertia body
                for i in n.findall("inertial"):
                    n.remove(i)
        else:
            n.find("parent").set("link", "arm_" + n.find("parent").get("link"))
            n.find("child").set("link", "arm_" + n.find("child").get("link"))
        for m in n.iter("mesh"):
            m.set("filename", f"{mesh_prefix}/so101/{os.path.basename(m.get('filename'))}")
        for mat in n.iter("material"):
            if mat.get("name") == "3d_printed":
                mat.set("name", "so101_print")
        robot.append(n)
    add_fixed_joint(robot, "arm_mount_joint", "base_link", "arm_base_link", xyz=(arm_x, 0, arm_base_z), rpy=(0, 0, arm_yaw))

    # ---------------- wrist camera: v2 bracket + Waveshare USB camera tied at the rod / U-frame corner
    if wrist_mount == "v2":
        br = el(robot, "link", name="wrist_cam_bracket_link")
        ixx, iyy, izz = box_inertia(0.016, 0.10, 0.08, 0.04)
        add_inertial(br, 0.016, (0.030, -0.070, 0.005), ixx, iyy, izz)
        add_mesh(br, f"{mesh_prefix}/mount/wrist_cam_bracket_v2.stl", scale=0.001, material="printed_white")
        add_box(br, (0.040, 0.075, 0.020), xyz=(0.005, -0.065, 0.010), rpy=(0, 0, -0.6), visual=False, collision=True)
        # bracket STL frame -> gripper_link: Rx(pi) + (0, -0.218, 0.95) mm (same as the official wrist_roll_follower visual)
        add_fixed_joint(robot, "wrist_cam_bracket_joint", "arm_gripper_link", "wrist_cam_bracket_link",
                        xyz=(0, -0.000218, 0.000950), rpy=(math.pi, 0, 0))
        # camera body frame (ROS camera_link convention): x = view direction, y = left, z = up
        xv = AIM_TARGET - USB_CAM_LENS; xv /= np.linalg.norm(xv)
        up_hint = np.array([0.0, 0.0, -1.0])                      # bracket frame z points away from the fingers' up side
        yv = np.cross(up_hint, xv); yv /= np.linalg.norm(yv); zv = np.cross(xv, yv)
        Rcam = np.column_stack([xv, yv, zv])
        cam = el(robot, "link", name="wrist_camera_link")
        add_inertial(cam, 0.012, (-0.006, 0, 0), 1e-6, 1e-6, 1e-6)
        add_mesh(cam, f"{mesh_prefix}/rover/usb_camera.stl", xyz=(-0.017, 0, 0), material="camera_board")
        add_fixed_joint(robot, "wrist_camera_joint", "wrist_cam_bracket_link", "wrist_camera_link",
                        xyz=USB_CAM_LENS, rpy=rpy_from_matrix(Rcam))
        el(robot, "link", name="wrist_camera_optical_frame")
        add_fixed_joint(robot, "wrist_camera_optical_joint", "wrist_camera_link", "wrist_camera_optical_frame",
                        rpy=(-math.pi / 2, 0, -math.pi / 2))

    # ---------------- optional arm battery (3S 26700 Li-ion pack, 0.3 kg): there is no room on the 125 mm deck top,
    # so it stands on the lower body behind the Jetson, 27 mm thick along x, 80 mm across, 72 mm tall (clears the deck)
    if arm_battery:
        b = el(robot, "link", name="arm_battery_link")
        ixx, iyy, izz = box_inertia(0.30, 0.027, 0.080, 0.072)
        add_inertial(b, 0.30, (0, 0, 0.036), ixx, iyy, izz)
        add_box(b, (0.027, 0.080, 0.072), xyz=(0, 0, 0.036), material="battery_black", collision=True)
        add_fixed_joint(robot, "arm_battery_joint", "base_link", "arm_battery_link", xyz=(-0.050, 0, LOWER_BODY_TOP))

    # ---------------- simulator extensions (ignored by robot_state_publisher and Isaac Sim)
    gazebo_camera(robot, "3d_camera_link", "front_camera", math.radians(69), 1280, 960, 0.05, 20.0)   # 69 x 54 deg at 4:3
    if wrist_mount == "v2":
        gazebo_camera(robot, "wrist_camera_link", "wrist_camera", math.radians(120), 640, 480, 0.02, 5.0)
    g = el(robot, "gazebo", reference="base_lidar_link")
    s = el(g, "sensor", name="lidar", type="gpu_lidar")
    el(s, "update_rate").text = "10"; el(s, "always_on").text = "1"; el(s, "topic").text = "scan"
    el(s, "gz_frame_id").text = "base_lidar_link"
    ray = el(s, "lidar"); sc = el(ray, "scan"); hz = el(sc, "horizontal")
    el(hz, "samples").text = "456"; el(hz, "resolution").text = "1"; el(hz, "min_angle").text = "-3.14159"; el(hz, "max_angle").text = "3.14159"
    rg = el(ray, "range"); el(rg, "min").text = "0.02"; el(rg, "max").text = "12.0"; el(rg, "resolution").text = "0.01"
    mj = el(robot, "mujoco")
    el(mj, "compiler", meshdir=".", balanceinertia="true", discardvisual="false", strippath="false")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(robot).write(out_path, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-height", type=float, default=0.175, help="ground to top of the upper metal deck, m (measured 0.175)")
    ap.add_argument("--spacer", type=float, default=0.0, help="spacer height under the printed plate, m (0 = bolted flat, as built)")
    ap.add_argument("--lidar-to-yaw", type=float, default=0.065, help="LiDAR centre to arm yaw axis, m (from the photos)")
    ap.add_argument("--arm-facing", choices=["front", "rear"], default="front",
                    help="front = as built (plate triangle to the rear); rear = Codex's drawing orientation")
    ap.add_argument("--wrist-mount", choices=["v2", "none"], default="v2",
                    help="v2 = fixed bracket + Waveshare USB camera (as built); none = no wrist camera")
    ap.add_argument("--arm-battery", action="store_true", help="add the 3S Li-ion arm battery pack behind the Jetson")
    a = ap.parse_args()
    kw = dict(deck_height=a.deck_height, spacer=a.spacer, lidar_to_yaw=a.lidar_to_yaw, arm_facing=a.arm_facing,
              wrist_mount=a.wrist_mount, arm_battery=a.arm_battery)
    print("wrote", build("package://so101_rover_description/meshes", os.path.join(PKG, "urdf", "so101_rover.urdf"), **kw))
    print("wrote", build("../meshes", os.path.join(PKG, "mujoco", "so101_rover_sim.urdf"), **kw))
    with open(os.path.join(PKG, "urdf", "build_params.json"), "w") as f:      # read by tests/check_model.py
        json.dump(kw, f, indent=2)
