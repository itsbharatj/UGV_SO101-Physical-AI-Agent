#!/usr/bin/env python3
"""Compare wrist-mount variants: what the wrist camera sees (own body / jaws / floor), where the fingertip falls,
and clearance between the mount and the moving jaw over the gripper's range.  Writes tests/out/wrist_variants.png.
    python3 tests/compare_wrist_mounts.py
"""
import os, sys, json, math, subprocess, numpy as np
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(PKG)
PY = sys.executable
import mujoco, yourdfpy
from PIL import Image, ImageDraw


def seg_model(xml_path):
    """Same model without the skybox: MuJoCo's segmentation renderer returns garbage ids for sky pixels."""
    import re, tempfile
    txt = open(xml_path).read()
    txt = re.sub(r'<texture[^>]*name="sky"[^>]*/>', "", txt)
    tmp = os.path.join(os.path.dirname(xml_path), "_seg_tmp.xml")
    open(tmp, "w").write(txt)
    mm = mujoco.MjModel.from_xml_path(tmp); os.remove(tmp)
    return mm
reg = json.load(open("meshes/mount/wrist_mounts.json"))
POSE = {"shoulder_pan": 0.0, "shoulder_lift": -1.0, "elbow_flex": 1.0, "wrist_flex": 0.9, "wrist_roll": 0.0, "gripper": 0.3}
rows = []
for name in reg:
    subprocess.run([PY, "scripts/build_urdf.py", "--wrist-variant", name], check=True, capture_output=True)
    subprocess.run([PY, "scripts/build_mjcf.py"], check=True, capture_output=True)
    r = yourdfpy.URDF.load("mujoco/so101_rover_sim.urdf", build_scene_graph=True, load_meshes=False)
    cfg = {n: 0.0 for n in r.actuated_joint_names}; cfg.update(POSE); r.update_cfg(cfg)
    T = lambda f: r.get_transform(f, "base_footprint")
    cam = T("wrist_camera_optical_frame"); tip = T("arm_gripper_frame_link")[:3, 3]
    v = tip - cam[:3, 3]; dist = np.linalg.norm(v); off = math.degrees(math.acos(np.dot(cam[:3, 2], v) / dist))
    m = seg_model("mujoco/so101_rover.xml"); d = mujoco.MjData(m)      # one model, one renderer per variant (no skybox)
    for j, q in POSE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q
    mujoco.mj_forward(m, d)
    rr = mujoco.Renderer(m, 480, 640); c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    rr.update_scene(d, camera=c); rgb = rr.render().copy()
    rr.enable_segmentation_rendering(); rr.update_scene(d, camera=c); seg = rr.render()[..., 0].copy(); rr.disable_segmentation_rendering()
    rr.close()
    body_of = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gi])
    vis = [gi for gi in range(m.ngeom) if m.geom_group[gi] != 3]
    gname = lambda gi: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "")
    jaw = [gi for gi in vis if body_of(gi) == "arm_moving_jaw_so101_v1_link"]
    body = [gi for gi in vis if body_of(gi) in ("arm_gripper_link", "arm_wrist_link") and "wrist_cam" not in gname(gi) and "usb_camera" not in gname(gi)]
    mount = [gi for gi in vis if "wrist_cam" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "") or "usb_camera" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "")]
    floor = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")]
    pct = lambda ids: 100 * np.isin(seg, ids).mean()
    # jaw clearance: sweep the gripper joint, count contacts between mount geoms and jaw geoms
    gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper"); lo, hi = m.jnt_range[gj]
    hits = 0
    mount_col = [gi for gi in range(m.ngeom) if m.geom_group[gi] == 3 and "wrist_cam" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or "")]
    jaw_col = [gi for gi in range(m.ngeom) if m.geom_group[gi] == 3 and body_of(gi) == "arm_moving_jaw_so101_v1_link"]
    for q in np.linspace(lo, hi, 25):
        d.qpos[m.jnt_qposadr[gj]] = q; mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 in mount_col and g2 in jaw_col) or (g2 in mount_col and g1 in jaw_col):
                hits += 1
    # fingertip pixel position (MuJoCo camera: fovy, 640x480)
    cam_id = c; fovy = math.radians(m.cam_fovy[cam_id]); f_px = 240 / math.tan(fovy / 2)
    Rc = d.cam_xmat[cam_id].reshape(3, 3); pc = Rc.T @ (tip - d.cam_xpos[cam_id]); u = 320 + f_px * pc[0] / -pc[2]; vpx = 240 - f_px * pc[1] / -pc[2]
    rows.append((name, reg[name]["extend_mm"], reg[name].get("offset_mm", 0), reg[name]["tilt_deg"], dist * 1000, off, pct(jaw), pct(body), pct(mount), pct(floor), hits, (u, vpx), rgb))
    print(f"{name:16s} ext {reg[name]['extend_mm']:3.0f} side {reg[name].get('offset_mm', 0):3.0f} yaw {reg[name]['tilt_deg']:6.1f} | tip {dist*1000:5.0f} mm, {off:4.1f} deg off-axis, at px ({u:.0f},{vpx:.0f}) | "
          f"moving jaw {pct(jaw):4.1f}%  wrist+gripper body {pct(body):4.1f}%  own mount {pct(mount):4.1f}%  floor {pct(floor):4.1f}% | jaw-mount contacts: {hits}")
subprocess.run([PY, "scripts/build_urdf.py"], check=True, capture_output=True); subprocess.run([PY, "scripts/build_mjcf.py"], check=True, capture_output=True)
W, H = 640, 480; sheet = Image.new("RGB", (W * len(rows), H + 60), "white"); dr = ImageDraw.Draw(sheet)
for i, (name, ext, side, tilt, dist, off, pj, pb, pm, pf, hits, (u, vpx), rgb) in enumerate(rows):
    im = Image.fromarray(rgb); dd = ImageDraw.Draw(im); dd.ellipse((u - 8, vpx - 8, u + 8, vpx + 8), outline="red", width=3)
    sheet.paste(im, (i * W, 60))
    dr.text((i * W + 10, 8), f"{name}: back {ext:.0f} mm, sideways {side:.0f} mm, yaw {tilt:.0f} deg   (red ring = fingertip)", fill="black")
    dr.text((i * W + 10, 30), f"tip {dist:.0f} mm away | moving jaw {pj:.0f}%  wrist/gripper body {pb:.0f}%  floor {pf:.0f}%", fill="black")
os.makedirs("tests/out", exist_ok=True); sheet.save("tests/out/wrist_variants.png"); print("wrote tests/out/wrist_variants.png")
