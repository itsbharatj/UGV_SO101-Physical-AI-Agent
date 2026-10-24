# so101_rover_description

URDF / MJCF of a **Waveshare UGV Rover (Jetson Orin ROS 2 kit)** carrying an **SO-101 follower arm** on a
3D-printed adapter plate, with a **Waveshare 160° USB camera** on a fixed wrist bracket and the **OAK-D Lite**
at the rover's front. Built and checked for Isaac Sim, MuJoCo, Gazebo and RViz. Matches the robot as built on
24 Oct 2026 (see `reference_photos/`).

![render](export/so101_rover_views.png)

## What is in the model

| Part | Source | Notes |
|---|---|---|
| Lower body, 4 driven wheels, LiDAR, front OAK-D Lite | Waveshare `ugv_description/urdf/ugv_rover.urdf` (geometry and inertia unchanged) | pan-tilt removed; collision = box body + cylinder wheels |
| Upper deck (metal plate), 4 brass pillars, Jetson Orin Nano | this package, schematic boxes at measured heights | deck top 0.175 m above ground |
| Printed adapter plate | `SO101_UGV_Rover_Mount/FINAL_FILES/01_adapter_print_1.stl` | bolted flat (no spacers), triangle toward the rear, 0.105 kg |
| SO-101 follower arm | TheRobotStudio `Simulation/SO101/so101_new_calib.urdf` | link names prefixed `arm_`; joint names unchanged: `shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper` (LeRobot order); reaches **forward** |
| Wrist camera | `SO101_OAK_RGB_v2` fixed bracket + Waveshare USB camera module (`pt_camera` mesh), 160° lens | aimed at the fixed-finger pad per the bracket's `pose.json` |
| Optional arm battery | 3S 26700 Li-ion pack, 80×27×72 mm, 0.30 kg | `--arm-battery` |
| Optional spacers | `02_spacer_6mm` print, 6 mm | `--spacer 0.006`; not fitted as built |

Frames follow ROS conventions: `base_footprint` on the ground, `base_link` 0.08 m up, x forward, z up.
Extra frames: `front_camera_optical_frame`, `wrist_camera_optical_frame` (z = optical axis, x right, y down),
`arm_gripper_frame_link` (between the fingertips), `base_lidar_link`, `base_imu_link`.

## Files

```
urdf/so101_rover.urdf          ROS version   (mesh paths package://so101_rover_description/meshes/...)
mujoco/so101_rover_sim.urdf    plain version (mesh paths ../meshes/...)  -> Isaac Sim, MuJoCo, anything non-ROS
mujoco/so101_rover.xml         MJCF with actuators (position servos on the 6 arm joints, velocity on 4 wheels)
export/so101_rover_assembly.*  whole robot as one STL (m and mm) and a coloured GLB, arm in a display pose
export/view_*.png              renders from the same angles as the reference photos
meshes/rover|so101|mount       STL meshes (rover and arm in metres, printed parts in mm with scale 0.001)
scripts/build_urdf.py          regenerates both URDFs from the upstream files in source_models/
tests/check_model.py           parse + kinematics + 3 s MuJoCo simulation + MJCF export
tests/export_views.py          STL/GLB export and reference renders
reference_photos/              photos of the real robot the model was matched to
```

## Using it

**Isaac Sim**: File → Import → URDF, pick `mujoco/so101_rover_sim.urdf`. Keep "merge fixed joints" on, set the
wheel joints to velocity drive and the arm joints to position drive. Mesh paths are relative, no ROS needed.

**MuJoCo**: `python -m mujoco.viewer --mjcf mujoco/so101_rover.xml`. Control vector: 6 arm joint targets (rad,
LeRobot order) then 4 wheel speeds (rad/s). The URDF also loads directly; it carries a `<mujoco>` compiler block.

**ROS 2**: copy the folder into your workspace, `colcon build`, then
`ros2 launch so101_rover_description display.launch.py` for RViz with joint sliders.

**Regenerate** with different parameters, then re-run the checks:

```bash
python3 scripts/build_urdf.py                                   # as built
python3 scripts/build_urdf.py --lidar-to-yaw 0.060              # if you measure a different LiDAR-to-arm distance
python3 scripts/build_urdf.py --spacer 0.006                    # if the 6 mm spacers are fitted
python3 scripts/build_urdf.py --arm-battery                     # add the arm's Li-ion pack
python3 scripts/build_urdf.py --wrist-mount none                # arm without the wrist camera
python3 tests/check_model.py && python3 tests/export_views.py
```

## Key dimensions (as modelled)

| | value |
|---|---|
| Ground → top of the upper metal deck | 0.175 m (measured) |
| Ground → underside of the SO-101 base | 0.185 m (deck + 10 mm plate) |
| Top of the lower body (Waveshare mesh) → SO-101 base | 0.065 m in the model (0.085 m measured; Waveshare's lower-body mesh is ~2 cm taller than the Jetson-kit body, see below) |
| LiDAR centre → arm yaw axis | 0.055 m (estimated from the photos, see check 2) |
| Front OAK-D Lite centre | 0.018 m below the deck, at the deck's front edge |
| Total mass | 4.48 kg (rover 3.6, arm 0.63, plate + wrist camera 0.15, no battery) |

## Things to check against the real robot

1. **Arm facing.** As built the plate's triangle points to the rear, so the SO-101's motor-holder end is at the rear and
   the arm reaches forward over the LiDAR. The model does the same (`--arm-facing rear` gives the other orientation).
2. **Fore/aft position.** The arm's yaw axis is placed 55 mm behind the LiDAR centre, read off the side photos. Measure
   it once (LiDAR centre to the centre of the arm's first motor) and pass `--lidar-to-yaw` if it differs.
3. **Lower body height.** Waveshare's lower-body mesh tops out at 0.12 m; your measurements put the real top at about
   0.10 m. Only the look of the gap under the deck is affected; the deck, LiDAR, camera and arm are at measured heights.
4. **Wheels.** Waveshare models 4 driven wheels; the middle pair is part of the body mesh and not simulated.
5. **Wrist camera aim.** The USB camera is tied to a bracket designed for the OAK-D Lite, so its exact aim is approximate.
   Correct `wrist_camera_optical_joint` after calibrating against a real image if you need it precise.
6. Inertias of the printed parts, deck and Jetson are box approximations. Rover and arm inertias are the upstream values.

## Printing the parts

The two printed parts on the robot are in `meshes/mount/` at true size in millimetres:
`rover_adapter_plate.stl` (black, PETG, 60% infill) and `wrist_cam_bracket_v2.stl` (white, PETG or ASA, 40% infill).

## Licences

Rover meshes and URDF: Waveshare `ugv_ws` (open source, see upstream). SO-101 meshes and URDF: TheRobotStudio
SO-ARM100, Apache-2.0. Adapter, bracket and this package: MIT.
