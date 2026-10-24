# so101_rover_description

URDF / MJCF of a **Waveshare UGV Rover (Jetson Orin ROS 2 kit)** carrying an **SO-101 follower arm** on a
3D-printed adapter plate, with a **Waveshare 160° USB camera** on the wrist and the **OAK-D Lite** at the rover's
front. Checked for Isaac Sim, MuJoCo, Gazebo and RViz. Matched to the robot as built on 24 Oct 2026, against the
photos in `reference_photos/`.

![render](export/so101_rover_views.png)

## What is in the model

| Part | Source | Notes |
|---|---|---|
| Lower body, LiDAR, front OAK-D Lite, 4 driven wheels | Waveshare `ugv_description/urdf/ugv_rover.urdf` | body mesh cut at its real flat top (0.093 m); Waveshare's mesh carries a 2.8 cm cover the Jetson kit does not have |
| 2 passive middle wheels | added | the rover has 6 wheels, 4 driven; collision cylinders, the body mesh draws them |
| Upper deck, 4 brass standoffs, Jetson Orin Nano | added, schematic boxes | deck top 0.175 m above ground (measured) |
| Printed adapter plate | `SO101_UGV_Rover_Mount/FINAL_FILES/01_adapter_print_1.stl` | bolted flat on the deck, triangle toward the rear |
| SO-101 follower arm | TheRobotStudio `Simulation/SO101/so101_new_calib.urdf` | links prefixed `arm_`; joint names unchanged and in LeRobot order: `shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper`; reaches **forward** over the LiDAR |
| Wrist camera | `SO101_OAK_RGB_v2` bracket + Waveshare USB camera module | camera tied where the bracket rods meet the U-frame, aimed at the fixed-finger pad |
| Optional | 3S Li-ion arm battery (`--arm-battery`), 6 mm spacers (`--spacer 0.006`) | not fitted as built |

**Frames** follow ROS conventions. `base_footprint` is on the ground and `base_link` is 0.081 m up, with x forward and z up.

| Frame | What it is |
|---|---|
| `front_camera_optical_frame`, `wrist_camera_optical_frame` | camera frames: z along the optical axis, x right, y down |
| `wrist_camera_link` | camera body frame: x along the view direction |
| `arm_gripper_frame_link` | point between the fingertips |
| `base_lidar_link`, `base_imu_link` | LiDAR and IMU, from Waveshare |

**Sensors**

| Sensor | Field of view | Range |
|---|---|---|
| Front OAK-D Lite (RGB) | 69° × 54° | 0.05 to 20 m |
| Wrist USB camera | 160° fisheye; simulated as 120° pinhole | 0.02 to 5 m |
| LiDAR | 360° | 0.02 to 12 m |

All three are in the URDF's `<gazebo>` blocks. The MJCF carries the two cameras as `front_camera` and `wrist_camera`, plus a `gripper_frame` site.

## Files

```
urdf/so101_rover.urdf          ROS version   (mesh paths package://so101_rover_description/meshes/...)
mujoco/so101_rover_sim.urdf    plain version (mesh paths ../meshes/...)  -> Isaac Sim, Gazebo without ROS, other tools
mujoco/so101_rover.xml         MJCF: free-floating rover on a floor, actuators, cameras  -> MuJoCo
export/so101_rover_assembly.*  whole robot as one STL (m and mm) and a coloured GLB, arm in a display pose
export/view_*.png              renders from the same angles as reference_photos/
meshes/rover|so101|mount       STL meshes (rover and arm in metres, printed parts in mm with scale 0.001)
scripts/build_urdf.py          builds both URDFs from the upstream files in source_models/
scripts/build_mjcf.py          builds the MJCF from the plain URDF
tests/check_model.py           read-only checks: URDF tree, meshes, inertias, measured heights, 3 s MuJoCo run, camera images
tests/export_views.py          STL / GLB export and reference renders
reference_photos/              photos of the real robot the model was matched to
```

## Using it

**Isaac Sim**: File → Import → URDF and pick `mujoco/so101_rover_sim.urdf`.

- Keep "merge fixed joints" on.
- Set the four driven wheel joints to velocity drive, the arm joints to position drive, and the middle wheels to no drive.
- Mesh paths are relative, so no ROS is needed.
- The importer renames `3d_camera_link` to `_3d_camera_link`, because prim names can't start with a digit. The name is kept so the TF frame matches Waveshare's ROS stack.

**MuJoCo**: `python -m mujoco.viewer --mjcf mujoco/so101_rover.xml`. The control vector is:

1. the 6 arm joint targets in radians, in LeRobot order;
2. then the 4 driven wheel speeds in rad/s: left up, left down, right up, right down.

Use the MJCF, not the URDF. Loaded raw, the URDF welds the rover to the world.

**ROS 2**: copy the folder into your workspace, run `colcon build`, then `ros2 launch so101_rover_description display.launch.py` for RViz with joint sliders.

**Rebuild** after changing a parameter:

```bash
python3 scripts/build_urdf.py [options] && python3 scripts/build_mjcf.py
python3 tests/check_model.py && python3 tests/export_views.py
```

| Option | Default | Use |
|---|---|---|
| `--deck-height` | 0.175 | ground to top of the upper metal deck, m |
| `--lidar-to-yaw` | 0.065 | LiDAR centre to the arm's yaw axis, m (read off the photos) |
| `--spacer` | 0 | spacer height under the plate, m |
| `--arm-battery` | off | add the 3S arm battery on the deck |
| `--wrist-mount none` | v2 | arm without the wrist camera |
| `--arm-facing rear` | front | the plate's other orientation (arm reaches backward) |

Python needs `numpy scipy lxml yourdfpy mujoco trimesh pillow`.

## Key dimensions (as modelled)

| | value |
|---|---|
| Ground → top of the upper metal deck | 0.175 m (measured) |
| Ground → underside of the SO-101 base | 0.185 m (deck + 10 mm plate) |
| Ground → top of the lower body | 0.093 m (from the photos; 9.2 cm from the lower body to the arm base) |
| LiDAR centre → arm yaw axis | 0.065 m |
| Wheel diameter | 0.080 m, 6 wheels, 4 driven |
| Total mass | 4.46 kg (rover body 2.37 + 6 wheels 1.24, arm 0.63, plate 0.11, LiDAR + OAK 0.09, wrist camera 0.03; no battery) |

## Known approximations

1. **Lower body to arm base:** 9.2 cm in the model against your 8.5 cm tape measurement. The photos give 9.3 cm, so the model follows the photos.
2. **Arm fore/aft position:** read off the photos, good to about ±1 cm. Measure the LiDAR centre to the centre of the arm's first motor and pass `--lidar-to-yaw` to make it exact.
3. **Wrist camera:** the USB board is tied by hand to a bracket designed for the OAK-D Lite, so its pose is approximate. Calibrate it against a real image if precise extrinsics matter, then edit `USB_CAM_LENS` in `scripts/build_urdf.py`.
4. **Deck, standoffs and Jetson:** simple boxes at measured positions.
5. **Printed parts:** box inertias. The rover and arm inertias are the upstream values.

## Printing the parts

The two printed parts on the robot are in `meshes/mount/`, at true size in millimetres:

- `rover_adapter_plate.stl`: black PETG, 60% infill.
- `wrist_cam_bracket_v2.stl`: white PETG or ASA, 40% infill.

## Licences

| Files | Licence |
|---|---|
| Rover meshes and URDF | Waveshare `ugv_ws`; see that repo |
| SO-101 meshes and URDF | TheRobotStudio SO-ARM100, Apache-2.0 |
| Adapter, bracket and this package | MIT |
