# egolanes_lite_ros2

A ROS 2 package that wraps the EgoLanesLite ONNX model for real-time lane segmentation and provides a full downstream pipeline: 2D mask → Bird's-Eye-View costmap → 3D metric point clusters.

## Package Overview

The package contains three nodes that form a sequential pipeline:

```
Camera Image
    │
    ▼
egolanes_lite_node        → /perception/lane_detection/mask        (mono8)
                          → /perception/lane_detection/mask_vis     (rgb8)
    │
    ├──▶ egolanes_ipm_node      → /perception/lane_detection/costmap   (OccupancyGrid)
    │                           → /perception/lane_detection/ipm_vis   (rgb8)
    │
    └──▶ egolanes_vectorizer_node → /perception/lane_detection/left_cluster  (PointCloud2)
                                  → /perception/lane_detection/right_cluster (PointCloud2)
                                  → /perception/lane_detection/lane_markers  (MarkerArray)
```

### Node Descriptions

| Node | Executable | Role |
|---|---|---|
| `EgoLanesLiteNode` | `egolanes_lite_node` | Runs ONNX inference on camera images, publishes a `mono8` class-ID mask |
| `EgoLanesIPMNode` | `egolanes_ipm_node` | Applies Inverse Perspective Mapping to produce a top-down `OccupancyGrid` |
| `EgoLanesVectorizerNode` | `egolanes_vectorizer_node` | Back-projects mask pixels to 3D metric points in `base_link` frame |

### Mask Class IDs

| Value | Meaning | Visualization color |
|---|---|---|
| `0` | Background | Black |
| `1` | Ego-left lane boundary | Red |
| `2` | Ego-right lane boundary | Green |
| `3` | Other lanes | Blue |

---

## Prerequisites

- **ROS 2** Humble (recommended) or Galactic
- **Python packages:**
  ```bash
  pip3 install onnxruntime-gpu opencv-python numpy
  # Use onnxruntime (CPU-only) if no NVIDIA GPU is available
  ```
- **ROS 2 packages:** `cv_bridge`, `nav_msgs`, `sensor_msgs`, `visualization_msgs`, `tf2_ros`

---

## Installation

1. Clone into your workspace `src/`:
   ```bash
   cd ~/autoware/src/universe/autoware.universe/perception/
   git clone <your-repo-url> egolanes_lite_ros2
   ```

2. Build:
   ```bash
   cd ~/autoware
   colcon build --packages-select egolanes_lite_ros2 --symlink-install
   ```

3. Source:
   ```bash
   source install/setup.bash
   ```

4. Place your ONNX model where the node can find it (see `model_path` below):
   ```bash
   cp EgoLanesLite_best.onnx ~/autoware/
   ```

---

## Configuration

All parameters live in `config/params.yaml`. The launch file loads this file automatically.

### egolanes_lite_node

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `EgoLanesLite_best.onnx` | Path to ONNX model. Absolute or relative to workspace root. |
| `image_topic` | string | `/sensing/camera/traffic_light/image_raw` | Input camera topic. |
| `mask_topic` | string | `/perception/lane_detection/mask` | Output `mono8` class-ID mask. |
| `mask_vis_topic` | string | `/perception/lane_detection/mask_vis` | Output `rgb8` colored overlay. |
| `input_h` | int | `400` | Model input height (auto-corrected from ONNX graph if mismatched). |
| `input_w` | int | `800` | Model input width. |
| `threshold` | float | `0.5` | Sigmoid threshold for lane/background decision. |
| `mean` | float[] | `[0.485, 0.456, 0.406]` | ImageNet normalization mean. |
| `std` | float[] | `[0.229, 0.224, 0.225]` | ImageNet normalization std. |
| `use_cuda` | bool | `true` | Use CUDA if available; falls back to CPU silently. |

### egolanes_ipm_node

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mask_topic` | string | `/perception/lane_detection/mask` | Input mask topic. |
| `costmap_topic` | string | `/perception/lane_detection/costmap` | Output `OccupancyGrid`. |
| `vis_topic` | string | `/perception/lane_detection/ipm_vis` | Output BEV visualization image. |
| `src_points` | float[8] | see yaml | Flat array of 4 (x,y) pixel coords forming a trapezoid on the **800×400 mask**. Order: TL, TR, BR, BL. **Must be tuned to your camera.** |
| `dst_points` | float[8] | `[100,0, 300,0, 300,400, 100,400]` | Flat array of 4 (x,y) coords forming a rectangle in the BEV grid. |
| `grid_resolution` | float | `0.05` | Metres per pixel in the output costmap. |
| `grid_width` | int | `400` | Costmap width in pixels. |
| `grid_height` | int | `400` | Costmap height in pixels. |
| `publish_vis` | bool | `true` | Publish BEV visualization to `vis_topic`. |

### egolanes_vectorizer_node

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mask_topic` | string | `/perception/lane_detection/mask` | Input mask topic. |
| `left_topic` | string | `/perception/lane_detection/left_cluster` | Output left lane `PointCloud2` in `base_link`. |
| `right_topic` | string | `/perception/lane_detection/right_cluster` | Output right lane `PointCloud2` in `base_link`. |
| `marker_topic` | string | `/perception/lane_detection/lane_markers` | Output `MarkerArray` polylines for RViz. |
| `publish_markers` | bool | `true` | Enable marker publication. |
| `base_link_frame` | string | `base_link` | Target TF frame for output points. |
| `camera_frame` | string | `""` | Camera TF frame. Auto-detected from `camera_info` if empty. |
| `use_tf_extrinsics` | bool | `true` | Use TF tree for camera→base_link transform (preferred). |
| `camera_fx` / `camera_fy` | float | `600.0` | Focal lengths (fallback if no `camera_info`). |
| `camera_cx` / `camera_cy` | float | `-1.0` | Principal point (`-1` = image centre). |
| `camera_height_m` | float | `1.2` | Camera height above ground in metres (manual extrinsics fallback). |
| `camera_pitch_deg` | float | `0.0` | Camera downward tilt in degrees (manual extrinsics fallback). |
| `row_step` | int | `4` | Sample one centroid every N rows (lower = denser point cloud). |
| `min_pixels_per_row` | int | `5` | Minimum lane pixels per row to emit a point (noise filter). |
| `max_range_m` | float | `30.0` | Discard ground-projected points beyond this distance. |

---

## Running

### Launch all nodes

```bash
ros2 launch egolanes_lite_ros2 egolanes_lite.launch.py
```

This starts `egolanes_lite_node` and `egolanes_ipm_node` using `config/params.yaml`.

This starts all three nodes: `egolanes_lite_node`, `egolanes_ipm_node`, and `egolanes_vectorizer_node`.

### Verify topics are publishing

```bash
ros2 topic list | grep lane_detection
ros2 topic hz /perception/lane_detection/mask
```

---

## AWSIM Integration

1. Launch AWSIM and load your map.

2. Find the forward camera topic:
   ```bash
   ros2 topic list | grep camera
   ```

3. Update `image_topic` in `config/params.yaml`:
   ```yaml
   egolanes_lite_node:
     ros__parameters:
       image_topic: "/sensing/camera/front/image_raw"
   ```

4. Launch:
   ```bash
   ros2 launch egolanes_lite_ros2 egolanes_lite.launch.py
   ```

5. Open RViz and add these topics:
   - `/perception/lane_detection/mask_vis` — 2D colored lane overlay
   - `/perception/lane_detection/ipm_vis` — top-down BEV projection
   - `/perception/lane_detection/costmap` — OccupancyGrid

6. **Tune `src_points`:** Look at `ipm_vis`. If lanes are not parallel and straight, adjust the `src_points` trapezoid in `params.yaml` to match your camera's resolution and pitch. Valid range for an 800×400 mask: x ∈ [0, 800], y ∈ [0, 400].

---

## Autoware Integration

1. Locate the Autoware perception launch file (path varies by setup):
   ```
   autoware_launch/launch/tier4_perception_launch/launch/perception.launch.xml
   ```

2. Include the node inside the perception group:
   ```xml
   <group>
     <push-ros-namespace namespace="lane_detection"/>
     <include file="$(find-pkg-share egolanes_lite_ros2)/launch/egolanes_lite.launch.py"/>
   </group>
   ```

3. Configure Autoware's `freespace_planner` or `costmap_generator` to subscribe to:
   ```
   /perception/lane_detection/costmap
   ```
   The costmap is published in the `base_link` frame with lane boundaries marked at occupancy value `100`.

4. For planning with 3D lane clusters, subscribe to:
   ```
   /perception/lane_detection/left_cluster
   /perception/lane_detection/right_cluster
   ```

---

## Camera Calibration for the Vectorizer Node

The vectorizer node needs camera intrinsics and extrinsics to back-project pixels to 3D.

**Intrinsics (preferred — live topic):**
```bash
ros2 topic echo /sensing/camera/traffic_light/camera_info --once
```
Look for the `K` matrix: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`. Set `camera_fx`, `camera_fy`, `camera_cx`, `camera_cy` in `params.yaml`.

**Extrinsics (preferred — TF tree):**
If Autoware's sensor calibration is loaded, the node auto-detects the camera frame from `camera_info` and looks up the transform. No manual configuration needed.

**Extrinsics (fallback — manual):**
Set `use_tf_extrinsics: false` and provide:
- `camera_height_m`: measure from road surface to camera lens
- `camera_pitch_deg`: downward tilt (0 = horizontal, positive = nose-down)

---

## Troubleshooting

**Mask looks pitch black in RViz:**
This is expected for the raw `mask` topic — pixel values are 0–3, which appear black. Use `mask_vis` instead.

**`FileNotFoundError` for ONNX model:**
The node searches the current working directory and all parent directories. Use an absolute path in `model_path` to be safe:
```yaml
model_path: "/home/user/models/EgoLanesLite_best.onnx"
```

**IPM output looks distorted:**
The `src_points` trapezoid is sized for an 800×400 image (the model output mask). Do not use coordinates from the original camera resolution. Tune using the `ipm_vis` topic.

**Vectorizer publishes empty point clouds:**
Check that `camera_info` is being published on the expected topic. If TF lookup fails, the node falls back to manual extrinsics and logs a warning — verify `camera_height_m` and `camera_pitch_deg` are set correctly.

**CUDA not used despite `use_cuda: true`:**
Verify `onnxruntime-gpu` is installed and that `CUDAExecutionProvider` appears in:
```python
import onnxruntime; print(onnxruntime.get_available_providers())
```
