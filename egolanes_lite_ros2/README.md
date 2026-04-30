# EgoLanesLite ROS 2 Integration Guide

This repository contains the `egolanes_lite_ros2` package, a ROS 2 wrapper for performing real-time lane segmentation using the trained EgoLanesLite model via ONNX Runtime. This guide provides a full walkthrough for installing, configuring, and integrating the node with AWSIM and Autoware.

## Prerequisites

- **ROS 2:** Humble (recommended) or Galactic.
- **Autoware.Universe:** Installed and built.
- **AWSIM:** For simulation-based testing.
- **Python Dependencies:**
  ```bash
  pip3 install onnxruntime-gpu opencv-python numpy
  ```
  *(Note: You can use `onnxruntime` if you only plan to run on CPU, but GPU acceleration is highly recommended for real-time inference).*

## 1. Installation

1. **Clone the Package:**
   Clone or copy the `egolanes_lite_ros2` directory into the `src` folder of your Autoware workspace (or a standalone ROS 2 workspace):
   ```bash
   cd ~/autoware/src/universe/autoware.universe/perception/  # Example path
   git clone <your-repo-url> egolanes_lite_ros2
   ```

2. **Build the Package:**
   Navigate to the root of your workspace and build the package using `colcon`:
   ```bash
   cd ~/autoware
   colcon build --packages-select egolanes_lite_ros2 --symlink-install
   ```

3. **Source the Workspace:**
   ```bash
   source install/setup.bash
   ```

## 2. Configuration & Model Preparation

The node parameters are defined in `config/params.yaml`. 

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `EgoLanesLite_best.onnx` | Path to the ONNX model. Can be absolute or relative to the workspace/package. |
| `image_topic` | string | `/sensing/camera/traffic_light/image_raw` | Input camera image topic. |
| `mask_topic` | string | `/perception/lane_detection/mask` | Output segmentation mask topic. |
| `input_h` | int | `400` | Model input height. |
| `input_w` | int | `800` | Model input width. |
| `threshold` | float | `0.0` | Confidence threshold for masking logic. |
| `mean` | double[] | `[0.485, 0.456, 0.406]` | Image normalization mean values. |
| `std` | double[] | `[0.229, 0.224, 0.225]` | Image normalization std dev values. |
| `use_cuda` | bool | `true` | Enable CUDA (GPU) inference if available. |

**Model Placement:** Ensure your exported `EgoLanesLite_best.onnx` is located where the node can find it (e.g., in the workspace root, or use an absolute path in `params.yaml`).

## 3. AWSIM Integration Walkthrough

To validate the model in simulation using AWSIM:

1. **Launch AWSIM:** Start the AWSIM executable and load your desired map.
2. **Identify the Target Camera:** AWSIM publishes multiple camera feeds. Find the forward-facing camera topic using:
   ```bash
   ros2 topic list | grep camera
   ```
   *(Let's assume the topic is `/sensing/camera/front/image_raw`)*
3. **Update Configuration:** 
   Edit `config/params.yaml` and update the `image_topic` to match the AWSIM camera:
   ```yaml
   egolanes_lite_node:
     ros__parameters:
       image_topic: "/sensing/camera/front/image_raw"
       # ...
   ```
4. **Run the Node:**
   Execute the launch file:
   ```bash
   ros2 launch egolanes_lite_ros2 egolanes_lite.launch.py
   ```
5. **Visualize Results:**
   Open `rqt_image_view` to see the live segmentation mask published by the node:
   ```bash
   ros2 run rqt_image_view rqt_image_view
   ```
   Select `/perception/lane_detection/mask` from the dropdown. *Note: The output is a `mono8` image where pixel values map to class IDs (0: background, 1: left lane, 2: right lane, 3: other).*

## 4. Autoware Integration Walkthrough

To embed the EgoLanesLite node permanently into the Autoware perception stack:

1. **Locate Autoware Launch Files:**
   Find the launch file responsible for perception modules in your Autoware setup. This is typically located at:
   `autoware_launch/launch/tier4_perception_launch/launch/perception.launch.xml`
   *(Path may vary depending on your specific Autoware.Universe configuration).*

2. **Include the Node Launch:**
   Inject the `egolanes_lite.launch.py` into the perception launch file. You can override parameters directly in the XML to ensure it connects to the correct Autoware sensing topics.
   ```xml
   <!-- Add inside the perception group/pipeline -->
   <group>
     <push-ros-namespace namespace="lane_detection"/>
     <include file="$(find-pkg-share egolanes_lite_ros2)/launch/egolanes_lite.launch.py">
       <!-- Override config if necessary -->
     </include>
   </group>
   ```

3. **Downstream Pipeline Consideration:**
   The output of this node is a raw semantic mask. Autoware's planning modules usually expect lane vectors or costmaps. Depending on your system architecture, you may need to write or configure a post-processing node that subscribes to `/perception/lane_detection/mask`, extracts the lane centerlines (class 1 and 2), and converts them into `autoware_auto_perception_msgs/msg/PointClusters` or a format ingestible by the `lane_planner` or `freespace_planner`.
