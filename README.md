# EgoLanes & EgoLanesLite

**Real-time ego-lane segmentation for autonomous driving.**

EgoLanes predicts a **3-channel binary lane mask** from a single front-camera RGB image, identifying the lanes most relevant to the ego vehicle:

| Channel | Meaning                 |
| ------- | ----------------------- |
| 0       | Ego-left lane boundary  |
| 1       | Ego-right lane boundary |
| 2       | Other lanes             |

The active training and deployment path is **EgoLanesLite**, which pairs industry-standard backbones (EfficientNet-B0 via [timm](https://github.com/huggingface/pytorch-image-models)) with **DeepLabV3+** or **UNet++** decoders from [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch). A legacy custom pipeline (`EgoLanesNetwork`) is retained for backward compatibility.

> For the full architecture reference — model internals, loss formulations, data pipeline details, legacy modules, and academic references — see **[EGOLANES_ARCHITECTURE.md](EGOLANES_ARCHITECTURE.md)**.

---

## Key Features

- **3-channel ego-lane segmentation** — left, right, and other lanes as separate binary masks
- **Flexible encoder-decoder stack** — swap backbones via config; DeepLabV3+ (default) or UNet++ decoder
- **Multi-scale edge loss** — Sobel-based boundary supervision across 5 scales for sharp lane edges
- **Semantic-aware augmentation** — horizontal flips automatically swap left/right lane channels
- **ONNX-first deployment** — export to ONNX with dynamic batch axis for TensorRT
- **ROS 2 perception node** — drop-in ONNX Runtime node publishing `mono8` lane masks
- **Transfer learning controls** — partial encoder/decoder freeze and cross-task checkpoint loading
- **YAML-driven configuration** — all training, inference, and export settings in config files

---

## Repository Structure

```text
Repository/
├── model_components/           # Model architectures (Lite + Legacy)
│   └── lite_models/            # DeepLabV3+, UNet++, heads, modules
├── training/                   # Training entry point and trainer classes
├── data_utils/                 # Dataloaders, augmentation, losses, metrics
├── inference/                  # Standalone inference scripts (.pth / .onnx)
├── configs/                    # YAML configs for training and inference
├── exports/                    # Evaluation harness
├── egolanes_lite_ros2/         # ROS 2 ONNX perception node
├── export_to_onnx.py           # ONNX export utility
├── requirements.txt            # Python dependencies
└── EGOLANES_ARCHITECTURE.md    # Full architecture documentation
```

---

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python training/train_ego_lanes_lite.py -c configs/EgoLanesLite_train.yaml
```

### Inference (checkpoint)

```bash
python inference/ego_lanes_lite_infer.py \
  --config configs/EgoLanesLite_infer.yaml \
  --checkpoint best.pth \
  --input path/to/images \
  --output runs/inference/egolanes_lite
```

### Inference (ONNX)

```bash
python inference/ego_lanes_lite_infer.py \
  --config configs/EgoLanesLite_infer.yaml \
  --onnx EgoLanesLite.onnx \
  --input path/to/images \
  --output runs/inference/egolanes_lite_onnx
```

### Export to ONNX

```bash
python export_to_onnx.py \
  --config configs/EgoLanesLite_infer.yaml \
  --checkpoint best.pth \
  --output EgoLanesLite.onnx
```

### Evaluate

```bash
python exports/lite_models/eval_egolaneslite.py \
  --checkpoint best.pth \
  --datasets carla \
  --height 400 --width 800 \
  --batch-size 1 --device cuda \
  --out_dir runs/eval --viz 10
```

---

## ROS 2 Deployment

Build and launch the ONNX Runtime perception node:

```bash
colcon build --packages-select egolanes_lite_ros2
source install/setup.bash
ros2 launch egolanes_lite_ros2 egolanes_lite.launch.py model_path:=EgoLanesLite_best.onnx
```

The node subscribes to a camera image topic and publishes a `mono8` mask where pixel values encode lane class (`0`=background, `1`=ego-left, `2`=ego-right, `3`=other).

---

## Dataset Preparation

Convert single-channel lane masks (e.g., TuSimple) into 3-channel EgoLanes format:

```bash
python data_utils/convert_tulane_to_carla_egolanes.py \
  --src-root dataset/TuLaneConverted \
  --dst-root dataset/CarlaEgoLanes/processed \
  --preview-dir dataset/CarlaEgoLanes/previews
```

Validate converted masks:

```bash
python data_utils/verify_carla_egolanes.py \
  --data-root dataset/CarlaEgoLanes/processed
```

Expected dataset layout:

```text
<dataset_root>/processed/
├── image/   (*.jpg, *.png)
└── mask/    (*.png — 3-channel, binary 0/255)
```

---

## Documentation

| Document                                                           | Description                                                                                                                                    |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [EGOLANES_ARCHITECTURE.md](EGOLANES_ARCHITECTURE.md)               | Complete architecture guide — model internals, loss math, data pipeline, training details, legacy vs. Lite comparison, and academic references |
| [configs/EgoLanesLite_train.yaml](configs/EgoLanesLite_train.yaml) | Training configuration reference                                                                                                               |
| [configs/EgoLanesLite_infer.yaml](configs/EgoLanesLite_infer.yaml) | Inference / export configuration reference                                                                                                     |
| [training/train_ego_lanes.py](training/train_ego_lanes.py)         | Legacy training entry point                                                                                                                    |
| [training/ego_lanes_trainer.py](training/ego_lanes_trainer.py)     | Legacy trainer class bridging EgoLanesNetwork                                                                                                  |

---

## Acknowledgments

The original code belongs to [Autoware Vision Pilot](https://github.com/autowarefoundation/autoware_vision_pilot/tree/main) and has been adjusted and modified for the requirements of this repository.
