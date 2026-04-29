# EgoLanes / EgoLanesLite: Complete Architecture Guide

This repository contains **two lane-segmentation stacks**:

1. **Lite stack (active training path)** built on `segmentation_models_pytorch` encoders/decoders and configured through YAML.
2. **Legacy stack (`EgoLanesNetwork`)** with a custom encoder-fusion-context-neck-head architecture, currently used mainly for legacy inference compatibility.

The project predicts a **3-channel lane mask**:

- channel 0: ego-left lane
- channel 1: ego-right lane
- channel 2: other lanes

All lane masks are expected to be binary (`0` or `255`) and shaped `H x W x 3`.

---

## 1) High-level system architecture

```text
Dataset (processed image/mask pairs)
        |
        v
DataLoader + Augmentations
        |
        v
Model (Lite: DeepLabV3+ or Unet++; Legacy: EgoLanesNetwork)
        |
        v
Loss + Metrics
        |
        v
Trainer (checkpointing, validation, logging)
        |
        +--> Inference scripts (.pth / .onnx)
        |
        +--> Eval script (dataset-level mIoU / pixel-acc)
        |
        +--> ONNX export pipeline
```

---

## 2) Repository map (what each part owns)

| Path | Responsibility |
|---|---|
| `training/train_ego_lanes_lite.py` | Entry point for Lite training. Loads YAML, seeds RNG, starts trainer. |
| `training/lite_trainer_base.py` | Core trainer framework: dataset construction, model build, checkpoint resume, logging, optimizer stepping. |
| `training/ego_lanes_lite_trainer.py` | Lane-specific training loop, `LanesLoss`, lane validation/checkpoint policy. |
| `model_components/lite_models/*` | Lite model implementations (`DeepLabV3Plus`, `UnetPlusPlus`, regression heads, bottleneck/CBAM, initialization, load/freeze logic). |
| `data_utils/lite_models/dataloaders/*` | Dataset loaders (`Carla`, `TUSimple`, `CurveLanes`) and shared base loader. |
| `data_utils/lite_models/augmentation/*` | Albumentations pipelines for lane/segmentation/depth tasks. |
| `data_utils/lite_models/helpers/*` | Losses, optimizers/schedulers, validation metrics/visualization, W&B logger, depth helpers. |
| `inference/ego_lanes_lite_infer.py` | Inference for Lite checkpoints and ONNX exports. |
| `inference/ego_lanes_infer.py` | Inference for legacy `EgoLanesNetwork`. |
| `exports/lite_models/eval_egolaneslite.py` | Evaluation harness for `.pth` and `.onnx` Lite models. |
| `export_to_onnx.py` | ONNX export utility from Lite `.pth` checkpoint. |
| `data_utils/convert_tulane_to_carla_egolanes.py` | Converts single-channel TuLane masks into 3-channel EgoLanes format. |
| `data_utils/verify_carla_egolanes.py` | Validates converted mask shape, binary values, and channel exclusivity. |
| `model_components/ego_lanes_network.py` + siblings | Legacy custom model pipeline used by non-lite inference. |
| `EgoLanesLite_train.yaml` / `EgoLanesLite_infer.yaml` | Main train and infer configurations. |

---

## 3) Model stacks in detail

### 3.1 Lite stack (current training/inference default)

The Lite stack is built in `LiteTrainerBase._build_model_stack()` and supports:

- `network.model: "deeplabv3plus"` -> `model_components/lite_models/DeepLabv3Plus.py`
- `network.model: "unetplusplus"` -> `model_components/lite_models/UnetPlusPlus.py`

Both models inherit from `BaseModel` and use:

- SMP encoders via `get_encoder(...)`
- decoder-specific modules (DeepLabV3+ decoder or Unet++ decoder)
- a configurable `RegressionHead` for final lane logits (`C=3`)

`DeepLabV3Plus` additionally supports:

- optional bottleneck block (`none`, `fcn`, `fcn_cbam`, `fcn_skip`, `fcn_skip_cbam`)
- loading encoder/decoder from a segmentation checkpoint
- full or partial encoder loading + partial freeze

### 3.2 Legacy stack (`EgoLanesNetwork`)

Defined by:

- `model_components/backbone.py`
- `model_components/backbone_feature_fusion.py`
- `model_components/auto_steer_context.py`
- `model_components/ego_path_neck.py`
- `model_components/ego_lanes_head.py`
- wired in `model_components/ego_lanes_network.py`

Flow:

1. EfficientNet-B0 feature extraction.
2. Multi-scale feature pooling + concatenation.
3. Context branch (MLP + conv attention-like modulation).
4. Decoder neck with transpose-conv upsampling and skip links.
5. Final 3-channel logits head.

This stack is used by `inference/ego_lanes_infer.py`.

---

## 4) Data pipeline and dataset contracts

### 4.1 Expected processed format

Each lane dataset loader expects:

```text
<dataset_root>/processed/
  image/*.jpg|*.jpeg|*.png
  mask/*.png
```

Pairing is by basename (same filename stem).

### 4.2 Implemented lane loaders

| Loader | File | Notes |
|---|---|---|
| CarlaDataset | `data_utils/lite_models/dataloaders/CarlaDataset.py` | Accepts `.jpg/.jpeg/.png` images, deterministic split by index `% 10`, supports `test` split. |
| TUSimpleDataset | `.../TUSimpleDataset.py` | Reads `.jpg`; deterministic split `% 10`; caps val to 500. |
| CurveLanesDataset | `.../CurveLanesDataset.py` | Reads `.jpg`; deterministic split `% 10`; caps val to 500. |

### 4.3 BaseDataset behavior

`BaseDataset.__getitem__` does:

1. read RGB image
2. read GT mask (for lane task)
3. run augmentation pipeline on image+mask
4. cast image to CHW float32
5. cast lane mask to CHW float32 in `[0,1]` (from `0/255`)

Hard validation:

- lane masks must be 3-channel
- non-existing masks raise errors

### 4.4 Augmentation behavior (lane task)

`LanesAugmentation` supports:

- fixed resize or random crop scaling mode
- optional horizontal flip
- optional noise profile
- optional normalization

Important lane-specific logic:

- if horizontal flip is applied, lane channels 0 and 1 are swapped so left/right semantics remain correct.

---

## 5) Training architecture

### 5.1 Entry flow

`training/train_ego_lanes_lite.py`:

1. parse `--config`
2. load YAML
3. set seeds
4. create `EgoLanesLiteTrainer(cfg)`
5. call `trainer.run()`

### 5.2 Trainer build stages

`EgoLanesLiteTrainer.__init__` performs:

1. output directory initialization
2. W&B logger setup
3. train/val loader construction
4. network config normalization (`efficientnet_b0` -> `timm-efficientnet-b0`)
5. model creation
6. loss/optimizer/scheduler creation
7. training state setup
8. optional checkpoint resume or fine-tune bootstrap

### 5.3 Loop behavior

`run()` supports:

- `training.mode = "epoch"` or `"steps"`
- gradient accumulation via `grad_accum_steps`
- validation by steps or epochs
- `last.pth` save and best-model save by mean mIoU

### 5.4 Lane loss and metrics

`LanesLoss` combines for each channel:

- `BCEWithLogitsLoss`
- multi-scale edge loss (Sobel-based)

Total:

`2 * left + 2 * right + 1 * other`

Validation (`validate_lanes`) reports:

- loss
- mean IoU
- pixel accuracy
- per-class IoU
- per-class pixel accuracy
- optional visualization panels for logging/saving

### 5.5 Checkpoint schema

Trainer checkpoints (`last.pth` / `best.pth`) contain:

- `epoch`
- `step`
- `model_state`
- `optimizer_state`
- `scheduler_state`
- `best`
- `wandb_run_id`

---

## 6) Inference architecture

### 6.1 Lite inference (`inference/ego_lanes_lite_infer.py`)

Supports either:

- PyTorch checkpoint (`--checkpoint`)
- ONNX model (`--onnx`) through `onnxruntime` wrapper

Pipeline per image:

1. read BGR -> RGB
2. resize to config size
3. normalize
4. model forward -> logits
5. threshold (`logits > threshold`)
6. resize mask back to original resolution
7. save raw 3-channel mask and colored overlay, or compose video

### 6.2 Legacy inference (`inference/ego_lanes_infer.py`)

Uses fixed `EgoLanesNetwork` architecture.

Defaults:

- input size `640x320`
- ImageNet normalization

Optional `--config` can override preprocessing fields.

---

## 7) Evaluation and export

### 7.1 Evaluation (`exports/lite_models/eval_egolaneslite.py`)

Capabilities:

- evaluate `.pth` checkpoint (through full trainer/model stack)
- evaluate `.onnx` backend (uses ONNXRuntime model wrapper + trainer-built dataloaders/loss)

Outputs:

- mean IoU
- pixel accuracy
- validation loss
- per-dataset samples saved to `--out_dir`

### 7.2 ONNX export (`export_to_onnx.py`)

Process:

1. load infer config and build Lite model
2. load checkpoint weights
3. export with `torch.onnx.export(...)`
4. validate graph with `onnx.checker`

By default dynamic axis is enabled for batch dimension.

---

## 8) Dataset conversion and validation utilities

### 8.1 TuLane -> EgoLanes converter

`data_utils/convert_tulane_to_carla_egolanes.py` converts single-channel lane masks into 3-channel masks.

Methods:

- `bottom_up` (default): traces lane runs from the bottom upward
- `component_scan`: connected-components + center proximity logic

It can symlink/hardlink/copy images and optionally save preview masks.

### 8.2 Converted dataset validator

`data_utils/verify_carla_egolanes.py` checks:

- image/mask pairing
- 3-channel mask format
- binary values (`0/255`)
- channel overlap violations

---

## 9) Configuration reference (what code actually reads)

### 9.1 Training config consumers

| Code area | Expected key path |
|---|---|
| device + seed + output + wandb | `experiment.*` |
| training/validation dataset selection | `dataset.training_sets`, `dataset.validation_sets` |
| dataset roots | `dataset.<name>_root` (e.g., `dataset.carla_root`) |
| augmentations | `dataset.augmentations.*` |
| dataloader | `dataloader.*` |
| train mode/epochs/steps/accum | `training.*` |
| validation cadence | `training.validation.*` |
| logging cadence | `training.logging.*` |
| save policies | `training.save_best`, `training.save_last` |
| optimizer | `optimizer.*` |
| scheduler | `scheduler.*` |
| lane loss | `loss.downsample_factor` |
| architecture | `network.*` |
| checkpoint loading mode | `checkpoint.*` |

### 9.2 Inference config consumers (lite inference script)

Read from:

- `dataset.augmentations.rescaling.height/width`
- `dataset.augmentations.normalize.mean/std`
- `network.*` (for building model)

---

## 10) Known gotchas in current repository state

1. `LiteTrainerBase` expects augmentation config under `dataset.augmentations`, while `EgoLanesLite_train.yaml` currently places `augmentations` at top level.
2. Validation/logging/save options are read from `training.validation`, `training.logging`, `training.save_*`; in `EgoLanesLite_train.yaml` they are currently top-level keys.
3. `build_single_dataset(...)` includes paths for datasets whose loader files are not in this repository snapshot; lane training with `carla/tusimple/curvelanes` is the safe path.
4. Legacy utility files (`data_utils/load_data_ego_lanes.py`, `data_utils/augmentations.py`) are old pipeline code and are not used by the Lite trainer.

If you want exact behavior from YAML, match the key nesting expected by the trainer code.

---

## 11) Quickstart commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Train Lite model:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite_train.yaml
```

Lite inference with checkpoint:

```bash
python inference/ego_lanes_lite_infer.py \
  --config EgoLanesLite_infer.yaml \
  --checkpoint checkpoint.pth \
  --input path/to/images \
  --output runs/inference/egolanes_lite
```

Lite inference with ONNX:

```bash
python inference/ego_lanes_lite_infer.py \
  --config EgoLanesLite_infer.yaml \
  --onnx EgoLanesLite.onnx \
  --input path/to/images \
  --output runs/inference/egolanes_lite_onnx
```

Legacy inference:

```bash
python inference/ego_lanes_infer.py \
  --checkpoint checkpoint.pth \
  --input path/to/images \
  --output runs/inference/egolanes_legacy
```

Export to ONNX:

```bash
python export_to_onnx.py \
  --config EgoLanesLite_infer.yaml \
  --checkpoint checkpoint.pth \
  --output EgoLanesLite.onnx
```

Evaluate:

```bash
python exports/lite_models/eval_egolaneslite.py \
  --checkpoint checkpoint.pth \
  --datasets carla \
  --height 400 \
  --width 800 \
  --batch-size 1 \
  --device cuda \
  --out_dir runs/eval/egolanes_lite_eval \
  --viz 10
```

Convert TuLane masks:

```bash
python data_utils/convert_tulane_to_carla_egolanes.py \
  --src-root dataset/TuLaneConverted \
  --dst-root dataset/CarlaEgoLanes/processed \
  --preview-dir dataset/CarlaEgoLanes/previews
```

Verify converted masks:

```bash
python data_utils/verify_carla_egolanes.py \
  --data-root dataset/CarlaEgoLanes/processed
```

---

## 12) ROS2 ONNX node (AWSIM / desktop simulation)

This repo now includes a ROS2 node:

- `egolanes_lite_ros2/egolanes_lite_ros2/egolanes_lite_node.py`

It runs an EgoLanesLite ONNX model with ONNX Runtime, subscribes to an RGB camera topic, and publishes a `mono8` lane mask image.

Published mask semantics:

- `0` = background
- `1` = ego-left lane
- `2` = ego-right lane
- `3` = other lane

Build and source:

```bash
colcon build --packages-select egolanes_lite_ros2
source install/setup.bash
```

Run with launch file:

```bash
ros2 launch egolanes_lite_ros2 egolanes_lite.launch.py
```

Run directly:

```bash
ros2 run egolanes_lite_ros2 egolanes_lite_node \
  --ros-args \
  -p model_path:=EgoLanesLite_best.onnx \
  -p image_topic:=/sensing/camera/traffic_light/image_raw \
  -p mask_topic:=/perception/lane_detection/mask \
  -p input_h:=400 \
  -p input_w:=800 \
  -p threshold:=0.0 \
  -p use_cuda:=true
```

Node parameters:

- `model_path` (string): ONNX model path
- `image_topic` (string): input `sensor_msgs/Image` topic
- `mask_topic` (string): output `sensor_msgs/Image` topic
- `input_h`, `input_w` (int): model input size used for resize
- `mean`, `std` (float array): normalization values
- `threshold` (float): logit threshold for lane activation
- `use_cuda` (bool): try `CUDAExecutionProvider` before CPU
