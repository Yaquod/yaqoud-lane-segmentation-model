# EgoLanesLite Project Guide

This repository trains, fine-tunes, evaluates, and runs inference for an EgoLanesLite lane segmentation model. It also includes a separate inference script for the older non-lite `EgoLanesNetwork`. The lite model is configured as a 3-channel lane mask predictor:

- channel 0: ego-left lane
- channel 1: ego-right lane
- channel 2: other lanes

Masks must be binary PNGs with values `0` or `255` in each channel.

## Table of Contents

- [Project Layout](#project-layout)
- [Environment Setup](#environment-setup)
- [Dataset Formats](#dataset-formats)
- [Convert TuLaneConverted To 3-Channel Format](#convert-tulaneconverted-to-3-channel-format)
- [Train From Scratch](#train-from-scratch)
- [Fine-Tune With The Training Config](#fine-tune-with-the-training-config)
- [Resume Training](#resume-training)
- [Checkpoint Inference On Images](#checkpoint-inference-on-images)
- [Evaluate A Checkpoint](#evaluate-a-checkpoint)
- [Exporting `EgoLanesLite` to ONNX](#exporting-egolaneslite-to-onnx)
- [Check Dataset Health](#check-dataset-health)
- [Common Problems](#common-problems)
- [Recommended Workflow](#recommended-workflow)

## Project Layout

```text
.
├── EgoLanesLite_train.yaml            # Training config
├── EgoLanesLite_infer.yaml            # Inference config
├── checkpoint.pth                     # Example checkpoint
├── data_utils/
│   ├── convert_tulane_to_carla_egolanes.py
│   └── lite_models/
│       ├── dataloaders/
│       └── helpers/
├── exports/lite_models/eval_egolaneslite.py
├── inference/ego_lanes_infer.py
├── inference/ego_lanes_lite_infer.py
├── model_components/
├── training/train_ego_lanes_lite.py
└── requirements.txt
```

## Environment Setup

Install dependencies in your cloud or training environment:

```bash
pip install -r requirements.txt
```

Quick import check:

```bash
python -c "import torch, yaml, cv2, albumentations, segmentation_models_pytorch, timm, wandb"
```

Use a CUDA runtime if training on GPU.

## Dataset Formats

### Processed EgoLanesLite Format

The lane dataloaders in this repo expect this processed structure:

```text
<dataset_root>/processed/
  image/
    sample_000001.jpg
  mask/
    sample_000001.png
```

Rules:

- Every image must have a mask with the same basename.
- The Carla loader accepts `.jpg`, `.jpeg`, and `.png`; the TuSimple and CurveLanes loaders currently collect `.jpg`.
- Masks must be 3-channel PNGs.
- Mask shape must be `H x W x 3`.
- Mask values must be binary: `0` or `255`.
- Channels are RGB order: ego-left, ego-right, other.

For Carla-style converted data, the relevant dataset config keys are:

```yaml
dataset:
  training_sets: ["carla"]
  validation_sets: ["carla"]
  carla_root: "dataset/CarlaEgoLanes/processed"
```

`EgoLanesLite_train.yaml` is the training config. Its current dataset roots are `tusimple_root` and `curvelanes_root`, so change those paths if your processed training data lives somewhere else. `EgoLanesLite_infer.yaml` may still contain dataset keys for compatibility, but checkpoint inference only reads the network and preprocessing settings from it.

### Current TuLaneConverted Format

The existing TuLane-style data is:

```text
dataset/TuLaneConverted/
  images/{train,val,target_train,target_val,target_test}/*.jpg
  lane_masks/{train,val,target_train,target_val,target_test}/*.png
  list/*.txt
```

Those masks are single-channel binary masks. They are not directly compatible with the 3-channel EgoLanesLite training path unless converted.

## Convert TuLaneConverted To 3-Channel Format

Converter:

```bash
python data_utils/convert_tulane_to_carla_egolanes.py \
  --src-root dataset/TuLaneConverted \
  --dst-root dataset/CarlaEgoLanes/processed \
  --preview-dir dataset/CarlaEgoLanes/previews
```

By default the converter uses:

```bash
--method bottom_up
```

This traces lane strokes from the bottom of the image upward, then chooses the lanes closest to the image center near the bottom as ego-left and ego-right. Remaining lane pixels become `other`.

For a small test run:

```bash
python data_utils/convert_tulane_to_carla_egolanes.py \
  --limit 50 \
  --dst-root /tmp/CarlaEgoLanes_test/processed \
  --preview-dir /tmp/CarlaEgoLanes_test/previews
```

Important: conversion from a single binary mask to semantic left/right/other channels is heuristic. Always inspect preview images before training. If the previews mislabel lanes, fix the converter or dataset labels before fine-tuning.

## Train From Scratch

Use the training config:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite_train.yaml
```

Main config fields:

- `experiment.output_dir`: root folder for runs
- `dataset.training_sets`: datasets used for training
- `dataset.validation_sets`: datasets used for validation
- `checkpoint.load_from`: optional checkpoint path
- `checkpoint.fine_tune`: if true, load weights only and reset optimizer/scheduler
- `training.mode`: `steps` or `epoch`
- `training.max_steps`: used when mode is `steps`
- `training.max_epochs`: used when mode is `epoch`
- `training.validation.every_n_steps`: validation interval for step mode
- `network.output_channels`: should be `3` for EgoLanesLite lane masks

Training outputs are written under:

```text
runs/training/EgoLanesLite/<experiment_name>/
  checkpoints/
    best.pth
    last.pth
  logs/
```

## Fine-Tune With The Training Config

To fine-tune instead of training from scratch, edit `EgoLanesLite_train.yaml` and set `checkpoint.load_from` to the checkpoint path:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite_train.yaml
```

Fine-tuning uses:

```yaml
checkpoint:
  load_from: "checkpoint.pth"
  strict_load: true
  fine_tune: true
```

`fine_tune: true` means:

- model weights are loaded from the checkpoint
- optimizer state is not resumed
- scheduler state is not resumed
- epoch and step counters restart from zero

Use fine-tuning when adapting a pretrained checkpoint to a new dataset/domain.

## Resume Training

To resume an interrupted run, set:

```yaml
checkpoint:
  load_from: "runs/training/EgoLanesLite/<run_name>/checkpoints/last.pth"
  strict_load: true
  fine_tune: false
```

Then run:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite_train.yaml
```

Resume mode loads:

- model weights
- optimizer state
- scheduler state
- best metric
- epoch and global step counters

Use resume when continuing the same training run, not when adapting to a new dataset.

## Checkpoint Inference On Images

Use the lite-specific inference script for checkpoints trained with the configurable `EgoLanesLite` architecture:

```bash
python inference/ego_lanes_lite_infer.py \
  --config EgoLanesLite_infer.yaml \
  --checkpoint checkpoint.pth \
  --input path/to/images_or_single_image \
  --output runs/inference/egolanes_lite
```

Use the non-lite inference script for checkpoints trained with `model_components.EgoLanesNetwork`:

```bash
python inference/ego_lanes_infer.py \
  --checkpoint checkpoint.pth \
  --input path/to/images_or_single_image \
  --output runs/inference/egolanes
```

The non-lite script defaults to the original `EgoLanesNetwork` preprocessing size, `640x320`, with ImageNet RGB normalization. You can pass `--config some_config.yaml` if you want it to read `dataset.augmentations.rescaling` and `dataset.augmentations.normalize` from a YAML file.

Without `--video`, inference writes:

```text
runs/inference/<run_name>/
  masks/       # raw 3-channel binary predicted masks
  overlays/    # colored overlays on the original images
```

Useful options:

```bash
--device cuda
--threshold 0.0
--alpha 0.5
--video
--fps 30
```

To create a video from all the predicted overlays, add the `--video` flag. This saves `output_overlay.mp4`, generated at the specified `--fps` (default 30), directly in the output directory. In video mode the scripts do not save the per-image `masks/` and `overlays/` folders.

For `ego_lanes_lite_infer.py`, the script builds the model architecture from the YAML config, so the config must match the checkpoint architecture. For `ego_lanes_infer.py`, the architecture is the fixed non-lite `EgoLanesNetwork`, so the checkpoint must come from that model.

## Evaluate A Checkpoint

Evaluation script:

```bash
python exports/lite_models/eval_egolaneslite.py \
  --checkpoint checkpoint.pth \
  --datasets tusimple curvelanes \
  --backbone efficientnet_b0 \
  --height 400 \
  --width 800 \
  --head-upsampling 4 \
  --decoder-channels 64 \
  --batch-size 1 \
  --device cuda \
  --out_dir runs/eval/egolanes_lite_eval \
  --viz 10
```

This reports:

- validation loss
- mean IoU
- pixel accuracy
- per-class IoU and accuracy

It also saves visual validation samples into `--out_dir`.

Note: the eval helper has its own default config and dataset roots in `exports/lite_models/helpers.py`. Make sure those roots and the command-line architecture settings match the checkpoint. Although the dataloader factory supports `carla`, the eval helper does not currently expose a CLI argument for `carla_root`; add that root to its default config before evaluating with `--datasets carla`.

## Check Dataset Health

Useful checks for a processed dataset root such as `dataset/CarlaEgoLanes/processed`:

```bash
find dataset/CarlaEgoLanes/processed/image -maxdepth 1 -type f | wc -l
find dataset/CarlaEgoLanes/processed/mask -maxdepth 1 -type f | wc -l
```

Inspect mask shape and values:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

for p in sorted(Path("dataset/CarlaEgoLanes/processed/mask").glob("*.png"))[:10]:
    arr = np.array(Image.open(p))
    print(p.name, arr.shape, [int((arr[..., i] > 0).sum()) for i in range(3)], np.unique(arr).tolist())
PY
```

Expected:

- shape like `(H, W, 3)`
- unique values `[0, 255]`
- nonzero pixels in the expected channels

## Common Problems

### `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

### Checkpoint shape mismatch

Your YAML architecture does not match the checkpoint. Check:

- `network.backbone.type`
- `network.backbone.output_stride`
- `network.decoder.deeplabv3plus_decoder_channels`
- `network.head.head_upsampling`
- `network.output_channels`

### Training mask shape error

EgoLanesLite expects 3-channel masks. If masks are single-channel, convert or regenerate labels.

### Bad left/right conversion from TuLane

The TuLane converter is heuristic. Use `--limit` and `--preview-dir` first, then inspect previews. Do not train on converted data until the previews look correct enough for your task.

### W&B errors

Disable W&B in the config:

```yaml
experiment:
  wandb:
    enabled: false
```

## Exporting `EgoLanesLite` to ONNX

To deploy the PyTorch model (`.pth`) to production platforms like TensorRT, OpenVINO, or ONNX Runtime, you can use the provided script to export it to `.onnx`.

### 1. Install ONNX Dependencies

Ensure you have `onnx` and `onnxruntime` installed in your environment:

```bash
pip install onnx onnxruntime
```

### 2. Run the Export

Execute the script `export_to_onnx.py` to generate your ONNX model. By default, it uses `EgoLanesLite_infer.yaml` and `checkpoint.pth`.

```bash
python export_to_onnx.py
```

You can optionally override inputs on the command line:

```bash
python export_to_onnx.py \
  --config EgoLanesLite_infer.yaml \
  --checkpoint checkpoint.pth \
  --output my_model_weights.onnx \
  --width 800 --height 400
```

This script will read the necessary input dimensions from the config, rebuild the network, load the states, test the inference, dynamically determine output axes sizes, export to ONNX, and then finally run `onnx.checker` against the built structure to validate it. The output will be `EgoLanesLite.onnx` (or the configured `--output` name), ready for deployment.

## Recommended Workflow

1. Install requirements in the cloud environment.
2. Convert or prepare the dataset in the processed `image/` and `mask/` layout.
3. Run a small conversion with previews and visually check labels.
4. Run checkpoint inference on a few images to see baseline behavior.
5. Train or fine-tune using `EgoLanesLite_train.yaml`.
6. Monitor validation mIoU and saved visualizations.
7. Resume from `last.pth` if interrupted.
8. Use `best.pth` for final inference/evaluation.
