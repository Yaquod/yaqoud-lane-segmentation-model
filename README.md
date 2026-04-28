# EgoLanesLite Project Guide

This repository trains, fine-tunes, evaluates, and runs inference for an EgoLanesLite lane segmentation model. The lite model is configured as a 3-channel lane mask predictor:

- channel 0: ego-left lane
- channel 1: ego-right lane
- channel 2: other lanes

Masks must be binary PNGs with values `0` or `255` in each channel.

## Project Layout

```text
.
├── EgoLanesLite.yaml                  # Base training config
├── EgoLanesLite_carla.yaml            # Carla/TuLane-derived fine-tuning config
├── checkpoint.pth                     # Example checkpoint
├── data_utils/
│   ├── convert_tulane_to_carla_egolanes.py
│   └── lite_models/
│       ├── dataloaders/
│       └── helpers/
├── exports/lite_models/eval_egolaneslite.py
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

### Carla/EgoLanesLite Processed Format

The preferred format for this repo is:

```text
dataset/CarlaEgoLanes/processed/
  image/
    sample_000001.jpg
  mask/
    sample_000001.png
```

Rules:

- Every image must have a mask with the same basename.
- Images may be `.jpg`, `.jpeg`, or `.png`.
- Masks must be 3-channel PNGs.
- Mask shape must be `H x W x 3`.
- Mask values must be binary: `0` or `255`.
- Channels are RGB order: ego-left, ego-right, other.

The Carla config points here by default:

```yaml
dataset:
  training_sets: ["carla"]
  validation_sets: ["carla"]
  carla_root: "dataset/CarlaEgoLanes/processed"
```

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

Use the base config:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite.yaml
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

## Fine-Tune On Carla/TuLane-Derived Data

Use the Carla config:

```bash
python training/train_ego_lanes_lite.py -c EgoLanesLite_carla.yaml
```

The current fine-tune config uses:

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
python training/train_ego_lanes_lite.py -c EgoLanesLite_carla.yaml
```

Resume mode loads:

- model weights
- optimizer state
- scheduler state
- best metric
- epoch and global step counters

Use resume when continuing the same training run, not when adapting to a new dataset.

## Checkpoint Inference On Images

Use the lite-specific inference script:

```bash
python inference/ego_lanes_lite_infer.py \
  --config EgoLanesLite_carla.yaml \
  --checkpoint checkpoint.pth \
  --input path/to/images_or_single_image \
  --output runs/inference/egolanes_lite
```

Outputs:

```text
runs/inference/egolanes_lite/
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

To create a video from all the predicted overlays, add the `--video` flag. This will save a smooth MP4 file, `output_overlay.mp4`, generated at the specified `--fps` (default 30) directly in the output directory.

The inference script uses the model architecture from the YAML config, so the config must match the checkpoint architecture.

## Evaluate A Checkpoint

Evaluation script:

```bash
python exports/lite_models/eval_egolaneslite.py \
  --checkpoint checkpoint.pth \
  --datasets carla \
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

Note: the eval helper has its own default config. Make sure command-line architecture settings match the checkpoint. For Carla support, the default eval config may also need `carla_root` support if not already present in the branch you run.

## Check Dataset Health

Useful checks:

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

## Recommended Workflow

1. Install requirements in the cloud environment.
2. Convert or prepare the dataset in `dataset/CarlaEgoLanes/processed`.
3. Run a small conversion with previews and visually check labels.
4. Run checkpoint inference on a few images to see baseline behavior.
5. Fine-tune using `EgoLanesLite_carla.yaml`.
6. Monitor validation mIoU and saved visualizations.
7. Resume from `last.pth` if interrupted.
8. Use `best.pth` for final inference/evaluation.
