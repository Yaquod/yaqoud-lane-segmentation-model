#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def verify_dataset(data_root: Path):
    image_dir = data_root / "image"
    mask_dir = data_root / "mask"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        print(f"Error: Missing image or mask directory in {data_root}")
        return

    images = {p.stem: p for p in image_dir.glob("*.*") if p.is_file()}
    masks = {p.stem: p for p in mask_dir.glob("*.png") if p.is_file()}

    if not images:
        print("No images found to verify.")
        return

    errors = []

    print(f"Checking {len(images)} samples...")

    for stem, image_path in tqdm(images.items()):
        if stem not in masks:
            errors.append(f"Missing mask for image: {stem}")
            continue

        mask_path = masks[stem]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None:
            errors.append(f"Could not read mask: {mask_path.name}")
            continue

        if len(mask.shape) != 3 or mask.shape[2] != 3:
            errors.append(
                f"Mask is not 3-channel: {mask_path.name} (shape: {mask.shape})"
            )
            continue

        unique_vals = np.unique(mask)
        invalid_vals = [v for v in unique_vals if v not in (0, 255)]
        if invalid_vals:
            errors.append(
                f"Mask contains invalid values (not 0 or 255): {mask_path.name} (values: {invalid_vals})"
            )

        # Check mutual exclusivity (no pixel belongs to multiple channels)
        channel_sum = np.sum(mask > 0, axis=-1)
        if np.any(channel_sum > 1):
            errors.append(f"Mask has overlapping channels: {mask_path.name}")

    if errors:
        print("\nVerification Failed! Found errors:")
        for err in errors[:20]:
            print(f" - {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more errors.")
    else:
        print(
            "\nVerification Passed! All masks are valid H x W x 3 mutually exclusive binary masks."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verify Carla EgoLanes converted dataset.")
    parser.add_argument(
        "--data-root", type=Path, default=Path("dataset/CarlaEgoLanes/processed")
    )
    args = parser.parse_args()

    verify_dataset(args.data_root)
