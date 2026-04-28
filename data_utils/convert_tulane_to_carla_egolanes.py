#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_SPLITS = ["train", "val", "target_train", "target_val", "target_test"]


def collect_samples(root: Path, splits: list[str]) -> list[tuple[str, Path, Path]]:
    samples = []
    for split in splits:
        image_dir = root / "images" / split
        mask_dir = root / "lane_masks" / split

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(image_dir.glob(ext))

        for image_path in sorted(image_paths):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            out_stem = f"{split}_{image_path.stem}"
            samples.append((out_stem, image_path, mask_path))

    return samples


def component_bottom_x(labels: np.ndarray, component_id: int, bottom_band: int) -> float:
    ys, xs = np.where(labels == component_id)
    bottom_y = ys.max()
    band = ys >= max(0, bottom_y - bottom_band)
    return float(np.median(xs[band]))


def split_binary_lane_mask(
    mask: np.ndarray,
    *,
    min_area: int,
    bottom_band: int,
    center_x: float | None = None,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    center = float(width) * 0.5 if center_x is None else center_x

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = []
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        ys, _ = np.where(labels == component_id)
        bottom_y = int(ys.max())
        bx = component_bottom_x(labels, component_id, bottom_band)
        components.append(
            {
                "id": component_id,
                "area": area,
                "bottom_y": bottom_y,
                "bottom_x": bx,
                "center_dist": abs(bx - center),
            }
        )

    out = np.zeros((height, width, 3), dtype=np.uint8)
    if not components:
        return out

    left_candidates = [c for c in components if c["bottom_x"] < center]
    right_candidates = [c for c in components if c["bottom_x"] >= center]

    ego_left = min(left_candidates, key=lambda c: c["center_dist"], default=None)
    ego_right = min(right_candidates, key=lambda c: c["center_dist"], default=None)

    # If one side is missing, still keep two closest components as ego lanes.
    used = {c["id"] for c in (ego_left, ego_right) if c is not None}
    if len(used) < 2 and len(components) >= 2:
        for comp in sorted(components, key=lambda c: c["center_dist"]):
            if comp["id"] not in used:
                if comp["bottom_x"] < center and ego_left is None:
                    ego_left = comp
                elif comp["bottom_x"] >= center and ego_right is None:
                    ego_right = comp
                elif ego_left is None:
                    ego_left = comp
                elif ego_right is None:
                    ego_right = comp
                used.add(comp["id"])
            if len(used) >= 2:
                break

    if ego_left is not None:
        out[..., 0][labels == ego_left["id"]] = 255
    if ego_right is not None:
        out[..., 1][labels == ego_right["id"]] = 255

    for comp in components:
        if comp["id"] not in used:
            out[..., 2][labels == comp["id"]] = 255

    return out


def link_or_copy_image(src: Path, dst: Path, mode: str):
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {src}")
        cv2.imwrite(str(dst), image)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        "Convert TuLaneConverted binary masks into Carla-style 3-channel EgoLanes masks."
    )
    parser.add_argument("--src-root", default="dataset/TuLaneConverted")
    parser.add_argument("--dst-root", default="dataset/CarlaEgoLanes/processed")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument(
        "--image-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place images in the output image directory.",
    )
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--bottom-band", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    out_image_dir = dst_root / "image"
    out_mask_dir = dst_root / "mask"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(src_root, args.splits)
    if args.limit is not None:
        samples = samples[: args.limit]

    for out_stem, image_path, mask_path in tqdm(samples, desc="Converting TuLane"):
        image_dst = out_image_dir / f"{out_stem}{image_path.suffix.lower()}"
        mask_dst = out_mask_dir / f"{out_stem}.png"

        link_or_copy_image(image_path, image_dst, args.image_mode)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        mask3 = split_binary_lane_mask(
            mask,
            min_area=args.min_area,
            bottom_band=args.bottom_band,
        )
        cv2.imwrite(str(mask_dst), cv2.cvtColor(mask3, cv2.COLOR_RGB2BGR))

    print(f"Converted {len(samples)} samples into {dst_root}")
    print(f"Images: {out_image_dir}")
    print(f"Masks : {out_mask_dir}")


if __name__ == "__main__":
    main()
