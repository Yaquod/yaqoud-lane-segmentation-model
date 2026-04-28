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


def find_closest_components_from_center(
    labels: np.ndarray,
    valid_component_ids: set[int],
    center_x: int,
    scan_start_ratio: float,
) -> tuple[int | None, int | None]:
    height, width = labels.shape
    start_y = min(height - 1, max(0, int(height * scan_start_ratio)))

    left_id = None
    right_id = None

    for y in range(start_y, -1, -1):
        if left_id is None:
            for x in range(center_x - 1, -1, -1):
                component_id = int(labels[y, x])
                if component_id in valid_component_ids:
                    left_id = component_id
                    break

        if right_id is None:
            for x in range(center_x, width):
                component_id = int(labels[y, x])
                if component_id in valid_component_ids:
                    right_id = component_id
                    break

        if left_id is not None and right_id is not None:
            break

    return left_id, right_id


def trace_lanes_bottom_up(
    mask: np.ndarray,
    *,
    min_run_width: int,
    scan_start_ratio: float,
    center_x: float | None = None,
) -> np.ndarray:
    binary = mask > 0
    height, width = binary.shape
    center = int(round(float(width) * 0.5 if center_x is None else center_x))
    center = min(width - 1, max(0, center))

    out = np.zeros((height, width, 3), dtype=np.uint8)

    # State for tracking lines
    # track_id -> dict with info
    active_tracks = {}
    completed_tracks = []
    next_track_id = 0

    # A track dict will contain:
    # id: int
    # y_path: list[int]
    # x_path: list[float]
    # pixels: list[tuple[y: int, xs: np.ndarray]]
    # missed_frames: int

    for y in range(height - 1, -1, -1):
        xs = np.flatnonzero(binary[y])
        if xs.size == 0:
            for tid in list(active_tracks.keys()):
                active_tracks[tid]["missed_frames"] += 1
                if active_tracks[tid]["missed_frames"] > 50:
                    completed_tracks.append(active_tracks.pop(tid))
            continue

        breaks = np.where(np.diff(xs) > 1)[0] + 1
        runs = np.split(xs, breaks)
        # Filters out noise
        runs = [run for run in runs if run.size >= min_run_width]
        if not runs:
            for tid in list(active_tracks.keys()):
                active_tracks[tid]["missed_frames"] += 1
                if active_tracks[tid]["missed_frames"] > 50:
                    completed_tracks.append(active_tracks.pop(tid))
            continue

        # Match current active tracks to these runs
        # Use distance of center of runs to last x of track
        run_centers = [float(run.mean()) for run in runs]
        run_assigned = [False] * len(runs)
        track_assigned = {tid: False for tid in active_tracks}

        if active_tracks:
            # Compute distance matrix
            tids = list(active_tracks.keys())
            dists = np.zeros((len(tids), len(runs)), dtype=np.float32)

            for i, tid in enumerate(tids):
                # Predict next x based on slope from recent history
                history_len = min(5, len(active_tracks[tid]["x_path"]))
                if history_len > 1:
                    dx = (
                        active_tracks[tid]["x_path"][-1]
                        - active_tracks[tid]["x_path"][-history_len]
                    )
                    dy = (
                        active_tracks[tid]["y_path"][-1]
                        - active_tracks[tid]["y_path"][-history_len]
                    )
                    slope = dx / dy if dy != 0 else 0
                    # y is decreasing, so expected next x:
                    predicted_x = active_tracks[tid]["x_path"][-1] - slope
                else:
                    predicted_x = active_tracks[tid]["x_path"][-1]

                for j, rc in enumerate(run_centers):
                    dists[i, j] = abs(predicted_x - rc)

            # Greedy matching
            max_dist = (
                width * 0.05
            )  # Tighter constraint than 0.1 due to better prediction
            for _ in range(min(len(tids), len(runs))):
                min_idx = np.argmin(dists)
                min_i, min_j = np.unravel_index(min_idx, dists.shape)
                if dists[min_i, min_j] > max_dist:
                    break

                tid = tids[min_i]
                active_tracks[tid]["y_path"].append(y)
                active_tracks[tid]["x_path"].append(run_centers[min_j])
                active_tracks[tid]["pixels"].append((y, runs[min_j]))
                active_tracks[tid]["missed_frames"] = 0

                track_assigned[tid] = True
                run_assigned[min_j] = True

                # Invalidate these so they can't be matched again
                dists[min_i, :] = np.inf
                dists[:, min_j] = np.inf

        # For assigned tracks that didn't get a match, increase miss count
        for tid in list(active_tracks.keys()):
            if not track_assigned[tid]:
                active_tracks[tid]["missed_frames"] += 1
                if active_tracks[tid]["missed_frames"] > 50:
                    completed_tracks.append(active_tracks.pop(tid))

        # Unassigned runs become new tracks
        for j, a_flag in enumerate(run_assigned):
            if not a_flag:
                active_tracks[next_track_id] = {
                    "id": next_track_id,
                    "y_path": [y],
                    "x_path": [run_centers[j]],
                    "pixels": [(y, runs[j])],
                    "missed_frames": 0,
                }
                next_track_id += 1

    all_tracks = completed_tracks + list(active_tracks.values())
    if not all_tracks:
        return out

    # Filter out very short noisy artifact tracks
    MIN_TRACK_LENGTH = 10
    all_tracks = [t for t in all_tracks if len(t["y_path"]) >= MIN_TRACK_LENGTH]

    if not all_tracks:
        return out

    # Choose ego-left and ego-right lanes
    # Candidate tracks for ego must extend somewhat towards the bottom (e.g. start at y > height * scan_start_ratio)
    # Among those, we find the two closest to center right and center left
    start_y_threshold = int(height * scan_start_ratio)

    left_candidates = []
    right_candidates = []

    for t in all_tracks:
        start_y = t["y_path"][0]  # First matched row (closest to bottom)
        start_x = t["x_path"][0]

        if start_y >= start_y_threshold:
            if start_x < center:
                left_candidates.append(t)
            else:
                right_candidates.append(t)

    # Sort left_candidates by x descending (closest to center)
    left_candidates.sort(key=lambda t: t["x_path"][0], reverse=True)
    # Sort right_candidates by x ascending (closest to center)
    right_candidates.sort(key=lambda t: t["x_path"][0])

    ego_left = left_candidates[0] if left_candidates else None
    ego_right = right_candidates[0] if right_candidates else None

    ego_left_id = ego_left["id"] if ego_left else -1
    ego_right_id = ego_right["id"] if ego_right else -1

    for t in all_tracks:
        tid = t["id"]
        if tid == ego_left_id:
            channel = 0
        elif tid == ego_right_id:
            channel = 1
        else:
            channel = 2

        for y, xs in t["pixels"]:
            out[y, xs, channel] = 255

    return out


def split_binary_lane_mask(
    mask: np.ndarray,
    *,
    min_area: int,
    min_run_width: int,
    scan_start_ratio: float,
    method: str,
    center_x: float | None = None,
) -> np.ndarray:
    if method == "bottom_up":
        return trace_lanes_bottom_up(
            mask,
            min_run_width=min_run_width,
            scan_start_ratio=scan_start_ratio,
            center_x=center_x,
        )

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    center = int(round(float(width) * 0.5 if center_x is None else center_x))
    center = min(width - 1, max(0, center))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    components = {}
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        ys, xs = np.where(labels == component_id)
        components[component_id] = {
            "id": component_id,
            "area": area,
            "nearest_left_dist": (
                int(center - xs[xs < center].max()) if np.any(xs < center) else None
            ),
            "nearest_right_dist": (
                int(xs[xs >= center].min() - center) if np.any(xs >= center) else None
            ),
            "top_y": int(ys.min()),
        }

    out = np.zeros((height, width, 3), dtype=np.uint8)
    if not components:
        return out

    ego_left_id, ego_right_id = find_closest_components_from_center(
        labels=labels,
        valid_component_ids=set(components),
        center_x=center,
        scan_start_ratio=scan_start_ratio,
    )

    if ego_left_id is None:
        left_candidates = [
            comp
            for comp in components.values()
            if comp["nearest_left_dist"] is not None
        ]
        if left_candidates:
            ego_left_id = min(
                left_candidates,
                key=lambda comp: (comp["nearest_left_dist"], comp["top_y"]),
            )["id"]

    if ego_right_id is None:
        right_candidates = [
            comp
            for comp in components.values()
            if comp["nearest_right_dist"] is not None
        ]
        if right_candidates:
            ego_right_id = min(
                right_candidates,
                key=lambda comp: (comp["nearest_right_dist"], comp["top_y"]),
            )["id"]

    used = {
        component_id for component_id in (ego_left_id, ego_right_id) if component_id
    }

    if ego_left_id is not None:
        out[..., 0][labels == ego_left_id] = 255
    if ego_right_id is not None:
        out[..., 1][labels == ego_right_id] = 255

    for component_id in components:
        if component_id not in used:
            out[..., 2][labels == component_id] = 255

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


def save_preview(mask3: np.ndarray, path: Path):
    preview = np.zeros_like(mask3)
    preview[..., 0] = mask3[..., 0]
    preview[..., 1] = mask3[..., 1]
    preview[..., 2] = mask3[..., 2]
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


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
    parser.add_argument("--min-run-width", type=int, default=1)
    parser.add_argument(
        "--method",
        choices=["bottom_up", "component_scan"],
        default="bottom_up",
        help="bottom_up traces lanes from the bottom of the image upwards to correctly identify contiguous lane lines.",
    )
    parser.add_argument(
        "--scan-start-ratio",
        type=float,
        default=0.95,
        help="Vertical position to start center-out scanning from. 0.95 scans from near the bottom.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Optional directory for RGB mask previews.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    out_image_dir = dst_root / "image"
    out_mask_dir = dst_root / "mask"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

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
            min_run_width=args.min_run_width,
            scan_start_ratio=args.scan_start_ratio,
            method=args.method,
        )
        cv2.imwrite(str(mask_dst), cv2.cvtColor(mask3, cv2.COLOR_RGB2BGR))

        if preview_dir:
            save_preview(mask3, preview_dir / f"{out_stem}.png")

    print(f"Converted {len(samples)} samples into {dst_root}")
    print(f"Images: {out_image_dir}")
    print(f"Masks : {out_mask_dir}")


if __name__ == "__main__":
    main()
