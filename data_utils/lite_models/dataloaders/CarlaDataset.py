# dataloader/CarlaDataset.py

import glob
import os

from data_utils.lite_models.dataloaders.BaseDataset import BaseDataset


"""
Carla EgoLanes processed dataset.

Expected structure:

CarlaEgoLanes/processed/
  image/            (*.jpg, *.jpeg, *.png)
  mask/             (*.png)

Each image must have a corresponding 3-channel binary mask with the same basename.
Mask channels are expected to be:
    channel 0: ego-left
    channel 1: ego-right
    channel 2: other lane
"""


class CarlaDataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str,
        aug_cfg: dict = {},
        mode: str = "train",
        data_type: str = "LANE_DETECTION",
        pseudo_labeling: bool = False,
    ):
        super().__init__(
            dataset_root,
            aug_cfg=aug_cfg,
            mode=mode,
            data_type=data_type,
            pseudo_labeling=pseudo_labeling,
        )

        self.root = dataset_root

        if "processed" not in os.path.basename(self.root):
            print(
                "[CarlaDataset] WARNING: dataset_root does not point to 'processed/'. "
                "Appending '/processed'."
            )
            self.root = os.path.join(self.root, "processed")

        self.split = mode.lower()
        self.dataset_name = "carla"

        if self.data_type != "LANE_DETECTION":
            raise ValueError(
                f"[CarlaDataset] Unsupported data_type: {self.data_type}. "
                "Only 'LANE_DETECTION' is supported."
            )

        self.samples = self._build_file_list()

    def _build_file_list(self):
        max_val_samples = 500

        print(
            f"[CarlaDataset] Building file list for split='{self.split}', "
            f"data_type='{self.data_type}'"
        )

        image_root = os.path.join(self.root, "image")
        mask_root = os.path.join(self.root, "mask")

        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Missing mask directory: {mask_root}")

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_files.extend(glob.glob(os.path.join(image_root, ext)))
        img_files = sorted(img_files)

        if len(img_files) == 0:
            raise RuntimeError(f"[CarlaDataset] No images found in {image_root}")

        print(f"[CarlaDataset] Found {len(img_files)} images total.")

        samples = []
        for idx, img_path in enumerate(img_files):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(mask_root, f"{basename}.png")

            if not os.path.isfile(gt_path):
                print(f"[CarlaDataset] WARNING: Missing GT mask for {img_path}")
                continue

            is_val = idx % 10 == 0

            if self.split == "train" and not is_val:
                samples.append((img_path, gt_path))
            elif self.split == "val" and is_val and len(samples) < max_val_samples:
                samples.append((img_path, gt_path))
            elif self.split == "test":
                samples.append((img_path, gt_path))

        print(f"[CarlaDataset] Loaded {len(samples)} samples for split='{self.split}'.")

        if len(samples) == 0:
            raise RuntimeError(
                f"[CarlaDataset] Empty dataset split='{self.split}'. "
                "Check dataset path and split logic."
            )

        return samples
