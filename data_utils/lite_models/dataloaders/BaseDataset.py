import cv2
import numpy as np
from torch.utils.data import Dataset

from data_utils.lite_models.augmentation.factory import build_aug


class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        aug_cfg: dict = {},
        mode="train",
        data_type="SEGMENTATION",
        pseudo_labeling=False,
    ):
        """
        Pseudo labeling means that a larger model is used to generate labels for the unlabeled data (eventually used to generate depth maps from DepthAnythingV2 model).

        """

        self.dataset_root = dataset_root

        self.mode = mode

        self.data_type = data_type

        self.pseudo_labeling = pseudo_labeling

        # ---- Build augmentations (does not know about pseudo-labeling) ----

        self.aug_type = self.data_type

        self.aug = build_aug(
            data_type=self.aug_type,
            cfg=aug_cfg,
            mode=self.mode,
            pseudo_labeling=self.pseudo_labeling,
        )

        self.samples = []  # to be defined in child classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # --------------------------------------------------
        # 1) LOAD RAW IMAGE (BGR → RGB)
        # --------------------------------------------------
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # 2) LOAD / FAKE GT
        # --------------------------------------------------
        if self.pseudo_labeling is False:
            if self.data_type == "SEGMENTATION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            elif self.data_type == "DEPTH":
                label = np.load(gt_path)
            elif self.data_type == "LANE_DETECTION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if label is None:
                    raise FileNotFoundError(f"[BaseDataset] Missing GT mask: {gt_path}")
                if label.ndim == 3 and label.shape[2] == 4:
                    label = cv2.cvtColor(label, cv2.COLOR_BGRA2RGB)
                elif label.ndim == 3 and label.shape[2] == 3:
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(
                    f"[BaseDataset] ERROR: unsupported data_type: {self.data_type}"
                )
        else:
            # fake GT (placeholder, will be ignored)
            if self.data_type == "SEGMENTATION" or self.data_type == "LANE_DETECTION":
                label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            elif self.data_type == "DEPTH":
                label = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # --------------------------------------------------
        # 3) AUGMENTATION PIPELINE (IMAGE + GT)
        # --------------------------------------------------
        image, label = self.aug.apply_augmentation(
            img, label, dataset_name=self.dataset_name
        )

        # --------------------------------------------------
        # 4) FINAL CAST FOR MODEL
        # --------------------------------------------------
        image = image.astype(np.float32)

        if self.data_type == "LANE_DETECTION":
            # lane detection specific behaviour: normalize to [0,1] float32
            if label.ndim != 3 or label.shape[2] != 3:
                raise ValueError(
                    "[BaseDataset] LANE_DETECTION masks must be 3-channel PNGs "
                    f"with shape [H,W,3]. Got shape {label.shape} for {gt_path}"
                )

            label = label.astype(np.float32) / 255.0

            # CHW for BCE
            label = np.transpose(label, (2, 0, 1))  # [C,H,W]

            # debug check for non-binary values
            u = np.unique(label)
            if len(u) > 2:
                print("WARN: non-binary mask values:", u[:20])

        else:
            # default behaviour for other tasks
            label = label.astype(np.int64)

        image = np.transpose(image, (2, 0, 1))  # CHW

        # --------------------------------------------------
        # 5) RETURN SAMPLE
        # --------------------------------------------------
        sample = {
            "image": image,
            "gt": label,
        }

        return sample
