#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_components.lite_models.DeepLabv3Plus import DeepLabV3Plus

# ============================================================
# ONNX Wrapper
# ============================================================


class ONNXWrapper:
    def __init__(self, onnx_path, device="cuda"):
        providers = (
            ["CUDAExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def eval(self):
        return self

    def __call__(self, x):
        x_np = x.detach().cpu().numpy()
        out = self.session.run(None, {self.input_name: x_np})[0]
        return torch.from_numpy(out).to(x.device)


LANE_COLORS_RGB = np.array(
    [
        [255, 0, 0],  # ego-left (channel 0): red
        [0, 255, 0],  # ego-right (channel 1): green
        [0, 0, 255],  # other (channel 2): blue
    ],
    dtype=np.uint8,
)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(device_arg: str) -> torch.device:
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def normalize_image_rgb(image_rgb: np.ndarray, mean, std) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    network_cfg = cfg["network"]
    backbone_cfg = network_cfg.get("backbone", {})
    decoder_cfg = network_cfg.get("decoder", {})
    head_cfg = network_cfg.get("head", {})

    backbone_type = backbone_cfg["type"]
    if "timm" not in backbone_type:
        backbone_type = "timm-" + backbone_type.replace("_", "-")

    model = DeepLabV3Plus(
        encoder_name=backbone_type,
        encoder_weights=None,
        segmentation_ckpt=None,
        encoder_output_stride=backbone_cfg.get("output_stride", 16),
        decoder_atrous_rates=decoder_cfg.get("aspp_dilations", [12, 24, 36]),
        decoder_channels=decoder_cfg.get("deeplabv3plus_decoder_channels", 64),
        encoder_depth=backbone_cfg.get("encoder_depth", 5),
        encoder_partial_load=False,
        load_encoder=False,
        load_decoder=False,
        freeze_encoder=False,
        freeze_decoder=False,
        head_upsampling=head_cfg.get("head_upsampling", 1),
        head_activation=head_cfg.get("head_activation", None),
        head_depth=head_cfg.get("head_depth", 1),
        head_mid_channels=head_cfg.get("head_mid_channels", None),
        head_kernel_size=head_cfg.get("head_kernel_size", 3),
        output_channels=network_cfg.get("output_channels", 3),
    )

    return model.to(device).eval()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print("[WARN] Checkpoint loaded with key differences.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)


def collect_images(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def colorize_mask(mask_hwc: np.ndarray) -> np.ndarray:
    colored = np.zeros((*mask_hwc.shape[:2], 3), dtype=np.uint8)
    for channel_idx, color in enumerate(LANE_COLORS_RGB):
        colored[mask_hwc[..., channel_idx]] = color
    return colored


def run_one_image(
    model,
    image_path: Path,
    out_dir: Path,
    device: torch.device,
    height: int,
    width: int,
    mean,
    std,
    threshold: float,
    alpha: float,
    save_images: bool = True,
):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = normalize_image_rgb(resized_rgb, mean=mean, std=std)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[0]

    pred = (logits > threshold).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(
        bool
    )

    raw_mask = pred.astype(np.uint8) * 255
    colored = colorize_mask(pred)
    overlay_rgb = cv2.addWeighted(colored, alpha, image_rgb, 1.0 - alpha, 0.0)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    if save_images:
        mask_dir = out_dir / "masks"
        overlay_dir = out_dir / "overlays"
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        Image.fromarray(raw_mask, mode="RGB").save(mask_dir / f"{stem}.png")
        cv2.imwrite(str(overlay_dir / f"{stem}.png"), overlay_bgr)

    return overlay_bgr


def main():
    parser = argparse.ArgumentParser("EgoLanesLite checkpoint inference")
    parser.add_argument("--config", default="EgoLanesLite_infer.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    parser.add_argument(
        "--onnx", default=None, help="Path to .onnx model (overrides checkpoint)"
    )
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="runs/inference/egolanes_lite")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate a video from the predicted overlays",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for the output video if --video is used",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    aug_cfg = cfg["dataset"]["augmentations"]
    resize_cfg = aug_cfg["rescaling"]
    norm_cfg = aug_cfg["normalize"]

    device = resolve_device(args.device)

    if args.onnx:
        print("[INFO] Using ONNX backend")
        model = ONNXWrapper(args.onnx, device=args.device)
    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --onnx must be provided")
        model = build_model(cfg, device)
        load_checkpoint(model, args.checkpoint, device)

    images = collect_images(args.input)
    if len(images) == 0:
        raise RuntimeError(f"No images found in input: {args.input}")

    out_dir = Path(args.output)

    video_writer = None
    if args.video:
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / "output_overlay.mp4"

    for image_path in tqdm(images, desc="Running Inference"):
        overlay_bgr = run_one_image(
            model=model,
            image_path=image_path,
            out_dir=out_dir,
            device=device,
            height=int(resize_cfg["height"]),
            width=int(resize_cfg["width"]),
            mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
            threshold=args.threshold,
            alpha=args.alpha,
            save_images=not args.video,
        )

        if args.video:
            if video_writer is None:
                h, w = overlay_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, args.fps, (w, h)
                )
            video_writer.write(overlay_bgr)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved overlay video to: {video_path}")

    print(f"Saved {len(images)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
