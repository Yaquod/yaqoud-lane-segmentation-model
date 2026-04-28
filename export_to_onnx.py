import argparse
import sys
from pathlib import Path
import torch
import yaml
import onnx

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_components.lite_models.DeepLabv3Plus import DeepLabV3Plus


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
        encoder_output_stride=backbone_cfg.get("output_stride", 16),
        decoder_atrous_rates=decoder_cfg.get("aspp_dilations", [12, 24, 36]),
        decoder_channels=decoder_cfg.get("deeplabv3plus_decoder_channels", 64),
        encoder_depth=backbone_cfg.get("encoder_depth", 5),
        head_upsampling=head_cfg.get("head_upsampling", 1),
        head_activation=head_cfg.get("head_activation", None),
        head_depth=head_cfg.get("head_depth", 1),
        head_mid_channels=head_cfg.get("head_mid_channels", None),
        head_kernel_size=head_cfg.get("head_kernel_size", 3),
        output_channels=network_cfg.get("output_channels", 3),
    )
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Export EgoLanesLite from PTH to ONNX")
    parser.add_argument(
        "--config",
        type=str,
        default="EgoLanesLite_infer.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint.pth",
        help="Path to checkpoint pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="EgoLanesLite.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument("--width", type=int, default=800, help="Input width")
    parser.add_argument("--height", type=int, default=400, help="Input height")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load config to get properties (especially head output size or channel config)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Allow overriding width/height if it's in the config
    try:
        # First try to find it under rescaling (like in infer yaml)
        rescaling = cfg.get("dataset", {}).get("augmentations", {}).get("rescaling", {})
        if isinstance(rescaling, list):
            args.height, args.width = rescaling[0], rescaling[1]
            print(f"Using dimensions from config (list): {args.height}x{args.width}")
        elif (
            isinstance(rescaling, dict)
            and "height" in rescaling
            and "width" in rescaling
        ):
            args.height, args.width = rescaling["height"], rescaling["width"]
            print(
                f"Using dimensions from config (rescaling dict): {args.height}x{args.width}"
            )
        else:
            # If not in rescaling, it might be directly directly under augmentations (like in train yaml)
            augmentations = cfg.get("dataset", {}).get("augmentations", {})
            if "height" in augmentations and "width" in augmentations:
                args.height, args.width = (
                    augmentations["height"],
                    augmentations["width"],
                )
                print(
                    f"Using dimensions from config (augmentations dict): {args.height}x{args.width}"
                )
    except Exception as e:
        print(
            f"Could not parse dimensions from config: {e}. Using defaults {args.height}x{args.width}"
        )
        pass

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get(
        "model_state", ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    )

    missing, expected = model.load_state_dict(state, strict=True)
    if missing or expected:
        print("[WARN] Model loaded with some missing/unexpected keys.")
        print("Missing:", missing)
        print("Unexpected:", expected)

    print(f"Exporting model to {args.output}...")

    dummy_input = torch.randn(1, 3, args.height, args.width, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Validating ONNX model...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid and successfully exported!")


if __name__ == "__main__":
    main()
