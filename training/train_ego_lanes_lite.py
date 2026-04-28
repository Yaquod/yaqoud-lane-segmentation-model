import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.lite_models.helpers.training import load_yaml, set_global_seed


from training.ego_lanes_lite_trainer import EgoLanesLiteTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="EgoLanesLite_train.yaml",
        help="Path to lane detection training YAML config"
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = EgoLanesLiteTrainer(cfg)


    trainer.run()


if __name__ == "__main__":
    main()
