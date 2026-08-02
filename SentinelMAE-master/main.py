"""
main.py — SentinelMAE end-to-end pipeline entry point
"""

import argparse
import sys
from pathlib import Path

from utils.config_loader import load_config
from utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(description="SentinelMAE Pipeline")
    parser.add_argument("--config",  default="training/config.yaml")
    parser.add_argument("--mode",    default="train",
                        choices=["train", "preprocess", "predict"],
                        help="train | preprocess | predict")
    parser.add_argument("--video",      default=None, help="Video path (predict mode)")
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pt",
                        help="Checkpoint path (predict mode)")
    args = parser.parse_args()

    logger = get_logger("sentinel.main")
    cfg    = load_config(args.config)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Project: {cfg['project']['name']}")

    if args.mode == "preprocess":
        logger.info("Starting preprocessing...")
        from preprocessing.clip_generator import build_dataset
        build_dataset(
            raw_dir = cfg["data"].get("raw_path", "data/raw"),
            out_dir = "data/processed",
            num_frames  = cfg["data"]["num_frames"],
            image_size  = cfg["data"]["image_size"],
        )

    elif args.mode == "train":
        logger.info("Starting training...")
        from training.train import train
        train(cfg)

    elif args.mode == "predict":
        if not args.video:
            logger.error("--video is required in predict mode.")
            sys.exit(1)
        if not Path(args.checkpoint).exists():
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        from inference.predict_video import predict
        predict(
            video_path  = args.video,
            checkpoint  = args.checkpoint,
            config_path = args.config,
        )


if __name__ == "__main__":
    main()