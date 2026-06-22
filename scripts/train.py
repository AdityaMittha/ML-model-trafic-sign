#!/usr/bin/env python3
"""Train traffic sign detection model."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import TrafficSignTrainer
from src.config import DATA_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Train traffic sign YOLO model")
  parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights")
  parser.add_argument("--data", type=str, default=str(DATA_DIR / "processed" / "dataset.yaml"))
  parser.add_argument("--epochs", type=int, default=150)
  parser.add_argument("--batch", type=int, default=16)
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--device", type=str, default="0")
  parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoints")
  parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
  args = parser.parse_args()

  trainer = TrafficSignTrainer(
    model_weights=args.model,
    data_yaml=args.data,
    project_dir=OUTPUTS_DIR / "training",
  )

  if args.tune:
    logger.info("Running hyperparameter tuning...")
    results = trainer.tune_hyperparameters(iterations=5)
    logger.info("Tuning results: %s", results)
    return

  override = {
    "epochs": args.epochs,
    "batch": args.batch,
    "imgsz": args.imgsz,
    "device": args.device,
  }

  logger.info("Starting training with %s", args.model)
  results = trainer.train(resume=not args.no_resume, **override)
  logger.info("Training complete. Best weights: %s", results.get("best_weights"))


if __name__ == "__main__":
  main()
