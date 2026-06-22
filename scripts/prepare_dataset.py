#!/usr/bin/env python3
"""Dataset preparation: merge, clean, balance, split."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.merger import DatasetMerger
from src.dataset.validator import AnnotationValidator
from src.config import DATA_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Prepare traffic sign dataset")
  parser.add_argument("--output", type=str, default=str(DATA_DIR / "processed"))
  parser.add_argument("--visualize", action="store_true", help="Generate annotation visualization")
  args = parser.parse_args()

  output_dir = Path(args.output)
  merger = DatasetMerger(output_dir=output_dir)

  logger.info("Starting dataset preparation pipeline...")
  yaml_path = merger.run_full_pipeline()

  validator = AnnotationValidator()
  for split in ("train", "val", "test"):
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    if img_dir.exists():
      stats = validator.verify_split(img_dir, lbl_dir)
      logger.info("%s split stats: %d images, %d boxes", split, stats["total_images"], stats["total_boxes"])

      if args.visualize:
        vis_path = OUTPUTS_DIR / "reports" / f"annotations_{split}.png"
        validator.visualize_sample(img_dir, lbl_dir, vis_path)
        logger.info("Visualization saved: %s", vis_path)

  logger.info("Dataset ready: %s", yaml_path)
  print(f"\nDataset YAML: {yaml_path}")
  print("Next step: python scripts/train.py")


if __name__ == "__main__":
  main()
