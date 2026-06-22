#!/usr/bin/env python3
"""Test model with error analysis (FP/FN/failure cases)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.error_analysis import ErrorAnalyzer
from src.config import DATA_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Test traffic sign model with error analysis")
  parser.add_argument("--model", type=str, required=True, help="Path to model weights")
  parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
  parser.add_argument("--data-dir", type=str, default=str(DATA_DIR / "processed"))
  parser.add_argument("--conf", type=float, default=0.45)
  parser.add_argument("--iou", type=float, default=0.50)
  args = parser.parse_args()

  data_dir = Path(args.data_dir)
  images_dir = data_dir / "images" / args.split
  labels_dir = data_dir / "labels" / args.split
  output_dir = OUTPUTS_DIR / "testing" / args.split

  analyzer = ErrorAnalyzer(args.model, args.conf, args.iou)
  stats = analyzer.analyze_split(images_dir, labels_dir, output_dir)

  recommendations = analyzer.recommend_improvements(stats)

  print("\n=== Error Analysis ===")
  print(f"  True Positives:  {stats['true_positives']}")
  print(f"  False Positives: {stats['false_positives']}")
  print(f"  False Negatives: {stats['false_negatives']}")

  print("\n=== Recommendations ===")
  for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

  print(f"\nFailure case images saved to: {output_dir}")


if __name__ == "__main__":
  main()
