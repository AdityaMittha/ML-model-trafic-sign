#!/usr/bin/env python3
"""Validate trained model and generate metrics report."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import MetricsEvaluator
from src.config import DATA_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Validate traffic sign detection model")
  parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
  parser.add_argument("--data", type=str, default=str(DATA_DIR / "processed" / "dataset.yaml"))
  parser.add_argument("--conf", type=float, default=0.45)
  parser.add_argument("--iou", type=float, default=0.50)
  parser.add_argument("--compare-models", action="store_true", help="Compare YOLO variants")
  args = parser.parse_args()

  if args.compare_models:
    model_paths = {
      "yolov11n": "yolo11n.pt",
    }
    evaluator = MetricsEvaluator(args.model, args.data, args.conf, args.iou)
    comparison = evaluator.benchmark_models(model_paths, OUTPUTS_DIR / "evaluation")
    for name, metrics in comparison.items():
      logger.info(
        "%s: mAP50=%.3f, mAP50-95=%.3f, P=%.3f, R=%.3f, size=%.1fMB",
        name,
        metrics.get("mAP50", 0),
        metrics.get("mAP50-95", 0),
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("size_mb", 0),
      )
    return

  evaluator = MetricsEvaluator(args.model, args.data, args.conf, args.iou)
  report = evaluator.generate_report(OUTPUTS_DIR / "evaluation")

  val = report["val_metrics"]
  test = report["test_metrics"]
  print("\n=== Validation Metrics ===")
  print(f"  Precision:  {val['precision']:.4f}")
  print(f"  Recall:     {val['recall']:.4f}")
  print(f"  F1 Score:   {val['f1']:.4f}")
  print(f"  mAP50:      {val['mAP50']:.4f}")
  print(f"  mAP50-95:   {val['mAP50-95']:.4f}")

  print("\n=== Test Metrics ===")
  print(f"  Precision:  {test['precision']:.4f}")
  print(f"  Recall:     {test['recall']:.4f}")
  print(f"  F1 Score:   {test['f1']:.4f}")
  print(f"  mAP50:      {test['mAP50']:.4f}")
  print(f"  mAP50-95:   {test['mAP50-95']:.4f}")


if __name__ == "__main__":
  main()
