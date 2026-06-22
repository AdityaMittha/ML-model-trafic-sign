"""Metrics computation: precision, recall, F1, mAP, confusion matrix."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO

from src.config import get_class_names, get_model_config, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class MetricsEvaluator:
  """Comprehensive evaluation using Ultralytics validation + custom analysis."""

  def __init__(
    self,
    model_path: str,
    data_yaml: str,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
  ):
    model_cfg = get_model_config()
    inference_cfg = model_cfg.get("inference", {})

    self.model_path = model_path
    self.data_yaml = data_yaml
    self.conf = conf or inference_cfg.get("confidence_threshold", 0.45)
    self.iou = iou or inference_cfg.get("iou_threshold", 0.50)
    self.class_names = get_class_names()
    self.model = YOLO(model_path)

  def validate(self, split: str = "val") -> dict:
    """Run YOLO validation and extract metrics."""
    results = self.model.val(
      data=self.data_yaml,
      split=split,
      conf=self.conf,
      iou=self.iou,
      plots=True,
      save_json=True,
    )

    metrics = {
      "precision": float(results.box.mp),
      "recall": float(results.box.mr),
      "mAP50": float(results.box.map50),
      "mAP50-95": float(results.box.map),
      "f1": 0.0,
    }
    if metrics["precision"] + metrics["recall"] > 0:
      metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"]
        / (metrics["precision"] + metrics["recall"])
      )

    if hasattr(results.box, "maps") and results.box.maps is not None:
      metrics["per_class_ap50"] = {
        self.class_names[i]: float(results.box.maps[i])
        for i in range(min(len(self.class_names), len(results.box.maps)))
      }

    return metrics

  def benchmark_models(
    self,
    model_paths: Dict[str, str],
    output_dir: Path,
  ) -> dict:
    """Compare multiple YOLO model variants."""
    comparison = {}
    for name, path in model_paths.items():
      if not Path(path).exists():
        logger.warning("Model not found: %s", path)
        continue
      evaluator = MetricsEvaluator(path, self.data_yaml, self.conf, self.iou)
      metrics = evaluator.validate(split="val")
      model = YOLO(path)
      info = model.model
      comparison[name] = {
        **metrics,
        "model_path": path,
        "parameters": sum(p.numel() for p in model.model.parameters()) if info else 0,
        "size_mb": Path(path).stat().st_size / (1024 * 1024),
      }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_comparison.json", "w") as f:
      json.dump(comparison, f, indent=2)

    return comparison

  def plot_confusion_matrix(
    self,
    confusion_matrix: np.ndarray,
    output_path: Path,
  ) -> Path:
    """Plot per-class confusion matrix."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(
      confusion_matrix,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=self.class_names,
      yticklabels=self.class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Traffic Sign Detection Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path

  def generate_report(self, output_dir: Optional[Path] = None) -> dict:
    """Full evaluation report for val and test splits."""
    output_dir = output_dir or (OUTPUTS_DIR / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
      "model": self.model_path,
      "confidence": self.conf,
      "iou": self.iou,
      "val_metrics": self.validate(split="val"),
      "test_metrics": self.validate(split="test"),
    }

    with open(output_dir / "evaluation_report.json", "w") as f:
      json.dump(report, f, indent=2)

    self._plot_per_class_performance(report, output_dir)
    logger.info("Evaluation report saved to %s", output_dir)
    return report

  def _plot_per_class_performance(self, report: dict, output_dir: Path) -> None:
    val_ap = report.get("val_metrics", {}).get("per_class_ap50", {})
    if not val_ap:
      return

    classes = list(val_ap.keys())
    scores = list(val_ap.values())

    plt.figure(figsize=(12, 6))
    colors = ["green" if s >= 0.7 else "orange" if s >= 0.5 else "red" for s in scores]
    plt.barh(classes, scores, color=colors)
    plt.xlabel("AP50")
    plt.title("Per-Class AP50 Performance")
    plt.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good (0.7)")
    plt.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_ap50.png", dpi=150)
    plt.close()
