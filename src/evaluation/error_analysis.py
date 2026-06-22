"""False positive/negative analysis and failure case identification."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import get_class_names, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
  """Analyze detection errors: FP, FN, and failure cases."""

  def __init__(self, model_path: str, conf: float = 0.45, iou: float = 0.50):
    self.model = YOLO(model_path)
    self.conf = conf
    self.iou = iou
    self.class_names = get_class_names()

  def _safe_class_name(self, class_id: int) -> str:
    """Get class name with bounds checking to avoid IndexError."""
    if 0 <= class_id < len(self.class_names):
      return self.class_names[class_id]
    return f"unknown_{class_id}"

  def _load_ground_truth(self, label_path: Path) -> List[dict]:
    boxes = []
    if not label_path.exists():
      return boxes
    with open(label_path, "r") as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
          continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        boxes.append({"class_id": cls_id, "bbox_yolo": [cx, cy, w, h]})
    return boxes

  def _yolo_to_xyxy(self, bbox_yolo: list, img_w: int, img_h: int) -> list:
    cx, cy, w, h = bbox_yolo
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]

  def _compute_iou(self, box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

  def analyze_split(
    self,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    max_failure_samples: int = 50,
  ) -> dict:
    """Per-image FP/FN analysis on a dataset split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = output_dir / "false_positives"
    fn_dir = output_dir / "false_negatives"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    stats = {
      "false_positives": 0,
      "false_negatives": 0,
      "true_positives": 0,
      "per_class_fp": {},
      "per_class_fn": {},
      "failure_cases": [],
    }

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    iou_match_thresh = 0.5

    for img_path in image_paths:
      img = cv2.imread(str(img_path))
      if img is None:
        continue
      h, w = img.shape[:2]

      gt_boxes = self._load_ground_truth(labels_dir / f"{img_path.stem}.txt")
      gt_xyxy = [
        {
          "class_id": b["class_id"],
          "bbox": self._yolo_to_xyxy(b["bbox_yolo"], w, h),
        }
        for b in gt_boxes
      ]

      results = self.model.predict(
        img, conf=self.conf, iou=self.iou, verbose=False
      )[0]

      pred_boxes = []
      if results.boxes is not None:
        for box in results.boxes:
          pred_boxes.append({
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist(),
          })

      matched_gt = set()
      matched_pred = set()

      for pi, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_xyxy):
          if gi in matched_gt:
            continue
          iou_val = self._compute_iou(pred["bbox"], gt["bbox"])
          if iou_val > best_iou:
            best_iou = iou_val
            best_gi = gi

        if best_iou >= iou_match_thresh and best_gi >= 0:
          if pred["class_id"] == gt_xyxy[best_gi]["class_id"]:
            stats["true_positives"] += 1
            matched_gt.add(best_gi)
            matched_pred.add(pi)
          else:
            stats["false_positives"] += 1
            cls_name = self._safe_class_name(pred["class_id"])
            stats["per_class_fp"][cls_name] = stats["per_class_fp"].get(cls_name, 0) + 1
            matched_pred.add(pi)
            if len(stats["failure_cases"]) < max_failure_samples:
              self._save_failure(img, pred, gt_xyxy[best_gi], fp_dir, img_path.stem, "fp")
              stats["failure_cases"].append({
                "type": "false_positive",
                "image": img_path.name,
                "predicted": cls_name,
                "actual": self._safe_class_name(gt_xyxy[best_gi]["class_id"]),
              })
        else:
          stats["false_positives"] += 1
          cls_name = self._safe_class_name(pred["class_id"])
          stats["per_class_fp"][cls_name] = stats["per_class_fp"].get(cls_name, 0) + 1
          if len(stats["failure_cases"]) < max_failure_samples:
            self._save_failure(img, pred, None, fp_dir, img_path.stem, "fp")
            stats["failure_cases"].append({
              "type": "false_positive",
              "image": img_path.name,
              "predicted": cls_name,
              "actual": None,
            })

      for gi, gt in enumerate(gt_xyxy):
        if gi not in matched_gt:
          stats["false_negatives"] += 1
          cls_name = self._safe_class_name(gt["class_id"])
          stats["per_class_fn"][cls_name] = stats["per_class_fn"].get(cls_name, 0) + 1
          if len(stats["failure_cases"]) < max_failure_samples:
            vis = img.copy()
            x1, y1, x2, y2 = map(int, gt["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"MISS: {cls_name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(fn_dir / f"{img_path.stem}_fn.jpg"), vis)
            stats["failure_cases"].append({
              "type": "false_negative",
              "image": img_path.name,
              "missed": cls_name,
            })

    with open(output_dir / "error_analysis.json", "w") as f:
      json.dump(stats, f, indent=2)

    return stats

  def _save_failure(
    self, img, pred, gt, out_dir, stem, prefix
  ) -> None:
    vis = img.copy()
    x1, y1, x2, y2 = map(int, pred["bbox"])
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = self._safe_class_name(pred["class_id"])
    cv2.putText(vis, f"Pred: {label} ({pred['confidence']:.2f})", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if gt:
      gx1, gy1, gx2, gy2 = map(int, gt["bbox"])
      cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
      cv2.putText(vis, f"GT: {self._safe_class_name(gt['class_id'])}", (gx1, gy1 - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"{stem}_{prefix}.jpg"), vis)

  def recommend_improvements(self, error_stats: dict) -> List[str]:
    """Generate actionable recommendations from error analysis."""
    recommendations = []

    fn_by_class = error_stats.get("per_class_fn", {})
    if fn_by_class:
      worst_fn = max(fn_by_class, key=fn_by_class.get)
      recommendations.append(
        f"Class '{worst_fn}' has highest false negatives ({fn_by_class[worst_fn]}). "
        "Add more training samples, lower confidence threshold for this class, or apply targeted augmentation."
      )

    fp_by_class = error_stats.get("per_class_fp", {})
    if fp_by_class:
      worst_fp = max(fp_by_class, key=fp_by_class.get)
      recommendations.append(
        f"Class '{worst_fp}' has highest false positives ({fp_by_class[worst_fp]}). "
        "Increase confidence threshold, add hard-negative mining, or review similar sign pairs."
      )

    if error_stats.get("false_negatives", 0) > error_stats.get("false_positives", 0):
      recommendations.append(
        "FN > FP: Model misses signs. Consider lower conf threshold (0.35), "
        "more small-sign augmentation, and higher resolution (imgsz=640)."
      )
    else:
      recommendations.append(
        "FP > FN: Model over-detects. Increase conf threshold (0.55), "
        "add background images without signs, and review NMS IoU."
      )

    recommendations.extend([
      "Collect real Indian road footage for domain adaptation fine-tuning.",
      "Apply night/rain augmentation for underrepresented conditions.",
      "Consider collecting more data for underperforming classes.",
    ])

    return recommendations
