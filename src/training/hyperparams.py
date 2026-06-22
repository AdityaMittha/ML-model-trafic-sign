"""Hyperparameter definitions and tuning utilities."""

from dataclasses import dataclass
from typing import List


@dataclass
class HyperparamConfig:
  """Documented hyperparameter choices for traffic sign detection."""

  lr0: float = 0.001
  batch: int = 16
  epochs: int = 150
  weight_decay: float = 0.0005
  imgsz: int = 640
  confidence: float = 0.45
  iou: float = 0.50

  @staticmethod
  def rationale() -> dict:
    return {
      "lr0": (
        "0.001 with AdamW provides stable convergence for fine-tuning pretrained "
        "YOLO on traffic signs. Lower (0.0005) slows learning; higher (0.002) risks instability."
      ),
      "batch": (
        "16 balances GPU memory on Colab T4 (16GB) with gradient stability. "
        "8 for limited memory; 32 if GPU allows for faster epochs."
      ),
      "epochs": (
        "150 epochs with patience=25 early stopping. Traffic sign datasets "
        "typically converge between 80-120 epochs."
      ),
      "weight_decay": (
        "0.0005 standard for YOLO prevents overfitting on smaller Indian sign subsets."
      ),
      "imgsz": (
        "640px captures sign detail at 5-10m. 416 is faster but loses small sign features."
      ),
      "confidence": (
        "0.45 threshold reduces false positives in production while maintaining recall. "
        "Tune per deployment: lower for safety-critical signs, higher for alerts."
      ),
      "iou": (
        "0.50 NMS IoU standard for overlapping detections. 0.40 merges more boxes; "
        "0.60 keeps distinct nearby signs separate."
      ),
    }


HYPERPARAM_SEARCH_GRID = {
  "lr0": [0.0005, 0.001, 0.002],
  "batch": [8, 16, 32],
  "weight_decay": [0.0001, 0.0005, 0.001],
  "imgsz": [416, 640],
}

CONFIDENCE_IOU_GRID = {
  "confidence": [0.25, 0.35, 0.45, 0.55],
  "iou": [0.40, 0.50, 0.60],
}
