"""Structured detection output formatting."""

from datetime import datetime
from typing import Any, Dict, List, Optional


def format_detection(
  class_name: str,
  confidence: float,
  bounding_box: List[float],
  timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
  """
  Standard detection output format for future pipeline integration.

  Returns:
    {
      "class_name": "Stop",
      "confidence": 0.94,
      "timestamp": "2026-01-01T10:00:00",
      "bounding_box": [x1, y1, x2, y2]
    }
  """
  ts = timestamp or datetime.now()
  return {
    "class_name": class_name,
    "confidence": round(confidence, 4),
    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
    "bounding_box": [round(v, 2) for v in bounding_box],
  }


def format_detections_batch(
  detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
  """Validate and normalize a batch of detections."""
  return [
    format_detection(
      d["class_name"],
      d["confidence"],
      d["bounding_box"],
      datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else None,
    )
    for d in detections
  ]
