"""Annotation verification utilities."""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt

from src.config import get_class_names

logger = logging.getLogger(__name__)


class AnnotationValidator:
  """Visual and statistical validation of YOLO annotations."""

  def __init__(self):
    self.class_names = get_class_names()

  def verify_split(
    self, images_dir: Path, labels_dir: Path
  ) -> Dict[str, any]:
    """Run verification checks on a dataset split."""
    from collections import Counter

    stats = {
      "total_images": 0,
      "total_boxes": 0,
      "images_without_labels": 0,
      "labels_without_images": 0,
      "class_distribution": Counter(),
      "box_size_stats": {"min_area": 1.0, "max_area": 0.0, "mean_area": 0.0},
      "issues": [],
    }

    image_stems = set()
    for ext in ("*.jpg", "*.jpeg", "*.png"):
      for p in images_dir.glob(ext):
        image_stems.add(p.stem)

    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    stats["total_images"] = len(image_stems)

    for stem in image_stems - label_stems:
      stats["images_without_labels"] += 1
      stats["issues"].append(f"No label for image: {stem}")

    for stem in label_stems - image_stems:
      stats["labels_without_images"] += 1
      stats["issues"].append(f"No image for label: {stem}")

    areas = []
    for stem in image_stems & label_stems:
      label_path = labels_dir / f"{stem}.txt"
      with open(label_path, "r") as f:
        for line in f:
          parts = line.strip().split()
          if len(parts) != 5:
            continue
          cls_id = int(parts[0])
          w, h = float(parts[3]), float(parts[4])
          area = w * h
          areas.append(area)
          stats["class_distribution"][cls_id] += 1
          stats["total_boxes"] += 1

    if areas:
      stats["box_size_stats"] = {
        "min_area": min(areas),
        "max_area": max(areas),
        "mean_area": sum(areas) / len(areas),
      }

    stats["class_distribution"] = dict(stats["class_distribution"])
    return stats

  def visualize_sample(
    self,
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    num_samples: int = 16,
  ) -> Path:
    """Generate grid visualization of annotated samples."""
    stems = [p.stem for p in images_dir.glob("*.jpg")][:num_samples]
    if not stems:
      stems = [p.stem for p in images_dir.glob("*.png")][:num_samples]

    cols = 4
    rows = min(4, (len(stems) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for idx, stem in enumerate(stems[:rows * cols]):
      img_path = images_dir / f"{stem}.jpg"
      if not img_path.exists():
        img_path = images_dir / f"{stem}.png"
      img = cv2.imread(str(img_path))
      if img is None:
        continue
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      h, w = img.shape[:2]

      label_path = labels_dir / f"{stem}.txt"
      if label_path.exists():
        with open(label_path, "r") as f:
          for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
              continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

      axes_flat[idx].imshow(img)
      axes_flat[idx].axis("off")

    for idx in range(len(stems), len(axes_flat)):
      axes_flat[idx].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
