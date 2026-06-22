"""Dataset cleaning: duplicate removal, corruption detection, label validation."""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image

from src.config import get_class_names, get_label_mapping

logger = logging.getLogger(__name__)


class BKNode:
  """Node for BK-Tree used in fast metric space search."""
  def __init__(self, val):
    self.val = val
    self.children = {}  # distance -> BKNode


class BKTree:
  """BK-Tree data structure for fast perceptual hash distance queries."""
  def __init__(self):
    self.root = None

  def insert(self, val) -> bool:
    if self.root is None:
      self.root = BKNode(val)
      return True
    curr = self.root
    while True:
      dist = val - curr.val
      if dist == 0:
        return False
      if dist in curr.children:
        curr = curr.children[dist]
      else:
        curr.children[dist] = BKNode(val)
        return True

  def search(self, val, max_dist: int) -> list:
    if self.root is None:
      return []
    results = []
    candidates = [self.root]
    while candidates:
      curr = candidates.pop()
      dist = val - curr.val
      if dist <= max_dist:
        results.append(curr.val)
      min_d = max(0, dist - max_dist)
      max_d = dist + max_dist
      for d, child in curr.children.items():
        if min_d <= d <= max_d:
          candidates.append(child)
    return results


class DatasetCleaner:
  """Clean and validate YOLO-format detection datasets."""

  def __init__(self, max_hash_distance: int = 5):
    self.max_hash_distance = max_hash_distance
    self.class_names = get_class_names()
    self.label_mapping = get_label_mapping()
    self.num_classes = len(self.class_names)

  def is_image_valid(self, image_path: Path) -> bool:
    """Check if image file is readable and not corrupted."""
    try:
      img = cv2.imread(str(image_path))
      if img is None:
        return False
      if img.shape[0] < 10 or img.shape[1] < 10:
        return False
      return True
    except Exception:
      return False

  def compute_perceptual_hash(self, image_path: Path) -> Optional[imagehash.ImageHash]:
    try:
      with Image.open(image_path) as img:
        return imagehash.phash(img)
    except Exception:
      return None

  def find_duplicates(
    self, image_paths: List[Path]
  ) -> Tuple[List[Path], List[Path]]:
    """Find duplicate images using perceptual hashing and BK-tree."""
    bk_tree = BKTree()
    unique_paths: List[Path] = []
    duplicate_paths: List[Path] = []
    hash_to_path = {}

    for img_path in image_paths:
      phash = self.compute_perceptual_hash(img_path)
      if phash is None:
        duplicate_paths.append(img_path)
        continue

      matches = bk_tree.search(phash, self.max_hash_distance)
      if matches:
        duplicate_paths.append(img_path)
        logger.debug("Duplicate: %s matches %s", img_path, hash_to_path[matches[0]])
      else:
        bk_tree.insert(phash)
        hash_to_path[phash] = img_path
        unique_paths.append(img_path)

    logger.info(
      "Duplicates: %d removed, %d unique", len(duplicate_paths), len(unique_paths)
    )
    return unique_paths, duplicate_paths

  def validate_yolo_label(
    self, label_path: Path, img_width: int, img_height: int
  ) -> Tuple[bool, List[str]]:
    """Validate a single YOLO label file."""
    errors: List[str] = []
    if not label_path.exists():
      return False, ["Label file missing"]

    try:
      with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

      for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
          errors.append(f"Line {i}: expected 5 values, got {len(parts)}")
          continue

        cls_id = int(parts[0])
        if cls_id < 0 or cls_id >= self.num_classes:
          errors.append(f"Line {i}: invalid class id {cls_id}")

        cx, cy, w, h = map(float, parts[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
          errors.append(f"Line {i}: center out of bounds ({cx}, {cy})")
        if not (0 < w <= 1 and 0 < h <= 1):
          errors.append(f"Line {i}: dimensions invalid ({w}, {h})")

        x1 = (cx - w / 2) * img_width
        y1 = (cy - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cy + h / 2) * img_height
        if x2 <= x1 or y2 <= y1:
          errors.append(f"Line {i}: invalid box dimensions")

    except Exception as e:
      errors.append(f"Parse error: {e}")

    return len(errors) == 0, errors

  def map_label_name(self, raw_label: str) -> Optional[str]:
    """Map external dataset label to standardized class name."""
    normalized = raw_label.strip().lower()
    mapping = self.label_mapping

    if raw_label in mapping:
      return mapping[raw_label]
    if normalized in mapping:
      return mapping[normalized]

    for key, value in mapping.items():
      if key.lower() == normalized:
        return value

    return None

  def clean_dataset(
    self,
    images_dir: Path,
    labels_dir: Path,
    remove_duplicates: bool = True,
    remove_corrupted: bool = True,
    validate_labels: bool = True,
  ) -> dict:
    """Run full cleaning pipeline on a YOLO dataset split with single-pass optimization."""
    stats = {
      "total_images": 0,
      "corrupted_removed": 0,
      "duplicates_removed": 0,
      "invalid_labels_removed": 0,
      "valid_images": 0,
      "removed_files": [],
    }

    image_paths = sorted(
      list(images_dir.glob("*.jpg"))
      + list(images_dir.glob("*.jpeg"))
      + list(images_dir.glob("*.png"))
    )
    stats["total_images"] = len(image_paths)

    valid_paths: List[Path] = []
    bk_tree = BKTree()
    hash_to_path = {}

    for idx, img_path in enumerate(image_paths):
      if (idx + 1) % 5000 == 0 or (idx + 1) == len(image_paths):
        logger.info("Cleaning progress: processed %d/%d images...", idx + 1, len(image_paths))

      # Load image once
      img = None
      if remove_corrupted or validate_labels or remove_duplicates:
        try:
          img = cv2.imread(str(img_path))
        except Exception:
          img = None

        if img is None or img.shape[0] < 10 or img.shape[1] < 10:
          stats["corrupted_removed"] += 1
          stats["removed_files"].append(str(img_path))
          label_path = labels_dir / f"{img_path.stem}.txt"
          img_path.unlink(missing_ok=True)
          label_path.unlink(missing_ok=True)
          continue

      if validate_labels:
        label_path = labels_dir / f"{img_path.stem}.txt"
        valid, errors = self.validate_yolo_label(
          label_path, img.shape[1], img.shape[0]
        )
        if not valid:
          stats["invalid_labels_removed"] += 1
          stats["removed_files"].append(str(img_path))
          img_path.unlink(missing_ok=True)
          label_path.unlink(missing_ok=True)
          logger.warning("Invalid label %s: %s", label_path, errors)
          continue

      if remove_duplicates:
        # Compute phash directly from memory (OpenCV BGR image -> PIL image -> phash)
        try:
          img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          pil_img = Image.fromarray(img_rgb)
          phash = imagehash.phash(pil_img)
        except Exception:
          phash = None

        if phash is None:
          stats["duplicates_removed"] += 1
          stats["removed_files"].append(str(img_path))
          label_path = labels_dir / f"{img_path.stem}.txt"
          img_path.unlink(missing_ok=True)
          label_path.unlink(missing_ok=True)
          continue

        matches = bk_tree.search(phash, self.max_hash_distance)
        if matches:
          stats["duplicates_removed"] += 1
          stats["removed_files"].append(str(dup_path := img_path))
          label_path = labels_dir / f"{dup_path.stem}.txt"
          dup_path.unlink(missing_ok=True)
          label_path.unlink(missing_ok=True)
          logger.debug("Duplicate: %s matches %s", img_path, hash_to_path[matches[0]])
          continue
        else:
          bk_tree.insert(phash)
          hash_to_path[phash] = img_path

      valid_paths.append(img_path)

    stats["valid_images"] = len(valid_paths)
    logger.info("Cleaning complete: %s", stats)
    return stats
