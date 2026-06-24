"""Dataset merger: combine multiple sources into unified YOLO format."""

import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml

from src.config import (
  DATA_DIR,
  get_class_names,
  get_class_id,
  get_dataset_config,
  get_label_mapping,
  PROJECT_ROOT,
)
from src.dataset.cleaner import DatasetCleaner

logger = logging.getLogger(__name__)


class DatasetMerger:
  """Merge multiple traffic sign datasets into unified YOLO format."""

  def __init__(self, output_dir: Optional[Path] = None):
    self.output_dir = output_dir or (DATA_DIR / "processed")
    self.class_names = get_class_names()
    self.label_mapping = get_label_mapping()
    self.cleaner = DatasetCleaner()
    self.cfg = get_dataset_config()

  def _class_id_from_name(self, name: str) -> Optional[int]:
    try:
      return get_class_id(name)
    except ValueError:
      return None

  def convert_classification_to_detection(
    self,
    image_path: Path,
    class_id: int,
    output_images_dir: Path,
    output_labels_dir: Path,
    prefix: str = "cls",
  ) -> bool:
    """Convert single-label classification image to full-frame detection sample."""
    img = cv2.imread(str(image_path))
    if img is None:
      return False

    h, w = img.shape[:2]
    stem = f"{prefix}_{class_id}_{image_path.stem}"
    out_img = output_images_dir / f"{stem}.jpg"
    out_label = output_labels_dir / f"{stem}.txt"

    cv2.imwrite(str(out_img), img)
    with open(out_label, "w") as f:
      f.write(f"{class_id} 0.5 0.5 0.95 0.95\n")
    return True

  def import_gtsrb_or_kaggle(
    self,
    source_dir: Path,
    split_images_dir: Path,
    split_labels_dir: Path,
    prefix: str,
  ) -> int:
    """Import classification dataset (folder-per-class or CSV) as pseudo-detection."""
    count = 0
    gtsrb_mapping = {
      14: "Stop",
      17: "No Entry",
      33: "Left Turn",
      35: "Right Turn",
      37: "U-Turn",
      20: "Pedestrian Crossing",
      13: "Give Way",
      31: "One Way",
      11: "Railway Crossing",
      18: "Road Work",
      0: "Speed Limit 20",
      1: "Speed Limit 30",
      2: "Speed Limit 40",
      3: "Speed Limit 50",
      4: "Speed Limit 60",
      5: "Speed Limit 80",
    }

    for ext_class_id, class_name in gtsrb_mapping.items():
      our_class_id = self._class_id_from_name(class_name)
      if our_class_id is None:
        continue

      class_folder = source_dir / str(ext_class_id)
      if not class_folder.exists():
        class_folder = source_dir / class_name
      if not class_folder.exists():
        continue

      for img_path in class_folder.glob("*"):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm"):
          if self.convert_classification_to_detection(
            img_path, our_class_id, split_images_dir, split_labels_dir, prefix
          ):
            count += 1

    logger.info("Imported %d images from %s", count, prefix)
    return count

  def import_yolo_dataset(
    self,
    source_images: Path,
    source_labels: Path,
    split_images_dir: Path,
    split_labels_dir: Path,
    prefix: str,
    label_map: Optional[Dict[str, int]] = None,
  ) -> int:
    """Import existing YOLO detection dataset with optional label remapping."""
    count = 0
    label_map = label_map or {}

    for img_path in source_images.glob("*"):
      if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

      label_path = source_labels / f"{img_path.stem}.txt"
      if not label_path.exists():
        continue

      new_lines = []
      with open(label_path, "r") as f:
        for line in f:
          parts = line.strip().split()
          if len(parts) != 5:
            continue
          cls_id = int(parts[0])
          if label_map and cls_id in label_map:
            cls_id = label_map[cls_id]
          if cls_id < 0 or cls_id >= len(self.class_names):
            continue
          new_lines.append(f"{cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

      if not new_lines:
        continue

      stem = f"{prefix}_{img_path.stem}"
      shutil.copy2(img_path, split_images_dir / f"{stem}{img_path.suffix}")
      with open(split_labels_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(new_lines) + "\n")
      count += 1

    logger.info("Imported %d YOLO images from %s", count, prefix)
    return count

  def split_dataset(
    self,
    images_dir: Path,
    labels_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
  ) -> Dict[str, List[str]]:
    """Split combined dataset into train/val/test."""
    random.seed(seed)
    all_stems = [
      p.stem
      for p in images_dir.glob("*")
      if (labels_dir / f"{p.stem}.txt").exists()
    ]
    random.shuffle(all_stems)

    n = len(all_stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
      "train": all_stems[:n_train],
      "val": all_stems[n_train:n_train + n_val],
      "test": all_stems[n_train + n_val:],
    }

  def balance_classes(
    self,
    images_dir: Path,
    labels_dir: Path,
    min_samples: int = 50,
    strategy: str = "oversample",
  ) -> dict:
    """Balance class distribution via oversampling underrepresented classes."""
    class_counts: Counter = Counter()
    stem_by_class: Dict[int, List[str]] = {i: [] for i in range(len(self.class_names))}

    for label_path in labels_dir.glob("*.txt"):
      stem = label_path.stem
      img_path = images_dir / f"{stem}.jpg"
      if not img_path.exists():
        for ext in (".jpeg", ".png"):
          alt = images_dir / f"{stem}{ext}"
          if alt.exists():
            img_path = alt
            break
      if not img_path.exists():
        continue

      with open(label_path, "r") as f:
        for line in f:
          cls_id = int(line.strip().split()[0])
          class_counts[cls_id] += 1
          stem_by_class[cls_id].append(stem)

    max_count = max(class_counts.values()) if class_counts else 0
    added = 0

    if strategy == "oversample":
      for cls_id, stems in stem_by_class.items():
        current = len(stems)
        if current == 0:
          logger.warning("Class %d (%s) has no samples", cls_id, self.class_names[cls_id])
          continue
        if current < min_samples:
          target = max(min_samples, max_count // 2)
          while len(stems) < target:
            src_stem = random.choice(stem_by_class[cls_id])
            new_stem = f"bal_{cls_id}_{added}"
            src_img = images_dir / f"{src_stem}.jpg"
            if not src_img.exists():
              break
            shutil.copy2(src_img, images_dir / f"{new_stem}.jpg")
            shutil.copy2(
              labels_dir / f"{src_stem}.txt",
              labels_dir / f"{new_stem}.txt",
            )
            stems.append(new_stem)
            added += 1

    return {"added_samples": added, "class_counts": dict(class_counts)}

  def create_yolo_dataset_yaml(self, dataset_path: Path) -> Path:
    """Generate dataset.yaml for Ultralytics training."""
    yaml_content = {
      "path": str(dataset_path.resolve()),
      "train": "images/train",
      "val": "images/val",
      "test": "images/test",
      "nc": len(self.class_names),
      "names": self.class_names,
    }
    yaml_path = dataset_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
      yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path

  def organize_splits(
    self,
    all_images_dir: Path,
    all_labels_dir: Path,
    splits: Dict[str, List[str]],
  ) -> None:
    """Move files into train/val/test subdirectories."""
    for split_name, stems in splits.items():
      img_dir = self.output_dir / "images" / split_name
      lbl_dir = self.output_dir / "labels" / split_name
      img_dir.mkdir(parents=True, exist_ok=True)
      lbl_dir.mkdir(parents=True, exist_ok=True)

      for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
          src_img = all_images_dir / f"{stem}{ext}"
          if src_img.exists():
            shutil.copy2(src_img, img_dir / f"{stem}.jpg")
            break
        src_label = all_labels_dir / f"{stem}.txt"
        if src_label.exists():
          shutil.copy2(src_label, lbl_dir / f"{stem}.txt")

  def run_full_pipeline(
    self,
    sources: Optional[Dict[str, Path]] = None,
  ) -> Path:
    """Execute complete merge, clean, balance, and split pipeline."""
    self.output_dir.mkdir(parents=True, exist_ok=True)
    staging_img = self.output_dir / "staging" / "images"
    staging_lbl = self.output_dir / "staging" / "labels"
    staging_img.mkdir(parents=True, exist_ok=True)
    staging_lbl.mkdir(parents=True, exist_ok=True)

    sources = sources or {}
    raw_base = DATA_DIR / "raw"

    # Import Indonesia Traffic Sign Dataset
    indonesia_path = sources.get("indonesia_traffic_sign", raw_base / "indonesia_traffic_sign")
    imported_count = 0
    if indonesia_path.exists():
      for split in ("train", "valid", "test"):
        img_sub = indonesia_path / split / "images"
        lbl_sub = indonesia_path / split / "labels"
        if img_sub.exists() and lbl_sub.exists():
          imported_count += self.import_yolo_dataset(img_sub, lbl_sub, staging_img, staging_lbl, f"indo_{split}")

    if imported_count == 0:
      raise FileNotFoundError(
        f"No raw images imported from {indonesia_path}! "
        "Please ensure the dataset was successfully downloaded and unzipped. "
        "Run: python scripts/download_datasets.py --kaggle"
      )

    self.cleaner.clean_dataset(staging_img, staging_lbl, remove_duplicates=False, remove_corrupted=False, validate_labels=False)

    strategy_cfg = self.cfg.get("strategy", {})
    self.balance_classes(
      staging_img,
      staging_lbl,
      min_samples=strategy_cfg.get("min_samples_per_class", 50),
    )

    splits = self.split_dataset(
      staging_img,
      staging_lbl,
      train_ratio=strategy_cfg.get("train_ratio", 0.8),
      val_ratio=strategy_cfg.get("val_ratio", 0.1),
    )

    self.organize_splits(staging_img, staging_lbl, splits)
    yaml_path = self.create_yolo_dataset_yaml(self.output_dir)

    stats_path = self.output_dir / "merge_stats.json"
    with open(stats_path, "w") as f:
      json.dump(
        {
          "splits": {k: len(v) for k, v in splits.items()},
          "total_classes": len(self.class_names),
          "class_names": self.class_names,
        },
        f,
        indent=2,
      )

    logger.info("Dataset pipeline complete. YAML: %s", yaml_path)
    return yaml_path
