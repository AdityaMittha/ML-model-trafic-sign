#!/usr/bin/env python3
"""Download traffic sign datasets (Kaggle, manual paths)."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def download_kaggle_dataset(dataset: str, output_dir: Path) -> bool:
  output_dir.mkdir(parents=True, exist_ok=True)
  try:
    subprocess.run(
      [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir), "--unzip"],
      check=True,
    )
    logger.info("Kaggle dataset downloaded to %s", output_dir)
    return True
  except (subprocess.CalledProcessError, FileNotFoundError) as e:
    logger.error("Kaggle download failed: %s", e)
    logger.info(
      "Install kaggle CLI and configure ~/.kaggle/kaggle.json with API credentials."
    )
    return False


def print_manual_instructions() -> None:
  instructions = """
=== Manual Dataset Download Instructions ===

1. GTSRB (Classification - supplementary):
   URL: https://benchmark.ini.rub.de/gtsrb_dataset.html
   Place in: data/raw/gtsrb/Train/<class_id>/

2. Indian Traffic Sign Dataset (Primary - Detection):
   Search: "Indian Traffic Sign Dataset" on Kaggle/GitHub
   Place YOLO format in: data/raw/indian_traffic_sign/images/ and labels/

3. Mapillary Traffic Sign Dataset (Primary - Detection):
   URL: https://www.mapillary.com/dataset/trafficsign
   Register and download, place in: data/raw/mapillary/

4. Kaggle Traffic Signs Preprocessed:
   kaggle datasets download -d valentynsichkar/traffic-signs-preprocessed
   Place in: data/raw/kaggle_preprocessed/
"""
  print(instructions)


def main():
  parser = argparse.ArgumentParser(description="Download traffic sign datasets")
  parser.add_argument("--kaggle", action="store_true", help="Download Kaggle preprocessed dataset")
  parser.add_argument("--all", action="store_true", help="Download all available datasets")
  parser.add_argument("--instructions", action="store_true", help="Show manual download instructions")
  args = parser.parse_args()

  if args.instructions or not any([args.kaggle, args.all]):
    print_manual_instructions()

  if args.kaggle or args.all:
    download_kaggle_dataset(
      "adityabayhaqie/indonesia-traffic-sign-dataset-yolov11",
      DATA_RAW / "indonesia_traffic_sign",
    )


if __name__ == "__main__":
  main()
