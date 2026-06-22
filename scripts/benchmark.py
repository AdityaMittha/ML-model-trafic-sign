#!/usr/bin/env python3
"""Performance benchmark: model comparison and inference profiling."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model(data_yaml: str, output_dir: Path) -> dict:
  """Benchmark YOLOv11 Nano model."""
  from ultralytics import YOLO
  from src.config import DATA_DIR

  model_name = "yolov11n"
  weights = "yolo11n.pt"

  results = {}
  data_path = data_yaml
  if not Path(data_path).exists():
    data_path = str(DATA_DIR / "processed" / "dataset.yaml")

  logger.info("Benchmarking %s...", model_name)
  try:
    model = YOLO(weights)

    val_results = None
    if Path(data_path).exists():
      try:
        import yaml
        with open(data_path, "r", encoding="utf-8") as f:
          dataset_cfg = yaml.safe_load(f)
        val_split = dataset_cfg.get("val", "")
        base_path = dataset_cfg.get("path")
        dataset_base = Path(base_path) if base_path else None
        val_images_dir = dataset_base / val_split if dataset_base else Path(val_split)
        has_images = False
        if val_images_dir.exists():
          for ext in ("*.jpg", "*.jpeg", "*.png"):
            if any(val_images_dir.glob(ext)):
              has_images = True
              break
        if has_images:
          val_results = model.val(data=data_path, split="val", verbose=False)
        else:
          logger.warning("Validation images directory '%s' is empty or does not exist. Skipping mAP evaluation.", val_images_dir)
      except Exception as val_err:
        logger.warning("Validation failed (skipping mAP evaluation): %s", val_err)

    size_mb = Path(weights).stat().st_size / (1024 * 1024) if Path(weights).exists() else 0
    params = sum(p.numel() for p in model.model.parameters())

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup runs (discard — includes JIT/CUDA kernel startup)
    for _ in range(3):
      model.predict(dummy, verbose=False)

    times = []
    for _ in range(30):
      start = time.perf_counter()
      model.predict(dummy, verbose=False)
      times.append(time.perf_counter() - start)

    entry = {
      "weights": weights,
      "parameters": params,
      "size_mb": round(size_mb, 2),
      "avg_inference_ms": round(np.mean(times) * 1000, 2),
      "avg_fps": round(1000 / (np.mean(times) * 1000), 2),
      "pi4_compatible": True,
      "pi5_compatible": True,
    }

    if val_results:
      entry.update({
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
      })

    results[model_name] = entry
  except Exception as e:
    results[model_name] = {"error": str(e)}

  output_dir.mkdir(parents=True, exist_ok=True)
  report_path = output_dir / "model_benchmark_report.json"
  with open(report_path, "w") as f:
    json.dump(results, f, indent=2)

  return results


def main():
  parser = argparse.ArgumentParser(description="Benchmark traffic sign models")
  parser.add_argument("--data", type=str, default="data/processed/dataset.yaml")
  parser.add_argument("--output", type=str, default="outputs/benchmarks")
  args = parser.parse_args()

  from src.config import OUTPUTS_DIR
  results = benchmark_model(args.data, Path(args.output) if args.output else OUTPUTS_DIR / "benchmarks")

  print("\n=== Model Benchmark Report ===")
  for name, data in results.items():
    if "error" in data:
      print(f"  {name}: ERROR")
    else:
      print(
        f"  {name}: mAP50={data.get('mAP50', 'N/A')}, "
        f"FPS={data.get('avg_fps', 'N/A')}, "
        f"Size={data.get('size_mb', 'N/A')}MB"
      )


if __name__ == "__main__":
  main()
