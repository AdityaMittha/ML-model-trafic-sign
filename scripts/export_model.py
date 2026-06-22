#!/usr/bin/env python3
"""Export model to PyTorch, ONNX, TFLite and benchmark formats."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.exporter import ModelExporter
from src.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Export traffic sign detection model")
  parser.add_argument("--model", type=str, required=True, help="Path to trained .pt weights")
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--benchmark", action="store_true", help="Benchmark all export formats")
  parser.add_argument("--format", type=str, choices=["pt", "onnx", "tflite", "all"], default="all")
  args = parser.parse_args()

  exporter = ModelExporter(args.model, OUTPUTS_DIR / "exports")

  if args.benchmark or args.format == "all":
    results = exporter.benchmark_formats(imgsz=args.imgsz)
    print("\n=== Export Format Comparison ===")
    for fmt, data in results.items():
      if fmt == "recommendation":
        continue
      if "error" in data:
        print(f"  {fmt}: ERROR - {data['error']}")
      else:
        print(
          f"  {fmt}: {data['size_mb']:.1f} MB, "
          f"{data['avg_inference_ms']:.1f} ms, {data['avg_fps']:.1f} FPS"
        )

    rec = results.get("recommendation", {})
    print(f"\n  Recommended for Raspberry Pi: {rec.get('recommended_format', 'onnx')}")
    print(f"  {rec.get('rationale', '')}")
  else:
    if args.format == "pt":
      exporter.export_pytorch()
    elif args.format == "onnx":
      exporter.export_onnx(imgsz=args.imgsz)
    elif args.format == "tflite":
      exporter.export_tflite(imgsz=args.imgsz)


if __name__ == "__main__":
  main()
