#!/usr/bin/env python3
"""Real-time inference: webcam, video, image folder."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.detector import TrafficSignDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
  parser = argparse.ArgumentParser(description="Traffic sign real-time inference")
  parser.add_argument("--model", type=str, required=True, help="Path to model weights")
  parser.add_argument("--source", type=str, default="0", help="Webcam index, video path, or image folder")
  parser.add_argument("--output", type=str, default=None, help="Output video path")
  parser.add_argument("--conf", type=float, default=0.45)
  parser.add_argument("--iou", type=float, default=0.50)
  parser.add_argument("--no-display", action="store_true", help="Disable live display")
  parser.add_argument("--device", type=str, default="", help="cuda:0 or cpu")
  parser.add_argument("--max-frames", type=int, default=None)
  args = parser.parse_args()

  source: str | int = args.source
  if source.isdigit() and not Path(source).exists():
    source = int(source)

  detector = TrafficSignDetector(args.model, args.conf, args.iou, args.device)
  benchmark = detector.run_on_source(
    source=source,
    output_path=Path(args.output) if args.output else None,
    display=not args.no_display,
    save_json=True,
    max_frames=args.max_frames,
  )

  print("\n=== Inference Benchmark ===")
  print(f"  Frames processed: {benchmark['total_frames']}")
  print(f"  Detections:       {benchmark['total_detections']}")
  print(f"  Avg FPS:          {benchmark['avg_fps']}")
  print(f"  Min FPS:          {benchmark['min_fps']}")
  print(f"  Max FPS:          {benchmark['max_fps']}")
  print(f"  Avg Latency:      {benchmark['avg_latency_ms']:.1f} ms")
  print(f"  CPU:              {benchmark['system_metrics']['cpu_percent']:.0f}%")
  print(f"  Memory:           {benchmark['system_metrics']['memory_mb']:.0f} MB")

  target_fps = 15
  if benchmark["avg_fps"] >= target_fps:
    print(f"\n  ✓ Real-time target met ({benchmark['avg_fps']:.1f} FPS >= {target_fps} FPS)")
  else:
    print(f"\n  ✗ Below real-time target ({benchmark['avg_fps']:.1f} FPS < {target_fps} FPS)")


if __name__ == "__main__":
  main()
