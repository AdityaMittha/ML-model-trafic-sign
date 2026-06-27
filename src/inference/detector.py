"""Real-time traffic sign detector with performance monitoring."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

from src.config import get_class_names, get_model_config
from src.inference.output_formatter import format_detection

logger = logging.getLogger(__name__)


class TrafficSignDetector:
  """Production inference engine for traffic sign detection."""

  def __init__(
    self,
    model_path: str,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    device: Union[int, str] = "",
  ):
    import torch

    model_cfg = get_model_config()
    inference_cfg = model_cfg.get("inference", {})

    self.model = YOLO(model_path)
    self.conf = conf or inference_cfg.get("confidence_threshold", 0.45)
    self.iou = iou or inference_cfg.get("iou_threshold", 0.50)
    self.class_names = get_class_names()
    
    if device is not None and device != "":
      device_str = str(device)
      if device_str not in ("cpu", "CPU") and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to device='cpu'.")
        self.device = "cpu"
      else:
        self.device = device_str
    else:
      self.device = "cpu" if not torch.cuda.is_available() else "0"
    self._fps_history: List[float] = []
    self._latency_history: List[float] = []

  def detect_frame(self, frame: np.ndarray) -> List[Dict]:
    """Run detection on a single frame and return structured outputs."""
    start = time.perf_counter()
    results = self.model.predict(
      frame,
      conf=self.conf,
      iou=self.iou,
      verbose=False,
      device=self.device,
    )[0]
    latency_ms = (time.perf_counter() - start) * 1000
    self._latency_history.append(latency_ms)

    detections = []
    timestamp = datetime.now()

    if results.boxes is not None:
      for box in results.boxes:
        cls_id = int(box.cls[0])
        box_conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
        detections.append(
          format_detection(class_name, box_conf, [x1, y1, x2, y2], timestamp)
        )

    return detections

  def draw_detections(
    self,
    frame: np.ndarray,
    detections: List[Dict],
  ) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence on frame."""
    vis = frame.copy()
    for det in detections:
      x1, y1, x2, y2 = map(int, det["bounding_box"])
      label = f"{det['class_name']} {det['confidence']:.2f}"
      cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
      (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
      cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return vis

  def get_system_metrics(self) -> Dict:
    """CPU, memory, and GPU usage snapshot."""
    metrics = {
      "cpu_percent": psutil.cpu_percent(),
      "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
      "memory_percent": psutil.virtual_memory().percent,
    }

    try:
      import torch
      if torch.cuda.is_available():
        metrics["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        metrics["gpu_utilization"] = "available"
    except ImportError:
      pass

    if self._latency_history:
      metrics["avg_latency_ms"] = np.mean(self._latency_history[-100:])
      metrics["avg_fps"] = 1000.0 / metrics["avg_latency_ms"] if metrics["avg_latency_ms"] > 0 else 0

    return metrics

  def run_on_source(
    self,
    source: Union[int, str, Path],
    output_path: Optional[Path] = None,
    display: bool = True,
    save_json: bool = True,
    max_frames: Optional[int] = None,
  ) -> dict:
    """
    Run real-time inference on webcam, video, or image folder.

    Measures FPS, latency, CPU/GPU/memory usage.
    """
    from src.inference.video_source import VideoSource

    video = VideoSource(source)
    if not video.open():
      raise RuntimeError(f"Cannot open source: {source}")

    # GPU warm-up: run a dummy inference to initialize CUDA context
    # This avoids the ~1-2s cold start penalty on the first real frame
    ret, warmup_frame = video.read()
    if ret:
      logger.info("Warming up GPU with first frame...")
      _ = self.detect_frame(warmup_frame)
      self._fps_history.clear()
      self._latency_history.clear()
      logger.info("GPU warm-up complete. Starting real-time inference.")

    writer = None
    all_detections: List[Dict] = []
    frame_count = 0
    session_start = time.perf_counter()
    consecutive_failures = 0

    try:
      while True:
        if max_frames and frame_count >= max_frames:
          break

        ret, frame = video.read()
        if not ret:
          consecutive_failures += 1
          if consecutive_failures > 30:
            logger.warning("Too many consecutive frame read failures, stopping.")
            break
          continue
        consecutive_failures = 0

        frame_start = time.perf_counter()
        detections = self.detect_frame(frame)
        frame_time = time.perf_counter() - frame_start
        fps = 1.0 / frame_time if frame_time > 0 else 0
        self._fps_history.append(fps)

        for det in detections:
          det["frame_index"] = frame_count
          all_detections.append(det)

        vis = self.draw_detections(frame, detections)

        metrics = self.get_system_metrics()
        overlay = (
          f"FPS: {fps:.1f} | Latency: {frame_time*1000:.1f}ms | "
          f"CPU: {metrics['cpu_percent']:.0f}% | Mem: {metrics['memory_mb']:.0f}MB"
        )
        cv2.putText(vis, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if output_path:
          if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(
              str(output_path),
              cv2.VideoWriter_fourcc(*"mp4v"),
              video.get_fps(),
              (w, h),
            )
          writer.write(vis)

        if display:
          cv2.imshow("Traffic Sign Detection", vis)
          if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    finally:
      video.release()
      if writer:
        writer.release()
      if display:
        cv2.destroyAllWindows()

    session_time = time.perf_counter() - session_start
    benchmark = {
      "source": str(source),
      "total_frames": frame_count,
      "total_detections": len(all_detections),
      "session_time_sec": round(session_time, 2),
      "avg_fps": round(np.mean(self._fps_history) if self._fps_history else 0, 2),
      "min_fps": round(min(self._fps_history) if self._fps_history else 0, 2),
      "max_fps": round(max(self._fps_history) if self._fps_history else 0, 2),
      "avg_latency_ms": round(np.mean(self._latency_history) if self._latency_history else 0, 2),
      "system_metrics": self.get_system_metrics(),
      "model": str(self.model.ckpt_path) if hasattr(self.model, "ckpt_path") else "unknown",
      "confidence_threshold": self.conf,
      "iou_threshold": self.iou,
    }

    if save_json:
      json_dir = Path("outputs/inference")
      json_dir.mkdir(parents=True, exist_ok=True)
      ts = datetime.now().strftime("%Y%m%d_%H%M%S")
      with open(json_dir / f"detections_{ts}.json", "w") as f:
        json.dump({"benchmark": benchmark, "detections": all_detections}, f, indent=2)
      with open(json_dir / f"benchmark_{ts}.json", "w") as f:
        json.dump(benchmark, f, indent=2)

    return benchmark
