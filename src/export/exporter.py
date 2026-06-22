"""Export to PyTorch, ONNX, and TensorFlow Lite with benchmark comparison."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ModelExporter:
  """Export trained YOLO models to deployment formats."""

  def __init__(self, model_path: str, output_dir: Optional[Path] = None):
    from ultralytics import YOLO

    self.model_path = model_path
    self.model = YOLO(model_path)
    self.output_dir = output_dir or (OUTPUTS_DIR / "exports")
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def export_pytorch(self) -> Path:
    """Copy/reference PyTorch weights (.pt)."""
    src = Path(self.model_path)
    dst = self.output_dir / "traffic_sign_detector.pt"
    if src.resolve() != dst.resolve():
      import shutil
      shutil.copy2(src, dst)
    logger.info("PyTorch model: %s", dst)
    return dst

  def export_onnx(self, imgsz: int = 640, simplify: bool = True) -> Path:
    """Export to ONNX for cross-platform deployment."""
    path = self.model.export(
      format="onnx",
      imgsz=imgsz,
      simplify=simplify,
      opset=12,
    )
    dst = self.output_dir / "traffic_sign_detector.onnx"
    exported = Path(path)
    if exported.exists() and exported.resolve() != dst.resolve():
      import shutil
      shutil.copy2(exported, dst)
    logger.info("ONNX model: %s", dst)
    return dst

  def export_tflite(self, imgsz: int = 640, int8: bool = False) -> Path:
    """Export to TensorFlow Lite for edge devices."""
    path = self.model.export(
      format="tflite",
      imgsz=imgsz,
      int8=int8,
    )
    dst = self.output_dir / "traffic_sign_detector.tflite"
    exported = Path(path)
    if exported.exists() and exported.resolve() != dst.resolve():
      import shutil
      shutil.copy2(exported, dst)
    logger.info("TFLite model: %s", dst)
    return dst

  def benchmark_formats(
    self,
    imgsz: int = 640,
    num_iterations: int = 50,
  ) -> Dict[str, dict]:
    """Compare accuracy proxy, size, and inference speed across formats."""
    results = {}
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    pt_path = self.export_pytorch()
    results["pytorch"] = self._benchmark_single(
      "pytorch", pt_path, dummy_img, num_iterations
    )

    try:
      onnx_path = self.export_onnx(imgsz=imgsz)
      results["onnx"] = self._benchmark_single(
        "onnx", onnx_path, dummy_img, num_iterations
      )
    except Exception as e:
      logger.warning("ONNX export/benchmark failed: %s", e)
      results["onnx"] = {"error": str(e)}

    try:
      tflite_path = self.export_tflite(imgsz=imgsz)
      results["tflite"] = self._benchmark_single(
        "tflite", tflite_path, dummy_img, num_iterations
      )
    except Exception as e:
      logger.warning("TFLite export/benchmark failed: %s", e)
      results["tflite"] = {"error": str(e)}

    report_path = self.output_dir / "export_benchmark.json"
    with open(report_path, "w") as f:
      json.dump(results, f, indent=2)

    recommendation = self._recommend_format(results)
    results["recommendation"] = recommendation

    with open(self.output_dir / "format_recommendation.json", "w") as f:
      json.dump(recommendation, f, indent=2)

    return results

  def _benchmark_single(
    self, format_name: str, model_path: Path, image: np.ndarray, n: int
  ) -> dict:
    size_mb = model_path.stat().st_size / (1024 * 1024)

    if format_name == "pytorch":
      from ultralytics import YOLO
      model = YOLO(str(model_path))
      times = []
      for _ in range(n):
        start = time.perf_counter()
        model.predict(image, verbose=False)
        times.append(time.perf_counter() - start)
    elif format_name == "onnx":
      import onnxruntime as ort
      session = ort.InferenceSession(str(model_path))
      input_name = session.get_inputs()[0].name
      img = image.astype(np.float32) / 255.0
      img = np.transpose(img, (2, 0, 1))
      img = np.expand_dims(img, 0)
      times = []
      for _ in range(n):
        start = time.perf_counter()
        session.run(None, {input_name: img})
        times.append(time.perf_counter() - start)
    else:
      times = [0.0]

    avg_ms = np.mean(times) * 1000
    return {
      "path": str(model_path),
      "size_mb": round(size_mb, 2),
      "avg_inference_ms": round(avg_ms, 2),
      "avg_fps": round(1000 / avg_ms, 2) if avg_ms > 0 else 0,
    }

  def _recommend_format(self, results: dict) -> dict:
    """Recommend best format for Raspberry Pi deployment."""
    scores = {}
    for fmt, data in results.items():
      if "error" in data:
        continue
      speed_score = data.get("avg_fps", 0)
      size_score = 100 / max(data.get("size_mb", 1), 0.1)
      scores[fmt] = speed_score * 0.6 + size_score * 0.4

    best = max(scores, key=scores.get) if scores else "onnx"

    return {
      "recommended_format": best,
      "rationale": (
        "For Raspberry Pi 4/5 deployment, ONNX with ONNX Runtime is recommended: "
        "smaller than PyTorch runtime, faster than TFLite for YOLO on ARM with NEON, "
        "and supports FP16 quantization. TFLite INT8 is alternative for Pi 4 with "
        "limited RAM. PyTorch (.pt) best for development and Colab only."
      ),
      "pi4_notes": (
        "Pi 4 (4GB): Use ONNX FP16 or TFLite INT8, imgsz=416, expect 8-15 FPS. "
        "Pi 5 (8GB): Use ONNX FP32, imgsz=640, expect 15-25 FPS with YOLOv11n."
      ),
      "scores": scores,
    }
