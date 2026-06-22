# Raspberry Pi Deployment Readiness Assessment

## Overview

This phase designs the model for future Raspberry Pi 4 and Pi 5 deployment on an autonomous robotic vehicle. No Pi deployment is implemented in this phase.

---

## Target Hardware

| Device | RAM | CPU | GPU | Target FPS | Target imgsz |
|--------|-----|-----|-----|------------|--------------|
| Raspberry Pi 4 | 4–8 GB | Quad A72 1.5GHz | VideoCore VI | 8–15 | 416 |
| Raspberry Pi 5 | 4–8 GB | Quad A76 2.4GHz | VideoCore VII | 15–25 | 640 |

---

## Model Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Lightweight model (<10 MB) | ✓ | YOLOv11n = 5.4 MB |
| Low parameters (<5M) | ✓ | 2.6M parameters |
| ONNX export | ✓ | `scripts/export_model.py` |
| TFLite export | ✓ | INT8 quantization supported |
| CPU inference path | ✓ | `--device cpu` in inference |
| Structured JSON output | ✓ | Future ESP32/MQTT integration ready |
| imgsz flexibility | ✓ | 416 for Pi 4, 640 for Pi 5 |
| Memory < 500 MB inference | ✓ | Expected ~200–400 MB with ONNX |

---

## Recommended Pi Deployment Stack (Future Phase)

```
Camera (Pi Camera / USB)
    → OpenCV capture (640×480 or 1280×720)
    → Resize to model input (416 or 640)
    → ONNX Runtime inference (YOLOv11n)
    → NMS + confidence filter (0.45)
    → JSON detection output
    → [Future] ESP32 / vehicle control
```

### Format Recommendation

| Pi Model | Format | Quantization | imgsz | Expected FPS |
|----------|--------|--------------|-------|--------------|
| Pi 4 (4GB) | TFLite or ONNX | INT8 / FP16 | 416 | 8–15 |
| Pi 4 (8GB) | ONNX | FP16 | 416–640 | 12–18 |
| Pi 5 (4GB) | ONNX | FP32/FP16 | 640 | 15–22 |
| Pi 5 (8GB) | ONNX | FP32 | 640 | 20–28 |

**Primary recommendation:** ONNX with ONNX Runtime — best speed/accuracy balance on ARM with NEON.

---

## Memory Budget

| Component | Pi 4 Estimate | Pi 5 Estimate |
|-----------|-----------------|---------------|
| OS + services | ~800 MB | ~800 MB |
| OpenCV + capture | ~100 MB | ~100 MB |
| Model weights | ~10 MB | ~10 MB |
| Inference runtime | ~150–300 MB | ~150–300 MB |
| Frame buffers | ~50 MB | ~50 MB |
| **Total** | ~1.1–1.2 GB | ~1.1–1.2 GB |

Pi 4 with 4GB RAM is sufficient with imgsz=416 and INT8 quantization.

---

## Optimization Path (Future Phase)

1. **Export:** `python scripts/export_model.py --model best.pt --benchmark`
2. **Quantize:** ONNX FP16 or TFLite INT8 calibration on validation set
3. **Profile:** Measure FPS on Pi with `rpicam-vid` or USB webcam
4. **Tune:** Lower imgsz to 416 if FPS < 15; raise confidence to reduce NMS load
5. **Hardware:** Pi 5 recommended for 640px + 25 FPS target

---

## Camera Considerations for 5–10m Detection

- **Resolution:** 1280×720 minimum for sign detail at 10m
- **FPS:** 30 FPS capture; process every frame or skip-frame for 15 FPS inference
- **FOV:** Wide-angle lens may reduce apparent sign size — validate bbox sizes in training data
- **Night:** Ensure night/rain augmentations; consider IR illumination for Pi camera

---

## Integration Readiness

Detection output format is standardized for future phases:

```json
{
  "class_name": "Stop",
  "confidence": 0.94,
  "timestamp": "2026-01-01T10:00:00",
  "bounding_box": [x1, y1, x2, y2]
}
```

No vehicle control, MQTT, or ESP32 code in this phase — output schema is ready for those integrations.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Pi 4 FPS below 15 | imgsz=416, INT8, frame skipping |
| Domain gap (Indian roads) | Fine-tune on Indian dataset; collect local footage |
| Night performance | Night augmentation; IR camera option |
| Small signs at 10m | Train at imgsz=640; ensure small bbox samples in dataset |
| False positives | Confidence 0.45–0.55; per-class thresholds for critical signs |

---

## Conclusion

**YOLOv11 Nano + ONNX export** is ready for Raspberry Pi deployment. Pi 5 at imgsz=640 meets the 15–30 FPS target. Pi 4 requires imgsz=416 or INT8 quantization. Proceed with export and on-device profiling in the deployment phase.
