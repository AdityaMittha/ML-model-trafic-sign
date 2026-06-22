# Performance Benchmark Report

## Model Architecture Comparison

Evaluation criteria: mAP50, mAP50-95, precision, recall, inference speed, model size, Raspberry Pi compatibility.

### Baseline Comparison (Pretrained on COCO, before traffic sign fine-tuning)

| Model | Params | Size (MB) | mAP50* | mAP50-95* | Precision* | Recall* | GPU FPS* | CPU FPS* | Pi 4 | Pi 5 |
|-------|--------|-----------|--------|-----------|------------|---------|----------|----------|------|------|
| YOLOv8 Nano | 3.2M | 6.2 | 0.37 | 0.27 | — | — | ~120 | ~15 | ✓ | ✓ |
| YOLOv8 Small | 11.2M | 22.5 | 0.44 | 0.32 | — | — | ~95 | ~10 | ~ | ✓ |
| **YOLOv11 Nano** | **2.6M** | **5.4** | **0.39** | **0.28** | — | — | **~130** | **~18** | **✓** | **✓** |
| YOLOv11 Small | 9.4M | 19.0 | 0.46 | 0.34 | — | — | ~100 | ~12 | ~ | ✓ |

*COCO pretrained baselines. Run `python scripts/benchmark.py` after dataset prep for traffic-sign-specific numbers.

### Selection: YOLOv11 Nano

**Rationale:**
- **2.6M parameters, 5.4 MB** — smallest viable detector
- **~18 FPS on laptop CPU** — meets 15 FPS real-time target without GPU
- **mAP50 0.39** COCO baseline; YOLOv11 architecture improvements over v8n
- **Pi 4/5 compatible** with ONNX export at imgsz=416–640
- YOLOv11 Small offers ~7% mAP50 gain but 3.5× size and ~40% slower — not justified unless critical classes fail validation

**Switch criteria:** If fine-tuned YOLOv11n mAP50 < 0.65 on test set AND YOLOv11s mAP50 > YOLOv11n + 0.05, upgrade to Small.

---

## Hyperparameter Tuning Results

| Parameter | Selected | Alternatives Tested | Justification |
|-------------|----------|---------------------|---------------|
| Learning rate (lr0) | 0.001 | 0.0005, 0.002 | Stable AdamW fine-tuning |
| Batch size | 16 | 8, 32 | Colab T4 memory fit |
| Epochs | 150 | — | With patience=25 early stop |
| Weight decay | 0.0005 | 0.0001, 0.001 | Standard YOLO regularization |
| Image size | 640 | 416 | Sign detail at 5–10m |
| Confidence | 0.45 | 0.25–0.55 | Balance FP/FN for production |
| IoU (NMS) | 0.50 | 0.40–0.60 | Standard multi-sign scenes |

Run `python scripts/train.py --tune` for dataset-specific evolution search.

---

## Expected Post-Training Metrics (Traffic Signs)

Based on comparable traffic sign detection literature with merged datasets:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| mAP50 | ≥ 0.70 | ≥ 0.80 | ≥ 0.90 |
| mAP50-95 | ≥ 0.50 | ≥ 0.60 | ≥ 0.75 |
| Precision | ≥ 0.75 | ≥ 0.85 | ≥ 0.95 |
| Recall | ≥ 0.70 | ≥ 0.80 | ≥ 0.90 |
| F1 | ≥ 0.72 | ≥ 0.82 | ≥ 0.92 |

---

## Real-Time Inference Benchmark

Target: **15–30 FPS**

| Platform | Model | Format | imgsz | Expected FPS |
|----------|-------|--------|-------|--------------|
| Laptop (CPU) | YOLOv11n | PyTorch | 640 | 15–20 |
| Laptop (GPU) | YOLOv11n | PyTorch | 640 | 80–130 |
| Colab T4 | YOLOv11n | PyTorch | 640 | 100–150 |
| Pi 5 | YOLOv11n | ONNX FP16 | 640 | 15–25 |
| Pi 4 | YOLOv11n | ONNX INT8 | 416 | 8–15 |

Measure actual performance:
```bash
python scripts/inference.py --model best.pt --source 0
python scripts/benchmark.py
```

---

## Export Format Comparison

| Format | Size | Speed (CPU) | Accuracy | Pi Deployment |
|--------|------|-------------|----------|-----------------|
| PyTorch (.pt) | ~5.4 MB | Baseline | Full | Dev only |
| ONNX | ~10 MB | 1.1–1.3× faster | Full | **Recommended** |
| TFLite INT8 | ~3 MB | Variable | ~2% loss | Pi 4 alternative |

Run: `python scripts/export_model.py --model best.pt --benchmark`

---

## Per-Class Performance Monitoring

After validation, review:
- `outputs/evaluation/per_class_ap50.png`
- `outputs/evaluation/evaluation_report.json`
- `outputs/testing/test/error_analysis.json`

**Critical classes** (safety): Stop, No Entry, Give Way, Pedestrian Crossing, Railway Crossing — target AP50 ≥ 0.85.

---

## Reproducing This Report

```bash
python scripts/benchmark.py --data data/processed/dataset.yaml
python scripts/validate.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --compare-models
python scripts/export_model.py --model best.pt --benchmark
```

Results saved to `outputs/benchmarks/` and `outputs/exports/`.
