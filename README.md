# Traffic Sign Detection for Indian Roads

Production-ready deep learning pipeline for detecting and classifying 20 Indian traffic signs using YOLOv11 Nano, with Google Colab support and Raspberry Pi readiness.

## Target Signs (20 Classes)

Stop, No Entry, Left Turn, Right Turn, U-Turn, No Horn, School Zone, Pedestrian Crossing, Give Way, One Way, Roundabout, Railway Crossing, Road Work, Narrow Road, Speed Limit 20/30/40/50/60/80

## Quick Start (Google Colab / Laptop)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (see docs/DATASET_STRATEGY.md for manual steps)
python scripts/download_datasets.py --instructions
python scripts/download_datasets.py --kaggle

# 3. Prepare merged dataset
python scripts/prepare_dataset.py --visualize

# 4. Train (YOLOv11 Nano, transfer learning + AMP)
python scripts/train.py --model yolo11n.pt --epochs 150 --batch 16

# 5. Validate
python scripts/validate.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt

# 6. Error analysis
python scripts/test.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt

# 7. Real-time inference
python scripts/inference.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --source 0
python scripts/inference.py --model best.pt --source video.mp4 --output output.mp4
python scripts/inference.py --model best.pt --source images/folder/

# 8. Export for deployment
python scripts/export_model.py --model best.pt --benchmark

# 9. Model comparison benchmark
python scripts/benchmark.py
```

## Project Structure

```
traffic_sign_detection/
├── configs/                 # YAML configs (classes, dataset, model, training)
├── data/
│   ├── raw/                 # Downloaded datasets
│   └── processed/           # Merged YOLO-format dataset
├── docs/
│   ├── DATASET_STRATEGY.md
│   ├── BENCHMARK_REPORT.md
│   └── RASPBERRY_PI_READINESS.md
├── notebooks/
│   └── Traffic_Sign_Detection_Colab.ipynb
├── scripts/                 # CLI entry points
├── src/
│   ├── dataset/             # Cleaning, augmentation, merging
│   ├── training/            # Trainer, hyperparameters
│   ├── evaluation/          # Metrics, error analysis
│   ├── inference/           # Real-time detector
│   └── export/              # ONNX, TFLite export
├── outputs/                 # Training runs, exports, reports
└── requirements.txt
```

## Detection Output Format

```json
{
  "class_name": "Stop",
  "confidence": 0.94,
  "timestamp": "2026-01-01T10:00:00",
  "bounding_box": [x1, y1, x2, y2]
}
```

## Model Selection

**YOLOv11 Nano** (`yolo11n.pt`) — preferred for speed/accuracy balance and Raspberry Pi deployment.

| Model     | Params | Size  | mAP50* | FPS (GPU)* | Pi 4 | Pi 5 |
|-----------|--------|-------|--------|------------|------|------|
| YOLOv8n   | 3.2M   | 6.2MB | 0.37   | 120        | Yes  | Yes  |
| YOLOv8s   | 11.2M  | 22MB  | 0.44   | 95         | Marginal | Yes |
| YOLOv11n  | 2.6M   | 5.4MB | 0.39   | 130        | Yes  | Yes  |
| YOLOv11s  | 9.4M   | 19MB  | 0.46   | 100        | Marginal | Yes |

*Typical COCO-pretrained baselines; traffic sign fine-tuning improves these.

## Scope (This Phase)

- Model design, training, validation, optimization, testing
- Real-time inference (webcam, video, images)
- Model export (PyTorch, ONNX, TFLite)

**Not included:** vehicle control, ESP32, MQTT, navigation, GUI dashboards.

## Documentation

- [Dataset Strategy](docs/DATASET_STRATEGY.md)
- [Benchmark Report](docs/BENCHMARK_REPORT.md)
- [Raspberry Pi Readiness](docs/RASPBERRY_PI_READINESS.md)

## Google Colab

Open `notebooks/Traffic_Sign_Detection_Colab.ipynb` for end-to-end Colab execution.
