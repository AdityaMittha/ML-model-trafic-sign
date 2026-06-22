# Traffic Sign Detection — Run, Train & Deploy Guide

## Current Project Status

| Component | Status |
|-----------|--------|
| ✅ Environment (`.venv`) | Ready — all dependencies installed |
| ✅ Model weights (`yolo11n.pt`) | Ready — YOLOv11 Nano pretrained on COCO |
| ✅ Config files | Ready — 20 Indian traffic sign classes defined |
| ✅ Scripts & pipeline code | Ready and debugged |
| ❌ **Training data** | **Empty** — `data/processed/images/` has no images |

> [!IMPORTANT]
> You need to download a dataset before you can train. The pretrained `yolo11n.pt` can already detect COCO objects (cars, people, etc.) but hasn't been fine-tuned for your 20 Indian traffic sign classes yet.

---

## Phase 1 — Environment Setup ✅ (Already Done)

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Verify installation
python -c "from ultralytics import YOLO; print('Ready')"
```

---

## Phase 2 — Get Training Data

### Option A: Kaggle Dataset (Easiest)
```powershell
# Install Kaggle CLI if needed
pip install kaggle

# Set up API key: place kaggle.json in C:\Users\adity\.kaggle\
# Download from: https://www.kaggle.com/settings → "Create New Token"

# Download traffic sign dataset
python scripts/download_datasets.py --kaggle
```

### Option B: Manual Download
```powershell
# Show download instructions for all supported datasets
python scripts/download_datasets.py --instructions
```

Supported datasets:
| Dataset | Type | Best for |
|---------|------|----------|
| Indian Traffic Sign | Detection | Primary — directly relevant |
| Mapillary | Detection | Large-scale, high quality |
| GTSRB | Classification | Supplementary |
| Kaggle Preprocessed | Classification | Quick start |

### After downloading, prepare the dataset:
```powershell
# Merges sources, deduplicates, balances classes, creates train/val/test splits
python scripts/prepare_dataset.py

# Optional: visualize annotations to check quality
python scripts/prepare_dataset.py --visualize
```

This creates:
```
data/processed/
├── images/
│   ├── train/    (80% of data)
│   ├── val/      (10%)
│   └── test/     (10%)
├── labels/       (YOLO format .txt files)
└── dataset.yaml  (already exists, points to the above)
```

---

## Phase 3 — Train the Model

### Basic Training (CPU — will be slow)
```powershell
python scripts/train.py --device cpu --epochs 50 --batch 8
```

### Full Training (GPU recommended)
```powershell
# Uses settings from configs/training_config.yaml
# 150 epochs, batch 16, AdamW optimizer, cosine LR, mosaic augmentation
python scripts/train.py
```

### Training with Custom Settings
```powershell
python scripts/train.py --epochs 100 --batch 16 --imgsz 640 --device 0
```

### Resume Interrupted Training
```powershell
# Automatically finds last checkpoint
python scripts/train.py

# Or force fresh start
python scripts/train.py --no-resume
```

### Hyperparameter Tuning
```powershell
python scripts/train.py --tune
```

**Output**: Best weights saved to `outputs/training/traffic_sign_yolo11n/weights/best.pt`

---

## Phase 4 — Evaluate & Benchmark

### Validate Trained Model
```powershell
python scripts/validate.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt
```

### Error Analysis (FP/FN breakdown)
```powershell
python scripts/test.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt
```

### Benchmark Inference Speed
```powershell
python scripts/benchmark.py
```

---

## Phase 5 — Run Inference

### On Webcam (real-time)
```powershell
python scripts/inference.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --source 0
```

### On a Video File
```powershell
python scripts/inference.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --source path/to/video.mp4
```

### On an Image Folder
```powershell
python scripts/inference.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --source path/to/images/
```

### Quick test with pretrained model (no training needed)
```powershell
python scripts/inference.py --model yolo11n.pt --source 0
```

> [!NOTE]
> The pretrained `yolo11n.pt` detects COCO classes (80 objects). After training on your traffic sign data, it will detect the 20 Indian sign classes instead.

---

## Phase 6 — Deploy (Export for Raspberry Pi)

### Export to All Formats
```powershell
python scripts/export_model.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --benchmark
```

This exports to:
| Format | File | Best For |
|--------|------|----------|
| PyTorch | `.pt` | Development, GPU |
| ONNX | `.onnx` | **Raspberry Pi (recommended)** |
| TFLite | `.tflite` | Pi 4 with limited RAM |

### On Raspberry Pi
```bash
# Install on Pi
pip install onnxruntime opencv-python-headless numpy

# Copy the exported .onnx model to your Pi, then run inference
python scripts/inference.py --model traffic_sign_detector.onnx --source 0 --device cpu
```

### Expected Performance on Pi

| Device | Format | Resolution | Expected FPS |
|--------|--------|-----------|-------------|
| Pi 4 (4GB) | ONNX FP16 | 416×416 | 8–15 FPS |
| Pi 5 (8GB) | ONNX FP32 | 640×640 | 15–25 FPS |

---

## Quick Start — TL;DR

```powershell
# 1. Activate environment
.venv\Scripts\activate

# 2. Get data
python scripts/download_datasets.py --kaggle
python scripts/prepare_dataset.py

# 3. Train
python scripts/train.py --device cpu --epochs 50

# 4. Test on webcam
python scripts/inference.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --source 0

# 5. Export for Pi
python scripts/export_model.py --model outputs/training/traffic_sign_yolo11n/weights/best.pt --benchmark
```
