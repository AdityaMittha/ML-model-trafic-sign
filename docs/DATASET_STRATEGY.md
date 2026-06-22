# Dataset Strategy for Indian Traffic Sign Detection

## Executive Summary

**Recommended strategy:** Merge **Mapillary Traffic Sign Dataset** + **Indian Traffic Sign Dataset** as primary detection sources, supplemented by **GTSRB** and **Kaggle Preprocessed** as classification-to-detection augmentation for underrepresented classes.

---

## Dataset Comparison

| Dataset | Type | Images | Classes | Annotation | Indian Suitability | Role |
|---------|------|--------|---------|------------|-------------------|------|
| **GTSRB** | Classification | 51,839 | 43 | No bboxes | Low | Supplementary appearance priors |
| **Indian Traffic Sign** | Detection | ~5K–15K | Varies | YOLO/COCO | **High** | Primary Indian domain source |
| **Mapillary** | Detection | ~100K | 100+ | High quality | Medium | Primary scale + diversity |
| **Kaggle Preprocessed** | Classification | ~51K | 43 | No bboxes | Low | Supplementary, GTSRB-derived |

---

## Per-Dataset Analysis

### 1. GTSRB (German Traffic Sign Recognition Benchmark)

- **Quality:** High image quality, standardized crops, clean labels
- **Limitation:** Classification only — no bounding boxes; German signs, not Indian road context
- **Class overlap:** 16 of our 20 classes mappable (Stop, speed limits, Give Way, etc.)
- **Decision:** Import as pseudo-detection (full-image bbox 0.95×0.95) for sign appearance learning, not spatial localization

### 2. Indian Traffic Sign Dataset

- **Quality:** Variable; depends on source version
- **Strength:** Indian sign designs, road environments, lighting conditions
- **Limitation:** Often smaller scale, potential class imbalance
- **Decision:** **Primary detection source** — highest domain relevance

### 3. Mapillary Traffic Sign Dataset

- **Quality:** Excellent — professional annotations, diverse geographies
- **Strength:** Large scale, real street scenes with context, multiple weather/lighting
- **Limitation:** Not India-specific; label taxonomy differs (100+ classes)
- **Decision:** **Primary detection source** with label mapping via `configs/classes.yaml`

### 4. Kaggle Traffic Signs Preprocessed

- **URL:** https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed
- **Quality:** Preprocessed GTSRB — same limitations as GTSRB
- **Decision:** Supplementary classification data; easy Kaggle download for Colab

---

## Label Standardization

All datasets map to 20 standardized classes via `configs/classes.yaml`:

```
Stop, No Entry, Left Turn, Right Turn, U-Turn, No Horn, School Zone,
Pedestrian Crossing, Give Way, One Way, Roundabout, Railway Crossing,
Road Work, Narrow Road, Speed Limit 20/30/40/50/60/80
```

Mapillary labels like `regulatory--stop--g1` → `Stop` automatically.

**Classes requiring Indian-specific data:** No Horn, School Zone, Narrow Road — may be underrepresented in Mapillary/GTSRB. Prioritize Indian dataset samples and oversampling.

---

## Merge Pipeline Decisions

| Step | Action | Rationale |
|------|--------|-----------|
| Import | YOLO detection + classification→detection | Unified format |
| Clean | Remove corrupted, duplicate (perceptual hash) | Data quality |
| Validate | YOLO label bounds check | Prevent training errors |
| Balance | Oversample classes < 50 samples | Reduce class bias |
| Split | 80/10/10 train/val/test, seed=42 | Reproducible evaluation |

---

## Class Balance Strategy

- **Oversample** underrepresented classes (not undersample — preserves rare sign diversity)
- Target minimum 50 samples per class before oversampling
- Oversample toward `max_count/2` for severely imbalanced classes
- Monitor per-class AP50 after training; add real footage for weak classes

---

## Augmentation Rationale (Real-World Robustness)

| Augmentation | Real-World Condition |
|--------------|---------------------|
| Brightness/Contrast | Day/night, sun glare, overexposure |
| Motion Blur | Vehicle motion at 5–10m |
| Gaussian Blur | Rain on lens, defocus |
| Rain Simulation | Wet weather visibility |
| Fog Simulation | Winter haze (common in India) |
| Shadow | Trees, buildings on Indian roads |
| Rotation (±15°) | Sign mounting angle |
| Perspective | Lane position variation |
| Scale/Zoom | Distance 5m vs 10m |
| Noise | Low-light sensor noise |

YOLO built-in augmentations (mosaic, mixup, HSV) are enabled in training config.

---

## Recommended Workflow

1. Download Mapillary + Indian dataset manually (registration required for Mapillary)
2. `python scripts/download_datasets.py --kaggle` for supplementary data
3. Place in `data/raw/` per structure in download script
4. `python scripts/prepare_dataset.py --visualize`
5. Review `outputs/reports/annotations_*.png` for quality
6. Train and monitor per-class AP50

---

## Future Improvements

- Collect custom Indian road footage with manual annotation for domain gap
- Active learning: annotate failure cases from error analysis
- Semi-supervised: use high-confidence predictions on unlabeled video
