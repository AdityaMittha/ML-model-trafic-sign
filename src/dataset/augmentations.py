"""Realistic augmentations for outdoor traffic sign robustness."""

import logging
from typing import Optional

import albumentations as A
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_training_augmentations(image_size: int = 640) -> A.Compose:
  """
  Albumentations pipeline for offline augmentation / analysis.

  Each augmentation targets a specific real-world failure mode:
  - Brightness/Contrast: day/night, over/under-exposure
  - MotionBlur: vehicle motion at 5-10m distance
  - GaussianBlur: defocus, rain on lens
  - RandomRain: wet road conditions
  - RandomFog: foggy/hazy visibility
  - RandomShadow: trees, buildings, passing vehicles
  - Rotate: slight sign tilt from camera angle
  - Perspective: viewing angle variation
  - RandomScale/RandomCrop: distance variation (5-10m)
  - GaussNoise: low-light sensor noise
  """
  return A.Compose(
    [
      A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7,
      ),
      A.MotionBlur(blur_limit=7, p=0.4),
      A.GaussianBlur(blur_limit=(3, 7), p=0.3),
      A.RandomRain(
        slant_lower=-10,
        slant_upper=10,
        drop_length=15,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.9,
        rain_type="drizzle",
        p=0.3,
      ),
      A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.25),
      A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=3,
        shadow_dimension=5,
        p=0.4,
      ),
      A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
      A.Perspective(scale=(0.02, 0.06), p=0.3),
      A.RandomScale(scale_limit=0.3, p=0.5),
      A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
      A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=30,
        val_shift_limit=30,
        p=0.5,
      ),
    ],
    bbox_params=A.BboxParams(
      format="yolo",
      label_fields=["class_labels"],
      min_visibility=0.3,
    ),
  )


def get_validation_augmentations() -> A.Compose:
  """Minimal transforms for validation - no augmentation."""
  return A.Compose([])


def apply_augmentation(
  image: np.ndarray,
  bboxes: list,
  class_labels: list,
  augmenter: Optional[A.Compose] = None,
) -> tuple:
  """Apply augmentation pipeline to image and YOLO bboxes."""
  if augmenter is None:
    augmenter = get_training_augmentations()

  transformed = augmenter(image=image, bboxes=bboxes, class_labels=class_labels)
  return transformed["image"], transformed["bboxes"], transformed["class_labels"]


AUGMENTATION_RATIONALE = {
  "RandomBrightnessContrast": (
    "Simulates varying daylight, dusk/dawn, and artificial lighting. "
    "Critical for day/night operation."
  ),
  "MotionBlur": (
    "Models camera shake and vehicle motion blur when signs are viewed "
    "at 5-10m while moving."
  ),
  "GaussianBlur": (
    "Represents defocus, dirty lenses, and moderate rain reducing sharpness."
  ),
  "RandomRain": (
    "Wet weather reduces visibility and adds streak artifacts on the image."
  ),
  "RandomFog": (
    "Fog and haze significantly reduce contrast at distance - common in Indian winters."
  ),
  "RandomShadow": (
    "Trees, buildings, and vehicles cast shadows over signs on Indian roads."
  ),
  "Rotate": (
    "Signs may appear tilted due to mounting angle or camera pitch."
  ),
  "Perspective": (
    "Viewing angle changes as vehicle approaches sign from different lanes."
  ),
  "RandomScale": (
    "Sign apparent size varies with distance (5m vs 10m detection range)."
  ),
  "GaussNoise": (
    "Low-light and high-ISO camera noise in nighttime driving."
  ),
  "HueSaturationValue": (
    "Color shifts from weather, sun angle, and different camera sensors."
  ),
}
