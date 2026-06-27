"""Video source abstraction: webcam, file, folder, mobile stream."""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
  """Unified interface for multiple input sources."""

  def __init__(
    self,
    source: Union[int, str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
  ):
    self.source = source
    self.width = width
    self.height = height
    self.cap: Optional[cv2.VideoCapture] = None
    self.is_folder = False
    self.folder_images: list = []
    self.folder_index = 0
    self._needs_brightness_fix = False

  def open(self) -> bool:
    if isinstance(self.source, (str, Path)):
      path = Path(self.source)
      if path.is_dir():
        self.is_folder = True
        self.folder_images = sorted(
          list(path.glob("*.jpg"))
          + list(path.glob("*.jpeg"))
          + list(path.glob("*.png"))
        )
        logger.info("Opened image folder: %d images", len(self.folder_images))
        return len(self.folder_images) > 0

      self.cap = cv2.VideoCapture(str(path))
    else:
      import platform
      backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
      self.cap = cv2.VideoCapture(int(self.source), backend)

    if self.cap is None or not self.cap.isOpened():
      logger.error("Failed to open source: %s", self.source)
      return False

    # For webcam sources, configure camera properties and warm up
    if isinstance(self.source, int):
      # Set resolution
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width or 640)
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height or 480)

      # Try to fix exposure via camera driver
      self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto
      self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
      self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Try brighter exposure
      self.cap.set(cv2.CAP_PROP_GAIN, 200)  # Boost gain

      # Discard warmup frames so auto-exposure settles
      import time
      logger.info("Warming up webcam...")
      for _ in range(30):
        self.cap.read()
        time.sleep(0.03)

      # Check if camera still produces dark frames
      ret, test_frame = self.cap.read()
      if ret and test_frame is not None:
        mean_val = np.mean(test_frame)
        if mean_val < 60:
          self._needs_brightness_fix = True
          logger.info("Dark camera detected (mean=%.0f). Enabling software brightness correction.", mean_val)
        else:
          logger.info("Camera brightness OK (mean=%.0f).", mean_val)
      logger.info("Webcam ready.")
    else:
      if self.width:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
      if self.height:
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    return True

  def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
    if self.is_folder:
      if self.folder_index >= len(self.folder_images):
        return False, None
      img = cv2.imread(str(self.folder_images[self.folder_index]))
      self.folder_index += 1
      return img is not None, img

    if self.cap is None:
      return False, None
    ret, frame = self.cap.read()
    if ret and frame is not None and self._needs_brightness_fix:
      frame = self._correct_brightness(frame)
    return ret, frame

  def _correct_brightness(self, frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to light channel."""
    try:
      lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)
      clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
      cl = clahe.apply(l)
      merged = cv2.merge((cl, a, b))
      return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception as e:
      logger.error("Error during brightness correction: %s", e)
      return frame

  def get_fps(self) -> float:
    if self.is_folder:
      return 30.0
    if self.cap:
      return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    return 30.0

  def release(self) -> None:
    if self.cap:
      self.cap.release()

  def frames(self) -> Generator[cv2.Mat, None, None]:
    while True:
      ret, frame = self.read()
      if not ret:
        break
      yield frame
