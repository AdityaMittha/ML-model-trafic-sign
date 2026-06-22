"""Video source abstraction: webcam, file, folder, mobile stream."""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2

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
      self.cap = cv2.VideoCapture(int(self.source))

    if self.cap is None or not self.cap.isOpened():
      logger.error("Failed to open source: %s", self.source)
      return False

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
    return self.cap.read()

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
