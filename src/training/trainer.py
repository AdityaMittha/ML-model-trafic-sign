"""YOLO training with transfer learning, AMP, early stopping, and checkpoint recovery."""

import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.config import get_model_config, get_training_config, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class TrafficSignTrainer:
  """Production training pipeline for traffic sign detection."""

  def __init__(
    self,
    model_weights: Optional[str] = None,
    data_yaml: Optional[str] = None,
    project_dir: Optional[Path] = None,
  ):
    model_cfg = get_model_config()
    train_cfg = get_training_config()

    self.model_weights = model_weights or model_cfg.get("selected_weights", "yolo11n.pt")
    self.data_yaml = data_yaml or train_cfg["training"]["data"]
    self.project_dir = project_dir or (OUTPUTS_DIR / "training")
    self.train_params = train_cfg["training"]
    self.model: Optional[YOLO] = None

  def _resolve_checkpoint(self) -> str:
    """Find latest checkpoint for automatic recovery."""
    name = self.train_params.get("name", "traffic_sign_yolo11n")
    weights_dir = self.project_dir / name / "weights"
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
      logger.info("Resuming from checkpoint: %s", last_pt)
      return str(last_pt)
    return self.model_weights

  def load_model(self, resume: bool = True) -> YOLO:
    weights = self._resolve_checkpoint() if resume else self.model_weights
    self.model = YOLO(weights)
    logger.info("Loaded model: %s", weights)
    return self.model

  def train(self, resume: bool = True, **override_params) -> dict:
    """
    Train with transfer learning from pretrained weights.

    Features:
    - Mixed precision (AMP) for faster training
    - Early stopping via patience parameter
    - Cosine LR scheduling
    - Mosaic/mixup augmentation (built into YOLO)
    - Automatic checkpoint saving every save_period epochs
    """
    if self.model is None:
      self.load_model(resume=resume)

    import torch

    params = {k: v for k, v in self.train_params.items() if k != "data"}
    params.update(override_params)

    if "device" in params:
      device_val = params["device"]
      if device_val not in ("cpu", "CPU") and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to device='cpu'.")
        params["device"] = "cpu"


    data_path = Path(self.data_yaml)
    if not data_path.is_absolute():
      from src.config import PROJECT_ROOT
      data_path = PROJECT_ROOT / self.data_yaml

    if not data_path.exists():
      raise FileNotFoundError(
        f"Dataset YAML not found: {data_path}. Run prepare_dataset.py first."
      )

    results = self.model.train(
      data=str(data_path),
      resume=resume and (self.project_dir / params.get("name", "traffic_sign_yolo11n") / "weights" / "last.pt").exists(),
      **params,
    )

    logger.info("Training complete.")
    return {
      "results": str(results),
      "best_weights": str(
        self.project_dir / params.get("name", "traffic_sign_yolo11n") / "weights" / "best.pt"
      ),
    }

  def tune_hyperparameters(self, iterations: int = 5) -> dict:
    """Run Ultralytics hyperparameter evolution search."""
    if self.model is None:
      self.load_model(resume=False)

    from src.config import PROJECT_ROOT
    data_path = PROJECT_ROOT / self.data_yaml

    results = self.model.tune(
      data=str(data_path),
      iterations=iterations,
      optimizer="AdamW",
      plots=True,
      save=True,
      val=True,
    )
    return {"tune_results": str(results)}
