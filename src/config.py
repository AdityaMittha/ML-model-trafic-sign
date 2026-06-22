"""Traffic Sign Detection - Core configuration loader."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_classes_config() -> dict:
    return load_yaml(CONFIGS_DIR / "classes.yaml")


def get_dataset_config() -> dict:
    return load_yaml(CONFIGS_DIR / "dataset_config.yaml")


def get_model_config() -> dict:
    return load_yaml(CONFIGS_DIR / "model_config.yaml")


def get_training_config() -> dict:
    return load_yaml(CONFIGS_DIR / "training_config.yaml")


def get_class_names() -> list[str]:
    cfg = get_classes_config()
    classes = cfg["classes"]
    return [classes[i] for i in sorted(classes.keys())]


def get_class_id(name: str) -> int:
    cfg = get_classes_config()
    for idx, cls_name in cfg["classes"].items():
        if cls_name == name:
            return int(idx)
    raise ValueError(f"Unknown class: {name}")


def get_label_mapping() -> dict:
    return get_classes_config().get("label_mapping", {})
