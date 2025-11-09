import os
import yaml
from src.core_logger import logger

DEFAULT_CONFIG_PATH = "config.yaml"

def load_config(path: str = DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        logger.warning(f"No config.yaml found at {path}, using environment vars only.")
        return {}

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {path}")
    return cfg

def get_env_or_config(key: str, cfg: dict, default=None):
    return os.getenv(key) or cfg.get(key, default)
