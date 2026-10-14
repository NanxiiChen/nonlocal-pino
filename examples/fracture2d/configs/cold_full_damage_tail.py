"""Frozen continuation controls created only after this run's prefix audit."""
import json
from pathlib import Path
from ml_collections import ConfigDict


def get_config():
    path = Path('/root/autodl-tmp/tf-logs/fracture2d/cold_full_pinn_loading_0060_20261014/tail_driver_config.json')
    if not path.is_file():
        raise ValueError('fresh prefix and its complete native audit must precede continuation')
    return ConfigDict(json.loads(path.read_text()), allow_dotted_keys=True)
