"""Load an explicit study configuration: --config=this_file.py:/path/config.json."""
import json
from pathlib import Path
from ml_collections import ConfigDict
from examples.electropolish2d.horizon_study import monitor_config


def get_config(path):
    c = ConfigDict(json.loads(Path(path).read_text()))
    monitor_config(c)
    return c
