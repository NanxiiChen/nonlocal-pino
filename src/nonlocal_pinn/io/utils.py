"""Shared utilities for PINN training."""

import os
import time

from ml_collections import ConfigDict


# ── TensorBoard ──────────────────────────────────────────────────────────────

def log_config_text(writer, config: ConfigDict) -> None:
    """Log pretty-printed config JSON as text in TensorBoard."""
    import json as _json
    text = _json.dumps(_json.loads(config.to_json_best_effort()), indent=2)
    writer.add_text('config', text, 0)


# ── Example scaffolding ──────────────────────────────────────────────────────

_LOSS_CACHE = {}

def load_example_losses(caller_file: str):
    """Load ``losses.py`` from the same directory as *caller_file*.  Cached."""
    import importlib.util as _iu
    import os

    dir_path = os.path.dirname(os.path.abspath(caller_file))
    path = os.path.join(dir_path, 'losses.py')
    if path not in _LOSS_CACHE:
        spec = _iu.spec_from_file_location(
            os.path.basename(dir_path) + '_losses', path)
        _LOSS_CACHE[path] = _iu.module_from_spec(spec)
        spec.loader.exec_module(_LOSS_CACHE[path])
    return _LOSS_CACHE[path]


def setup_run_dir(config: ConfigDict) -> str:
    """Create and return the log directory for a training run."""
    run = config.log.run_name or time.strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(config.log.log_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
