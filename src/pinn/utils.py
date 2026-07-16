"""Shared utilities for PINN training."""

from ml_collections import ConfigDict

def log_config_text(writer, config: ConfigDict) -> None:
    """Log pretty-printed config JSON as text in TensorBoard."""
    import json as _json
    text = _json.dumps(_json.loads(config.to_json_best_effort()), indent=2)
    writer.add_text('config', text, 0)
