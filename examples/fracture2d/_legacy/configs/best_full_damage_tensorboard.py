"""CPU monitor settings; scalar cadence does not change the numerical solver."""
from ml_collections import ConfigDict


def get_config():
    return ConfigDict(dict(
        solution_root='/root/autodl-tmp/tf-logs/fracture2d/best_full_pinn_loading_1024_20261014/fine',
        log_root='/root/autodl-tmp/tf-logs',
        event_dir='/root/autodl-tmp/tf-logs/fracture2d/events/best_full_pinn_loading_1024_20261014',
        scalar_every=8, poll_seconds=5., flush_seconds=5,
        log_source_endpoint=True, figure_dpi=100, cpu_threads=2))
