"""Original-residual/nodal-data strategy for 50 s corrosion.

Ordinary FP64 selected from full42/2026 controls; seed is runtime data.
The historical best single bounded checkpoint is documented separately.
See the example README and artifacts/index.yaml for transport limitations.
"""
from examples.corrosion2d.configs.residual_pilot import get_strategy_config as matched_config


def get_config():
    c = matched_config("corrosion2d", "phase")
    c.train.gpu_policy = "ordinary"
    c.train.compute_dtype = "float64"
    c.experiment.bounded_eta = 0.1
    c.experiment.data_loss_mode = "s_bc"
    c.experiment.initial_sampling = "matched_graph"
    c.experiment.boundary_sampling = "matched_graph"
    c.experiment.transport_evaluation = "kks_raw_edges"
    c.experiment.ch_norm = "l1"
    c.log.run_name = ""  # fresh timestamped run; never overwrite the saved winner
    return c
