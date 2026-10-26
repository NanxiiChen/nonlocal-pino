"""Original-residual/random-data strategy for 5 s; seed is runtime data.

Ordinary FP32 selected from full42/2026 controls; seed is runtime data.
See the example README for the independent-seed evidence.
"""
from examples.corrosion2d.configs.residual_pilot import get_strategy_config as matched_config


def get_config():
    c = matched_config("electropolish2d", "phase")
    c.train.gpu_policy = "ordinary"
    c.train.compute_dtype = "float32"
    c.experiment.data_loss_mode = "physical"
    c.experiment.initial_sampling = "random_interior"
    c.experiment.boundary_sampling = "random"
    c.experiment.transport_evaluation = "physical_difference"
    c.experiment.ch_norm = "l1"
    c.log.run_name = ""  # fresh cold start; old models load by explicit path
    return c
