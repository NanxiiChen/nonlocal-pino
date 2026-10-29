"""One global network with PDE and thermodynamic integral balances."""
from examples.confined_ch2d.configs.train_coarsening_global_basis import get_config as baseline
from examples.confined_ch2d.time_integration import integration_nodes


def get_config():
    c=baseline();c.loss.energy_weight=1.
    nodes=tuple(integration_nodes(count=160,duration=c.physical.terminal_time,time_scale=c.loss.phase_metric.time_scale)[1:])
    c.sample.Nt=len(nodes);c.sample.temporal.nodes=nodes
    c.train.refinement.time_points=len(nodes);c.train.refinement.time_nodes=nodes
    return c
