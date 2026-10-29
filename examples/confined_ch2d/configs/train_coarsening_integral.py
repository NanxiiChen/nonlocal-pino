"""Global Volterra residual; one network, no intermediate-state propagation."""
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_eliminated_blend import get_config as baseline
from examples.confined_ch2d.time_integration import integration_nodes


def get_config():
    c=baseline();scale=.005
    nodes=tuple(integration_nodes(count=32,duration=c.physical.terminal_time,time_scale=scale)[1:])
    c.sample.Nt=len(nodes)
    c.sample.temporal=ConfigDict(dict(kind='fixed_nodes',nodes=nodes))
    c.loss.phase_metric=ConfigDict(dict(kind='integral_resolvent',time_scale=scale))
    c.train.refinement.time_points=len(nodes);c.train.refinement.time_nodes=nodes
    return c
