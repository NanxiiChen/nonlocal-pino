"""One spatial MLP/global time expansion; all coefficients trained together."""
from examples.confined_ch2d.configs.train_coarsening_integral import get_config as baseline
from examples.confined_ch2d.time_integration import integration_nodes


def get_config():
    c=baseline();c.arch.model='global_basis';c.arch.rank=24
    c.arch.output_scale=.01;c.arch.initial_feature=True
    c.phase.lifting='logarithmic_additive';c.phase.initial_time_scale=c.loss.phase_metric.time_scale
    nodes=tuple(integration_nodes(count=80,duration=c.physical.terminal_time,time_scale=c.loss.phase_metric.time_scale)[1:])
    c.sample.Nt=len(nodes);c.sample.temporal.nodes=nodes
    c.train.refinement.time_points=len(nodes);c.train.refinement.time_nodes=nodes
    return c
