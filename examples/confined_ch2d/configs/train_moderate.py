"""Intermediate-difficulty three-wave case; independent reference required."""
from examples.confined_ch2d.configs.train_coarsening_average_lowfreq import get_config as baseline
from examples.confined_ch2d.time_integration import integration_nodes


def get_config():
    c=baseline()
    c.initial.waves=((1,1,.8,0.),(2,-1,.7,.8),(1,-2,.6,1.6))
    c.physical.terminal_time=.75
    c.reference.snapshots=11
    c.reference.root='/root/autodl-tmp/data/confined_ch2d/moderate_multimode3'
    c.reference.path=c.reference.root+'/reference.npz'
    nodes=tuple(integration_nodes(count=80,duration=c.physical.terminal_time,
        time_scale=c.loss.phase_metric.time_scale)[1:])
    c.sample.Nt=len(nodes);c.sample.temporal.nodes=nodes
    c.train.refinement.time_points=len(nodes);c.train.refinement.time_nodes=nodes
    return c
