"""Original-variable global L-BFGS finisher of the audited finer-time model."""
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_single_time_refined import get_config as baseline


def get_config():
    c=baseline();del c.train.continuation
    c.train.refinement=ConfigDict(dict(
        parent_run='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-time-refine-single-time-01',
        optimizer='lbfgs',iterations=250,time_batch=8,history_size=20,
        max_evaluations=1200,max_line_search=30,gradient_tolerance=1e-6,function_tolerance=1e-9))
    c.log.eval_every=25;c.log.checkpoint_every=100
    return c
