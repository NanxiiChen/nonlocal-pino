"""Log-time residual measure and expanding horizon; unchanged CH physics."""
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening import get_config as baseline


def get_config():
    c=baseline()
    c.sample.include_initial_time=False
    c.sample.temporal=ConfigDict(dict(kind='log_uniform',time_scale=.005,
        curriculum=True,initial_horizon=.03,warmup_iterations=3000))
    s=(np.arange(c.train.refinement.time_points)+.5)/c.train.refinement.time_points
    c.train.refinement.time_nodes=tuple(.005*np.expm1(s*np.log1p(c.physical.terminal_time/.005)))
    return c
