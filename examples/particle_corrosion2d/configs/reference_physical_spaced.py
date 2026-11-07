"""Independent local/native reference for six separated ceramic inclusions."""
from examples.particle_corrosion2d.configs.train_physical_spaced import get_config as training_config


def get_config():
    c=training_config()
    c.reference.rtol=2e-8
    c.reference.atol=2e-10
    c.reference.max_step=.00625
    return c
