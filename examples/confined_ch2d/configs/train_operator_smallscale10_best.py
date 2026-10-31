"""Historical FP64 width96 recipe for a fixed10mode geometry-operator run.

The measured best checkpoint was obtained by complete-state10k+10k+10k stages.
This cold entry shares its numerical settings and does not require parent logs.
The current operator SOTA is selected by configs/train_operator.py.
"""
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch_wide import get_config as baseline


def get_config():
    c=baseline()
    c.train.iterations=30000
    return c
