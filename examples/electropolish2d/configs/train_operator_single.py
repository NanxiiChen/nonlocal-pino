"""Fixed input0/B1 CViT fitting control; retain source dataset and2000Adam budget."""
from examples.electropolish2d.configs.train_operator_legacy import get_config as operator_config


def get_config():
    c = operator_config()
    c.operator.training_indices = (0,)
    c.operator.condition_batch = 1
    return c
