"""Matched direct damage baseline; mechanics/model/stagger match train_base."""
from examples.fracture2d.configs.train_base import get_config as train_config


def get_config():
    c = train_config()
    c.output.root = '/root/autodl-tmp/tf-logs/fracture2d/runs/nonlocal_direct'
    return c
