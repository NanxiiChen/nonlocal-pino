"""Continuous online initial interfaces; no finite train bank; B8 Adam pilot."""
from ml_collections import ConfigDict
from examples.electropolish2d.configs.train_operator import get_config as operator_config


def get_config():
    c = operator_config()
    # Existing train/test counts describe the REFERENCE BANK only in online mode.
    c.operator.condition_batch = 8
    c.operator.online = ConfigDict()
    c.operator.online.candidate_factor = 4
    c.operator.online.curve_points = 2000
    c.operator.online.sdf_window = 16
    c.operator.online.height_min = .005
    c.operator.online.height_max = .495
    c.train.optimizer = 'adam'  # fixed-input control: lower pooled error and cheaper than SOAP
    c.train.iterations = 5000
    return c
