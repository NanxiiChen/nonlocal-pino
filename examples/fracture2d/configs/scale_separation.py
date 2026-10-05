"""Explicit coarse pilot for the separately validated horizon branch."""

from ml_collections import ConfigDict


def get_config():
    c = ConfigDict()
    c.data = ConfigDict()
    c.data.cloud = "/root/autodl-tmp/data/fracture2d/scale_separation_20261001/cloud_h008_api.npz"
    c.data.cloud_sha256 = "2121855eea8355ee8dca827a1dec41f1306af0deba5b4d9e535f00a55753d960"
    c.geometry = ConfigDict()
    c.geometry.grip_extent = 0.048
    c.geometry.h_grip = 0.008
    c.nl = ConfigDict()
    c.nl.delta_u = 0.016
    c.nl.delta_d = 0.048
    c.nl.visibility_tolerance = 1e-12
    c.physical = ConfigDict()
    c.physical.E = 210000.0
    c.physical.nu = 0.3
    c.physical.k_res = 1e-8
    c.physical.rank_rtol = 1e-12
    c.physical.Gc = 2.7
    c.physical.l0 = 0.024
    c.validation = ConfigDict()
    c.validation.load = 1e-4
    c.validation.cg_rtol = 1e-9
    c.validation.cg_atol = 1e-10
    c.validation.cg_maxiter = 10000
    c.validation.affine_atol = 1e-12
    c.validation.gradient_rtol = 1e-9
    c.validation.directional_step = 1e-6
    c.validation.directional_rtol = 1e-5
    c.validation.work_step = 1e-6
    c.validation.work_relative_limit = 1e-5
    c.validation.roundoff_factor = 1024.0
    c.validation.widths = [0.008, 0.024, 0.032, 0.048, 0.096]
    c.output = ConfigDict()
    c.output.dir = "/root/autodl-tmp/tf-logs/fracture2d/scale_sep_h008_du016_20261001"
    return c
