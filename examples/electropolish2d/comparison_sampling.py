"""Method-independent prescribed inputs for paired derivative experiments.

The original GPU sampler still chooses profiles and evaluates variable query
sets. Shared encoder/IC/Dirichlet inputs are evaluated together on CPU in FP64,
then cast once to FP32. Their fixed layout cannot depend on halo/AD row counts.
No evolved reference field is used. Production presets retain the old sampler.
"""
import jax
import numpy as np

from examples.electropolish2d.losses import _rough_surface, _sdf_to_curve
from examples.electropolish2d.online_conditions import make_sampler


KIND = "host_float64_shared"


def prescribed_shared_fields(coefficients, points, c):
    """Known initial data on the original FP32-rounded polyline vertices."""
    points = np.asarray(points, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or coefficients.ndim != 3
            or coefficients.shape[1:] != (c.surface.n_modes, 3)
            or not np.isfinite(points).all() or not np.isfinite(coefficients).all()):
        raise ValueError("Finite shared coordinates and complete profile triples required")
    curve_x = np.linspace(0, 1, c.operator.online.curve_points).astype(np.float32).astype(np.float64)
    k = np.sqrt(c.physical.omega_phi / (2 * c.physical.alpha_phi)) * c.nondim.L_c
    fields = []
    for profile in coefficients:
        curve_y = _rough_surface(curve_x, c.surface.y0, profile)
        surface_y = _rough_surface(points[:, 0], c.surface.y0, profile)
        distance = _sdf_to_curve(points[:, 0], points[:, 1], curve_x, curve_y,
                                 surface_y, window=c.operator.online.sdf_window)
        phi = .5 * (1 + np.tanh(k * distance))
        phi[np.isclose(points[:, 1], .5)] = 0.
        phi[np.isclose(points[:, 1], 0.)] = 1.
        fields.append(np.stack((phi, c.physical.c_se * (3 * phi**2 - 2 * phi**3)), axis=-1))
    return np.asarray(fields, dtype=np.float32)


def make_comparison_sampler(c):
    if (c.operator.online.input_evaluation != KIND
            or c.operator.method_control != "rolling_derivative_pair"
            or c.arch.local_input != "zero"
            or c.nl.method not in ("sparse", "local_ad", "fem_weak")):
        raise ValueError("Shared prescribed-input mode requires the declared image-only comparison")
    original, geometry = make_sampler(c)
    _, height, width = c.operator.image_shape
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, .5, height), indexing="xy")
    sensors = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    shared_count = c.operator.ic_points + 2 * c.sample.Nx
    side_count = 2 * (c.sample.Ny - 2) if c.nl.method == "local_ad" else 0

    def sample(key, points, nearest):
        # Keep the original random draws, acceptance and variable-query values.
        _, local, coefficients, count = original(key, points, nearest)
        first = len(points) - shared_count - side_count
        if first <= 0:
            raise ValueError("Missing PDE/core rows before the common IC/Dirichlet queries")
        common = np.asarray(points[first:first + shared_count], dtype=np.float32)
        fields = prescribed_shared_fields(np.asarray(coefficients),
                                         np.concatenate((sensors, common)), c)
        image = jax.device_put(fields[:, :len(sensors), 0].reshape(-1, 1, height, width))
        known = jax.device_put(fields[:, len(sensors):])
        local = local.at[:, first:first + shared_count].set(known)
        return image, local, coefficients, count

    return sample, geometry
