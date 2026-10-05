"""Rank, affine, variational/work and failed-band gates for split horizons."""

import json
import os
from pathlib import Path

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_flags
from scipy.sparse import diags

from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.split_horizon_model import build_model, read_cloud


DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "scale_separation.py")
CONFIG = config_flags.DEFINE_config_file("config", DEFAULT, "Explicit independent-horizon validation config")


def run(config):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Path(config.output.dir)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir()
    atomic_json(output / "config.json", json.loads(config.to_json_best_effort()))
    cloud = read_cloud(config.data.cloud, config.data.cloud_sha256)
    model = build_model(cloud, delta_u=config.nl.delta_u, delta_d=config.nl.delta_d,
        grip_extent=config.geometry.grip_extent, h_grip=config.geometry.h_grip,
        visibility_tolerance=config.nl.visibility_tolerance, E=config.physical.E,
        nu=config.physical.nu, k_res=config.physical.k_res, rank_rtol=config.physical.rank_rtol,
        Gc=config.physical.Gc, l0=config.physical.l0)
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    points = jnp.asarray(model.geometry.coordinates)
    gradient = jnp.asarray([[1e-4, 2e-5], [-3e-5, 2e-4]])
    q = (points @ gradient.T).reshape(-1)
    damage = jnp.full(model.n_material, 0.2, jnp.float64)
    factor = backend._edge_factor(damage)
    deformation = backend._deformation_gradient(factor, q)
    affine_error = float(jnp.max(abs(deformation - (jnp.eye(2) + gradient)[None, :, :])))
    if affine_error > config.validation.affine_atol:
        raise ValueError(f"split affine patch failed: {affine_error}")
    explicit = backend._residual_from_factor(factor, q)
    automatic = jax.grad(lambda v: backend._energy_from_factor(factor, v))(q)
    gradient_defect = float(jnp.max(abs(explicit - automatic))) / float(jnp.max(abs(explicit)))
    if gradient_defect > config.validation.gradient_rtol:
        raise ValueError(f"split energy gradient differs: {gradient_defect}")
    direction = np.zeros(model.n_dofs)
    free = np.asarray(backend.free_dofs)
    direction[free] = np.random.default_rng(0).normal(size=len(free))
    direction /= np.linalg.norm(direction)
    direction = jnp.asarray(direction)
    epsilon = config.validation.directional_step
    finite = float((backend._energy_from_factor(factor, q + epsilon * direction)
                    - backend._energy_from_factor(factor, q - epsilon * direction)) / (2 * epsilon))
    derivative = float(explicit @ direction)
    directional_defect = abs(finite - derivative) / abs(derivative)
    if not np.isfinite(directional_defect) or directional_defect > config.validation.directional_rtol:
        raise ValueError(f"split independent energy finite difference failed: {directional_defect}")
    D = model.material_operator.to_scipy().tocsr()
    scalar_norm = max(abs(D.diagonal()))
    constant = float(max(abs(np.asarray(residual.operator.deterministic_matvec(jnp.ones(model.n_material, jnp.float64))))))
    allowed = config.validation.roundoff_factor * np.finfo(np.float64).eps * scalar_norm
    if constant > allowed:
        raise ValueError(f"split scalar constant residual exceeds roundoff: {constant} > {allowed}")
    paired = diags(model.geometry.material_reference_volumes) @ D
    asymmetry = paired - paired.T
    skew = float(max(abs(asymmetry.data))) if asymmetry.nnz else 0.0
    if skew > config.validation.roundoff_factor * np.finfo(np.float64).eps * max(abs(paired.data)):
        raise ValueError("split scalar volume pairing is not symmetric")
    summary = {"validation_only": True, "not_a_loading_solution": True,
        "model_digest": model.model_digest, "static_digest": model.static_digest,
        "cloud_sha256": config.data.cloud_sha256, "material_nodes": model.n_material,
        "union_nodes": model.n_union, "mechanics_edges": model.static.n_edges,
        "damage_edges": len(model.geometry.material_damage_edges),
        "delta_u": model.geometry.delta, "delta_d": model.geometry.damage_delta,
        "grip_extent": model.geometry.grip_extent, "minimum_BA_rank": int(min(model.static.shape_rank)),
        "maximum_BA_condition": float(max(model.static.shape_condition)),
        "affine_error": affine_error, "gradient_relative_defect": gradient_defect,
        "independent_directional_defect": directional_defect,
        "scalar_constant_inf": constant, "scalar_pairing_skew_inf": skew}
    atomic_json(output / "validation.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    zero = np.zeros(model.n_material)
    def solve(d, load):
        return backend.solve_strict(d, load, config.validation.cg_rtol,
                                   config.validation.cg_atol, config.validation.cg_maxiter)
    load = config.validation.load
    center = solve(zero, load)
    minus, plus = solve(zero, load - config.validation.work_step), solve(zero, load + config.validation.work_step)
    work = float(plus.energy - minus.energy) / (2 * config.validation.work_step)
    reaction = float(center.top_reaction)
    work_defect = abs(work - reaction) / abs(reaction)
    if not np.isfinite(work_defect) or work_defect > config.validation.work_relative_limit:
        raise ValueError(f"split reaction/work consistency failed: {work_defect}")
    summary.update({"intact_reaction": reaction, "reaction_work_relative_defect": work_defect})
    atomic_json(output / "validation.json", summary)
    np.savez_compressed(output / "intact_fields.npz", q=np.asarray(center.displacement), d=zero,
                        coordinates=np.asarray(model.geometry.coordinates))
    probes = []
    coords = np.asarray(model.geometry.coordinates[:model.n_material])
    for width in config.validation.widths:
        d = np.asarray(abs(coords[:, 1] - 0.5) <= width / 2, np.float64)
        result = solve(d, load)
        probe = {"width": width, "width_over_horizon": width / model.geometry.delta,
                 "damaged_nodes": int(np.count_nonzero(d)), "reaction": float(result.top_reaction),
                 "reaction_over_intact": float(result.top_reaction) / reaction,
                 "explicit_converged": bool(result.explicit_converged),
                 "cg_relative_residual": float(result.free_residual_relative_norm),
                 "prescribed_damage_probe": True, "not_AT2_solution": True}
        probes.append(probe)
        atomic_json(output / "strip_transmission.json", probes)
        np.savez_compressed(output / f"strip_{len(probes):02d}.npz", q=np.asarray(result.displacement), d=d)
        print(json.dumps(probe), flush=True)
    summary["operator_checks_passed"] = True
    summary["transmission_probe_completed"] = True
    atomic_json(output / "validation.json", summary)
    return summary


def main(_):
    run(flags.FLAGS.config)


if __name__ == "__main__":
    app.run(main)
