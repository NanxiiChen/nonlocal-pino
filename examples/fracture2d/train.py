"""Full-graph single-load nonlocal fracture PINN experiment.

Uses the previous archived state *only* for the committed history, and uses
the current reference state for held-out diagnostics, never as a target loss.
No curve convergence or strict post-peak claim is implied by a finite loss.
"""

import json
import os
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
# The stiff nonlocal residual amplifies matmul rounding; TF32 relaxed
# precision made eager and jitted evaluations differ at the 1e-4 level.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import optax
from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

from pinn.arch import make_pinn

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.fp32 import FP32A2Backend, build_fp32_model
from examples.fracture2d.losses import FractureResidual


_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "train_base.py")
_CONFIG = _config_flags.DEFINE_config_file("config", _DEFAULT, "Config file")
FLAGS = _absl_flags.FLAGS


def _check_file(path, expected_hash):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if direct._sha256(path) != expected_hash:
        raise ValueError(f"SHA256 mismatch for {path}")
    return path


def _build_grid(config):
    cloud_path = _check_file(config.data.cloud, config.data.cloud_sha256)
    cloud = direct._load_cloud(cloud_path)
    geometry = direct.build_union_geometry_from_arrays(
        cloud["coordinates"], cloud["triangles"], cloud["volumes"],
        cloud["bottom"], cloud["top"], int(cloud["anchor_particle"]),
        cloud["d048"], config.nl.delta, config.nl.h_g,
        config.nl.visibility_tolerance,
    )
    model = direct.build_engineering_grip_model(
        geometry, config.physical.E, config.physical.nu,
        config.physical.k_res, config.physical.rank_rtol,
        config.physical.Gc, config.physical.l0,
    )
    return model


def _nonnegative_damage(raw, floor):
    """Smooth one-sided floor for the phase-field damage output.

    ``0.5 * (raw + sqrt(raw^2 + floor^2))`` equals ``raw`` well above the
    floor, decays to zero from above well below it, and keeps a finite
    gradient at zero, so the damage stays strictly nonnegative (the direct
    bulk value is exactly zero) without freezing the optimizer at the
    boundary.
    """
    return 0.5 * (raw + jnp.sqrt(raw * raw + floor * floor))


def _zero_field_output(net):
    """Zero the final field projection so raw outputs start at zero."""
    if getattr(net, "out_heads", None) is not None:
        heads = tuple(
            eqx.tree_at(
                lambda head: (head.weight, head.bias), head,
                (jnp.zeros_like(head.weight), jnp.zeros_like(head.bias)),
            )
            for head in net.out_heads
        )
        return eqx.tree_at(lambda n: n.out_heads, net, heads)
    if getattr(net, "trunk_output", None) is not None:
        out = net.trunk_output
        return eqx.tree_at(
            lambda n: (n.trunk_output.weight, n.trunk_output.bias), net,
            (jnp.zeros_like(out.weight), jnp.zeros_like(out.bias)),
        )
    if getattr(net, "output", None) is None:
        raise ValueError("the configured network does not expose a field output layer")
    out = net.output
    return eqx.tree_at(
        lambda n: (n.output.weight, n.output.bias), net,
        (jnp.zeros_like(out.weight), jnp.zeros_like(out.bias)),
    )


def _build_network(config, key, out_dim=4):
    """Build one ``pinn.arch`` network per field group with a zeroed output.

    ``train.py`` requests all four fields from one network; ``stagger.py``
    requests ``out_dim=2`` for displacement and ``out_dim=1`` for damage.
    """
    arch = config.arch
    if out_dim not in (1, 2, 4):
        raise ValueError("fracture2d network output dimension must be 1, 2, or 4")
    net = make_pinn(
        spatial_dim=2,
        out_dim=out_dim,
        trunk_type=str(arch.type),
        hidden_dim=int(arch.hidden_dim),
        num_layers=int(arch.num_layers),
        activation=str(arch.activation),
        use_spatial_emb=bool(arch.use_spatial_emb),
        use_temporal_emb=False,
        static=True,
        spatial_emb_dim=int(arch.spatial_emb_dim),
        spatial_scale=arch.spatial_scale,
        tensor_split=bool(arch.tensor_split),
        multi_head=bool(arch.multi_head),
        film_hidden_dim=int(arch.film_hidden_dim),
        film_num_layers=int(arch.film_num_layers),
        key=key,
    )
    net = jax.tree.map(lambda array: array.astype(jnp.float32), net)
    return _zero_field_output(net)


def _validate_config(config):
    if config.train.dtype not in ("float32", "float64"):
        raise ValueError("train.dtype must be 'float32' or 'float64'")
    damage_floor = config.physical.damage_floor
    if not np.isfinite(damage_floor) or not (0.0 < float(damage_floor) < 1.0):
        raise ValueError("physical.damage_floor must lie strictly between 0 and 1")
    if int(config.data.state_index) != config.data.state_index or config.data.state_index < 1:
        raise ValueError("data.state_index must be an integer >= 1")
    if int(config.train.steps) != config.train.steps or config.train.steps < 1:
        raise ValueError("train.steps must be a positive integer")
    if int(config.train.report_every) != config.train.report_every or config.train.report_every < 1:
        raise ValueError("train.report_every must be a positive integer")
    if int(config.train.seed) != config.train.seed or config.train.seed < 0:
        raise ValueError("train.seed must be a nonnegative integer")
    for name in ("learning_rate",):
        value = config.train[name]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"train.{name} must be positive and finite")
    for name in ("mechanics_scale", "damage_scale", "history_scale"):
        value = config.loss[name]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{name} must be positive and finite")
    if config.loss.scaling not in ("fixed", "initial_driver_rms"):
        raise ValueError("loss.scaling must be 'fixed' or 'initial_driver_rms'")
    if config.arch.type not in ("mlp", "modified_mlp"):
        raise ValueError("arch.type must be 'mlp' or 'modified_mlp' for the static PINN")
    if config.arch.activation not in ("tanh", "relu", "gelu", "silu", "swish"):
        raise ValueError(f"unknown arch.activation {config.arch.activation!r}")
    for name in ("hidden_dim", "num_layers", "spatial_emb_dim",
                 "film_hidden_dim", "film_num_layers"):
        value = config.arch[name]
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"arch.{name} must be a positive integer")
    for name in ("use_spatial_emb", "multi_head", "tensor_split"):
        if not isinstance(config.arch[name], (bool, np.bool_)):
            raise ValueError(f"arch.{name} must be boolean")
    scales = config.arch.spatial_scale
    scales = scales if isinstance(scales, (tuple, list)) else (scales,)
    if not len(scales) or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("arch.spatial_scale must be positive and finite")
    field = {
        "delta": config.nl.delta, "h_g": config.nl.h_g,
        "visibility_tolerance": config.nl.visibility_tolerance,
        "youngs_modulus": config.physical.E, "poisson_ratio": config.physical.nu,
        "k_res": config.physical.k_res, "rank_rtol": config.physical.rank_rtol,
        "Gc": config.physical.Gc, "l0": config.physical.l0,
    }
    for name, value in field.items():
        expected = direct.SCHEME[name]
        if float(value) != expected:
            raise ValueError(f"adopted {name} must equal {expected}")


def _reference(config, model, *, diagnostic_warning=False):
    if not isinstance(diagnostic_warning, bool):
        raise ValueError("diagnostic_warning must be boolean")
    path = _check_file(config.data.reference, config.data.reference_sha256)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata["model_digest"] != model.model_digest:
            raise ValueError("reference model digest differs from frozen model")
        if metadata["static_digest"] != model.static_digest:
            raise ValueError("reference static digest differs from frozen model")
        index = int(config.data.state_index)
        if index >= len(data["loads"]):
            raise ValueError("state index exceeds the reference load path")
        if not diagnostic_warning and (index > 59 or bool(data["warning"][index])):
            raise ValueError("single-state training requires a pre-warning strict reference state (1..59)")
        if diagnostic_warning and not 60 <= index <= 75:
            raise ValueError("warning diagnostics require an archived load in 60..75")
        if not np.array_equal(data["coordinates"], model.geometry.coordinates[:model.n_material]):
            raise ValueError("reference material coordinates differ from frozen cloud")
        load = np.float64(data["loads"][index])
        prev_H = np.asarray(data["history"][index - 1], dtype=np.float64)
        prev_d = np.asarray(data["damage"][index - 1], dtype=np.float64)
        q_ref = np.asarray(data["union_q"][index], dtype=np.float64)
        d_ref = np.asarray(data["damage"][index], dtype=np.float64)
        H_ref = np.asarray(data["history"][index], dtype=np.float64)
        reaction = float(data["top_reaction"][index])
        warning = bool(data["warning"][index])
    return load, prev_H, prev_d, q_ref, d_ref, H_ref, reaction, warning


def _warning_region(config, model):
    """Report every warning load and its drift count without dropping particles."""
    with np.load(config.data.reference, allow_pickle=False) as data:
        loads = np.asarray(data["loads"], dtype=np.float64)
        status = np.asarray(data["warning"], dtype=bool)
        q = np.asarray(data["union_q"], dtype=np.float64)
    if loads.shape != (76,) or status.shape != loads.shape or q.shape != (76, model.n_dofs):
        raise ValueError("reference warning-region arrays do not match the adopted scheme")
    if not np.array_equal(np.flatnonzero(status), [*range(60, 69), 71]):
        raise ValueError("reference warning states differ from the adopted per-state flags")
    return [
        {"load": float(loads[i]), "warning": bool(status[i]), "drift_count": int(np.count_nonzero(
            np.linalg.norm(q[i, :2*model.n_material].reshape(-1, 2), axis=1)
            / loads[i] > direct.DRIFT_ENVELOPE))}
        for i in range(60, 76)
    ]


def _relative_l2(actual, reference):
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1.0e-15))


def run(config):
    _validate_config(config)
    if config.train.dtype != "float32":
        raise ValueError(
            "train.py implements the rebuilt float32 single-network experiment; "
            "the float64 path is implemented by stagger.py"
        )
    reference_model = _build_grid(config)
    load, prev_H, prev_d, q_ref, d_ref, H_ref, reaction_ref, warning = _reference(config, reference_model)
    print(json.dumps({"reference_warning_region": _warning_region(config, reference_model),
                      "comparison": "diagnostic only; no warning state is a training target"}), flush=True)
    model = build_fp32_model(reference_model)
    print(json.dumps({"dtype": "float32", "graph": "independently rebuilt strict float32 horizon",
                      "candidate_added": model.candidate_added,
                      "candidate_removed": model.candidate_removed,
                      "material_added": model.material_added,
                      "material_removed": model.material_removed,
                      "max_ba_condition": model.max_condition}), flush=True)
    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if not devices:
        raise RuntimeError("fracture2d PINN training requires a visible JAX GPU")
    backend = FP32A2Backend(model, devices[0])
    residual = FractureResidual(model, backend)
    residual.validate_fields(
        q_ref.astype(np.float32), d_ref.astype(np.float32),
        H_ref.astype(np.float32), prev_H.astype(np.float32), np.float32(load)
    )
    points = jax.device_put(model.geometry.coordinates, devices[0])
    previous = jax.device_put(prev_H.astype(np.float32), devices[0])
    load_jax = jax.device_put(np.float32(load), devices[0])
    net = _build_network(config, jax.random.PRNGKey(int(config.train.seed)))
    if any(leaf.dtype != np.float32 for leaf in jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array))
           if leaf is not None):
        raise ValueError("the PINN network must contain only float32 trainable arrays")
    optimizer = optax.adam(float(config.train.learning_rate))
    opt_state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))

    def fields(network):
        raw = jax.vmap(network)(points)
        # Uniform affine loading is only a network initialization/lifting; the
        # same free A2 DOFs are optimized and all PDE residuals remain unchanged.
        base_q = jnp.stack((jnp.zeros_like(points[:, 1]), load_jax * points[:, 1]), axis=1)
        q = residual.lift((base_q + load_jax * raw[:, :2]).reshape(-1), load_jax)
        history_scale = config.physical.E * load_jax**2
        damage_scale = history_scale / (config.physical.Gc / config.physical.l0)
        damage = _nonnegative_damage(
            damage_scale * raw[:model.n_material, 2], config.physical.damage_floor
        )
        return (q, damage, history_scale * raw[:model.n_material, 3])

    q_check, d_check, H_check = fields(net)
    r_check = residual.evaluate(q_check, d_check, H_check, previous)
    for name, array in (
        ("q", q_check), ("damage", d_check), ("history", H_check),
        ("mechanics residual", r_check.mechanics),
        ("damage residual", r_check.damage),
        ("Amor residual", r_check.history), ("reaction", r_check.reaction),
    ):
        if array.dtype != jnp.float32:
            raise ValueError(f"PINN {name} must remain float32, got {array.dtype}")

    if config.loss.scaling == "initial_driver_rms":
        mechanics_scale = float(jnp.sqrt(jnp.mean(r_check.mechanics**2)))
        driver_scale = float(jnp.sqrt(jnp.mean(r_check.driver**2)))
        loss_scales = (mechanics_scale, 2.0 * driver_scale, driver_scale)
    else:
        loss_scales = (float(config.loss.mechanics_scale),
                       float(config.loss.damage_scale),
                       float(config.loss.history_scale))
    if not np.all(np.isfinite(loss_scales)) or np.any(np.asarray(loss_scales) <= 0.0):
        raise ValueError(f"loss scales must be finite and strictly positive: {loss_scales}")
    mechanics_scale, damage_scale, history_scale = (
        jax.device_put(np.float32(scale), devices[0]) for scale in loss_scales
    )
    print(json.dumps({"loss_scaling": str(config.loss.scaling),
                      "trunk_type": str(config.arch.type),
                      "use_spatial_emb": bool(config.arch.use_spatial_emb),
                      "mechanics_scale": loss_scales[0],
                      "damage_scale": loss_scales[1],
                      "history_scale": loss_scales[2]}), flush=True)

    def component_losses(network):
        q, d, H = fields(network)
        r = residual.evaluate(q, d, H, previous)
        return (jnp.mean((r.mechanics / mechanics_scale) ** 2),
                jnp.mean((r.damage / damage_scale) ** 2),
                jnp.mean((r.history / history_scale) ** 2))

    def objective(network):
        mechanics, damage, history = component_losses(network)
        return mechanics + damage + history

    @eqx.filter_jit
    def step(network, state):
        value, grads = eqx.filter_value_and_grad(objective)(network)
        updates, state = optimizer.update(grads, state, network)
        return eqx.apply_updates(network, updates), state, value

    report = None
    for iteration in range(1, int(config.train.steps) + 1):
        net, opt_state, value = step(net, opt_state)
        if iteration % int(config.train.report_every) == 0 or iteration == config.train.steps:
            q, d, H = fields(net)
            r = residual.evaluate(q, d, H, previous)
            components = component_losses(net)
            q_np, d_np, H_np = map(np.asarray, (q, d, H))
            drift = int(np.count_nonzero(
                np.linalg.norm(q_np.reshape(-1, 2)[:model.n_material], axis=1) / float(load) > direct.DRIFT_ENVELOPE
            ))
            report = {
                "iteration": iteration, "loss_before_update": float(value), "load": float(load),
                "loss_after_update": float(sum(components)),
                "loss_mechanics": float(components[0]),
                "loss_damage": float(components[1]),
                "loss_history": float(components[2]),
                "loss_scaling": str(config.loss.scaling),
                "reference_warning": warning, "comparison_region": "warning" if warning else "strict",
                "reference_field_equations_validated": False,
                "mechanics_inf": float(jnp.max(jnp.abs(r.mechanics))),
                "damage_inf": float(jnp.max(jnp.abs(r.damage))),
                "history_inf": float(jnp.max(jnp.abs(r.history))),
                "u_relative_l2": _relative_l2(q_np[:2*model.n_material], q_ref[:2*model.n_material]),
                "d_relative_l2": _relative_l2(d_np, d_ref),
                "H_relative_l2": _relative_l2(H_np, H_ref),
                "reaction": float(r.reaction), "reference_reaction": reaction_ref,
                "drift_count": drift, "damage_min": float(np.min(d_np)),
                "reference_drift_count": int(np.count_nonzero(
                    np.linalg.norm(q_ref[:2*model.n_material].reshape(-1, 2), axis=1)
                    / float(load) > direct.DRIFT_ENVELOPE)),
                "damage_max": float(np.max(d_np)),
                "history_decrease_max": float(np.max(np.maximum(prev_H - H_np, 0.0))),
                "damage_healing_max": float(np.max(np.maximum(prev_d - d_np, 0.0))),
            }
            if not np.isfinite([entry for entry in report.values() if isinstance(entry, (float, int))]).all():
                raise FloatingPointError("training diagnostics are non-finite")
            print(json.dumps(report, sort_keys=True), flush=True)
    assert report is not None
    return report


def main(_):
    run(FLAGS.config)


if __name__ == "__main__":
    _absl_app.run(main)
