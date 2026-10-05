"""Two-equation stagger on one strict load or an explicit warning trial.

Mechanics and damage use independent Equinox networks; choose
``train.dtype = "float64"`` for the adopted reference graph or the base
config's ``float32`` for the separately rebuilt graph. The material Amor
history is computed exactly from displacement, not fitted by a third output.
Each completed sweep is
evaluated against explicit reference-comparison gates and reported; advancing
an accepted state to the next load is implemented by separate continuation
entry points. Warning-region comparisons never produce strict acceptance.
"""

import hashlib
import json
import os
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
# Matches `train.py`: full-precision float32 matmuls keep eager and jitted
# evaluations of the stiff residual consistent.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import optax
from absl import app as _absl_app, flags as _absl_flags
from tensorboardX import SummaryWriter

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.fp32 import FP32A2Backend, build_fp32_model
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.pinn_paths import physical_pinn_path
from examples.fracture2d.stagger_checkpoint import (
    SCHEMA, SCHEMA_FP64, SCHEMA_ACCEPTED, SCHEMA_WARNING_PROGRESS,
    SCHEMA_DIAGNOSTIC, read_checkpoint, write_checkpoint,
)
from examples.fracture2d.train import (
    _build_grid, _build_network, _nonnegative_damage, _reference, _relative_l2,
    _validate_config, _warning_region,
)


def _validate_stagger(config):
    _validate_config(config)
    if "stagger_lr_policy" in config.train:
        if config.train.stagger_lr_policy not in ("constant", "linear_tail"):
            raise ValueError("stagger_lr_policy must be constant or linear_tail")
        if config.train.stagger_lr_policy == "linear_tail":
            for name in ("stagger_tail_fraction", "stagger_tail_ratio"):
                value = config.train[name]
                if not np.isfinite(value) or not 0 < value <= 1:
                    raise ValueError(f"train.{name} must be in (0, 1]")
    if "coupling_omega" in config.stagger:
        value = config.stagger.coupling_omega
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("explicit numerical coupling omega must lie in (0, 1]")
    for name in ("sweeps", "mechanics_steps", "damage_steps"):
        value = config.stagger[name]
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"stagger.{name} must be a positive integer")
    if not np.isfinite(config.stagger.omega) or float(config.stagger.omega) != direct.SCHEME["omega"]:
        raise ValueError("stagger.omega must equal the adopted direct Picard relaxation")
    for name in ("mechanics_loss_limit", "displacement_l2_limit", "reaction_relative_limit"):
        value = config.stagger[name]
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"stagger.{name} must be positive and finite")
    if not isinstance(config.stagger.gate_displacement_l2, (bool, np.bool_)):
        raise ValueError("stagger.gate_displacement_l2 must be boolean")
    if (not np.isfinite(config.stagger.damage_rmse_limit)
            or float(config.stagger.damage_rmse_limit) < 0.0):
        raise ValueError("stagger.damage_rmse_limit must be nonnegative and finite")
    if not np.isfinite(config.stagger.damage_range_atol) or float(config.stagger.damage_range_atol) < 0.0:
        raise ValueError("stagger.damage_range_atol must be nonnegative and finite")
    if (not np.isfinite(config.stagger.healing_atol)
            or not 0.0 <= float(config.stagger.healing_atol) < 1.0):
        raise ValueError("stagger.healing_atol must lie in [0, 1)")
    if config.stagger.mechanics_scale_policy not in (
        "first_load_initial", "first_load_times_load_ratio", "per_load_warm_start"
    ):
        raise ValueError("stagger.mechanics_scale_policy must select an explicit scale")
    if not isinstance(config.stagger.accept_on_first_passing_report, (bool, np.bool_)):
        raise ValueError("stagger.accept_on_first_passing_report must be boolean")
    for name in ("log_every", "checkpoint_every"):
        value = config.output[name]
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"output.{name} must be a positive integer")
    if not isinstance(config.output.resume, (bool, np.bool_)):
        raise ValueError("output.resume must be boolean")
    if not isinstance(config.output.dir, str) or not config.output.dir:
        raise ValueError("output.dir must be a nonempty explicit path")
    if not isinstance(config.output.tensorboard_root, str) or not config.output.tensorboard_root:
        raise ValueError("output.tensorboard_root must be a nonempty explicit path")
    run_dir = Path(config.output.dir)
    tensorboard_root = Path(config.output.tensorboard_root)
    if tensorboard_root == run_dir or tensorboard_root in run_dir.parents:
        raise ValueError("TensorBoard root must be separate from the data run directory")


def _acceptance_gates(config, result):
    """Evaluate each independent completed-sweep acceptance criterion."""
    range_atol = float(config.stagger.damage_range_atol)
    gates = {
        "finite": all(
            np.isfinite(value) for value in result.values()
            if isinstance(value, (float, int))
        ),
        "mechanics_loss": result["mechanics_loss"] <= float(config.stagger.mechanics_loss_limit),
        "reaction": result["reaction_relative"] <= float(config.stagger.reaction_relative_limit),
        "damage_range": bool(result["damage_min"] >= -range_atol
                             and result["damage_max"] <= 1.0 + range_atol),
        "healing": result["damage_healing_max"] <= float(config.stagger.healing_atol),
        "history_monotonic": result["history_decrease_max"] <= 0.0,
        "drift": result["drift_count"] == 0,
    }
    if config.stagger.gate_displacement_l2:
        gates["displacement_l2"] = (
            result["u_relative_l2"] <= float(config.stagger.displacement_l2_limit)
        )
    if config.stagger.damage_rmse_limit > 0.0:
        gates["damage_rmse"] = result["d_rmse"] <= float(config.stagger.damage_rmse_limit)
    return gates


def _warning_advancement_gates(config, result):
    """Only existing physics/bounds gates, not warning-reference comparators."""
    gates = _acceptance_gates(config, result)
    for name in ("reaction", "damage_rmse", "displacement_l2"):
        gates.pop(name, None)
    return gates


def _load_ratio_scale(first_scale, load, first_load):
    """Scale a first-load force RMS by the prescribed displacement amplitude."""
    if (not all(np.isfinite(x) for x in (first_scale, load, first_load))
            or first_scale <= 0.0 or first_load <= 0.0 or load < first_load):
        raise ValueError("load-ratio mechanics scale requires finite increasing positive loads")
    scale = float(first_scale) * (float(load) / float(first_load))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("load-ratio mechanics scale must be finite and positive")
    return scale


def _damage_error_metrics(actual, reference):
    """Dimensionless absolute damage errors; defined even at zero damage."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (actual.ndim != 1 or actual.shape != reference.shape or not actual.size
            or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference))):
        raise ValueError("damage comparison requires equal nonempty finite material fields")
    difference = actual - reference
    return {
        "d_rmse": float(np.sqrt(np.mean(difference**2))),
        "d_max_abs_error": float(np.max(np.abs(difference))),
    }


def _inner_update_ratio(config, step, count):
    """Optional numerical taper; legacy configs retain their constant update."""
    if "stagger_lr_policy" not in config.train or config.train.stagger_lr_policy == "constant":
        return 1.
    if not 1 <= step <= count:
        raise ValueError("inner update index must lie inside its declared block")
    first_tail = int(np.floor((1 - config.train.stagger_tail_fraction) * count))
    if step <= first_tail:
        return 1.
    fraction = (step - first_tail) / (count - first_tail)
    return 1 - (1 - config.train.stagger_tail_ratio) * fraction


def run(config, *, resume=False, initial_state=None, parent_sha256=None,
         return_state=False, warning_diagnostic=False, parent_schema=None,
         sweep_observer=None, trial_initial_state=None, after_stop=None):
    _validate_stagger(config)
    if after_stop is not None and ("coupling" not in config or not config.coupling.after_stop):
        raise ValueError("coupled mechanics-after stop requires explicit numerical policy")
    if not isinstance(warning_diagnostic, bool):
        raise ValueError("warning_diagnostic must be boolean")
    if warning_diagnostic and parent_schema not in (SCHEMA_ACCEPTED, SCHEMA_DIAGNOSTIC):
        raise ValueError("warning diagnostic parent schema must identify an accepted or trial snapshot")
    if not warning_diagnostic and parent_schema is not None:
        raise ValueError("strict stagger cannot have a diagnostic parent schema")
    run_dir = physical_pinn_path(config.output.dir)
    tensorboard_root = Path(config.output.tensorboard_root)
    tensorboard_dir = tensorboard_root / run_dir.name
    if not tensorboard_root.parent.is_dir():
        raise FileNotFoundError(f"TensorBoard root parent does not exist: {tensorboard_root.parent}")
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"cannot resume a missing run directory: {run_dir}")
        if not tensorboard_dir.is_dir():
            raise FileNotFoundError(f"cannot resume without TensorBoard directory: {tensorboard_dir}")
    else:
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite an existing run: {run_dir}")
        if tensorboard_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing TensorBoard log: {tensorboard_dir}")
        if not run_dir.parent.is_dir():
            raise FileNotFoundError(f"run parent directory does not exist: {run_dir.parent}")
    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if not devices:
        raise RuntimeError("fracture2d stagger experiment requires a visible JAX GPU")
    device = devices[0]
    precision = str(config.train.dtype)
    if precision not in ("float32", "float64"):
        raise ValueError("train.dtype must be 'float32' or 'float64'")
    dtype = np.float64 if precision == "float64" else np.float32
    if warning_diagnostic and precision != "float64":
        raise ValueError("warning diagnostics require the adopted float64 graph")
    checkpoint_schema = (SCHEMA_WARNING_PROGRESS if warning_diagnostic else
                         SCHEMA_FP64 if precision == "float64" else SCHEMA)
    reference_model = _build_grid(config)
    load, H_committed, d_committed, q_ref, d_ref, H_ref, reaction_ref, warning = (
        _reference(config, reference_model, diagnostic_warning=warning_diagnostic)
    )
    if warning and not warning_diagnostic:
        raise ValueError("the stagger experiment requires a strict reference prefix state")
    index = int(config.data.state_index)
    if warning_diagnostic and index == 60 and parent_schema != SCHEMA_ACCEPTED:
        raise ValueError("first warning trial requires an accepted state-59 parent")
    if warning_diagnostic and index > 60 and parent_schema != SCHEMA_DIAGNOSTIC:
        raise ValueError("later warning trials require a diagnostic predecessor")
    if index == 1 and initial_state is not None:
        raise ValueError("state 1 must start from the archived zero-load state")
    if index == 1 and parent_sha256 is not None:
        raise ValueError("state 1 cannot have an accepted parent digest")
    if index > 1 and (initial_state is None or parent_sha256 is None):
        raise ValueError("subsequent loads require an accepted parent state and digest")
    if index > 1 and (not isinstance(parent_sha256, str) or len(parent_sha256) != 64
                      or any(c not in "0123456789abcdef" for c in parent_sha256)):
        raise ValueError("accepted parent digest must be a lowercase SHA256")
    if initial_state is not None:
        if len(initial_state) != 7:
            raise ValueError("accepted parent must contain networks, optimizer states, q, d, H")
        parent_q, parent_d, parent_H = map(np.asarray, initial_state[4:])
        if (parent_q.shape != (reference_model.n_dofs,)
                or parent_d.shape != (reference_model.n_material,)
                or parent_H.shape != (reference_model.n_material,)):
            raise ValueError("accepted parent field shapes do not match the model")
        if any(x.dtype != dtype for x in (parent_q, parent_d, parent_H)):
            raise ValueError("accepted parent fields must match the stagger precision")
        if not all(np.all(np.isfinite(x)) for x in (parent_q, parent_d, parent_H)):
            raise ValueError("accepted parent fields must be finite")
        with np.load(config.data.reference, allow_pickle=False) as archive:
            previous_load = float(archive["loads"][index - 1])
        constrained = reference_model.constraints.constrained_dofs
        if not np.array_equal(
            parent_q[constrained],
            reference_model.constraints.prescribed_vector(previous_load)[constrained],
        ):
            raise ValueError("accepted parent does not satisfy its previous-load constraints")
        if np.min(parent_d) < -float(config.stagger.damage_range_atol) or np.max(parent_d) > 1.0:
            raise ValueError("accepted parent damage is outside physical bounds")
        if np.min(parent_H) < 0.0:
            raise ValueError("accepted parent Amor history must be nonnegative")
    print(json.dumps({"reference_warning_region": _warning_region(config, reference_model),
                      "comparison": "diagnostic only; no warning state is a training target"}), flush=True)
    if precision == "float64":
        model = reference_model
        backend = FP64A2Backend(model.static, model.constraints, device)
        print(json.dumps({"dtype": precision,
                          "graph": "adopted reference union horizon"}), flush=True)
    else:
        model = build_fp32_model(reference_model)
        print(json.dumps({"dtype": precision, "graph": "independently rebuilt strict float32 horizon",
                          "candidate_added": model.candidate_added,
                          "candidate_removed": model.candidate_removed,
                          "material_added": model.material_added,
                          "material_removed": model.material_removed,
                          "max_ba_condition": model.max_condition}), flush=True)
        backend = FP32A2Backend(model, device)
    residual = FractureResidual(model, backend)
    points = jax.device_put(model.geometry.coordinates.astype(dtype), device)
    committed = jax.device_put(H_committed.astype(dtype), device)
    damage_iter = jax.device_put(d_committed.astype(dtype), device)
    load_jax = jax.device_put(np.asarray(load, dtype), device)
    q_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed)), out_dim=2)
    d_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed) + 1), out_dim=1)
    if precision == "float64":
        q_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), q_net)
        d_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), d_net)
    optim = optax.adam(float(config.train.learning_rate))
    q_state = optim.init(eqx.filter(q_net, eqx.is_inexact_array))
    d_state = optim.init(eqx.filter(d_net, eqx.is_inexact_array))
    if initial_state is not None:
        q_net, q_state, d_net, d_state = initial_state[:4]
        committed = jax.device_put(parent_H.astype(dtype), device)
        damage_iter = jax.device_put(parent_d.astype(dtype), device)
        d_committed = parent_d
        H_committed = parent_H
    if trial_initial_state is not None:
        if not warning_diagnostic or not 60 <= index <= 75 or precision != "float64" or initial_state is None:
            raise ValueError("separate numerical trial seeds require old FP64 warning state 60..75")
        if (len(trial_initial_state) != 7 or "trial_seed_sha256" not in config.output
                or "trial_seed_path" not in config.output):
            raise ValueError("numerical trial seed must have explicit path/digest provenance")
        seed_path = physical_pinn_path(config.output.trial_seed_path)
        if hashlib.sha256(seed_path.read_bytes()).hexdigest() != config.output.trial_seed_sha256:
            raise ValueError("numerical trial seed digest differs")
        seed_q, seed_d, seed_H = map(np.asarray, trial_initial_state[4:])
        residual.validate_fields(seed_q, seed_d, seed_H, parent_H, np.asarray(load, dtype))
        if (min(seed_d) < -config.stagger.damage_range_atol or max(seed_d) > 1 or min(seed_H) < 0
                or max(np.maximum(parent_d - seed_d, 0)) > config.stagger.healing_atol
                or min(seed_H - parent_H) < 0):
            raise ValueError("numerical trial seed has invalid damage/history range")
        q_net, q_state, d_net, d_state = trial_initial_state[:4]
        damage_iter = jax.device_put(seed_d.astype(dtype), device)
        print(json.dumps({"event": "numerical_seed_restored_without_commit",
                           ("accepted_parent_sha256" if index == 60 else "diagnostic_parent_sha256"): parent_sha256,
                          "trial_seed_sha256": config.output.trial_seed_sha256,
                          "history_floor_inf": float(np.max(parent_H)),
                          "ignored_trial_history_inf": float(np.max(seed_H))}), flush=True)
        # Keep d_committed/H_committed and `committed` from the accepted parent;
        # seed_H is an uncommitted trial field, never a history lower bound.

    def displacement(network):
        raw = jax.vmap(network)(points)
        base = jnp.stack((jnp.zeros_like(points[:, 1]), load_jax * points[:, 1]), axis=1)
        return residual.lift((base + load_jax * raw).reshape(-1), load_jax)

    def damage(network):
        scale = (config.physical.E / (config.physical.Gc / config.physical.l0)) * load_jax**2
        raw = scale * jax.vmap(network)(points[:model.n_material])[:, 0]
        return _nonnegative_damage(raw, float(config.physical.damage_floor))

    if config.stagger.mechanics_scale_policy in (
        "first_load_initial", "first_load_times_load_ratio"
    ):
        with np.load(config.data.reference, allow_pickle=False) as archive:
            first_load = np.asarray(archive["loads"][1], dtype=dtype)
            zero_damage = np.asarray(archive["damage"][0], dtype=dtype)
        if (not np.isfinite(first_load) or first_load <= 0.0
                or zero_damage.shape != (model.n_material,)
                or not np.array_equal(zero_damage, np.zeros_like(zero_damage))):
            raise ValueError("first-load normalization requires the archived zero-damage initial state")
        first_load_jax = jax.device_put(first_load, device)
        affine = jnp.stack(
            (jnp.zeros_like(points[:, 1]), first_load_jax * points[:, 1]), axis=1
        )
        initial_q = residual.lift(affine.reshape(-1), first_load_jax)
        factor_initial = backend._edge_factor(jax.device_put(zero_damage, device))
    else:
        initial_q = displacement(q_net)
        factor_initial = backend._edge_factor(damage_iter)
    initial = backend._residual_from_factor(factor_initial, initial_q)[backend.free_dofs]
    mechanics_scale_value = float(jnp.sqrt(jnp.mean(initial**2)))
    if not np.isfinite(mechanics_scale_value) or mechanics_scale_value <= 0.0:
        raise ValueError("initial free-row mechanics RMS must be finite and positive")
    if config.stagger.mechanics_scale_policy == "first_load_times_load_ratio":
        mechanics_scale_value = _load_ratio_scale(
            mechanics_scale_value, float(load), float(first_load)
        )
    mechanics_scale = jax.device_put(np.asarray(mechanics_scale_value, dtype), device)

    def _numerical_config():
        """Canonical config identity; the control-only resume flag is excluded."""
        values = json.loads(config.to_json_best_effort())
        values["output"]["resume"] = False
        return json.dumps(values, sort_keys=True)

    identity = {
        "config_sha256": hashlib.sha256(_numerical_config().encode()).hexdigest(),
        "cloud_sha256": str(config.data.cloud_sha256),
        "reference_sha256": str(config.data.reference_sha256),
        "direct_model_digest": reference_model.model_digest,
        "precision": precision,
    }
    if parent_sha256 is not None:
        parent_key = ("parent_diagnostic_sha256" if warning_diagnostic and index > 60
                      else "parent_accepted_sha256")
        identity[parent_key] = parent_sha256
    if warning_diagnostic:
        identity["warning_diagnostic"] = True
        identity["parent_schema"] = parent_schema
    if precision == "float32":
        graph_hash = hashlib.sha256()
        for array in (
            model.geometry.coordinates, model.static.source, model.static.target,
            model.static.eta, model.material_volumes, model.grip_volumes,
        ):
            values = np.ascontiguousarray(array)
            graph_hash.update(str(values.shape).encode("ascii"))
            graph_hash.update(values.dtype.str.encode("ascii"))
            graph_hash.update(values.tobytes())
        identity["fp32_graph_sha256"] = graph_hash.hexdigest()
    checkpoint_path = run_dir / "latest.ckpt.zip"
    if not resume:
        tensorboard_root.mkdir(exist_ok=True)
        run_dir.mkdir()
        tensorboard_dir.mkdir()
        (run_dir / "config.json").write_text(_numerical_config(), encoding="utf-8")
    else:
        stored_config = (run_dir / "config.json").read_text(encoding="utf-8")
        if stored_config != _numerical_config():
            raise ValueError("resume configuration differs from stored run configuration")

    @eqx.filter_jit
    def mechanics_step(network, state, fixed_damage, update_ratio):
        fixed_damage = jax.lax.stop_gradient(fixed_damage)
        def objective(n):
            q = displacement(n)
            factor = backend._edge_factor(fixed_damage)
            r = backend._residual_from_factor(factor, q)[backend.free_dofs]
            return jnp.mean((r / mechanics_scale)**2)
        value, grad = eqx.filter_value_and_grad(objective)(network)
        update, state = optim.update(grad, state, network)
        if "stagger_lr_policy" in config.train and config.train.stagger_lr_policy == "linear_tail":
            update = jax.tree.map(lambda value: update_ratio * value, update)
        return eqx.apply_updates(network, update), state, value

    @eqx.filter_jit
    def damage_step(network, state, fixed_history, scale, update_ratio):
        fixed_history = jax.lax.stop_gradient(fixed_history)
        def objective(n):
            d = damage(n)
            rho = (residual.gc_over_l0 * d
                   + residual.gc_l0 * residual.operator.deterministic_matvec(d)
                   + 2.0 * fixed_history * (d - 1.0))
            return jnp.mean((rho / scale)**2)
        value, grad = eqx.filter_value_and_grad(objective)(network)
        update, state = optim.update(grad, state, network)
        if "stagger_lr_policy" in config.train and config.train.stagger_lr_policy == "linear_tail":
            update = jax.tree.map(lambda value: update_ratio * value, update)
        return eqx.apply_updates(network, update), state, value

    def history(q, fixed_damage):
        fields = residual.evaluate(q, fixed_damage, None, committed)
        return fields.trial_history

    initial_history = history(displacement(q_net), damage_iter)
    coupled_damage_scale_value = float(2.0 * jnp.sqrt(jnp.mean(initial_history**2)))
    if not np.isfinite(coupled_damage_scale_value) or coupled_damage_scale_value <= 0.0:
        raise ValueError("initial full-graph coupled damage scale must be finite and positive")
    coupled_damage_scale = jax.device_put(np.asarray(coupled_damage_scale_value, dtype), device)

    def coupled_loss(fields):
        """Diagnostic normalized MSE for both current-field governing residuals."""
        return (jnp.mean((fields.mechanics / mechanics_scale)**2)
                + jnp.mean((fields.damage / coupled_damage_scale)**2))

    start_report = {"mode": "two_network_stagger_diagnostic", "dtype": precision,
                    "mechanics_scale_policy": str(config.stagger.mechanics_scale_policy),
                    "initial_mechanics_scale": mechanics_scale_value,
                    "initial_coupled_damage_scale": coupled_damage_scale_value}
    if precision == "float32":
        start_report.update({
            "candidate_added": model.candidate_added,
            "candidate_removed": model.candidate_removed,
            "material_added": model.material_added,
            "material_removed": model.material_removed,
        })
    print(json.dumps(start_report), flush=True)

    n_q = int(config.stagger.mechanics_steps)
    n_d = int(config.stagger.damage_steps)
    coupling_omega = (config.stagger.coupling_omega if "coupling_omega" in config.stagger
                      else config.stagger.omega)
    steps_per_sweep = 2 * n_q + n_d
    phases = ("before", "damage", "after")

    def global_step(sweep, phase, step):
        offsets = {"before": 0, "damage": n_q, "after": n_q + n_d,
                   "complete": steps_per_sweep}
        return (sweep - 1) * steps_per_sweep + offsets[phase] + step

    q_before = jnp.zeros(model.n_dofs, dtype=dtype)
    H_used = committed
    d_trial = damage_iter

    def state_tuple():
        return (q_net, q_state, d_net, d_state,
                damage_iter, q_before, H_used, d_trial)

    if resume:
        metadata, restored = read_checkpoint(
            checkpoint_path, identity, state_tuple(), schema=checkpoint_schema)
        q_net, q_state, d_net, d_state, damage_iter, q_before, H_used, d_trial = restored
        start_sweep, start_phase, start_step = (
            metadata["sweep"], metadata["phase"], metadata["step"]
        )
        if start_phase == "complete":
            raise ValueError("the requested stagger run is already complete")
        if start_sweep > config.stagger.sweeps:
            raise ValueError("checkpoint sweep exceeds the configured budget")
        if start_step > (n_d if start_phase == "damage" else n_q):
            raise ValueError("checkpoint step exceeds its stage budget")
    else:
        start_sweep, start_phase, start_step = 1, "before", 0

    writer = SummaryWriter(
        logdir=str(tensorboard_dir),
        purge_step=(global_step(start_sweep, start_phase, start_step) + 1 if resume else None),
    )
    print(json.dumps({"tensorboard_dir": str(tensorboard_dir),
                      "checkpoint": str(checkpoint_path)}), flush=True)
    if not resume:
        writer.add_scalar("diag/mechanics_scale", mechanics_scale_value, 0)
        writer.add_scalar("diag/coupled_damage_scale", coupled_damage_scale_value, 0)
        writer.add_scalar("eval/reaction_reference", reaction_ref, 0)
        writer.flush()

    def save(sweep, phase, step):
        metadata = write_checkpoint(
            checkpoint_path, identity, sweep, phase, step, state_tuple(),
            schema=checkpoint_schema)
        print(json.dumps({"checkpoint": str(checkpoint_path), "sweep": sweep,
                          "phase": phase, "step": step,
                          "state_sha256": metadata["state_sha256"]}, sort_keys=True), flush=True)

    def rho_at(d, H):
        return (residual.gc_over_l0 * d
                + residual.gc_l0 * residual.operator.deterministic_matvec(d)
                + 2.0 * H * (d - 1.0))

    phase_loss_tag = {"before": "mechanics_before", "damage": "damage",
                      "after": "mechanics_after"}

    def stage_report(sweep, phase, step, damage_scale=None):
        q = displacement(q_net)
        d = (damage(d_net) if phase == "damage"
             else damage_iter if phase == "before" else d_trial)
        # One Amor reduction supplies the trial history and every value logged
        # at this step, so the loss and all diagnostics refer to the same
        # post-update state.
        trial = residual.evaluate(q, d, None, committed)
        if phase == "damage":
            if damage_scale is None:
                raise ValueError("damage-stage reporting requires its fixed scale")
            stage_loss = jnp.mean((rho_at(d, H_used) / damage_scale)**2)
        else:
            stage_loss = jnp.mean((trial.mechanics / mechanics_scale)**2)
        q_np = np.asarray(q)
        d_np = np.asarray(d)
        H_np = np.asarray(trial.trial_history)
        position = global_step(sweep, phase, step)
        diagnostics = {
            "mechanics_residual_inf": float(jnp.max(jnp.abs(trial.mechanics))),
            "damage_residual_inf": float(jnp.max(jnp.abs(trial.damage))),
            "history_residual_inf": float(jnp.max(jnp.abs(trial.history))),
            "coupled_loss": float(coupled_loss(trial)),
            "damage_healing_max": float(np.max(np.maximum(d_committed - d_np, 0.0))),
            "drift_count": int(np.count_nonzero(
                np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)
                / float(load) > direct.DRIFT_ENVELOPE)),
            "damage_min": float(np.min(d_np)),
            "damage_max": float(np.max(d_np)),
        }
        comparisons = {
            "rel_L2_u": _relative_l2(q_np[:2*model.n_material], q_ref[:2*model.n_material]),
            "rel_L2_d": _relative_l2(d_np, d_ref),
            "rel_L2_H": _relative_l2(H_np, H_ref),
            "reaction": float(trial.reaction),
            **_damage_error_metrics(d_np, d_ref),
        }
        if not np.all(np.isfinite([float(stage_loss), *diagnostics.values(),
                                   *comparisons.values()])):
            raise FloatingPointError("stage report contains non-finite values")
        writer.add_scalar(f"loss/{phase_loss_tag[phase]}", float(stage_loss), position)
        for name, value in diagnostics.items():
            writer.add_scalar(f"diag/{name}", value, position)
        for name, value in comparisons.items():
            writer.add_scalar(f"eval/{name}", value, position)
        writer.flush()
        report = {
            "event": "stage_progress", "sweep": sweep, "phase": phase,
            "step": step, "global_step": position, "stage_loss": float(stage_loss),
            "mechanics_inf": diagnostics["mechanics_residual_inf"],
            "damage_inf": diagnostics["damage_residual_inf"],
            "reaction": comparisons["reaction"],
            "reference_reaction": reaction_ref,
            "u_relative_l2": comparisons["rel_L2_u"],
            "d_rmse": comparisons["d_rmse"],
            "d_max_abs_error": comparisons["d_max_abs_error"],
            "drift_count": diagnostics["drift_count"],
            "damage_min": diagnostics["damage_min"],
            "damage_max": diagnostics["damage_max"],
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        return q, trial

    def candidate_gates(q, fields):
        q_np = np.asarray(q)
        d_np = np.asarray(d_trial)
        H_np = np.asarray(fields.trial_history)
        candidate = {
            "mechanics_loss": float(jnp.mean((fields.mechanics / mechanics_scale)**2)),
            "u_relative_l2": _relative_l2(
                q_np[:2*model.n_material], q_ref[:2*model.n_material]
            ),
            "reaction_relative": abs(float(fields.reaction) - reaction_ref)
            / max(abs(reaction_ref), 1.0e-30),
            **_damage_error_metrics(d_np, d_ref),
            "damage_min": float(np.min(d_np)),
            "damage_max": float(np.max(d_np)),
            "damage_healing_max": float(np.max(np.maximum(d_committed - d_np, 0.0))),
            "history_decrease_max": float(np.max(np.maximum(H_committed - H_np, 0.0))),
            "drift_count": int(np.count_nonzero(
                np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)
                / float(load) > direct.DRIFT_ENVELOPE
            )),
        }
        return (_warning_advancement_gates(config, candidate) if warning_diagnostic
                else _acceptance_gates(config, candidate))

    result = None
    try:
        for sweep in range(start_sweep, int(config.stagger.sweeps) + 1):
            after_sample = None
            stop_step = n_q
            for phase in phases:
                if sweep == start_sweep and phases.index(phase) < phases.index(start_phase):
                    continue
                first_step = (start_step if sweep == start_sweep and phase == start_phase else 0)
                count = n_d if phase == "damage" else n_q
                H_scale = None
                if phase == "damage":
                    H_scale_value = float(2.0 * jnp.sqrt(jnp.mean(H_used**2)))
                    if not np.isfinite(H_scale_value) or H_scale_value <= 0.0:
                        raise ValueError("fixed-history AT2 driving scale must be finite and positive")
                    H_scale = jax.device_put(np.asarray(H_scale_value, dtype), device)
                if (resume and sweep == start_sweep and phase == start_phase == "after"
                        and first_step > 0 and sweep == config.stagger.sweeps
                        and config.stagger.accept_on_first_passing_report):
                    q_restored = displacement(q_net)
                    trial_restored = residual.evaluate(q_restored, d_trial, None, committed)
                    if all(candidate_gates(q_restored, trial_restored).values()):
                        after_sample = (q_restored, trial_restored)
                        stop_step = first_step
                steps = range(first_step + 1, count + 1) if after_sample is None else ()
                for step in steps:
                    update_ratio = jnp.asarray(_inner_update_ratio(config, step, count), dtype=dtype)
                    if phase == "damage":
                        if H_scale is None:
                            raise RuntimeError("damage stage has no fixed-history scale")
                        d_net, d_state, _ = damage_step(d_net, d_state, H_used, H_scale, update_ratio)
                    else:
                        fixed_d = damage_iter if phase == "before" else d_trial
                        q_net, q_state, _ = mechanics_step(q_net, q_state, fixed_d, update_ratio)
                    if step % config.output.log_every == 0 or step == count:
                        sample = stage_report(
                            sweep, phase, step,
                            H_scale if phase == "damage" else None,
                        )
                        if phase == "after" and step == count:
                            after_sample = sample
                        if phase == "after" and after_stop is not None and all(candidate_gates(*sample).values()):
                            candidate_q, candidate_fields = sample
                            payload = {"q": np.asarray(candidate_q), "d": np.asarray(d_trial),
                                       "H": np.asarray(candidate_fields.trial_history),
                                       "reaction": float(candidate_fields.reaction),
                                       "at2_local": np.asarray(residual.gc_over_l0 * d_trial),
                                       "at2_nonlocal": np.asarray(residual.gc_l0 * residual.operator.deterministic_matvec(d_trial)),
                                       "at2_history": np.asarray(2 * candidate_fields.trial_history * (d_trial - 1))}
                            if after_stop(sweep, payload):
                                after_sample, stop_step = sample, step
                                save(sweep, phase, step)
                                break
                        if (phase == "after" and sweep == config.stagger.sweeps
                                and config.stagger.accept_on_first_passing_report
                                and all(candidate_gates(*sample).values())):
                            after_sample = sample
                            stop_step = step
                            save(sweep, phase, step)
                            break
                    if step % config.output.checkpoint_every == 0 or step == count:
                        save(sweep, phase, step)

                if phase == "before":
                    q_before = displacement(q_net)
                    H_used = history(q_before, damage_iter)
                    save(sweep, "damage", 0)
                elif phase == "damage":
                    d_solved = damage(d_net)
                    d_trial = damage_iter + dtype(coupling_omega) * (d_solved - damage_iter)
                    save(sweep, "after", 0)

            if after_sample is None:
                # A resumed last step may already be in the checkpoint; in
                # that case evaluate the final candidate once after restoring.
                q_after = displacement(q_net)
                fields = residual.evaluate(q_after, d_trial, None, committed)
            else:
                q_after, fields = after_sample
            H_next = fields.trial_history
            mechanics_before = backend._residual_from_factor(
                backend._edge_factor(damage_iter), q_before
            )[backend.free_dofs]
            rho_solved = rho_at(damage(d_net), H_used)
            q_np, d_np, H_np = map(np.asarray, (q_after, d_trial, H_next))
            fixed = model.constraints.constrained_dofs
            prescribed = model.constraints.prescribed_vector(float(load_jax)).astype(dtype)
            if not np.array_equal(q_np[fixed], prescribed[fixed]):
                raise RuntimeError("stagger PINN violated an exact collar/anchor displacement")
            reaction_relative = abs(float(fields.reaction) - reaction_ref) / max(abs(reaction_ref), 1.0e-30)
            result = {
                "mode": "two_network_stagger_diagnostic",
                "sweep": sweep, "load": float(load), "step": stop_step,
                "global_step": global_step(sweep, "after", stop_step),
                "mechanics_before_inf": float(jnp.max(jnp.abs(mechanics_before))),
                "damage_solved_inf": float(jnp.max(jnp.abs(rho_solved))),
                "coupled_loss": float(coupled_loss(fields)),
                "mechanics_inf": float(jnp.max(jnp.abs(fields.mechanics))),
                "damage_inf": float(jnp.max(jnp.abs(fields.damage))),
                "history_inf": float(jnp.max(jnp.abs(fields.history))),
                "damage_change_inf": float(jnp.max(jnp.abs(d_trial - damage_iter))),
                "history_change_inf": float(jnp.max(jnp.abs(H_next - H_used))),
                "displacement_change_inf": float(jnp.max(jnp.abs(q_after - q_before))),
                "reaction": float(fields.reaction), "reference_reaction": reaction_ref,
                "u_relative_l2": _relative_l2(q_np[:2*model.n_material], q_ref[:2*model.n_material]),
                "d_relative_l2": _relative_l2(d_np, d_ref),
                **_damage_error_metrics(d_np, d_ref),
                "H_relative_l2": _relative_l2(H_np, H_ref),
                "drift_count": int(np.count_nonzero(
                    np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)
                    / float(load) > direct.DRIFT_ENVELOPE)),
                "damage_min": float(np.min(d_np)), "damage_max": float(np.max(d_np)),
                "damage_healing_max": float(np.max(np.maximum(d_committed - d_np, 0.0))),
                "history_decrease_max": float(np.max(np.maximum(H_committed - H_np, 0.0))),
                "reaction_relative": reaction_relative,
                "mechanics_loss": float(jnp.mean((fields.mechanics / mechanics_scale)**2)),
            }
            if not np.all(np.isfinite([value for value in result.values()
                                       if isinstance(value, (float, int))])):
                raise FloatingPointError("stagger diagnostics contain non-finite values")
            gates = (_warning_advancement_gates(config, result) if warning_diagnostic
                     else _acceptance_gates(config, result))
            ready = bool(all(gates.values()))
            accepted = ready and not warning_diagnostic
            if stop_step < n_q and not ready:
                raise RuntimeError("a passing final-stage report failed the same-pass acceptance gates")
            result["gates"] = {name: bool(value) for name, value in gates.items()}
            result["accepted"] = accepted
            if warning_diagnostic:
                result["diagnostic_only"] = True
                result["reference_warning_label"] = bool(warning)
                result["diagnostic_usable"] = ready
            position = result["global_step"]
            for name, value in result["gates"].items():
                writer.add_scalar(f"gate/{name}", float(value), position)
            writer.add_scalar("diag/accepted", float(accepted), position)
            if warning_diagnostic:
                writer.add_scalar("diag/diagnostic_usable", float(ready), position)
            writer.flush()
            print(json.dumps(result, sort_keys=True), flush=True)
            report_path = run_dir / f"sweep_{sweep:04d}.json"
            temporary = report_path.with_name(report_path.name + ".tmp")
            temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            os.replace(temporary, report_path)
            if sweep_observer is not None:
                sweep_observer(result, {
                    "q": q_np, "d": d_np, "H": H_np,
                    "H_used": np.asarray(H_used),
                    "d_solved": np.asarray(damage(d_net)),
                    "d_previous": np.asarray(damage_iter),
                    "at2_local": np.asarray(residual.gc_over_l0 * d_trial),
                    "at2_nonlocal": np.asarray(
                        residual.gc_l0 * residual.operator.deterministic_matvec(d_trial)),
                    "at2_history": np.asarray(2.0 * H_next * (d_trial - 1.0)),
                    "at2_fixed_relaxed": np.asarray(rho_at(d_trial, H_used)),
                    "at2_fixed_solved": np.asarray(rho_solved),
                    "coordinates": np.asarray(model.geometry.coordinates[:model.n_material]),
                })
            damage_iter = d_trial
            if sweep < config.stagger.sweeps:
                save(sweep + 1, "before", 0)
            else:
                if ready and return_state:
                    accepted_state = (
                        q_net, q_state, d_net, d_state, q_after, d_trial, H_next
                    )
                    write_checkpoint(
                        run_dir / ("diagnostic.ckpt.zip" if warning_diagnostic
                                   else "accepted.ckpt.zip"),
                        {**identity, "state_index": index},
                        sweep, "complete", stop_step, accepted_state,
                        schema=SCHEMA_DIAGNOSTIC if warning_diagnostic else SCHEMA_ACCEPTED,
                    )
                save(sweep, "complete", stop_step)
    finally:
        writer.close()
    if return_state:
        accepted_state = (
            q_net, q_state, d_net, d_state, q_after, d_trial, H_next
        ) if (result["diagnostic_usable"] if warning_diagnostic else result["accepted"]) else None
        return result, accepted_state
    return result


def main(_):
    config = _absl_flags.FLAGS.config
    run(config, resume=bool(config.output.resume))


if __name__ == "__main__":
    _absl_app.run(main)
