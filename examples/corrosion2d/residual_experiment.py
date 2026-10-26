"""Matched graph references and the shared soft-IC KKS training engine.

Canonical training starts in each case's train.py with its train_base preset.
This advanced CLI retains historical research presets for explicit controls.
Run from the repository root using this project's .venv. Reference integration
uses CPU float64 SciPy BDF; training uses JAX and the production architectures.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from tensorboardX import SummaryWriter

from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residuals import (bounded_phase_weight, make_loss_vector,
    make_normalized_data_loss_vector, make_transport_loss_vector, phase_only_constraint,
    validate_normalized_data, validate_transport_options)
from examples.corrosion2d.transport_optimizer import make_tail_optimizer, make_tail_schedule
from examples.corrosion2d.training_precision import (
    computation_dtype, cast_floating, audit_floating,
)
from examples.corrosion2d.training_data import (
    corrosion_boundary_points, corrosion_initial_points,
    electropolish_boundary_points, electropolish_initial_points,
)
from nonlocal_pinn.nn.arch import make_pinn
from nonlocal_pinn.nn.constraints import ConstrainedForward, kks_constraint
from nonlocal_pinn.operators import build_D


def grid(config):
    x = np.linspace(0.0, 1.0, config.sample.Nx)
    y = np.linspace(0.0, 0.5, config.sample.Ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy, np.stack([xx.ravel(), yy.ravel()], axis=-1)


def boundary_values(config, points):
    x, y = points.T
    if config.experiment.case == "corrosion2d":
        zero = np.zeros(len(points), dtype=bool)
        for cx, cy, radius in config.pits.entries:
            zero |= np.isclose(y, 0.0) & (np.hypot(x - cx, y - cy) < radius * 0.5 - 1e-8)
        one = np.isclose(x, 0.0) | np.isclose(x, 1.0) | np.isclose(y, 0.5)
    elif config.experiment.case == "electropolish2d":
        zero, one = np.isclose(y, 0.5), np.isclose(y, 0.0)
    else:
        raise ValueError(config.experiment.case)
    if np.any(zero & one) or not np.any(zero) or not np.any(one):
        raise ValueError("Dirichlet masks must be nonempty and disjoint")
    values = np.zeros((len(points), 2), dtype=np.float64)
    values[one] = 1.0
    return zero, one, values


def nondim(config):
    p, s = config.physical, config.nondim
    return (p.D * s.T_c / s.L_c**2, 2.0 * p.A * p.L * s.T_c,
            p.L * p.omega_phi * s.T_c, p.L * p.alpha_phi * s.T_c / s.L_c**2,
            p.c_se - p.c_le, p.c_le)


def reference_spec(config):
    case = config.experiment.case
    geometry = config.pits.to_dict() if case == "corrosion2d" else config.surface.to_dict()
    return json.loads(json.dumps({
        "case": case, "physical": config.physical.to_dict(),
        "nondim": config.nondim.to_dict(), "geometry": geometry,
        "Nx": config.sample.Nx, "Ny": config.sample.Ny,
        "domain": [[0.0, 1.0], [0.0, 0.5]],
        "delta": config.nl.delta, "kernel": config.nl.kernel,
        "operator": "in-domain symmetric pair graph, D=sum(w*(value_i-value_j))",
        "dirichlet": "pit mouth zero; left/right/top one" if case == "corrosion2d" else "top zero; bottom one",
        "other_boundary": "no external graph edges",
        "initial_condition": "production analytic pit/SDF profile, c=h(phi)*c_se; boundary nodes prescribed",
    }))


def initial_fields(config, points):
    losses = importlib.import_module(f"examples.{config.experiment.case}.losses")
    if config.experiment.case == "corrosion2d":
        phi, c = losses.ic_values(points, config.physical, config.nondim.L_c, config.pits.entries)
    else:
        phi, c = losses.ic_values(points, config.physical, config.nondim.L_c, config.surface)
    return np.stack([np.asarray(phi), np.asarray(c)], axis=-1)


def graph_matrix(op, n):
    row, col = np.asarray(op._row_idx), np.asarray(op._col_idx)
    weights = -np.asarray(op._vals, dtype=np.float64)
    if np.any(weights < 0) or np.any(row == col):
        raise ValueError("Reference requires nonnegative off-diagonal pair weights")
    matrix = sparse.diags(np.bincount(row, weights=weights, minlength=n)) - sparse.csr_matrix((weights, (row, col)), shape=(n, n))
    asymmetry = matrix - matrix.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-8:
        raise ValueError("Reference graph must be symmetric")
    if np.max(np.abs(matrix @ np.ones(n))) > 1e-8:
        raise ValueError("Reference graph must annihilate constants")
    return matrix.tocsr()


def solve_reference(config, matrix, points, rtol, atol, max_step):
    """Independent physical-field ODE and analytic sparse Jacobian."""
    n_ch, n1, n2, n3, dc, c_le = nondim(config)
    zero, one, boundary = boundary_values(config, points)
    free = ~(zero | one)
    ids = np.flatnonzero(free)
    nf = len(ids)
    ic = np.asarray(initial_fields(config, points), dtype=np.float64)
    ic[~free] = boundary[~free]
    ff = matrix[ids][:, ids].tocsr()

    def unpack(state):
        fields = boundary.copy()
        fields[free, 0], fields[free, 1] = state[:nf], state[nf:]
        return fields[:, 0], fields[:, 1]

    def rhs(t, state):
        phi, c = unpack(state)
        hp = 6.0 * phi * (1.0 - phi)
        gp = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        q = c - (3.0 * phi**2 - 2.0 * phi**3) * dc - c_le
        dp = -n3 * (matrix @ phi) - n2 * gp + n1 * q * dc * hp
        dconc = -n_ch * (matrix @ q)
        return np.concatenate([dp[free], dconc[free]])

    jac_calls = 0

    def jac(t, state):
        nonlocal jac_calls
        jac_calls += 1
        if jac_calls % 25 == 0:
            print(f"reference Jacobian {jac_calls}, t*={t:.6g}", flush=True)
        phi, c = unpack(state)
        p = phi[free]
        hp, hpp = 6.0 * p * (1.0 - p), 6.0 - 12.0 * p
        gpp = 12.0 * p**2 - 12.0 * p + 2.0
        q = c[free] - (3.0 * p**2 - 2.0 * p**3) * dc - c_le
        local = -n2 * gpp + n1 * dc * (q * hpp - dc * hp**2)
        return sparse.bmat([
            [-n3 * ff + sparse.diags(local), sparse.diags(n1 * dc * hp)],
            [n_ch * dc * ff @ sparse.diags(hp), -n_ch * ff],
        ], format="csc")

    times = np.linspace(0.0, 1.0, config.experiment.reference_snapshots)
    start = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, 1.0), np.concatenate([ic[free, 0], ic[free, 1]]),
                    method="BDF", jac=jac, t_eval=times, rtol=rtol, atol=atol,
                    max_step=max_step)
    if not sol.success or not np.array_equal(sol.t, times):
        raise RuntimeError(f"Reference integration failed: {sol.message}")
    fields = np.stack([np.stack(unpack(y), axis=-1) for y in sol.y.T])
    if not np.all(np.isfinite(fields)) or fields.min() < -1e-5 or fields.max() > 1.00001:
        raise ValueError(f"Reference physical range violation: [{fields.min()}, {fields.max()}]")
    q = fields[..., 1] - (3.0 * fields[..., 0]**2 - 2.0 * fields[..., 0]**3) * dc - c_le
    info = {"rtol": rtol, "atol": atol, "max_step": max_step,
            "nfev": sol.nfev, "njev": sol.njev, "nlu": sol.nlu,
            "seconds": time.perf_counter() - start,
            "field_min": float(fields.min()), "field_max": float(fields.max()),
            "q_min": float(q.min()), "q_max": float(q.max()),
            "max_network_q_bound_violation": float(max(0.0, q.max() - (1.0 - config.physical.c_se), -c_le - q.min()))}
    print(json.dumps(info), flush=True)
    return times, fields, info


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def generate_reference(config):
    if not jax.config.x64_enabled or jax.default_backend() != "cpu":
        raise ValueError("Reference generation requires JAX_ENABLE_X64=true JAX_PLATFORMS=cpu")
    path = Path(config.experiment.reference_path)
    coarse_path = path.with_name(path.stem + "_coarse.npz")
    if path.exists() or path.with_suffix(".json").exists() or coarse_path.exists():
        raise FileExistsError(path)
    xx, yy, points = grid(config)
    # The production graph weights are recorded directly, without hashes.
    op = build_D(jnp.asarray(points, dtype=jnp.float32), config.nl.delta, method="sparse", kernel=config.nl.kernel)
    matrix = graph_matrix(op, len(points))
    e = config.experiment
    times, coarse, coarse_info = solve_reference(config, matrix, points, e.reference_rtol, e.reference_atol, e.reference_max_step)
    _, fine, fine_info = solve_reference(config, matrix, points, e.refined_rtol, e.refined_atol, e.refined_max_step)
    errors = []
    for k in range(2):
        errors.append(np.linalg.norm(fine[..., k] - coarse[..., k], axis=1) / np.linalg.norm(fine[..., k], axis=1))
    max_error = float(np.max(errors))
    if max_error >= e.reference_error_tolerance:
        raise ValueError(f"Reference refinement relative L2 {max_error} >= {e.reference_error_tolerance}")
    shape = (len(times), config.sample.Ny, config.sample.Nx)
    metadata = {"spec": reference_spec(config), "solver": "SciPy BDF, analytic sparse Jacobian, physical phi/c",
                "coarse": coarse_info, "fine": fine_info,
                "refinement_max_relative_l2": max_error,
                "refinement_relative_l2_by_time": np.asarray(errors).tolist(),
                "continuum_convergence_claimed": False}
    write_npz(path, X=xx, Y=yy, time=times, physical_time=times * config.nondim.T_c,
              phi=fine[..., 0].reshape(shape), c=fine[..., 1].reshape(shape),
              graph_row=np.asarray(op._row_idx), graph_col=np.asarray(op._col_idx),
              graph_weight=-np.asarray(op._vals), metadata_json=np.array(json.dumps(metadata)))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_npz(coarse_path, X=xx, Y=yy, time=times, phi=coarse[..., 0].reshape(shape), c=coarse[..., 1].reshape(shape))
    print(f"Validated reference: {path}, max refinement L2={max_error:.3e}", flush=True)


def load_reference(config):
    path = Path(config.experiment.reference_path)
    with np.load(path) as data:
        reference = {key: data[key] for key in data.files}
    required = {"X", "Y", "time", "physical_time", "phi", "c", "graph_row", "graph_col", "graph_weight", "metadata_json"}
    if not required.issubset(reference):
        raise ValueError(f"Reference lacks alignment metadata/arrays: {path}: {sorted(required - reference.keys())}")
    metadata = json.loads(str(reference["metadata_json"]))
    if metadata["spec"] != reference_spec(config):
        raise ValueError(f"Reference model/BC/grid mismatch: {path}")
    if metadata["refinement_max_relative_l2"] >= config.experiment.reference_error_tolerance:
        raise ValueError("Reference refinement check did not pass")
    expected = (config.experiment.reference_snapshots, config.sample.Ny, config.sample.Nx)
    for field in ("phi", "c"):
        if reference[field].shape != expected or not np.all(np.isfinite(reference[field])):
            raise ValueError(f"Invalid reference {field} shape/values")
    xx, yy, _ = grid(config)
    if not np.array_equal(reference["X"], xx) or not np.array_equal(reference["Y"], yy):
        raise ValueError("Reference coordinate values/order mismatch")
    times = reference["time"]
    if times.shape != (config.experiment.reference_snapshots,) or times[0] != 0 or times[-1] != 1 or not np.all(np.diff(times) > 0):
        raise ValueError("Reference snapshots must be ordered and cover normalized time0..1")
    if not np.allclose(reference["physical_time"], reference["time"] * config.nondim.T_c, rtol=0, atol=1e-12):
        raise ValueError("Reference physical times mismatch")
    _, _, points = grid(config)
    zero, one, _ = boundary_values(config, points)
    fields = np.stack([reference["phi"], reference["c"]], axis=-1)
    if boundary_error(fields, zero, one) > 1e-10:
        raise ValueError("Reference Dirichlet field values do not match the declared boundary model")
    return reference


def build_model(config, key):
    a = config.arch
    raw = make_pinn(spatial_dim=2, out_dim=2, trunk_type=a.type,
                    hidden_dim=a.hidden_dim, num_layers=a.num_layers,
                    activation=a.activation, use_spatial_emb=a.use_spatial_emb,
                    use_temporal_emb=a.use_temporal_emb, spatial_emb_dim=a.spatial_emb_dim,
                    spatial_scale=a.spatial_scale, temporal_emb_dim=a.temporal_emb_dim,
                    temporal_scale=a.temporal_scale, tensor_split=a.tensor_split,
                    film_hidden_dim=a.film_hidden_dim, film_num_layers=a.film_num_layers, key=key)
    if not a.use_hard_constraint:
        return ConstrainedForward(raw, phase_only_constraint)
    return ConstrainedForward(raw, lambda r: kks_constraint(r, config.physical.c_se, config.physical.c_le))


def make_step(loss_vector, optimizer, config):
    bias = jnp.asarray(config.loss.gradnorm_bias)
    n_losses = len(config.loss.gradnorm_bias)
    alpha, cap = config.train.alpha_gradnorm, config.loss.gradnorm_weight_cap

    @eqx.filter_jit
    def step(net, state, weights, times, x, sub_data, mask, update_weights, active):
        if active.shape != (n_losses,) or active.dtype != jnp.bool_:
            raise ValueError(f"active must be a fixed-shape boolean vector of length{n_losses}")
        values, pullback = eqx.filter_vjp(lambda n: loss_vector(n, times, x, sub_data, mask), net)

        def refresh(_):
            norms = []
            for i in range(n_losses):
                grad = pullback(jax.nn.one_hot(i, n_losses))[0]
                norms.append(sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            norms = jnp.clip(jnp.stack(norms), 1e-6, 1e6)
            proposed = jnp.clip(jnp.mean(norms) / norms, 1e-6, cap)
            return jax.lax.stop_gradient(alpha * proposed + (1.0 - alpha) * weights)

        new_weights = jax.lax.cond(update_weights, refresh, lambda _: weights, operand=None)
        grad = pullback(new_weights * bias * active)[0]
        finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)
        ])) & jnp.all(jnp.isfinite(new_weights)) & jnp.all(new_weights > 0)
        updates, state = optimizer.update(grad, state, net)
        net = eqx.apply_updates(net, updates)
        return net, state, new_weights, values, finite

    return step


def stagger_phase(iteration, stagger_steps, switch_every):
    """0=AC-only, 1=CH-only, 2=joint; iteration is zero-based training time."""
    return (iteration // switch_every) % 2 if iteration < stagger_steps else 2


def interface_points(xx, yy, phase):
    import contourpy
    paths = contourpy.contour_generator(x=xx[0], y=yy[:, 0], z=phase).lines(0.5)
    return np.concatenate(paths) if paths else np.empty((0, 2))


def field_metrics(prediction, reference):
    target = np.stack([reference["phi"], reference["c"]], axis=-1)
    error = prediction - target
    metrics = {}
    for i, name in enumerate(("phi", "c")):
        metrics[f"pooled_rel_l2_{name}"] = float(np.linalg.norm(error[..., i]) / np.linalg.norm(target[..., i]))
        metrics[f"terminal_rel_l2_{name}"] = float(np.linalg.norm(error[-1, ..., i]) / np.linalg.norm(target[-1, ..., i]))
        metrics[f"rel_l2_by_time_{name}"] = (np.linalg.norm(error[..., i], axis=(1, 2)) / np.linalg.norm(target[..., i], axis=(1, 2))).tolist()
    a = interface_points(reference["X"], reference["Y"], prediction[-1, ..., 0])
    b = interface_points(reference["X"], reference["Y"], target[-1, ..., 0])
    metrics["terminal_interface_present"] = bool(len(a) and len(b))
    if len(a) and len(b):
        distances = np.concatenate([cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]])
        metrics["terminal_interface_mean_distance"] = float(np.mean(distances))
        metrics["terminal_interface_max_distance"] = float(np.max(distances))
    else:
        metrics["terminal_interface_mean_distance"] = None
        metrics["terminal_interface_max_distance"] = None
    return metrics


def boundary_error(prediction, zero, one):
    """Physical BC error on flattened graph nodes, independent of grid shape."""
    fields = prediction.reshape(prediction.shape[0], len(zero), 2)
    boundary = fields[:, zero | one]
    expected = np.broadcast_to(one[zero | one][None, :, None], boundary.shape)
    return float(np.max(np.abs(boundary - expected)))


def training_boundary_data(config, points, random_zero, random_one):
    """Choose boundary coordinates without changing prescribed values."""
    mode = config.experiment.boundary_sampling
    if mode == "matched_graph":
        zero, one, _ = boundary_values(config, np.asarray(points))
        x_zero, x_one = np.asarray(points)[zero], np.asarray(points)[one]
    elif mode == "random":
        x_zero, x_one = np.asarray(random_zero), np.asarray(random_one)
    else:
        raise ValueError(f"Unknown boundary sampling: {mode}")
    for x in (x_zero, x_one):
        if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.all(np.isfinite(x)):
            raise ValueError("Boundary coordinates must be nonempty finite (N,2) arrays")
    return jnp.asarray(x_zero, dtype=jnp.float32), jnp.asarray(x_one, dtype=jnp.float32)


def training_initial_data(config, points, interior_samples):
    """Explicit IC sampling; graph mode matches reference boundary override at t=0."""
    mode = config.experiment.initial_sampling
    if mode == "random_interior":
        x = np.asarray(interior_samples)
        fields = initial_fields(config, x)
    elif mode == "matched_graph":
        x = np.asarray(points)
        fields = initial_fields(config, x)
        zero, one, values = boundary_values(config, x)
        fields[zero | one] = values[zero | one]
    else:
        raise ValueError(f"Unknown initial sampling: {mode}")
    if fields.shape != (len(x), 2) or not np.all(np.isfinite(fields)):
        raise ValueError("Invalid training initial data")
    return jnp.asarray(x, dtype=jnp.float32), jnp.asarray(fields, dtype=jnp.float32)


def validate_loss_schedule(config):
    switch = config.experiment.ch_rms_after
    if (isinstance(switch, (bool, np.bool_)) or not isinstance(switch, (int, np.integer))
            or not 0 <= switch < config.train.iterations):
        raise ValueError("ch_rms_after must be an integer in [0, iterations)")
    if switch and (config.experiment.ch_norm != "l1"
                   or switch % config.train.gradnorm_update_every != 0):
        raise ValueError("ch_rms_after requires initial L1 and a GradNorm refresh at the switch")


def validate_time_window(config):
    """Research-only sampling curriculum; full horizon is always reached."""
    steps = config.loss.t_ramp_steps
    if (isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer))
            or not 0 <= steps < config.train.iterations):
        raise ValueError("t_ramp_steps must be an integer in [0, iterations)")
    if not np.isfinite(config.sample.t_min) or not 0 <= config.sample.t_min < 1:
        raise ValueError("sample.t_min must be finite and in [0,1)")
    if steps:
        early, late = config.loss.t_ramp_early, config.loss.t_ramp_late
        if not np.isfinite(early) or not config.sample.t_min < early < 1:
            raise ValueError("Enabled time ramp requires t_min < t_ramp_early < 1")
        if late != 1.0:
            raise ValueError("Research time ramp must reach the full horizon: t_ramp_late=1")


def time_window_max(config, iteration):
    """Linear upper bound on sampled nondimensional time, no solution prior."""
    if config.loss.t_ramp_steps == 0:
        return 1.0
    fraction = min(iteration / config.loss.t_ramp_steps, 1.0)
    return config.loss.t_ramp_early + (1.0 - config.loss.t_ramp_early) * fraction


def sample_training_times(config, key_time, iteration):
    return jr.uniform(jr.fold_in(key_time, iteration), (config.sample.Nt,),
                      minval=config.sample.t_min, maxval=time_window_max(config, iteration))


def make_training_losses(config, nd, x_ic, ic, x_zero, x_one):
    """Fixed-shape loss pair; the caller carries the same optimizer state."""
    validate_loss_schedule(config)
    if (config.experiment.transport_evaluation != "physical_difference"
            or config.experiment.ch_norm != "l1"):
        initial = make_transport_loss_vector(config, nd, x_ic, ic, x_zero, x_one)
    elif config.experiment.data_loss_mode == "physical":
        initial = make_loss_vector(config, nd, x_ic, ic, x_zero, x_one)
    else:
        initial = make_normalized_data_loss_vector(config, nd, x_ic, ic, x_zero, x_one,
            normalize_initial=config.experiment.data_loss_mode == "s_bc_ic")
    if not config.experiment.ch_rms_after:
        return initial, initial
    final_config = config.copy_and_resolve_references()
    final_config.experiment.ch_norm = "rms"
    final = make_transport_loss_vector(final_config, nd, x_ic, ic, x_zero, x_one)
    return initial, final


def train(config):
    if config.experiment.spatial_operator in ("local_ad_tensor", "local_ad_joint"):
        from examples.corrosion2d.local_training import train_local
        return train_local(config)
    if config.experiment.spatial_operator != "nonlocal_graph":
        raise ValueError(f"Unknown spatial operator: {config.experiment.spatial_operator}")
    dtype = computation_dtype(config)
    jax.config.update("jax_enable_x64", dtype == np.dtype("float64"))
    validate_loss_schedule(config)
    validate_time_window(config)
    learning_rate = make_tail_schedule(config)
    validate_transport_options(config)
    if config.experiment.data_loss_mode not in ("physical", "s_bc", "s_bc_ic"):
        raise ValueError(f"Unknown data loss mode: {config.experiment.data_loss_mode}")
    if config.experiment.data_loss_mode != "physical":
        validate_normalized_data(config)
    if config.experiment.initial_sampling not in ("random_interior", "matched_graph"):
        raise ValueError(f"Unknown initial sampling: {config.experiment.initial_sampling}")
    if config.experiment.boundary_sampling not in ("random", "matched_graph"):
        raise ValueError(f"Unknown boundary sampling: {config.experiment.boundary_sampling}")
    for name, minimum in (("stagger_steps", 0), ("stagger_switch_every", 1)):
        value = config.loss[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"loss.{name} must be an integer >= {minimum}, got {value}")
    bias = np.asarray(config.loss.gradnorm_bias)
    if bias.shape != (5,) or not np.all(np.isfinite(bias)) or np.any(bias <= 0):
        raise ValueError("loss.gradnorm_bias must contain five finite positive values")
    if config.experiment.formulation in ("bounded", "bounded_frozen"):
        bounded_phase_weight(0.5, config.experiment.bounded_eta)
    for name in ("ac_scale", "ch_scale"):
        value = config.loss[name]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{name} must be finite and positive, got {value}")
    reference = load_reference(config)
    jax.config.update("jax_default_matmul_precision", config.train.matmul_precision)
    if not config.log.run_name:
        config = config.copy_and_resolve_references()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        config.log.run_name = f"{stamp}-{config.experiment.formulation}-seed{config.train.seed}"
    run = Path(config.log.log_dir) / config.log.run_name
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(config.to_json(indent=2) + "\n")
    source = run / "source"
    source.mkdir()
    repo = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), Path(__file__).with_name("residuals.py"),
             Path(__file__).with_name("transport_optimizer.py"),
             Path(__file__).parent / "configs/residual_pilot.py",
             repo / f"examples/{config.experiment.case}/losses.py",
             repo / "src/nonlocal_pinn/nn/constraints.py",
             repo / "src/nonlocal_pinn/training/runtime.py",
             Path(__file__).with_name("training_data.py"),
             Path(__file__).with_name("training_entry.py"),
             Path(__file__).with_name("training_precision.py"),
             repo / f"examples/{config.experiment.case}/train.py",
             repo / f"examples/{config.experiment.case}/configs/train_base.py",
             repo / "examples/corrosion2d/configs/train_1pit.py",
             repo / "examples/electropolish2d/configs/train_legacy.py"]
    for f in files:
        target = source / f.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
    (source / "files.json").write_text(json.dumps([str(f.relative_to(repo)) for f in files], indent=2))
    # Common stored inputs and RNG draws are generated in FP32 for both modes.
    with jax.enable_x64(False):
        xx, yy, points = grid(config)
        x_grid = jnp.asarray(points, dtype=jnp.float32)
        op = build_D(x_grid, config.nl.delta, method=config.nl.method, kernel=config.nl.kernel)
        if not np.array_equal(np.asarray(op._row_idx), reference["graph_row"]) or not np.array_equal(np.asarray(op._col_idx), reference["graph_col"]) or not np.array_equal(-np.asarray(op._vals), reference["graph_weight"]):
            raise ValueError("Training/reference graph edges or weights differ")
        zero, one, _ = boundary_values(config, points)
        free_np = ~(zero | one)
        dc, cle = config.physical.c_se - config.physical.c_le, config.physical.c_le
        reference_cl = reference["c"] - dc * (3 * reference["phi"]**2 - 2 * reference["phi"]**3)
        reference_q = reference_cl - cle
        diagnostic_matrix = graph_matrix(op, len(points))
        reference_dq_terminal = diagnostic_matrix @ reference_q[-1].ravel()
        if config.experiment.case == "corrosion2d":
            x_zero, x_one = corrosion_boundary_points(config.pits.entries, config.sample.N_bc_dir, config.sample.N_bc_solid)
            x_ic = corrosion_initial_points(config.pits.entries, config.sample.N_ic_global, config.sample.N_ic_local)
        else:
            x_zero, x_one = electropolish_boundary_points(config)
            x_ic = electropolish_initial_points(config)
        x_zero, x_one = training_boundary_data(config, points, x_zero, x_one)
        x_ic, ic = training_initial_data(config, points, x_ic)
        # Match the independently prescribed initial state, without using trajectory data.
        if config.experiment.initial_sampling == "matched_graph":
            expected = np.stack([reference["phi"][0].ravel(), reference["c"][0].ravel()], axis=-1)
            if not np.allclose(np.asarray(ic), expected, rtol=0, atol=3e-7):
                raise ValueError("Graph training IC differs from the reference initial state")
        key_net, key_time = jr.split(jr.PRNGKey(config.train.seed))
        net = build_model(config, key_net)
        pool = op.build_subgraph_pool(config.sample.K_core, config.sample.K_pool)
        batches = []
        free = jnp.asarray(~(zero | one))
        for ids, sub_data, core_mask, _ in pool:
            mask = core_mask & free[ids]
            if not np.any(np.asarray(mask)):
                raise ValueError("Graph batch contains no free PDE core nodes")
            batches.append((x_grid[ids], sub_data, mask))
        del pool
    x_grid, x_ic, ic, x_zero, x_one, net, batches = cast_floating(
        (x_grid, x_ic, ic, x_zero, x_one, net, batches), dtype)
    row_sums = np.bincount(np.asarray(op._row_idx),
                           weights=-np.asarray(op._vals, dtype=np.float64),
                           minlength=len(points))
    diagonal_defect = np.asarray(op._diag, dtype=np.float64) - row_sums
    loss_vector, final_loss_vector = make_training_losses(config, nondim(config), x_ic, ic, x_zero, x_one)
    eqx.tree_serialise_leaves(run / "model_initial.eqx", net)
    optimizer = make_tail_optimizer(config)
    state = optimizer.init(eqx.filter(net, eqx.is_array))
    precision_record = {
        "compute_dtype": str(dtype),
        "input_policy": "Common FP32 initialization/data/time draws and stored graph coefficients, exactly promoted",
        "graph_diagonal_policy": "Preserve stored FP32 diagonal; no row-sum correction",
        "graph_constant_defect_max": float(np.max(np.abs(diagonal_defect))),
        "graph_diagonal_relative_defect_max": float(np.max(np.abs(diagonal_defect)/row_sums)),
        "parameters": audit_floating(net, dtype),
        "optimizer_initial": audit_floating(state, dtype),
        "data": audit_floating((x_grid, x_ic, ic, x_zero, x_one, batches), dtype),
        "lr_schedule": "Existing Optax FP32 scalar schedule; updates cast to parameter dtype",
    }
    (run / "precision.json").write_text(json.dumps(precision_record, indent=2) + "\n")
    step = make_step(loss_vector, optimizer, config)
    final_step = make_step(final_loss_vector, optimizer, config) if config.experiment.ch_rms_after else step
    writer = SummaryWriter(str(run))
    times_eval = jnp.asarray(np.asarray(reference["time"], dtype=np.float32), dtype=dtype)

    @eqx.filter_jit
    def predict(n):
        raw = jax.vmap(lambda t: jax.vmap(n.net, in_axes=(0, None))(x_grid, t))(times_eval)
        fields = n.constraint(raw)
        if config.arch.use_hard_constraint:
            cl = (1 - config.physical.c_se + config.physical.c_le) * (.5 * jnp.tanh(raw[..., 1]) + .5)
        else:
            phi = fields[..., 0]
            cl = fields[..., 1] - dc * (3 * phi**2 - 2 * phi**3)
        return fields, cl

    weights = jnp.ones(5)
    # Precompute all fixed-shape masks, keeping IC/BC active in every stage.
    stage_masks = (jnp.array([True, False, True, True, True]),
                   jnp.array([False, True, True, True, True]),
                   jnp.ones(5, dtype=bool))
    stage_names = ("ac_only", "ch_only", "joint")
    activation_counts = dict.fromkeys(stage_names, 0)
    start = time.perf_counter()
    history_path = run / "metrics.jsonl"

    def evaluate(completed):
        fields, cl = predict(net)
        prediction = np.asarray(fields).reshape(len(times_eval), *xx.shape, 2)
        cl = np.asarray(cl, dtype=np.float64).reshape(len(times_eval), *xx.shape)
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError(f"Nonfinite physical prediction after {completed} updates")
        metrics = field_metrics(prediction, reference)
        q = cl - cle
        for name, predicted, target in (("cL", cl, reference_cl), ("q", q, reference_q)):
            metrics[f"terminal_rel_l2_{name}"] = float(np.linalg.norm(predicted[-1]-target[-1])/np.linalg.norm(target[-1]))
            metrics[f"pooled_rel_l2_{name}"] = float(np.linalg.norm(predicted-target)/np.linalg.norm(target))
        dq_terminal = diagnostic_matrix @ q[-1].ravel()
        metrics["terminal_rel_l2_Dq_free"] = float(np.linalg.norm((dq_terminal-reference_dq_terminal)[free_np])/
                                                  np.linalg.norm(reference_dq_terminal[free_np]))
        qflat = q.reshape(len(times_eval), -1)
        refqflat = reference_q.reshape(len(times_eval), -1)
        for name, nodes, target in (("pit", zero, -cle), ("solid", one, 0.0)):
            metrics[f"bc_{name}_q_rmse_over_cle"] = float(np.sqrt(np.mean((qflat[:, nodes]-target)**2))/cle)
            metrics[f"bc_{name}_c_max_error"] = float(np.max(np.abs(prediction.reshape(len(times_eval), -1, 2)[:, nodes, 1]-(1.0 if name == "solid" else 0.0))))
        metrics["initial_q_rmse_over_cle_free"] = float(np.sqrt(np.mean((qflat[0, free_np]-refqflat[0, free_np])**2))/cle)
        dq0, refdq0 = diagnostic_matrix @ qflat[0], diagnostic_matrix @ refqflat[0]
        metrics["initial_rel_l2_Dq_free"] = float(np.linalg.norm((dq0-refdq0)[free_np])/np.linalg.norm(refdq0[free_np]))
        metrics["boundary_max_abs_error"] = boundary_error(prediction, zero, one)
        concentration = prediction[..., 1]
        metrics["concentration_min"] = float(concentration.min())
        metrics["concentration_max"] = float(concentration.max())
        metrics["concentration_fraction_outside_0_1"] = float(np.mean((concentration < 0) | (concentration > 1)))
        metrics["step"] = completed
        metrics["elapsed_seconds"] = time.perf_counter() - start
        for k, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"eval/{k}", value, completed)
        with history_path.open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(f"[{config.experiment.case}/{config.experiment.formulation} {completed}] "
              f"terminal phi L2={metrics['terminal_rel_l2_phi']:.6g}, c L2={metrics['terminal_rel_l2_c']:.6g}, "
              f"q L2={metrics['terminal_rel_l2_q']:.6g}, Dq L2={metrics['terminal_rel_l2_Dq_free']:.6g} "
              f"({metrics['elapsed_seconds']:.1f}s)", flush=True)
        return metrics, prediction

    evaluate(0)
    previous_stage = None
    for iteration in range(config.train.iterations):
        with jax.enable_x64(False):
            times = sample_training_times(config, key_time, iteration)
        times = times.astype(dtype)
        x, sub_data, mask = batches[iteration % len(batches)]
        stage = stagger_phase(iteration, config.loss.stagger_steps, config.loss.stagger_switch_every)
        active = stage_masks[stage]
        refresh = iteration % config.train.gradnorm_update_every == 0
        current_step = (final_step if config.experiment.ch_rms_after and iteration >= config.experiment.ch_rms_after else step)
        net, state, weights, losses, finite = current_step(
            net, state, weights, times, x, sub_data, mask,
            jnp.asarray(refresh), active)
        if not bool(finite):
            raise FloatingPointError(f"Nonfinite loss/gradient at update {iteration+1}: {np.asarray(losses)}")
        completed = iteration + 1
        activation_counts[stage_names[stage]] += 1
        if completed % config.log.log_every == 0 or stage != previous_stage:
            effective_weights = np.asarray(weights) * bias * np.asarray(active)
            for k, value in zip(("ac", "ch", "bc_zero", "bc_one", "ic"), np.asarray(losses)):
                writer.add_scalar(f"loss/{k}", float(value), completed)
            for k, value in zip(("ac", "ch", "bc_zero", "bc_one", "ic"), effective_weights):
                writer.add_scalar(f"weight/{k}", float(value), completed)
            for k, value in zip(("ac", "ch", "bc_zero", "bc_one", "ic"), np.asarray(weights)):
                writer.add_scalar(f"adaptive_weight/{k}", float(value), completed)
            with (run / "loss_history.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": completed, "losses": np.asarray(losses).tolist(),
                                         "weights": effective_weights.tolist(),
                                         "adaptive_weights": np.asarray(weights).tolist(),
                                         "active": np.asarray(active).tolist(), "stage": stage_names[stage],
                                         "update_weights": refresh}) + "\n")
            print(f"update {completed} ({stage_names[stage]}): losses={np.asarray(losses)}, weights={effective_weights}", flush=True)
        previous_stage = stage
        if completed % config.log.eval_every == 0 or completed == config.train.iterations:
            final_metrics, prediction = evaluate(completed)
        if completed % config.experiment.checkpoint_every == 0:
            eqx.tree_serialise_leaves(run / f"ckpt_{completed:06d}.eqx", net)
    precision_record["parameters_final"] = audit_floating(net, dtype)
    precision_record["optimizer_final"] = audit_floating(state, dtype)
    (run / "precision.json").write_text(json.dumps(precision_record, indent=2) + "\n")
    eqx.tree_serialise_leaves(run / "model_final.eqx", net)
    np.savez_compressed(run / "prediction.npz", X=xx, Y=yy, time=reference["time"],
                        phi=prediction[..., 0], c=prediction[..., 1])
    summary = {"run": str(run), "reference": config.experiment.reference_path,
               "model_path": "model_final.eqx", "updates": config.train.iterations,
               "formulation": config.experiment.formulation, "hard_initial_condition": False,
               "concentration_map": "kks" if config.arch.use_hard_constraint else "direct",
               "ac_scale": config.loss.ac_scale, "ch_scale": config.loss.ch_scale,
               "bounded_eta": config.experiment.bounded_eta if config.experiment.formulation in ("bounded", "bounded_frozen") else None,
               "stagger_steps": config.loss.stagger_steps, "stagger_switch_every": config.loss.stagger_switch_every,
               "activation_counts": activation_counts,
               "adaptive_weight_state": "unmasked and unbiased; weights in loss_history are effective",
               "data_loss_mode": config.experiment.data_loss_mode,
               "transport_evaluation": config.experiment.transport_evaluation,
               "ch_norm": config.experiment.ch_norm,
               "ch_rms_after": config.experiment.ch_rms_after,
               "final_ch_norm": "rms" if config.experiment.ch_rms_after else config.experiment.ch_norm,
               "time_sampling": {"ramp_steps": config.loss.t_ramp_steps,
                   "initial_max": time_window_max(config, 0),
                   "last_update_max": time_window_max(config, config.train.iterations - 1),
                   "full_horizon_updates": config.train.iterations - config.loss.t_ramp_steps},
               "ch_rms_floor": config.experiment.ch_rms_floor,
               "learning_rate_profile": {"tail_start": config.experiment.lr_tail_start,
                   "tail_end": config.experiment.lr_tail_end,
                   "initial_rate": float(learning_rate(0)),
                   "last_update_rate": float(learning_rate(config.train.iterations - 1))},
               "initial_sampling": config.experiment.initial_sampling,
               "initial_training_nodes": len(x_ic),
               "boundary_sampling": config.experiment.boundary_sampling,
               "boundary_training_nodes": {"pit": len(x_zero), "solid": len(x_one)},
               "xla_flags": os.environ["XLA_FLAGS"] if "XLA_FLAGS" in os.environ else None,
               "final_metrics": final_metrics, "wall_seconds": time.perf_counter() - start}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("reference", "train"))
    parser.add_argument("--case", choices=("corrosion2d", "electropolish2d"), required=True)
    parser.add_argument("--formulation", choices=("phase", "latent", "bounded", "bounded_frozen"), default="phase")
    parser.add_argument("--bounded-eta", type=float,
                        help="Positive bounded-mode parameter; weight cap is 1+1/eta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concentration-map", choices=("kks", "direct"), default="kks")
    parser.add_argument("--ac-scale", type=float)
    parser.add_argument("--ch-scale", type=float)
    parser.add_argument("--stagger-steps", type=int,
                        help="Number of initial updates alternating AC/CH; 0 disables")
    parser.add_argument("--stagger-switch-every", type=int,
                        help="Updates in each AC-only or CH-only block")
    parser.add_argument("--data-loss-mode", choices=("physical", "s_bc", "s_bc_ic"))
    parser.add_argument("--initial-sampling", choices=("random_interior", "matched_graph"))
    parser.add_argument("--boundary-sampling", choices=("random", "matched_graph"))
    parser.add_argument("--transport-evaluation", choices=("physical_difference", "kks_raw_edges"))
    parser.add_argument("--ch-norm", choices=("l1", "rms"))
    parser.add_argument("--lr-tail-start", type=int, help="SOAP: start late rate decay; 0 keeps legacy")
    parser.add_argument("--lr-tail-end", type=float, help="SOAP: target rate after all updates")
    parser.add_argument("--ch-rms-after", type=int, help="Switch CH L1 to RMS after this many updates; 0 disables")
    parser.add_argument("--time-ramp-steps", type=int, help="Grow sampled time window; 0 keeps full horizon")
    parser.add_argument("--time-ramp-early", type=float, help="Initial nondimensional time upper bound")
    args = parser.parse_args()
    config = get_config(args.case, args.formulation, args.seed)
    if args.stagger_steps is not None or args.stagger_switch_every is not None:
        if args.action != "train":
            parser.error("Stagger overrides apply only to training")
        if args.stagger_steps is not None:
            if args.stagger_steps < 0:
                parser.error("--stagger-steps must be nonnegative")
            config.loss.stagger_steps = args.stagger_steps
        if args.stagger_switch_every is not None:
            if args.stagger_switch_every <= 0:
                parser.error("--stagger-switch-every must be positive")
            config.loss.stagger_switch_every = args.stagger_switch_every
    if args.bounded_eta is not None:
        if args.action != "train" or args.formulation not in ("bounded", "bounded_frozen"):
            parser.error("--bounded-eta requires training with --formulation bounded")
        if not np.isfinite(args.bounded_eta) or args.bounded_eta <= 0:
            parser.error("--bounded-eta must be finite and positive")
        config.experiment.bounded_eta = args.bounded_eta
    if (args.concentration_map != "kks" or args.ac_scale is not None or args.ch_scale is not None
            or args.formulation in ("bounded", "bounded_frozen")):
        if args.action != "train":
            parser.error("Output mapping and loss-scale overrides apply only to training")
        config.arch.use_hard_constraint = args.concentration_map == "kks"
        if args.ac_scale is not None:
            config.loss.ac_scale = args.ac_scale
        if args.ch_scale is not None:
            config.loss.ch_scale = args.ch_scale
        prefix = (f"20261025-residual-{args.formulation}-eta{config.experiment.bounded_eta}"
                  if args.formulation in ("bounded", "bounded_frozen") else f"20261024-residual-{args.formulation}")
        config.log.run_name = (f"{prefix}-map{args.concentration_map}"
                               f"-ac{config.loss.ac_scale}-ch{config.loss.ch_scale}-seed{args.seed}")
    if args.stagger_steps is not None or args.stagger_switch_every is not None:
        config.log.run_name = (f"20261025-stagger{config.loss.stagger_steps}-every{config.loss.stagger_switch_every}"
                               f"-{args.formulation}-map{args.concentration_map}"
                               f"-ac{config.loss.ac_scale}-ch{config.loss.ch_scale}-seed{args.seed}")
    if args.data_loss_mode is not None or args.initial_sampling is not None or args.boundary_sampling is not None:
        if args.action != "train":
            parser.error("Data-loss and initial-sampling overrides apply only to training")
        if args.data_loss_mode is not None:
            config.experiment.data_loss_mode = args.data_loss_mode
        if args.initial_sampling is not None:
            config.experiment.initial_sampling = args.initial_sampling
        if args.boundary_sampling is not None:
            config.experiment.boundary_sampling = args.boundary_sampling
        config.log.run_name = (f"20261025-data-{config.experiment.data_loss_mode}"
                               f"-ic{config.experiment.initial_sampling}-bc{config.experiment.boundary_sampling}-{args.formulation}"
                               f"-stagger{config.loss.stagger_steps}-seed{args.seed}")
    if args.transport_evaluation is not None or args.ch_norm is not None:
        if args.action != "train":
            parser.error("Transport-evaluation and CH-norm overrides apply only to training")
        if args.transport_evaluation is not None:
            config.experiment.transport_evaluation = args.transport_evaluation
        if args.ch_norm is not None:
            config.experiment.ch_norm = args.ch_norm
        config.log.run_name = (f"20261025-transport-{config.experiment.transport_evaluation}"
            f"-ch{config.experiment.ch_norm}-{config.experiment.data_loss_mode}"
            f"-ic{config.experiment.initial_sampling}-bc{config.experiment.boundary_sampling}"
            f"-{args.formulation}-seed{args.seed}")
    if args.lr_tail_start is not None or args.lr_tail_end is not None:
        if args.action != "train":
            parser.error("Late learning-rate controls apply only to training")
        if args.lr_tail_start is not None:
            config.experiment.lr_tail_start = args.lr_tail_start
        if args.lr_tail_end is not None:
            if config.experiment.lr_tail_start == 0:
                parser.error("--lr-tail-end requires a positive --lr-tail-start")
            config.experiment.lr_tail_end = args.lr_tail_end
        config.log.run_name += (f"-tail{config.experiment.lr_tail_start}"
                                f"-end{config.experiment.lr_tail_end}")
    if args.ch_rms_after is not None:
        if args.action != "train":
            parser.error("CH norm continuation applies only to training")
        config.experiment.ch_rms_after = args.ch_rms_after
        config.log.run_name += f"-rmsafter{config.experiment.ch_rms_after}"
    if args.time_ramp_steps is not None or args.time_ramp_early is not None:
        if args.action != "train":
            parser.error("Time-window controls apply only to training")
        if args.time_ramp_steps is not None:
            config.loss.t_ramp_steps = args.time_ramp_steps
        if args.time_ramp_early is not None:
            config.loss.t_ramp_early = args.time_ramp_early
        validate_time_window(config)
        config.log.run_name += f"-timeramp{config.loss.t_ramp_steps}-early{config.loss.t_ramp_early}"
    if args.action == "reference":
        generate_reference(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
