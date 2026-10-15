"""FP64 split-horizon neural loading with damage/reaction-first warning reports.

Only learned predecessor fields supply cumulative history and irreversibility
diagnostics. Direct fields are held-out comparisons, never loss targets.
User-accepted warning comparisons require an explicit SHA-bound policy.
Warning progression is explicitly distinct from a strictly validated solution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorboardX import SummaryWriter

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.losses import FractureResidual
from examples.fracture2d._legacy.split_horizon_model import build_model, read_cloud
from examples.fracture2d._legacy.split_reference import reference_state_root
from examples.fracture2d._legacy.stagger_checkpoint import read_checkpoint, write_checkpoint
from examples.fracture2d._legacy.split_loading import reference_indices, check_fork_loading
from examples.fracture2d._legacy.inner_stopping import StopWindow, validate_controls
from examples.fracture2d._legacy.train import _build_network, _nonnegative_damage


SCHEMA = "fracture2d-split-pinn-warning-progress-v1"


def resume_identity(stored, current):
    """Endpoint-only extension preserves the identity of all earlier snapshots."""
    old_end = stored["config"]["data"]["end_state_index"]
    new_end = current["config"]["data"]["end_state_index"]
    normalized = json.loads(json.dumps(current))
    normalized["config"]["data"]["end_state_index"] = old_end
    if stored != normalized or new_end < old_end:
        raise ValueError("split PINN resume permits endpoint extension only")


def checkpoint_resume_identity(path, expected):
    with zipfile.ZipFile(path) as archive:
        stored = json.loads(archive.read("metadata.json"))["identity"]
    resume_identity(stored, expected)
    if stored["index"] > stored["config"]["data"]["end_state_index"]:
        raise ValueError("split PINN snapshot index exceeds its declared endpoint")
    return stored


def neural_state_root(root, index):
    root = Path(root)
    visited = set()
    while True:
        if root.resolve() in visited:
            raise ValueError("split neural fork ancestry is cyclic")
        visited.add(root.resolve())
        output = json.loads((root / "identity.json").read_text())["config"]["output"]
        # Schema-v1 roots created before prefix forks have no fork fields.
        if "fork_from" in output and output["fork_from"] and index <= output["fork_after_index"]:
            root = Path(output["fork_from"])
        else:
            return root / f"state_{index:04d}"


def fork_checkpoint_identity(path, expected):
    with zipfile.ZipFile(path) as archive:
        stored = json.loads(archive.read("metadata.json"))["identity"]
    for name in ("model_digest", "reference_sha256", "parent_sha256", "index"):
        if stored[name] != expected[name]:
            raise ValueError(f"split neural fork {name} differs")
    if "reference_provenance" in stored or "reference_provenance" in expected:
        if ("reference_provenance" not in stored or "reference_provenance" not in expected
                or stored["reference_provenance"] != expected["reference_provenance"]):
            raise ValueError("split neural fork reference provenance differs")
    for name in ("arch", "physical", "nl", "geometry"):
        if stored["config"][name] != expected["config"][name]:
            raise ValueError(f"split neural fork {name} differs")
    check_fork_loading(stored["config"]["data"], expected["config"]["data"], expected["index"])
    return stored


def accepted_warning_policy(root, expected_sha256):
    """Validate explicit engineering acceptance without relabeling any fields."""
    path = Path(root) / "policy.json"
    payload = path.read_bytes()
    if not isinstance(expected_sha256, str) or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("accepted-warning reference policy SHA differs")
    policy = json.loads(payload)
    if (policy["schema"] != "fracture2d-split-warning-v1"
            or policy["diagnostic_only"] is not True or policy["fork_after_index"] < 1):
        raise ValueError("accepted-warning comparisons require a bound diagnostic warning root")
    return policy


def comparison_reference(root, index, model_digest, n_dofs, n_material, load, *,
                         reference_policy="strict", policy_sha256=None, return_metadata=False):
    """Strict by default; accepted warning data remain explicitly diagnostic."""
    if reference_policy not in ("strict", "user_accepted_warning"):
        raise ValueError("unknown PINN reference comparison policy")
    policy = None
    if reference_policy == "user_accepted_warning":
        policy = accepted_warning_policy(root, policy_sha256)
        if policy["model_digest"] != model_digest:
            raise ValueError("accepted-warning reference model differs")
    state_dir = reference_state_root(root, index)
    report_bytes = (state_dir / "result.json").read_bytes()
    report = json.loads(report_bytes)
    is_warning = report["accepted_reference"] is False
    if report["model_digest"] != model_digest or abs(report["load"] - load) > 1e-12:
        raise ValueError("split PINN comparison requires a matching strict reference and field digest")
    if is_warning:
        if reference_policy != "user_accepted_warning":
            raise ValueError("split PINN comparison requires a matching strict reference and field digest")
        if policy is None:
            raise ValueError("accepted-warning reference lacks its bound policy")
        if (report["diagnostic_only"] is not True or report["warning_usable"] is not True
                or report["strict_accepted"] is not False or report["field_equations_validated"] is not False
                or set(report["hard_gates"]) != {"finite", "damage_range", "history_monotonic"}
                or not all(report["hard_gates"].values()) or index <= policy["fork_after_index"]
                or report["inherited_warning"] != (index > policy["fork_after_index"] + 1)):
            raise ValueError("invalid accepted-warning reference status/ancestry")
        previous = reference_state_root(root, index - 1)
        parent_report = json.loads((previous / "result.json").read_text())
        parent_strict = parent_report["accepted_reference"] is True
        parent_path = previous / ("committed.npz" if parent_strict else "warning.npz")
        parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        parent_key = "committed_sha256" if parent_strict else "warning_sha256"
        if (parent_sha != report["parent_sha256"] or parent_sha != parent_report[parent_key]
                or parent_report["model_digest"] != model_digest):
            raise ValueError("accepted-warning reference parent SHA/model differs")
        path, digest_key = state_dir / "warning.npz", "warning_sha256"
    else:
        if report["accepted_reference"] is not True:
            raise ValueError("split PINN comparison requires a matching strict reference and field digest")
        path, digest_key = state_dir / "committed.npz", "committed_sha256"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report[digest_key]:
        raise ValueError("split PINN comparison requires a matching strict reference and field digest")
    with np.load(path, allow_pickle=False) as archive:
        q, d = (archive[name] for name in ("q", "d"))
    if (q.shape != (n_dofs,) or d.shape != (n_material,)
            or any(x.dtype != np.float64 or not np.all(np.isfinite(x)) for x in (q, d))
            or min(d) < -1e-10 or max(d) > 1 or not np.isfinite(report["reaction"])):
        raise ValueError("split reference fields do not match model shape/dtype")
    result = (q, d, float(report["reaction"]), digest)
    if return_metadata:
        metadata = {"status": "warning" if is_warning else "strict",
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "comparison_policy": reference_policy, "policy_sha256": policy_sha256,
                    "warning_failures": report["warning_failures"] if is_warning else [],
                    "diagnostic_only": is_warning}
        return (*result, metadata)
    return result


def classify_report(report, limits, *, inherited_warning):
    """Optionally gate on reference comparisons; retain every diagnostic failure."""
    if not all(np.isfinite(x) for x in report.values() if isinstance(x, (float, int))):
        raise FloatingPointError("split PINN diagnostics are nonfinite")
    checks = {
        "damage_rmse": (report["damage_rmse"], limits.damage_rmse_limit),
        "reaction_relative": (report["reaction_relative"], limits.reaction_relative_limit),
        "mechanics_loss": (report["mechanics_loss"], limits.mechanics_loss_limit),
        "coupled_AT2_relative": (report["coupled_AT2_relative"], limits.coupled_AT2_relative_limit),
        "damage_change": (report["damage_change"], limits.damage_change_limit),
        "displacement_relative": (report["displacement_relative"], limits.displacement_relative_limit),
        "healing": (report["healing"], limits.healing_limit),
        "damage_range_violation": (max(0., -report["damage_min"], report["damage_max"] - 1), limits.range_atol),
        "drift_count": (report["drift_count"], 0),
        "history_decrease": (report["history_decrease"], 0),
    }
    metrics = {key: {"value": float(value), "limit": float(limit), "pass": bool(value <= limit)}
               for key, (value, limit) in checks.items()}
    failed = [name for name, entry in metrics.items() if not entry["pass"]]
    primary_pass = all(metrics[name]["pass"] for name in ("damage_rmse", "reaction_relative"))
    comparison_gate = limits.comparison_gate if "comparison_gate" in limits else True
    if type(comparison_gate) is not bool:
        raise ValueError("reporting.comparison_gate must be boolean")
    strict = comparison_gate and primary_pass and not failed and not inherited_warning
    advance = primary_pass if comparison_gate else True
    result = {**report, "checks": metrics, "failed_checks": failed,
             "inherited_warning": bool(inherited_warning), "research_advance": advance,
             "strict_accepted": strict, "field_equations_validated": False,
             "status": "strict_checks_pass" if strict else "warning" if advance else "rejected"}
    if "comparison_gate" in limits:
        result.update(comparison_gate_enforced=comparison_gate, comparison_checks_pass=primary_pass)
    return result


def plot_outputs(state_dir, coordinates, triangles, damage, reference, report, curve):
    """Lead with the damage field and reaction curve, including status labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    mesh = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    damage_upper = max(float(np.max(damage)), float(np.max(reference)))
    error_upper = max(float(np.max(np.abs(damage - reference))), np.finfo(np.float64).eps)
    for ax, values, title in zip(axes, (damage, reference, damage - reference),
                                 ("PINN damage", "Direct comparison damage", "PINN minus reference")):
        error_plot = title == "PINN minus reference"
        artist = ax.tripcolor(mesh, values, shading="gouraud", cmap="coolwarm" if error_plot else "viridis",
                             vmin=-error_upper if error_plot else 0., vmax=error_upper if error_plot else damage_upper)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(artist, ax=ax)
    fig.suptitle(f"Load {report['load']:.6g}; {report['status']}; damage RMSE {report['damage_rmse']:.3g}")
    fig.savefig(state_dir / "damage_field.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot([0., *[row["load"] for row in curve]], [0., *[row["reference_reaction"] for row in curve]], "k.-", label="Direct comparison")
    ax.plot([0., *[row["load"] for row in curve]], [0., *[row["reaction"] for row in curve]], "o--", label="PINN research path")
    warning = [row for row in curve if row["status"] != "strict_checks_pass"]
    ax.scatter([row["load"] for row in warning], [row["reaction"] for row in warning], marker="x", color="red", label="Warning / rejected")
    ax.set(xlabel="Prescribed displacement", ylabel="Reaction")
    ax.legend()
    fig.savefig(state_dir.parent / "reaction_curve.png", dpi=160)
    plt.close(fig)


def run(c):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if c.train.dtype != "float64":
        raise ValueError("split PINN requires explicit FP64")
    targets = reference_indices(c.data)
    adaptive = "inner_stop" in c and c.inner_stop.enabled
    if "inner_stop" in c and type(c.inner_stop.enabled) is not bool:
        raise ValueError("inner_stop.enabled must be boolean")
    if adaptive:
        validate_controls(c.inner_stop, min(c.stagger.mechanics_steps, c.stagger.damage_steps))
        if c.stagger.accept_on_first_passing_report:
            raise ValueError("adaptive blocks require reference first-passage stopping disabled")
    if c.train.mechanics_objective not in ("residual", "energy"):
        raise ValueError("mechanics_objective must explicitly select residual or energy")
    # Older schema-v1 configs deliberately retain strict-only comparison behavior.
    reference_policy = c.data.reference_policy if "reference_policy" in c.data else "strict"
    if reference_policy not in ("strict", "user_accepted_warning"):
        raise ValueError("unknown PINN reference comparison policy")
    policy_sha = None
    if reference_policy == "user_accepted_warning":
        policy_sha = c.data.reference_policy_sha256
        accepted_warning_policy(c.data.reference_root, policy_sha)
    # Schema-v1 configs predate this optional implementation selector. Their
    # missing selector retains the original AD path and unchanged config bytes.
    operator_mode = c.train.operator_mode if "operator_mode" in c.train else "autodiff"
    if operator_mode not in ("autodiff", "analytic", "analytic_padded_damage"):
        raise ValueError("operator_mode must select autodiff, analytic or analytic_padded_damage")
    if operator_mode != "autodiff" and c.train.mechanics_objective != "energy":
        raise ValueError("analytic operator modes require the energy mechanics objective")
    for name, value in (("load_step", c.data.load_step), ("learning_rate", c.train.learning_rate),
                        ("damage_floor", c.physical.damage_floor)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite")
    for name, value in (("end_state_index", c.data.end_state_index), ("sweeps", c.stagger.sweeps),
                        ("mechanics_steps", c.stagger.mechanics_steps), ("damage_steps", c.stagger.damage_steps),
                        ("log_every", c.output.log_every)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(c.stagger.omega) or not 0 < c.stagger.omega <= 1:
        raise ValueError("neural Picard omega must lie in (0, 1]")
    if not isinstance(c.stagger.accept_on_first_passing_report, bool):
        raise ValueError("first-passage reporting control must be boolean")
    if "comparison_gate" in c.reporting:
        if type(c.reporting.comparison_gate) is not bool:
            raise ValueError("reporting.comparison_gate must be boolean")
        if not c.reporting.comparison_gate and c.stagger.accept_on_first_passing_report:
            raise ValueError("diagnostic reference comparisons require reference first-passage stopping disabled")
    if (isinstance(c.output.fork_after_index, bool) or int(c.output.fork_after_index) != c.output.fork_after_index
            or not 0 <= c.output.fork_after_index < c.data.end_state_index
            or bool(c.output.fork_from) != bool(c.output.fork_after_index)):
        raise ValueError("neural fork requires an explicit nonempty prefix before the endpoint")
    for name, value in c.reporting.items():
        if name == "comparison_gate":
            continue
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"reporting.{name} must be nonnegative finite")
    root = Path(c.output.dir)
    if not root.parent.is_dir():
        raise FileNotFoundError(root.parent)
    if not c.output.resume:
        root.mkdir()  # Never overwrite a previous run.
    elif not root.is_dir():
        raise FileNotFoundError(root)
    values = json.loads(c.to_json_best_effort())
    values["output"]["resume"] = False
    identity = {"schema": SCHEMA, "config": values}
    if c.output.resume:
        stored_identity = json.loads((root / "identity.json").read_text())
        resume_identity(stored_identity, identity)
        if stored_identity != identity:
            atomic_json(root / "identity.json", identity)
    else:
        atomic_json(root / "identity.json", identity)
    build_start = time.perf_counter()
    model = build_model(read_cloud(c.data.cloud, c.data.cloud_sha256),
        delta_u=c.nl.delta_u, delta_d=c.nl.delta_d, grip_extent=c.geometry.grip_extent,
        h_grip=c.geometry.h_grip, visibility_tolerance=c.nl.visibility_tolerance,
        E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
        rank_rtol=c.physical.rank_rtol, Gc=c.physical.Gc, l0=c.physical.l0)
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("split PINN training requires GPU")
    backend = FP64A2Backend(model.static, model.constraints, device)
    residual = FractureResidual(model, backend)
    energy_function = backend._energy_from_factor
    damage_matvec = residual.operator.deterministic_matvec
    if operator_mode != "autodiff":
        from examples.fracture2d._legacy.split_pinn_gradients import make_analytic_energy, make_analytic_matvec, make_padded_matvec
        energy_function = make_analytic_energy(backend)
        if operator_mode == "analytic":
            damage_matvec = make_analytic_matvec(residual.operator)
        else:
            damage_matvec, layout = make_padded_matvec(residual.operator)
            print(json.dumps({"operator_mode": operator_mode, "damage_layout": layout}), flush=True)
    model_build_seconds = time.perf_counter() - build_start
    points = jax.device_put(model.geometry.coordinates, device)
    q_net = jax.tree.map(lambda x: x.astype(jnp.float64), _build_network(c, jax.random.PRNGKey(c.train.seed), out_dim=2))
    d_net = jax.tree.map(lambda x: x.astype(jnp.float64), _build_network(c, jax.random.PRNGKey(c.train.seed + 1), out_dim=1))
    optimizer = optax.adam(c.train.learning_rate)
    q_state, d_state = (optimizer.init(eqx.filter(net, eqx.is_inexact_array)) for net in (q_net, d_net))
    d_parent = jnp.zeros(model.n_material, dtype=jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    q_parent = jnp.asarray(model.constraints.prescribed_vector(0.))
    parent_sha, inherited_warning, curve = None, False, []
    gc0, gc1 = model.Gc / model.l0, model.Gc * model.l0
    union_volume = float(np.sum(model.geometry.union_volumes))

    def displacement(net, load):
        raw = jax.vmap(net)(points)
        base = jnp.stack((jnp.zeros_like(points[:, 1]), load * points[:, 1]), axis=1)
        return residual.lift((base + load * raw).reshape(-1), load)

    def damage(net, load):
        raw = c.physical.E / gc0 * load**2 * jax.vmap(net)(points[:model.n_material])[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)

    def rho(d, H):
        return gc0 * d + gc1 * damage_matvec(d) + 2 * H * (d - 1)

    @eqx.filter_jit
    def mechanics_step(net, state, d, load, scale):
        def objective(n):
            factor = backend._edge_factor(jax.lax.stop_gradient(d))
            q = displacement(n, load)
            if c.train.mechanics_objective == "energy":
                energy_scale = c.physical.E * load**2 * union_volume
                return energy_function(factor, q) / energy_scale
            r = backend._residual_from_factor(factor, q)
            return jnp.mean((r[backend.free_dofs] / scale)**2)
        loss, grads = eqx.filter_value_and_grad(objective)(net)
        updates, state = optimizer.update(grads, state, net)
        return eqx.apply_updates(net, updates), state, loss

    @eqx.filter_jit
    def damage_step(net, state, H, load, scale):
        def objective(n):
            return jnp.mean((rho(damage(n, load), jax.lax.stop_gradient(H)) / scale)**2)
        loss, grads = eqx.filter_value_and_grad(objective)(net)
        updates, state = optimizer.update(grads, state, net)
        return eqx.apply_updates(net, updates), state, loss

    @eqx.filter_jit
    def mechanics_monitor(net, d, load, scale):
        q = displacement(net, load)
        factor = backend._edge_factor(d)
        evaluation = backend._evaluate_from_factor(factor, q)
        objective = energy_function(factor, q) / (c.physical.E * load**2 * union_volume)
        norm = jnp.mean((evaluation["residual"][backend.free_dofs] / scale)**2)
        return q, objective, norm, evaluation["top_reaction"]

    @eqx.filter_jit
    def damage_monitor(net, H, load, H_scale):
        image = damage(net, load)
        terms = (gc0*image, gc1*damage_matvec(image), 2*H*(image-1))
        defect = sum(terms)
        term_scale = sum(jnp.max(jnp.abs(t)) for t in terms)
        return image, jnp.mean((defect/H_scale)**2), jnp.max(jnp.abs(defect))/term_scale, term_scale

    first_load = jnp.asarray(c.data.load_step, dtype=jnp.float64)
    affine = displacement(q_net, first_load)
    first_scale = jnp.sqrt(jnp.mean(backend._residual_from_factor(backend._edge_factor(d_parent), affine)[backend.free_dofs]**2))
    if not np.isfinite(float(first_scale)) or float(first_scale) <= 0:
        raise ValueError("first-load mechanics scale must be positive finite")
    event_dir = Path(c.output.event_dir) if "event_dir" in c.output else root / "events"
    writer = SummaryWriter(logdir=str(event_dir))
    try:
        for index, reference_index in enumerate(targets, 1):
            load = jnp.asarray(reference_index * c.data.load_step, dtype=jnp.float64)
            q_ref, d_ref, reaction_ref, reference_sha, reference_metadata = comparison_reference(c.data.reference_root,
                reference_index, model.model_digest, model.n_dofs, model.n_material, float(load),
                reference_policy=reference_policy, policy_sha256=policy_sha, return_metadata=True)
            state_dir = root / f"state_{index:04d}"
            inherited = bool(c.output.fork_from) and index <= c.output.fork_after_index
            if inherited:
                state_dir = neural_state_root(c.output.fork_from, index)
            state_identity = {"model_digest": model.model_digest, "config": values,
                               "reference_sha256": reference_sha, "parent_sha256": parent_sha, "index": index}
            if "reference_policy" in c.data:
                state_identity["reference_provenance"] = reference_metadata
            if "reference_indices" in c.data:
                state_identity.update(reference_index=reference_index, load=float(load))
            template = (q_net, q_state, d_net, d_state, q_parent, d_parent, H_parent)
            if (state_dir / "result.json").exists():
                report = json.loads((state_dir / "result.json").read_text())
                if not report["research_advance"]:
                    raise ValueError("completed rejected PINN load requires an explicit new trial root")
                if "reference_policy" in c.data:
                    if (report["reference_provenance"] != reference_metadata
                            or reference_metadata["status"] == "warning" and report["strict_accepted"]):
                        raise ValueError("restored PINN reference provenance/status differs")
                checkpoint_identity = (fork_checkpoint_identity(state_dir / "state.zip", state_identity) if inherited
                                       else checkpoint_resume_identity(state_dir / "state.zip", state_identity))
                _, template = read_checkpoint(state_dir / "state.zip", checkpoint_identity, template, schema=SCHEMA)
                q_net, q_state, d_net, d_state, q_parent, d_parent, H_parent = template
                for name, array, shape in (("q", q_parent, (model.n_dofs,)),
                                           ("d", d_parent, (model.n_material,)),
                                           ("H", H_parent, (model.n_material,))):
                    host = np.asarray(array)
                    if host.shape != shape or host.dtype != np.float64 or not np.all(np.isfinite(host)):
                        raise ValueError(f"restored neural {name} must be finite FP64 of model shape")
                fixed = model.constraints.constrained_dofs
                if (not np.array_equal(np.asarray(q_parent)[fixed], model.constraints.prescribed_vector(float(load))[fixed])
                        or np.min(np.asarray(H_parent)) < 0):
                    raise ValueError("restored neural parent has invalid boundary/history fields")
                parent_sha = hashlib.sha256((state_dir / "state.zip").read_bytes()).hexdigest()
                inherited_warning |= not report["strict_accepted"]
                curve.append(report)
                continue
            if inherited:
                raise ValueError("split neural fork prefix contains an unfinished load")
            load_start = time.perf_counter()
            phase_seconds = {"before": 0., "damage": 0., "after": 0.}
            phase_updates = {"before": 0, "damage": 0, "after": 0}
            d_iter = d_parent
            d_previous = d_parent
            H_used = H_parent
            first_sweep, first_phase = 1, "before"
            if state_dir.exists():
                if not c.output.resume:
                    raise FileExistsError(state_dir)
                progress_template = (q_net, q_state, d_net, d_state, d_iter, d_previous, H_used)
                checkpoint_identity = checkpoint_resume_identity(state_dir / "progress.zip", state_identity)
                metadata, restored = read_checkpoint(state_dir / "progress.zip", checkpoint_identity,
                                                       progress_template, schema=SCHEMA)
                q_net, q_state, d_net, d_state, d_iter, d_previous, H_used = restored
                first_sweep, first_phase = metadata["sweep"], metadata["phase"]
                if first_sweep > c.stagger.sweeps or first_phase == "complete":
                    raise ValueError("invalid split PINN unfinished-load progress")
            else:
                state_dir.mkdir()
                write_checkpoint(state_dir / "progress.zip", state_identity, 1, "before", 0,
                    (q_net, q_state, d_net, d_state, d_iter, d_previous, H_used), schema=SCHEMA)
            scale = first_scale * reference_index
            report, q, fields = None, None, None
            sweep = 0
            for sweep in range(first_sweep, c.stagger.sweeps + 1):
                stop_step = c.stagger.mechanics_steps
                block_stops = []
                for phase in ("before", "damage", "after"):
                    if sweep == first_sweep and ("before", "damage", "after").index(phase) < ("before", "damage", "after").index(first_phase):
                        if adaptive:
                            block_stops.append(json.loads((state_dir / f"inner_stop_{sweep:04d}_{phase}.json").read_text()))
                        continue
                    count = c.stagger.damage_steps if phase == "damage" else c.stagger.mechanics_steps
                    phase_start = time.perf_counter()
                    reason, monitor_count, last_monitor, previous_field = "budget_exhausted", 0, None, None
                    window = None
                    if adaptive:
                        minimum = c.inner_stop.first_load_min_updates if parent_sha is None else c.inner_stop.min_updates
                        window = StopWindow(c.inner_stop, phase, minimum)
                    H_scale = None
                    actual_updates = 0
                    if phase == "damage":
                        H_used = residual.evaluate(displacement(q_net, load), d_iter, None, H_parent).trial_history
                        H_scale = 2 * jnp.sqrt(jnp.mean(H_used**2))
                        if not np.isfinite(float(H_scale)) or float(H_scale) <= 0:
                            raise ValueError("fixed-history damage scale must be positive finite")
                    for step in range(1, count + 1):
                        actual_updates = step
                        stop_inner = False
                        phase_updates[phase] += 1
                        if phase == "damage":
                            if H_scale is None:
                                raise RuntimeError("damage block requires a fixed-history scale")
                            d_net, d_state, loss = damage_step(d_net, d_state, H_used, load, H_scale)
                        else:
                            q_net, q_state, loss = mechanics_step(q_net, q_state, d_iter, load, scale)
                        if adaptive and (step % c.inner_stop.check_every == 0 or step == count):
                            if phase == "damage":
                                current, objective, defect, term_scale = damage_monitor(d_net, H_used, load, H_scale)
                                if not np.isfinite(float(term_scale)) or float(term_scale) <= 0:
                                    raise FloatingPointError("inner AT2 term scale must be positive finite")
                                change = None if previous_field is None else float(jnp.max(jnp.abs(current-previous_field)))
                                reaction = 0.
                                physical = bool(jnp.all(jnp.isfinite(current)) & (jnp.min(current) >= 0) & (jnp.max(current) <= 1))
                            else:
                                current, objective, defect, reaction = mechanics_monitor(q_net, d_iter, load, scale)
                                change = None if previous_field is None else float(jnp.max(jnp.abs(current-previous_field))/load)
                                physical = bool(jnp.all(jnp.isfinite(current)))
                            last_monitor = dict(step=step, objective=float(objective), residual=float(defect),
                                                field_change=change, reaction=float(reaction))
                            if window is None:
                                raise RuntimeError("adaptive block lacks its stopping window")
                            reason_checked = window.observe(step, **{k:last_monitor[k] for k in ("objective", "residual", "field_change", "reaction")}, physical_pass=physical)
                            previous_field = current
                            monitor_count += 1
                            if reason_checked is not None:
                                reason = reason_checked
                                if phase == "after":
                                    stop_step = step
                                stop_inner = True
                        if step % c.output.log_every == 0 or step == count or stop_inner:
                            if not np.isfinite(float(loss)):
                                raise FloatingPointError("split PINN training loss is nonfinite")
                            print(json.dumps({"index": index, "sweep": sweep, "phase": phase,
                                              "step": step, "loss_before_update": float(loss)}), flush=True)
                            offset = (index - 1) * c.stagger.sweeps + sweep - 1
                            block_step = offset * count + step
                            writer.add_scalar(f"loss/{phase}", float(loss), block_step)
                            writer.flush()
                            if (phase == "after" and sweep == c.stagger.sweeps
                                    and c.stagger.accept_on_first_passing_report):
                                check = residual.evaluate(displacement(q_net, load), d_iter, None, H_parent)
                                reaction_error = abs(float(check.reaction) - reaction_ref) / max(abs(reaction_ref), 1e-30)
                                damage_error = float(jnp.sqrt(jnp.mean((d_iter - jnp.asarray(d_ref))**2)))
                                if (reaction_error <= c.reporting.reaction_relative_limit
                                        and damage_error <= c.reporting.damage_rmse_limit):
                                    stop_step = step
                                    reason = "reference_first_pass"
                                    break
                        if stop_inner:
                            break
                    if adaptive:
                        if actual_updates == 0:
                            raise RuntimeError("adaptive block executed no updates")
                        block = dict(sweep=sweep, phase=phase, updates=actual_updates, reason=reason,
                                     checkpoints=monitor_count, last=last_monitor,
                                     fixed_history=phase == "damage", monitors_unrelaxed_image=phase == "damage")
                        block_stops.append(block)
                        atomic_json(state_dir / f"inner_stop_{sweep:04d}_{phase}.json", block)
                    if phase == "damage":
                        d_iter = d_iter + c.stagger.omega * (damage(d_net, load) - d_iter)
                    if phase != "after":
                        next_phase = "damage" if phase == "before" else "after"
                        write_checkpoint(state_dir / "progress.zip", state_identity, sweep, next_phase, 0,
                            (q_net, q_state, d_net, d_state, d_iter, d_previous, H_used), schema=SCHEMA)
                    jax.block_until_ready(eqx.filter((q_net, q_state, d_net, d_state), eqx.is_array))
                    phase_seconds[phase] += time.perf_counter() - phase_start
                q = displacement(q_net, load)
                fields = residual.evaluate(q, d_iter, None, H_parent)
                q_np, d_np, H_np = map(np.asarray, (q, d_iter, fields.trial_history))
                fixed = model.constraints.constrained_dofs
                if not np.array_equal(q_np[fixed], model.constraints.prescribed_vector(float(load))[fixed]):
                    raise RuntimeError("split PINN exact boundary constraints violated")
                parts = (gc0 * d_iter, gc1 * residual.operator.deterministic_matvec(d_iter), 2 * fields.trial_history * (d_iter - 1))
                at2_scale = sum(float(jnp.max(jnp.abs(part))) for part in parts)
                if not np.isfinite(at2_scale) or at2_scale <= 0:
                    raise ValueError("coupled AT2 diagnostic scale must be positive finite")
                report = classify_report({"index": index, "load": float(load), "sweep": sweep,
                    "final_mechanics_step": stop_step,
                    "reaction": float(fields.reaction), "reference_reaction": reaction_ref,
                    "reaction_relative": abs(float(fields.reaction) - reaction_ref) / max(abs(reaction_ref), 1e-30),
                    "damage_rmse": float(np.sqrt(np.mean((d_np - d_ref)**2))),
                    "damage_max_error": float(np.max(abs(d_np - d_ref))),
                    "damage_min": float(min(d_np)), "damage_max": float(max(d_np)),
                    "damage_inf": float(jnp.max(jnp.abs(fields.damage))),
                    "coupled_AT2_relative": float(jnp.max(jnp.abs(fields.damage))) / at2_scale,
                    "mechanics_loss": float(jnp.mean((fields.mechanics / scale)**2)),
                    "mechanics_inf": float(jnp.max(jnp.abs(fields.mechanics))),
                    "displacement_relative": float(np.linalg.norm(q_np - q_ref) / np.linalg.norm(q_ref)),
                    "damage_change": float(jnp.max(jnp.abs(d_iter - (d_parent if sweep == 1 else d_previous)))),
                    "healing": float(jnp.max(jnp.maximum(d_parent - d_iter, 0))),
                    "history_decrease": float(jnp.max(jnp.maximum(H_parent - fields.trial_history, 0))),
                    "drift_count": int(np.count_nonzero(np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)
                                                        / float(load) > direct.DRIFT_ENVELOPE))},
                    c.reporting, inherited_warning=inherited_warning or reference_metadata["status"] == "warning")
                if "reference_policy" in c.data:
                    report["reference_provenance"] = reference_metadata
                if "reference_indices" in c.data:
                    previous_load = 0. if index == 1 else targets[index-2] * c.data.load_step
                    report.update(reference_index=reference_index, load_increment=float(load)-previous_load)
                if adaptive:
                    inner_warning = any(b["reason"] == "stagnation_warning" for b in block_stops)
                    report.update(inner_stops=block_stops, inner_stagnation_warning=inner_warning)
                    if inner_warning and report["strict_accepted"]:
                        report.update(strict_accepted=False, status="warning")
                d_previous = d_iter
                print(json.dumps(report), flush=True)
                atomic_json(state_dir / f"sweep_{sweep:04d}.json", report)
                np.savez_compressed(state_dir / f"fields_{sweep:04d}.npz", q=q_np, d=d_np, H=H_np,
                                    d_reference=d_ref, coordinates=model.geometry.material_reference_coordinates)
                position = (index - 1) * c.stagger.sweeps + sweep
                for name in ("reaction", "reference_reaction", "damage_rmse", "damage_max_error"):
                    writer.add_scalar(f"eval/{name}", report[name], position)
                for name, check in report["checks"].items():
                    writer.add_scalar(f"diag/{name}", check["value"], position)
                writer.flush()
                if sweep < c.stagger.sweeps:
                    write_checkpoint(state_dir / "progress.zip", state_identity, sweep + 1, "before", 0,
                        (q_net, q_state, d_net, d_state, d_iter, d_previous, H_used), schema=SCHEMA)
            if report is None or q is None or fields is None:
                raise RuntimeError("split PINN load did not complete a sweep")
            state = (q_net, q_state, d_net, d_state, q, d_iter, fields.trial_history)
            write_checkpoint(state_dir / "state.zip", state_identity, sweep, "complete", 0, state, schema=SCHEMA)
            atomic_json(state_dir / "result.json", report)
            curve.append(report)
            atomic_json(root / "reaction_curve.json", curve)
            plot_outputs(state_dir, model.geometry.material_reference_coordinates,
                         model.geometry.material_reference_triangles, np.asarray(d_iter), d_ref, report, curve)
            atomic_json(state_dir / "timing.json", {
                "scope": "work in this invocation; phase times include logging/checkpoint overhead",
                "operator_mode": operator_mode, "model_build_seconds": model_build_seconds,
                "load_wall_seconds": time.perf_counter() - load_start,
                "phase_seconds": phase_seconds, "phase_updates": phase_updates})
            if not report["research_advance"]:
                break
            parent_sha = hashlib.sha256((state_dir / "state.zip").read_bytes()).hexdigest()
            q_parent, d_parent, H_parent = state[4:]
            inherited_warning |= not report["strict_accepted"]
    finally:
        writer.close()
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mechanics-steps", type=int)
    parser.add_argument("--damage-steps", type=int)
    parser.add_argument("--sweeps", type=int)
    parser.add_argument("--last-index", type=int)
    parser.add_argument("--mechanics-objective", choices=("residual", "energy"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fork-from", type=Path)
    parser.add_argument("--fork-after-index", type=int)
    args = parser.parse_args()
    c = runpy.run_path(str(args.config))["get_config"]()
    if args.output is not None:
        c.output.dir = str(args.output)
    for arg, field in ((args.mechanics_steps, "mechanics_steps"),
                       (args.damage_steps, "damage_steps"), (args.sweeps, "sweeps")):
        if arg is not None:
            c.stagger[field] = arg
    if args.last_index is not None:
        c.data.end_state_index = args.last_index
    if args.mechanics_objective is not None:
        c.train.mechanics_objective = args.mechanics_objective
    if args.resume:
        c.output.resume = True
    if args.fork_from is not None:
        c.output.fork_from = str(args.fork_from)
    if args.fork_after_index is not None:
        c.output.fork_after_index = args.fork_after_index
    run(c)


if __name__ == "__main__":
    main()
