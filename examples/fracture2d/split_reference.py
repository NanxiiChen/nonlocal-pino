"""Strict new-model reference loading; array solves are never PINN targets.

Fixed-H AT2 uses an explicitly checked Jacobi-PCG solve of the unchanged
volume-multiplied SPD equation. Mechanics uses the adopted FP64 A2 backend.
Only coupled, finite, bounded, monotone, non-drifting states advance a load.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.anderson import safeguarded_step
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.split_horizon_model import build_model, read_cloud
from examples.fracture2d.trial_storage import FIELDS as STORAGE_FIELDS, prune_trial_fields, resolve_storage_policy


def damage_pcg(D, volumes, gc0, gc1, H, initial, *, rtol, atol, maxiter):
    """Same fixed-H AT2 equation, with no clipping or fallback solver."""
    if (H.shape != volumes.shape or initial.shape != H.shape or not np.all(np.isfinite(H))
            or min(H) < 0 or min(volumes) <= 0 or min(gc0, gc1) <= 0):
        raise ValueError("fixed-H PCG physical/shape validation failed")
    diagonal = volumes * (gc0 + 2 * H + gc1 * D.diagonal())
    if min(diagonal) <= 0 or not np.all(np.isfinite(diagonal)):
        raise ValueError("fixed-H PCG diagonal is not strictly positive")
    n = len(H)
    A = LinearOperator((n, n), matvec=lambda d: volumes * ((gc0 + 2 * H) * d + gc1 * (D @ d)), dtype=np.float64)
    M = LinearOperator((n, n), matvec=lambda r: r / diagonal, dtype=np.float64)
    rhs = 2 * volumes * H
    iterations = [0]
    def count(_):
        iterations[0] += 1
    d, info = cg(A, rhs, x0=initial, rtol=rtol, atol=atol, maxiter=maxiter, M=M, callback=count)
    rho = gc0 * d + gc1 * (D @ d) + 2 * H * (d - 1)
    explicit_norm = np.linalg.norm(volumes * rho)
    limit = max(atol, rtol * np.linalg.norm(rhs))
    if info != 0 or not np.all(np.isfinite(d)) or explicit_norm > limit * (1 + 1e-3):
        raise RuntimeError(f"fixed-H PCG failed explicit check: info={info}, residual={explicit_norm}, limit={limit}")
    return d, {"iterations": iterations[0], "volume_residual_norm": float(explicit_norm),
               "strong_residual_inf": float(max(abs(rho))), "limit": float(limit)}


def make_jax_damage_pcg(operator, volumes, gc0, gc1, *, rtol, atol, maxiter):
    """Reuse checked volume-SPD CG; host residual is an independent audit only."""
    from examples.fracture2d.at2_cg import damage_solver

    solve = damage_solver(operator, volumes, gc0, gc1, tol=rtol, atol=atol, maxiter=maxiter)
    D = operator.to_scipy().tocsr()

    def checked(H, initial):
        start = time.monotonic()
        d = np.asarray(solve(jnp.asarray(H), jnp.asarray(initial)))
        rho = gc0 * d + gc1 * (D @ d) + 2 * H * (d - 1)
        norm = float(np.linalg.norm(volumes * rho))
        limit = max(atol, rtol * float(np.linalg.norm(2 * volumes * H)))
        if not np.isfinite(norm) or norm > limit * (1 + 1e-3):
            raise RuntimeError(f"JAX AT2 CG failed independent host residual: {norm} > {limit}")
        return d, {"iterations": None, "iteration_count_available": False,
                   "backend": "jax_cg", "volume_residual_norm": norm,
                   "strong_residual_inf": float(max(abs(rho))), "limit": limit,
                   "seconds": time.monotonic() - start}

    return checked


def reference_state_root(root, index):
    """Resolve a read-only forked prefix without copying accepted arrays."""
    root = Path(root)
    visited = set()
    while True:
        if root.resolve() in visited:
            raise ValueError("split reference fork ancestry is cyclic")
        visited.add(root.resolve())
        policy = json.loads((root / "policy.json").read_text())
        if "fork_from" in policy and policy["fork_from"] is not None and index <= policy["fork_after_index"]:
            root = Path(policy["fork_from"])
        else:
            return root / f"state_{index:04d}"


def read_trial_seed(state_dir, *, model_digest, parent_sha256, load, n_dofs, n_material):
    """An uncommitted trial is an initial guess, never a history lower bound."""
    state_dir = Path(state_dir)
    progress = json.loads((state_dir / "progress.json").read_text())
    path = trial_archive_path(state_dir, progress)
    if (progress["model_digest"] != model_digest
            or progress["parent_sha256"] != parent_sha256
            or abs(progress["load"] - load) > 1e-12
            or hashlib.sha256(path.read_bytes()).hexdigest() != progress["trial_sha256"]):
        raise ValueError("trial seed model/load/parent/digest differs")
    with np.load(path, allow_pickle=False) as archive:
        q, d = (archive[name] for name in ("q", "d"))
    if (q.shape != (n_dofs,) or d.shape != (n_material,)
            or q.dtype != np.float64 or d.dtype != np.float64
            or not all(np.all(np.isfinite(x)) for x in (q, d))
            or min(d) < 0 or max(d) > 1):
        raise ValueError("trial seed must contain finite bounded FP64 fields of model shape")
    return q, d


def trial_archive_path(state_dir, progress):
    """Legacy fixed-name trials remain readable; new progress pins a generation."""
    name = progress["trial_file"] if "trial_file" in progress else "trial.npz"
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
        raise ValueError("reference trial filename must be a local NPZ basename")
    return Path(state_dir) / name


def write_trial_fields(state_dir, sweep, *, q, d, H, extra_fields=None):
    """Publish immutable fields before atomically updating the progress pointer.

    Replacing a fixed trial.npz before its progress.json could invalidate the
    previous resume point on interruption. A new generation leaves it intact.
    """
    path = Path(state_dir) / f"trial_{sweep:04d}.npz"
    temporary = path.with_suffix(".npz.tmp")
    fields = {"q": q, "d": d, "H": H}
    if extra_fields is not None:
        if set(fields).intersection(extra_fields):
            raise ValueError("extra trial fields cannot replace physical q/d/H")
        fields.update(extra_fields)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)
    return {"trial_file": path.name, "trial_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args):
    storage_policy = resolve_storage_policy(args)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    damage_backend = args.damage_backend if "damage_backend" in vars(args) else "cpu"
    if damage_backend not in ("cpu", "gpu"):
        raise ValueError("damage backend must be cpu or gpu")
    if damage_backend == "gpu" and jax.devices()[0].platform != "gpu":
        raise RuntimeError("GPU damage loading requires a visible JAX GPU")
    # Prior schema-v1 programmatic callers have no optional relaxation selector.
    method = args.relaxation_policy if "relaxation_policy" in vars(args) else "picard"
    if method not in ("picard", "anderson_residual"):
        raise ValueError("reference relaxation must be picard or anderson_residual")
    if method == "anderson_residual":
        if (args.anderson_depth < 1 or args.anderson_max_backtracks < 0
                or not 0 < args.anderson_sufficient_decrease < 1):
            raise ValueError("invalid explicit Anderson numerical controls")
    restart_policy = args.anderson_restart_policy if "anderson_restart_policy" in vars(args) else "reject_or_rank"
    rank_strategy = args.anderson_rank_strategy if "anderson_rank_strategy" in vars(args) else "restart"
    build_start = time.monotonic()
    config = runpy.run_path(str(args.config))["get_config"]()
    model = build_model(read_cloud(config.data.cloud, config.data.cloud_sha256),
        delta_u=config.nl.delta_u, delta_d=config.nl.delta_d, grip_extent=config.geometry.grip_extent,
        h_grip=config.geometry.h_grip, visibility_tolerance=config.nl.visibility_tolerance,
        E=config.physical.E, nu=config.physical.nu, k_res=config.physical.k_res,
        rank_rtol=config.physical.rank_rtol, Gc=config.physical.Gc, l0=config.physical.l0)
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    D = model.material_operator.to_scipy().tocsr()
    volumes = np.asarray(model.geometry.material_reference_volumes)
    gc0, gc1 = model.Gc / model.l0, model.Gc * model.l0
    if damage_backend == "gpu":
        solve_damage = make_jax_damage_pcg(model.material_operator, volumes, gc0, gc1,
                                         rtol=1e-11, atol=1e-14, maxiter=2000)
    else:
        def solve_damage(H, initial):
            return damage_pcg(D, volumes, gc0, gc1, H, initial,
                              rtol=1e-11, atol=1e-14, maxiter=2000)
    build_seconds = time.monotonic() - build_start
    print(json.dumps({"event": "reference_model_ready", "model_build_seconds": build_seconds,
                      "damage_backend": damage_backend, "model_digest": model.model_digest}), flush=True)
    root = args.output
    if not root.parent.is_dir():
        raise FileNotFoundError(root.parent)
    policy = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for name in STORAGE_FIELDS:
        if name in policy:
            del policy[name]
    if storage_policy is not None:
        policy.update(storage_policy)
    if damage_backend == "cpu" and "damage_backend" in policy:
        del policy["damage_backend"]  # Preserve legacy CPU policy identities.
    if method == "picard":
        for key in ("relaxation_policy", "anderson_depth", "anderson_max_backtracks", "anderson_sufficient_decrease",
                    "anderson_restart_policy", "anderson_rank_strategy"):
            if key in policy:
                del policy[key]
    elif restart_policy == "reject_or_rank" and "anderson_restart_policy" in policy:
        del policy["anderson_restart_policy"]  # Preserve earlier method-v1 identities.
    if rank_strategy == "restart" and "anderson_rank_strategy" in policy:
        del policy["anderson_rank_strategy"]
    del policy["omega"]
    if args.trial_from is None:
        del policy["trial_from"]
    else:
        seed_progress = json.loads((args.trial_from / "progress.json").read_text())
        policy["trial_seed_sha256"] = seed_progress["trial_sha256"]
        policy["trial_seed_load"] = seed_progress["load"]
    policy["resume"] = False
    if args.fork_from is None:
        del policy["fork_from"]
        del policy["fork_after_index"]
    policy.update({"schema": "fracture2d-split-reference-v1", "model_digest": model.model_digest,
                   "cloud_sha256": config.data.cloud_sha256, "model_config": json.loads(config.to_json_best_effort()),
                    "omega": args.omega, "mechanics_cg_rtol": config.validation.cg_rtol,
                   "mechanics_cg_atol": config.validation.cg_atol, "mechanics_cg_maxiter": config.validation.cg_maxiter,
                   "damage_cg_rtol": 1e-11, "damage_cg_atol": 1e-14, "damage_cg_maxiter": 2000,
                   "damage_change_atol": 1e-5, "q_change_relative_limit": 1e-5,
                   "H_change_relative_limit": 1e-5, "reaction_change_relative_limit": 1e-5,
                   "coupled_AT2_relative_inf_limit": 1e-6,
                   "healing_atol": 1e-3, "range_atol": 1e-10,
                   "consecutive_passing_sweeps": 3, "reference_only": True})
    if args.fork_from is not None:
        old = json.loads((args.fork_from / "policy.json").read_text())
        current = policy.copy()
        for key in ("output", "last_load", "max_sweeps", "fork_from", "fork_after_index",
                    "omega", "trial_from", "trial_seed_sha256", "trial_seed_load", *STORAGE_FIELDS):
            if key in old:
                del old[key]
            if key in current:
                del current[key]
        for key in ("relaxation_policy", "anderson_depth", "anderson_max_backtracks", "anderson_sufficient_decrease",
                    "anderson_restart_policy", "anderson_rank_strategy"):
            if key in old:
                del old[key]
            if key in current:
                del current[key]
        if old != current or args.fork_after_index < 1:
            raise ValueError("reference fork must preserve model, load step and all acceptance thresholds")
    if args.resume:
        stored = json.loads((root / "policy.json").read_text())
        previous_endpoint = stored["last_load"]
        stored["last_load"] = policy["last_load"]
        if stored != policy or args.last_load < previous_endpoint:
            raise ValueError("split reference resume differs from declared policy/model")
        if args.last_load > previous_endpoint:
            atomic_json(root / "policy.json", policy)
    else:
        root.mkdir()
        atomic_json(root / "policy.json", policy)
    q_comm = model.constraints.prescribed_vector(0.0)
    d_comm, H_comm = np.zeros(model.n_material), np.zeros(model.n_material)
    parent_digest = None
    curve = []
    loads = np.arange(args.load_step, args.last_load + args.load_step * 0.5, args.load_step)
    for index, load in enumerate(loads, 1):
        load_start = time.monotonic()
        state_dir = root / f"state_{index:04d}"
        inherited = args.fork_from is not None and index <= args.fork_after_index
        if inherited:
            state_dir = reference_state_root(args.fork_from, index)
        if state_dir.exists() and (state_dir / "committed.npz").is_file():
            report = json.loads((state_dir / "result.json").read_text())
            if (not report["accepted_reference"] or report["model_digest"] != model.model_digest
                    or report["parent_sha256"] != parent_digest or abs(report["load"] - load) > 1e-12):
                raise ValueError("split reference committed load/parent differs")
            digest = hashlib.sha256((state_dir / "committed.npz").read_bytes()).hexdigest()
            if digest != report["committed_sha256"]:
                raise ValueError("split reference committed field digest differs")
            with np.load(state_dir / "committed.npz", allow_pickle=False) as archive:
                q_comm, d_comm, H_comm = (archive[name] for name in ("q", "d", "H"))
            parent_digest = digest
            curve.append(report)
            continue
        if inherited:
            raise ValueError("reference fork prefix contains an uncommitted load")
        if state_dir.exists() and (state_dir / "result.json").exists():
            raise ValueError("completed rejected reference load requires an explicit retry branch")
        d_iter = d_comm.copy()
        q_guess = q_comm.copy()
        fixed = model.constraints.constrained_dofs
        q_guess[fixed] = model.constraints.prescribed_vector(float(load))[fixed]
        previous_q, previous_H, previous_reaction = None, None, None
        consecutive = 0
        start_sweep = 1
        history_x, history_f = [], []
        if state_dir.exists():
            progress = json.loads((state_dir / "progress.json").read_text())
            if (progress["model_digest"] != model.model_digest or progress["parent_sha256"] != parent_digest
                    or abs(progress["load"] - load) > 1e-12):
                raise ValueError("split reference progress committed context differs")
            trial_path = trial_archive_path(state_dir, progress)
            if hashlib.sha256(trial_path.read_bytes()).hexdigest() != progress["trial_sha256"]:
                raise ValueError("split reference progress field digest differs")
            with np.load(trial_path, allow_pickle=False) as archive:
                q_guess, d_iter, previous_H = (archive[name] for name in ("q", "d", "H"))
                if method == "anderson_residual":
                    hx, hf = archive["anderson_x"], archive["anderson_f"]
                    count = progress["anderson_history_count"]
                    if (hx.shape != (count, model.n_material) or hf.shape != hx.shape
                            or count > args.anderson_depth or hx.dtype != np.float64 or hf.dtype != np.float64
                            or not np.all(np.isfinite(hx)) or not np.all(np.isfinite(hf))):
                        raise ValueError("reference Anderson progress history is malformed")
                    history_x, history_f = list(hx), list(hf)
            previous_q = q_guess.copy()
            previous_reaction = progress["reaction"]
            consecutive = progress["consecutive"]
            start_sweep = progress["sweep"] + 1
        else:
            state_dir.mkdir()
            if args.trial_from is not None and index == args.fork_after_index + 1:
                q_guess, d_iter = read_trial_seed(args.trial_from,
                    model_digest=model.model_digest, parent_sha256=parent_digest,
                    load=float(load), n_dofs=model.n_dofs, n_material=model.n_material)
        if start_sweep > args.max_sweeps:
            raise ValueError("reference progress exhausted its budget without a committed result; use an explicit retry branch")
        def bounds_pass(candidate):
            return (np.all(np.isfinite(candidate)) and min(candidate) >= -1e-10 and max(candidate) <= 1.
                    and max(np.maximum(d_comm - candidate, 0)) <= 1e-3)

        def evaluate_map(candidate):
            solved = backend.solve_strict(candidate, float(load), config.validation.cg_rtol,
                config.validation.cg_atol, config.validation.cg_maxiter, initial_guess=q_guess)
            fields_at_candidate = residual.evaluate(solved.displacement, jnp.asarray(candidate), None, jnp.asarray(H_comm))
            candidate_H = np.asarray(fields_at_candidate.trial_history)
            image, solve_info = solve_damage(candidate_H, candidate)
            payload = {"solve": solved, "fields": fields_at_candidate, "d": candidate,
                       "H": candidate_H, "damage_info": solve_info}
            return image, payload

        def physics_pass(payload):
            q_candidate = np.asarray(payload["solve"].displacement)
            ratios = np.linalg.norm(q_candidate[:2 * model.n_material].reshape(-1, 2), axis=1) / load
            return (bounds_pass(payload["d"]) and np.all(np.isfinite(q_candidate))
                    and np.all(np.isfinite(payload["H"])) and min(payload["H"] - H_comm) >= 0
                    and max(ratios) <= direct.DRIFT_ENVELOPE)

        if method == "anderson_residual":
            current_image, _ = evaluate_map(d_iter)
        else:
            before = backend.solve_strict(d_iter, float(load), config.validation.cg_rtol,
                config.validation.cg_atol, config.validation.cg_maxiter, initial_guess=q_guess)
        report = None
        for sweep in range(start_sweep, args.max_sweeps + 1):
            sweep_start = time.monotonic()
            anderson_info = None
            if method == "anderson_residual":
                d, current_image, payload, anderson_info, history_x, history_f = safeguarded_step(
                    d_iter, current_image, history_x, history_f, evaluate=evaluate_map,
                    bounds_pass=bounds_pass, physics_pass=physics_pass,
                    depth=args.anderson_depth, beta=args.omega,
                    sufficient_decrease=args.anderson_sufficient_decrease,
                    max_backtracks=args.anderson_max_backtracks, restart_policy=restart_policy, rank_strategy=rank_strategy)
                after, fields, damage_info = payload["solve"], payload["fields"], payload["damage_info"]
            else:
                H_used = np.asarray(residual.evaluate(before.displacement, jnp.asarray(d_iter), None,
                                                      jnp.asarray(H_comm)).trial_history)
                d_solved, damage_info = solve_damage(H_used, d_iter)
                d = d_iter + args.omega * (d_solved - d_iter)
                after = backend.solve_strict(d, float(load), config.validation.cg_rtol,
                    config.validation.cg_atol, config.validation.cg_maxiter,
                    initial_guess=np.asarray(before.displacement))
                fields = residual.evaluate(after.displacement, jnp.asarray(d), None, jnp.asarray(H_comm))
            q_guess = np.asarray(after.displacement)
            H = np.asarray(fields.trial_history)
            reaction = float(fields.reaction)
            local, nonlocal_term, history = gc0 * d, gc1 * (D @ d), 2 * H * (d - 1)
            rho = local + nonlocal_term + history
            scale = max(abs(local)) + max(abs(nonlocal_term)) + max(abs(history))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("split reference AT2 term scale must be finite and positive")
            ratios = np.linalg.norm(q_guess[:2 * model.n_material].reshape(-1, 2), axis=1) / load
            changes = None
            if previous_q is not None:
                if previous_H is None or previous_reaction is None:
                    raise RuntimeError("reference previous-iterate diagnostics are incomplete")
                changes = {"q_relative_inf": float(max(abs(q_guess - previous_q)) / load),
                    "H_relative_inf": float(max(abs(H - previous_H)) / max(max(abs(H)), max(abs(previous_H)))),
                    "reaction_relative": abs(reaction - previous_reaction) / max(abs(reaction), abs(previous_reaction))}
            gates = {"finite": all(np.all(np.isfinite(v)) for v in (q_guess, d, H, rho)),
                "damage_range": min(d) >= -1e-10 and max(d) <= 1.0,
                "healing": max(np.maximum(d_comm - d, 0)) <= 1e-3,
                "history_monotonic": min(H - H_comm) >= 0,
                "drift": max(ratios) <= direct.DRIFT_ENVELOPE,
                "coupled_damage": max(abs(rho)) / scale <= 1e-6,
                "damage_change": max(abs(d - d_iter)) <= 1e-5,
                "iterate_changes": changes is not None and max(changes.values()) <= 1e-5}
            consecutive = consecutive + 1 if all(gates.values()) else 0
            report = {"load": float(load), "state_index": index, "sweep": sweep,
                "reference_only": True, "model_digest": model.model_digest,
                "parent_sha256": parent_digest,
                "reaction": reaction, "damage_max": float(max(d)), "damage_min": float(min(d)),
                "damage_inf": float(max(abs(rho))), "damage_relative_inf": float(max(abs(rho)) / scale),
                "damage_change_inf": float(max(abs(d - d_iter))), "max_displacement_ratio": float(max(ratios)),
                "drift_count": int(np.count_nonzero(ratios > direct.DRIFT_ENVELOPE)),
                "healing": float(max(np.maximum(d_comm - d, 0))), "history_max": float(max(H)),
                "changes": changes, "gates": {name: bool(value) for name, value in gates.items()},
                "consecutive": consecutive, "damage_solve": damage_info,
                "mechanics_iterations": int(after.cg_iterations),
                "mechanics_relative_residual": float(after.free_residual_relative_norm)}
            if damage_backend == "gpu":
                report["timing"] = {"model_build_seconds": build_seconds,
                                    "sweep_seconds": time.monotonic() - sweep_start,
                                    "load_seconds": time.monotonic() - load_start}
            extra_fields = None
            if anderson_info is not None:
                report.update({"anderson": anderson_info, "anderson_history_count": len(history_x)})
                extra_fields = {"anderson_x": np.asarray(history_x, dtype=np.float64).reshape(-1, model.n_material),
                                "anderson_f": np.asarray(history_f, dtype=np.float64).reshape(-1, model.n_material)}
            print(json.dumps(report), flush=True)
            atomic_json(state_dir / f"sweep_{sweep:04d}.json", report)
            trial_binding = write_trial_fields(state_dir, sweep, q=q_guess, d=d, H=H, extra_fields=extra_fields)
            atomic_json(state_dir / "progress.json", {**report, **trial_binding})
            if storage_policy is not None and storage_policy["trial_retention"] == "sparse":
                prune_trial_fields(state_dir, trial_binding, tail=storage_policy["trial_tail"],
                                   stride=storage_policy["trial_stride"])
            d_iter = d
            previous_q, previous_H, previous_reaction = q_guess.copy(), H.copy(), reaction
            if not all(gates[name] for name in ("finite", "damage_range", "healing", "history_monotonic", "drift")):
                report["accepted_reference"] = False
                break
            if anderson_info is not None and not anderson_info["selected_physics_pass"]:
                report["accepted_reference"] = False
                break
            if consecutive >= 3:
                with (state_dir / "committed.npz").open("xb") as stream:
                    np.savez_compressed(stream, q=q_guess, d=d, H=H)
                report.update({"accepted_reference": True, "parent_sha256": parent_digest,
                    "committed_sha256": hashlib.sha256((state_dir / "committed.npz").read_bytes()).hexdigest()})
                parent_digest = report["committed_sha256"]
                q_comm, d_comm, H_comm = q_guess.copy(), d.copy(), H.copy()
                break
            # Same d/load as the converged previous after-solve; no redundant
            # mechanics re-training or changed-history accumulation.
            before = after
        else:
            if report is None:
                raise RuntimeError("reference load did not execute a sweep")
            report["accepted_reference"] = False
        if report is None:
            raise RuntimeError("reference load has no completed report")
        atomic_json(state_dir / "result.json", report)
        curve.append(report)
        atomic_json(root / "loading_summary.json", curve)
        if not report["accepted_reference"]:
            atomic_json(root / "result.json", {"complete": False, "rejected_load": float(load), "last": report})
            return
    atomic_json(root / "result.json", {"complete": True, "last": curve[-1]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-step", type=float, required=True)
    parser.add_argument("--last-load", type=float, required=True)
    parser.add_argument("--max-sweeps", type=int, required=True)
    parser.add_argument("--damage-backend", choices=("cpu", "gpu"), default="cpu",
                        help="Explicit damage CG implementation; gpu has no CPU fallback")
    parser.add_argument("--trial-retention", choices=("sparse", "all"), default=None,
                        help="Fresh roots default to sparse; resume inherits stored or legacy all policy")
    parser.add_argument("--trial-tail", type=int, default=None,
                        help="Latest published generations retained (fresh default 4; minimum 4)")
    parser.add_argument("--trial-stride", type=int, default=None,
                        help="Periodic full-field snapshot interval (fresh default 50)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fork-from", type=Path)
    parser.add_argument("--fork-after-index", type=int, default=0)
    parser.add_argument("--omega", type=float, default=0.5,
                        help="Explicit outer Picard relaxation; stored in numerical identity")
    parser.add_argument("--trial-from", type=Path,
                        help="Verified first new-load trial initialization, not committed history")
    parser.add_argument("--relaxation-policy", choices=("picard", "anderson_residual"), default="picard")
    parser.add_argument("--anderson-depth", type=int, default=5)
    parser.add_argument("--anderson-max-backtracks", type=int, default=4)
    parser.add_argument("--anderson-sufficient-decrease", type=float, default=0.01)
    parser.add_argument("--anderson-restart-policy", choices=("reject_or_rank", "rank_only"), default="reject_or_rank")
    parser.add_argument("--anderson-rank-strategy", choices=("restart", "drop_oldest"), default="restart")
    args = parser.parse_args()
    if not np.all(np.isfinite([args.load_step, args.last_load])) or min(args.load_step, args.last_load) <= 0 or args.max_sweeps < 3:
        raise ValueError("split reference requires positive load controls and at least three sweeps")
    steps = args.last_load / args.load_step
    if abs(steps - round(steps)) > 1e-8:
        raise ValueError("split reference endpoint must be an integer number of explicit load increments")
    if args.fork_from is None and args.fork_after_index != 0:
        raise ValueError("fork-after-index requires a declared fork source")
    if not np.isfinite(args.omega) or not 0 < args.omega <= 1:
        raise ValueError("Picard omega must be finite in (0, 1]")
    if args.trial_from is not None and args.fork_from is None:
        raise ValueError("trial initialization requires an explicit accepted-prefix fork")
    run(args)


if __name__ == "__main__":
    main()
