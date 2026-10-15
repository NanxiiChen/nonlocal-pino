"""FP64 stagger: direct nonlocal mechanics / nonlocal PINN damage loading.

Only this hybrid path's own fields provide history. External direct solutions
are held-out comparisons. Strict fixed-d mechanical solves do not certify the
staggered fracture state; every fixed-H and final consistency defect is reported.
There is no joint u/d optimizer or differentiation through the mechanics solve.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

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
from examples.fracture2d._legacy.split_loading import reference_indices
from examples.fracture2d._legacy.split_pinn import (
    accepted_warning_policy, checkpoint_resume_identity, classify_report,
    comparison_reference, plot_outputs, resume_identity, fork_checkpoint_identity,
    neural_state_root,
)
from examples.fracture2d._legacy.outer_stopping import validate_controls as validate_outer_controls, evaluate as evaluate_outer
from examples.fracture2d._legacy.stagger_checkpoint import read_checkpoint, write_checkpoint
from examples.fracture2d._legacy.train import _build_network, _nonnegative_damage


SCHEMA = "fracture2d-direct-mechanics-pinn-damage-v1"
SCHEMA_GATED = "fracture2d-direct-mechanics-pinn-damage-gated-v1"


def checkpoint_schema(config):
    if "outer_stop" in config:
        enabled = config["outer_stop"]["enabled"]
        if type(enabled) is not bool:
            raise ValueError("outer_stop.enabled must be boolean")
        if enabled:
            return SCHEMA_GATED
    return SCHEMA


def validate_completed(metadata, report, config):
    if metadata["phase"] != "complete":
        raise ValueError("invalid completed hybrid checkpoint phase")
    if checkpoint_schema(config) == SCHEMA_GATED:
        outer = report["outer_convergence"]
        if (not outer["converged"] or metadata["step"] != outer["consecutive"]
                or outer["consecutive"] < config["outer_stop"]["consecutive_passes"]
                or not config["outer_stop"]["min_sweeps"] <= metadata["sweep"] <= config["outer_stop"]["max_sweeps"]):
            raise ValueError("completed gated hybrid checkpoint is not stagger-ready")
    elif metadata["sweep"] != config["stagger"]["sweeps"] or metadata["step"] != 0:
        raise ValueError("invalid completed fixed-budget hybrid checkpoint phase")


def validate_config(c):
    if c.train.dtype != "float64" or c.train.mechanics_solver != "direct":
        raise ValueError("hybrid loading requires explicit FP64/direct mechanics")
    if c.train.operator_mode not in ("autodiff", "analytic", "analytic_padded_damage"):
        raise ValueError("unknown hybrid damage operator mode")
    if (isinstance(c.output.fork_after_index, bool) or int(c.output.fork_after_index) != c.output.fork_after_index
            or not 0 <= c.output.fork_after_index < c.data.end_state_index
            or bool(c.output.fork_from) != bool(c.output.fork_after_index)):
        raise ValueError("hybrid fork requires an explicit nonempty prefix before the endpoint")
    if "outer_stop" in c:
        validate_outer_controls(c.outer_stop)
    if "inner_stop" in c and c.inner_stop.enabled:
        raise ValueError("hybrid loading currently requires fixed damage block budgets")
    if c.stagger.accept_on_first_passing_report is not False:
        raise ValueError("hybrid blocks must not stop on external reference agreement")
    if type(c.output.resume) is not bool or type(c.reporting.comparison_gate) is not bool:
        raise ValueError("resume and comparison_gate must be boolean")
    for name, value in (("load_step", c.data.load_step), ("learning_rate", c.train.learning_rate),
                        ("damage_floor", c.physical.damage_floor)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive finite")
    for name, value in (("sweeps", c.stagger.sweeps), ("damage_steps", c.stagger.damage_steps),
                        ("log_every", c.output.log_every), ("cg_maxiter", c.validation.cg_maxiter)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(c.stagger.omega) or not 0 < c.stagger.omega <= 1:
        raise ValueError("hybrid Picard omega must lie in (0, 1]")
    for name in ("cg_rtol", "cg_atol"):
        if not np.isfinite(c.validation[name]) or c.validation[name] < 0:
            raise ValueError(f"validation.{name} must be nonnegative finite")
    if c.validation.cg_rtol == c.validation.cg_atol == 0:
        raise ValueError("CG tolerances cannot both be zero")
    for name, value in c.reporting.items():
        if name != "comparison_gate" and (not np.isfinite(value) or value < 0):
            raise ValueError(f"reporting.{name} must be nonnegative finite")
    if c.data.reference_policy not in ("strict", "user_accepted_warning"):
        raise ValueError("unknown hybrid comparison policy")
    return reference_indices(c.data)


def run(c):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    targets = validate_config(c)
    schema = checkpoint_schema(c)
    gated = schema == SCHEMA_GATED
    max_sweeps = c.outer_stop.max_sweeps if gated else c.stagger.sweeps
    policy_sha = c.data.reference_policy_sha256 if c.data.reference_policy == "user_accepted_warning" else None
    if policy_sha is not None:
        accepted_warning_policy(c.data.reference_root, policy_sha)
    root = Path(c.output.dir)
    if not root.parent.is_dir():
        raise FileNotFoundError(root.parent)
    values = json.loads(c.to_json_best_effort())
    values["output"]["resume"] = False
    identity = {"schema": schema, "config": values}
    if c.output.resume:
        stored = json.loads((root / "identity.json").read_text())
        resume_identity(stored, identity)
        if stored != identity:
            atomic_json(root / "identity.json", identity)
    else:
        root.mkdir()
        atomic_json(root / "identity.json", identity)
    build_start = time.perf_counter()
    model = build_model(read_cloud(c.data.cloud, c.data.cloud_sha256),
        delta_u=c.nl.delta_u, delta_d=c.nl.delta_d, grip_extent=c.geometry.grip_extent,
        h_grip=c.geometry.h_grip, visibility_tolerance=c.nl.visibility_tolerance,
        E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
        rank_rtol=c.physical.rank_rtol, Gc=c.physical.Gc, l0=c.physical.l0)
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("hybrid nonlocal loading requires GPU")
    backend = FP64A2Backend(model.static, model.constraints, device)
    residual = FractureResidual(model, backend)
    damage_matvec = residual.operator.deterministic_matvec
    if c.train.operator_mode != "autodiff":
        from examples.fracture2d._legacy.split_pinn_gradients import make_analytic_matvec, make_padded_matvec
        if c.train.operator_mode == "analytic":
            damage_matvec = make_analytic_matvec(residual.operator)
        else:
            damage_matvec, layout = make_padded_matvec(residual.operator)
            print(json.dumps({"operator_mode": c.train.operator_mode, "damage_layout": layout}), flush=True)
    build_seconds = time.perf_counter() - build_start
    points = jax.device_put(model.geometry.material_reference_coordinates, device)
    d_net = jax.tree.map(lambda a: a.astype(jnp.float64),
                        _build_network(c, jax.random.PRNGKey(c.train.seed + 1), out_dim=1))
    optimizer = optax.adam(c.train.learning_rate)
    d_state = optimizer.init(eqx.filter(d_net, eqx.is_inexact_array))
    d_parent = jnp.zeros(model.n_material, jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    q_parent = jnp.asarray(model.constraints.prescribed_vector(0.))
    parent_sha, inherited_warning, curve = None, False, []
    gc0, gc1 = model.Gc / model.l0, model.Gc * model.l0

    def damage(net, load):
        raw = c.physical.E / gc0 * load**2 * jax.vmap(net)(points)[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)

    @eqx.filter_jit
    def damage_step(net, state, H, load, scale):
        def objective(n):
            d = damage(n, load)
            r = gc0*d + gc1*damage_matvec(d) + 2*jax.lax.stop_gradient(H)*(d-1)
            return jnp.mean((r/scale)**2)
        loss, grads = eqx.filter_value_and_grad(objective)(net)
        updates, state = optimizer.update(grads, state, net)
        return eqx.apply_updates(net, updates), state, loss

    def validate_fields(q, d, H, load, floor_H):
        for name, a, shape in (("q", q, (model.n_dofs,)), ("d", d, (model.n_material,)),
                               ("H", H, (model.n_material,))):
            a = np.asarray(a)
            if a.shape != shape or a.dtype != np.float64 or not np.all(np.isfinite(a)):
                raise ValueError(f"hybrid {name} must be finite FP64 of model shape")
        fixed = model.constraints.constrained_dofs
        if not np.array_equal(np.asarray(q)[fixed], model.constraints.prescribed_vector(float(load))[fixed]):
            raise ValueError("hybrid exact displacement constraints differ")
        if np.min(np.asarray(d)) < 0 or np.max(np.asarray(d)) > 1:
            raise ValueError("hybrid damage must lie in [0, 1] for direct mechanics")
        if np.min(np.asarray(H)) < 0 or np.any(np.asarray(H) < np.asarray(floor_H)):
            raise ValueError("hybrid cumulative history decreased")

    def solve_mechanics(d, load, guess):
        result = backend.solve_strict(d, load, c.validation.cg_rtol, c.validation.cg_atol,
                                      c.validation.cg_maxiter, initial_guess=residual.lift(guess, load))
        diagnostics = {"converged": result.converged, "iterations": int(result.cg_iterations),
                       "free_residual_norm": float(result.free_residual_norm),
                       "free_residual_relative": float(result.free_residual_relative_norm),
                       "convergence_limit": float(result.cg_convergence_limit)}
        return result.displacement, diagnostics

    first_load = jnp.asarray(c.data.load_step, jnp.float64)
    union_points = jnp.asarray(model.geometry.coordinates)
    affine = residual.lift(jnp.stack((jnp.zeros_like(union_points[:, 1]),
                                     first_load*union_points[:, 1]), axis=1).reshape(-1), first_load)
    first_scale = jnp.sqrt(jnp.mean(backend._residual_from_factor(
        backend._edge_factor(d_parent), affine)[backend.free_dofs]**2))
    if not np.isfinite(float(first_scale)) or float(first_scale) <= 0:
        raise ValueError("hybrid first-load affine mechanics scale must be positive finite")
    writer = SummaryWriter(logdir=c.output.event_dir)
    try:
        for index, reference_index in enumerate(targets, 1):
            load = jnp.asarray(reference_index*c.data.load_step, jnp.float64)
            reference = comparison_reference(
                c.data.reference_root, reference_index, model.model_digest, model.n_dofs,
                model.n_material, float(load), reference_policy=c.data.reference_policy,
                policy_sha256=policy_sha, return_metadata=True)
            if len(reference) != 5:
                raise RuntimeError("hybrid comparison requires reference provenance")
            q_ref, d_ref, reaction_ref, reference_sha, provenance = reference
            state_dir = root / f"state_{index:04d}"
            inherited = bool(c.output.fork_from) and index <= c.output.fork_after_index
            if inherited:
                state_dir = neural_state_root(c.output.fork_from, index)
            state_identity = {"model_digest": model.model_digest, "config": values,
                              "reference_sha256": reference_sha, "reference_provenance": provenance,
                              "parent_sha256": parent_sha, "index": index,
                              "reference_index": reference_index, "load": float(load)}
            template = (d_net, d_state, q_parent, d_parent, H_parent)
            if (state_dir / "result.json").exists():
                report = json.loads((state_dir / "result.json").read_text())
                if not report["research_advance"]:
                    raise ValueError("rejected hybrid load requires a new trial root")
                if (report["reference_provenance"] != provenance or report["mechanics_solver"] != "direct"
                        or report["index"] != index or report["load"] != float(load)):
                    raise ValueError("restored hybrid report provenance/load differs")
                expected = (fork_checkpoint_identity(state_dir / "state.zip", state_identity) if inherited
                            else checkpoint_resume_identity(state_dir / "state.zip", state_identity))
                metadata, state = read_checkpoint(state_dir / "state.zip", expected, template,
                                                  schema=checkpoint_schema(expected["config"]))
                validate_completed(metadata, report, expected["config"])
                d_net, d_state, q, d, H = state
                validate_fields(q, d, H, load, H_parent)
                q_parent, d_parent, H_parent = q, d, H
                parent_sha = hashlib.sha256((state_dir / "state.zip").read_bytes()).hexdigest()
                inherited_warning |= not report["strict_accepted"]
                curve.append(report)
                continue
            if inherited:
                raise ValueError("hybrid fork prefix contains an unfinished load")
            load_start = time.perf_counter()
            phase_seconds = {"before": 0., "damage": 0., "after": 0.}
            phase_updates = {"before": 0, "damage": 0, "after": 0}
            q_iter = residual.lift(q_parent, load)
            d_iter, d_previous, H_used = d_parent, d_parent, H_parent
            q_prior, H_prior, R_prior = q_iter, H_parent, jnp.asarray(0., jnp.float64)
            previous_consecutive = 0
            first_sweep, first_phase = 1, "before"
            if state_dir.exists():
                if not c.output.resume:
                    raise FileExistsError(state_dir)
                progress_template = (d_net, d_state, q_iter, d_iter, d_previous, H_used)
                if gated:
                    progress_template = (*progress_template, q_prior, H_prior, R_prior)
                expected = checkpoint_resume_identity(state_dir / "progress.zip", state_identity)
                metadata, progress = read_checkpoint(state_dir / "progress.zip", expected, progress_template, schema=schema)
                d_net, d_state, q_iter, d_iter, d_previous, H_used = progress[:6]
                if gated:
                    q_prior, H_prior, R_prior = progress[6:]
                    previous_consecutive = metadata["step"]
                first_sweep, first_phase = metadata["sweep"], metadata["phase"]
                if (first_sweep > max_sweeps or first_phase == "complete"
                        or not gated and metadata["step"] != 0
                        or gated and (previous_consecutive > max_sweeps
                                      or first_sweep == 1 and previous_consecutive != 0)):
                    raise ValueError("invalid unfinished hybrid checkpoint phase")
                validate_fields(q_iter, d_iter, H_used, load, H_parent)
                if gated:
                    validate_fields(q_prior, d_previous, H_prior, load, H_parent)
                    prior_reaction = np.asarray(R_prior)
                    if prior_reaction.shape != () or prior_reaction.dtype != np.float64 or not np.isfinite(prior_reaction):
                        raise ValueError("restored outer previous reaction must be finite FP64 scalar")
            else:
                state_dir.mkdir()
                write_checkpoint(state_dir / "progress.zip", state_identity, 1, "before", 0,
                    (d_net, d_state, q_iter, d_iter, d_previous, H_used,
                     * ((q_prior, H_prior, R_prior) if gated else ())), schema=schema)
            scale = first_scale*reference_index
            report, fields, d_np = None, None, None
            sweep = 0
            for sweep in range(first_sweep, max_sweeps + 1):
                mechanics = None
                for phase in ("before", "damage", "after"):
                    if sweep == first_sweep and ("before", "damage", "after").index(phase) < ("before", "damage", "after").index(first_phase):
                        continue
                    phase_start = time.perf_counter()
                    if phase == "damage":
                        H_used = residual.evaluate(q_iter, d_iter, None, H_parent).trial_history
                        H_scale = 2*jnp.sqrt(jnp.mean(H_used**2))
                        if not np.isfinite(float(H_scale)) or float(H_scale) <= 0:
                            raise ValueError("hybrid fixed-H damage scale must be positive finite")
                        for step in range(1, c.stagger.damage_steps + 1):
                            d_net, d_state, loss = damage_step(d_net, d_state, H_used, load, H_scale)
                            phase_updates[phase] += 1
                            if step % c.output.log_every == 0 or step == c.stagger.damage_steps:
                                if not np.isfinite(float(loss)):
                                    raise FloatingPointError("hybrid damage loss is nonfinite")
                                print(json.dumps({"index": index, "sweep": sweep, "phase": phase,
                                                  "step": step, "loss_before_update": float(loss)}), flush=True)
                                writer.add_scalar("loss/damage", float(loss),
                                    ((index-1)*max_sweeps+sweep-1)*c.stagger.damage_steps+step)
                        d_iter = d_iter + c.stagger.omega*(damage(d_net, load)-d_iter)
                    else:
                        q_iter, mechanics = solve_mechanics(d_iter, load, q_iter)
                        atomic_json(state_dir / f"mechanics_{sweep:04d}_{phase}.json", mechanics)
                    if phase != "after":
                        write_checkpoint(state_dir / "progress.zip", state_identity, sweep,
                            "damage" if phase == "before" else "after", previous_consecutive,
                            (d_net, d_state, q_iter, d_iter, d_previous, H_used,
                             * ((q_prior, H_prior, R_prior) if gated else ())), schema=schema)
                    jax.block_until_ready(eqx.filter((d_net, d_state, q_iter, d_iter), eqx.is_array))
                    phase_seconds[phase] += time.perf_counter()-phase_start
                if mechanics is None:
                    raise RuntimeError("hybrid sweep lacks its final direct mechanical solve")
                fields = residual.evaluate(q_iter, d_iter, None, H_parent)
                validate_fields(q_iter, d_iter, fields.trial_history, load, H_parent)
                q_np, d_np, H_np = map(np.asarray, (q_iter, d_iter, fields.trial_history))
                parts = (gc0*d_iter, gc1*residual.operator.deterministic_matvec(d_iter),
                         2*fields.trial_history*(d_iter-1))
                at2_scale = sum(float(jnp.max(jnp.abs(p))) for p in parts)
                image = damage(d_net, load)
                image_parts = (gc0*image, gc1*residual.operator.deterministic_matvec(image), 2*H_used*(image-1))
                image_scale = sum(float(jnp.max(jnp.abs(p))) for p in image_parts)
                if any(not np.isfinite(s) or s <= 0 for s in (at2_scale, image_scale)):
                    raise FloatingPointError("hybrid damage term scales must be positive finite")
                report = classify_report({"index": index, "load": float(load), "sweep": sweep,
                    "reaction": float(fields.reaction), "reference_reaction": reaction_ref,
                    "reaction_relative": abs(float(fields.reaction)-reaction_ref)/max(abs(reaction_ref), 1e-30),
                    "damage_rmse": float(np.sqrt(np.mean((d_np-d_ref)**2))),
                    "damage_max_error": float(np.max(abs(d_np-d_ref))),
                    "damage_min": float(min(d_np)), "damage_max": float(max(d_np)),
                    "damage_inf": float(jnp.max(jnp.abs(fields.damage))),
                    "coupled_AT2_relative": float(jnp.max(jnp.abs(fields.damage)))/at2_scale,
                    "mechanics_loss": float(jnp.mean((fields.mechanics/scale)**2)),
                    "mechanics_inf": float(jnp.max(jnp.abs(fields.mechanics))),
                    "displacement_relative": float(np.linalg.norm(q_np-q_ref)/np.linalg.norm(q_ref)),
                    "damage_change": float(jnp.max(jnp.abs(d_iter-d_previous))),
                    "healing": float(jnp.max(jnp.maximum(d_parent-d_iter, 0))),
                    "history_decrease": float(jnp.max(jnp.maximum(H_parent-fields.trial_history, 0))),
                    "drift_count": int(np.count_nonzero(np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)
                                                       /float(load) > direct.DRIFT_ENVELOPE))},
                    c.reporting, inherited_warning=inherited_warning or provenance["status"] == "warning")
                report.update(mechanics_solver="direct", damage_solver="nonlocal_pinn", mechanics=mechanics,
                    reference_provenance=provenance, reference_index=reference_index,
                    fixed_H_image_AT2_relative=float(jnp.max(jnp.abs(sum(image_parts))))/image_scale,
                    fixed_H_image_damage_min=float(jnp.min(image)), fixed_H_image_damage_max=float(jnp.max(image)),
                    mechanics_scale_kind="first_load_affine_residual_times_reference_index")
                if gated:
                    has_previous = sweep > 1
                    report.update(
                        q_change_over_load=float(jnp.max(jnp.abs(q_iter-q_prior))/load) if has_previous else None,
                        H_change_relative=float(jnp.max(jnp.abs(fields.trial_history-H_prior)) /
                            max(float(jnp.max(jnp.abs(fields.trial_history))), float(jnp.max(jnp.abs(H_prior))), c.outer_stop.scale_floor)) if has_previous else None,
                        reaction_change_relative=abs(report["reaction"]-float(R_prior)) /
                            max(abs(report["reaction"]), abs(float(R_prior)), c.outer_stop.scale_floor) if has_previous else None)
                    outer = evaluate_outer(report, c.outer_stop, c.reporting, sweep=sweep,
                                           previous_consecutive=previous_consecutive)
                    previous_consecutive = outer["consecutive"]
                    report["outer_convergence"] = outer
                    report["research_advance"] &= outer["converged"]
                    report["strict_accepted"] &= outer["converged"]
                    if not outer["converged"]:
                        report["status"] = "stagger_budget_exhausted" if sweep == max_sweeps else "stagger_iterating"
                print(json.dumps(report), flush=True)
                atomic_json(state_dir / f"sweep_{sweep:04d}.json", report)
                np.savez_compressed(state_dir / f"fields_{sweep:04d}.npz", q=q_np, d=d_np, H=H_np,
                    H_used=np.asarray(H_used), d_image=np.asarray(image), d_previous=np.asarray(d_previous),
                    d_reference=d_ref, coordinates=model.geometry.material_reference_coordinates)
                position = (index-1)*max_sweeps+sweep
                for name in ("reaction", "reference_reaction", "damage_rmse", "damage_max_error"):
                    writer.add_scalar(f"eval/{name}", report[name], position)
                for name, check in report["checks"].items():
                    writer.add_scalar(f"diag/{name}", check["value"], position)
                writer.add_scalar("diag/fixed_H_image_AT2_relative", report["fixed_H_image_AT2_relative"], position)
                writer.flush()
                d_previous = d_iter
                if gated:
                    q_prior, H_prior, R_prior = q_iter, fields.trial_history, jnp.asarray(report["reaction"], jnp.float64)
                    if report["outer_convergence"]["converged"]:
                        break
                if sweep < max_sweeps:
                    write_checkpoint(state_dir / "progress.zip", state_identity, sweep+1, "before", previous_consecutive,
                        (d_net, d_state, q_iter, d_iter, d_previous, H_used,
                         * ((q_prior, H_prior, R_prior) if gated else ())), schema=schema)
            if report is None or fields is None or d_np is None:
                raise RuntimeError("hybrid load did not complete a stagger sweep")
            state = (d_net, d_state, q_iter, d_iter, fields.trial_history)
            write_checkpoint(state_dir / "state.zip", state_identity, sweep, "complete", previous_consecutive, state, schema=schema)
            atomic_json(state_dir / "result.json", report)
            curve.append(report)
            atomic_json(root / "reaction_curve.json", curve)
            plot_outputs(state_dir, model.geometry.material_reference_coordinates,
                model.geometry.material_reference_triangles, d_np, d_ref, report, curve)
            atomic_json(state_dir / "timing.json", {"scope": "work in this invocation; includes logging/checkpoints",
                "model_build_seconds": build_seconds, "load_wall_seconds": time.perf_counter()-load_start,
                "phase_seconds": phase_seconds, "phase_updates": phase_updates})
            if not report["research_advance"]:
                break
            parent_sha = hashlib.sha256((state_dir / "state.zip").read_bytes()).hexdigest()
            q_parent, d_parent, H_parent = state[2:]
            inherited_warning |= not report["strict_accepted"]
    finally:
        writer.close()
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--last-index", type=int)
    args = parser.parse_args()
    c = runpy.run_path(str(args.config))["get_config"]()
    if args.resume:
        c.output.resume = True
    if args.last_index is not None:
        c.data.end_state_index = args.last_index
    run(c)


if __name__ == "__main__":
    main()
