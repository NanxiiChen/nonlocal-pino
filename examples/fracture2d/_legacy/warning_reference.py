"""Explicit diagnostic-only loading; original strict gate failures remain visible."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.losses import FractureResidual
from examples.fracture2d._legacy.split_horizon_model import build_model, read_cloud
from examples.fracture2d._legacy.split_reference import (
    damage_pcg, make_jax_damage_pcg, read_trial_seed, reference_state_root,
    trial_archive_path, write_trial_fields,
)
from examples.fracture2d._legacy.trial_storage import FIELDS, prune_trial_fields, resolve_storage_policy


def publish_warning_endpoint(folder, q, d, H):
    """Atomic no-clobber field publication; recover an interrupted report write."""
    endpoint = Path(folder)/"warning.npz"
    if endpoint.exists():
        with np.load(endpoint,allow_pickle=False) as archive:
            for name,value in (("q",q),("d",d),("H",H)):
                if not np.array_equal(archive[name],value):
                    raise ValueError("unbound warning endpoint differs from progress fields")
    else:
        temporary = endpoint.with_name(f"warning.{uuid.uuid4().hex}.npz.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream,q=q,d=d,H=H)
            os.link(temporary,endpoint)
        finally:
            if temporary.exists():
                temporary.unlink()
    return hashlib.sha256(endpoint.read_bytes()).hexdigest()


def gates(d, H, q, rho, *, floor_d, floor_H, previous_d, load, n_material, scale, previous, reaction, limits):
    """Original strict checks plus explicit diagnostic warning advancement."""
    ratios = np.linalg.norm(q[:2*n_material].reshape(-1, 2), axis=1) / load
    changes = None
    if previous is not None:
        old_q, _, old_H, old_reaction = previous
        if max(max(abs(H)),max(abs(old_H))) <= 0 or max(abs(reaction),abs(old_reaction)) <= 0:
            raise ValueError("warning adjacent history/reaction scales must be positive")
        changes = dict(q_relative_inf=float(max(abs(q-old_q))/load),
            H_relative_inf=float(max(abs(H-old_H))/max(max(abs(H)), max(abs(old_H)))),
            reaction_relative=abs(reaction-old_reaction)/max(abs(reaction), abs(old_reaction)))
    d_change = float(max(abs(d-previous_d)))
    healing = float(max(np.maximum(floor_d-d, 0)))
    original = dict(finite=all(np.all(np.isfinite(x)) for x in (q, d, H, rho)),
        damage_range=min(d) >= -limits["range_atol"] and max(d) <= 1.,
        healing=healing <= limits["healing_atol"], history_monotonic=min(H-floor_H) >= 0.,
        drift=max(ratios) <= direct.DRIFT_ENVELOPE,
        coupled_damage=max(abs(rho))/scale <= limits["coupled_AT2_relative_inf_limit"],
        damage_change=d_change <= limits["damage_change_atol"],
        iterate_changes=changes is not None and all(changes[name] <= limits[key] for name, key in (
            ("q_relative_inf", "q_change_relative_limit"),
            ("H_relative_inf", "H_change_relative_limit"),
            ("reaction_relative", "reaction_change_relative_limit"))))
    hard = {name: bool(original[name]) for name in ("finite", "damage_range", "history_monotonic")}
    return original, hard, changes, d_change, healing, ratios


def run(args, *, return_model=False):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if args.sweeps_per_load < 3 or not 0 < args.omega <= 1:
        raise ValueError("warning loading requires >=3 sweeps and omega in (0,1]")
    if (args.seed_state is None) != (args.seed_sha256 is None):
        raise ValueError("seed state and SHA must both be explicit or both absent")
    if args.damage_backend not in ("gpu", "cpu"):
        raise ValueError("explicit warning damage backend must be gpu or cpu")
    if args.damage_backend == "gpu" and jax.devices()[0].platform != "gpu":
        raise RuntimeError("warning GPU loading requires a visible JAX GPU")
    source_policy = json.loads((args.source_root/"policy.json").read_text())
    if source_policy["schema"] != "fracture2d-split-reference-v1" or args.fork_after_index < 1:
        raise ValueError("warning source must be an explicitly indexed strict reference")
    if source_policy["consecutive_passing_sweeps"] != 3:
        raise ValueError("warning source must preserve the original three-confirmation contract")
    step = source_policy["load_step"]
    end = int(round(args.last_load/step))
    if (not np.isfinite(args.last_load) or args.last_load <= step*args.fork_after_index
            or abs(args.last_load/step-end) > 1e-8):
        raise ValueError("warning endpoint must extend the prefix by whole load increments")
    storage = resolve_storage_policy(args)
    if storage is None:
        raise ValueError("warning roots require an explicit stored retention policy")
    start = time.monotonic()
    c = runpy.run_path(str(args.config))["get_config"]()
    if json.loads(c.to_json_best_effort()) != source_policy["model_config"]:
        raise ValueError("warning model config differs from strict source")
    model = build_model(read_cloud(c.data.cloud, c.data.cloud_sha256),
        delta_u=c.nl.delta_u, delta_d=c.nl.delta_d, grip_extent=c.geometry.grip_extent,
        h_grip=c.geometry.h_grip, visibility_tolerance=c.nl.visibility_tolerance,
        E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
        rank_rtol=c.physical.rank_rtol, Gc=c.physical.Gc, l0=c.physical.l0)
    if model.model_digest != source_policy["model_digest"]:
        raise ValueError("warning model digest differs from strict source")
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    volumes = np.asarray(model.geometry.material_reference_volumes)
    D = model.material_operator.to_scipy().tocsr()
    gc0, gc1 = model.Gc/model.l0, model.Gc*model.l0
    if args.damage_backend == "gpu":
        solve_d = make_jax_damage_pcg(model.material_operator, volumes, gc0, gc1,
                                     rtol=1e-11, atol=1e-14, maxiter=2000)
    else:
        def solve_d(H, initial):
            return damage_pcg(D, volumes, gc0, gc1, H, initial, rtol=1e-11, atol=1e-14, maxiter=2000)
    parent_sha = None
    q_comm = model.constraints.prescribed_vector(0.)
    d_comm, H_comm = np.zeros(model.n_material), np.zeros(model.n_material)
    summaries = []
    for index in range(1, args.fork_after_index+1):
        folder = reference_state_root(args.source_root, index)
        report = json.loads((folder/"result.json").read_text())
        path = folder/"committed.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (not report["accepted_reference"] or report["model_digest"] != model.model_digest
                or report["parent_sha256"] != parent_sha or digest != report["committed_sha256"]
                or abs(report["load"]-index*step) > 1e-12):
            raise ValueError("warning inherited strict prefix identity differs")
        with np.load(path, allow_pickle=False) as archive:
            q_comm, d_comm, H_comm = (archive[name] for name in ("q", "d", "H"))
        parent_sha = digest
        summaries.append(report)
    residual.validate_fields(q_comm,d_comm,H_comm,H_comm,np.asarray(args.fork_after_index*step))
    policy = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    for name in FIELDS:
        if name in policy:
            del policy[name]
    policy.update(storage)
    policy.update(schema="fracture2d-split-warning-v1", model_digest=model.model_digest,
        model_config=json.loads(c.to_json_best_effort()), load_step=step, resume=False,
        fork_from=str(args.source_root), strict_parent_sha256=parent_sha,
        warning_checks=["healing", "drift", "coupled_damage", "damage_change", "iterate_changes"],
        hard_checks=["finite", "damage_range", "history_monotonic", "exact_BC", "linear_solves"],
        original_limits={name: source_policy[name] for name in (
            "coupled_AT2_relative_inf_limit", "damage_change_atol", "q_change_relative_limit",
            "H_change_relative_limit", "reaction_change_relative_limit", "healing_atol", "range_atol")},
        original_confirmations_required=3,
        drift_envelope=direct.DRIFT_ENVELOPE, diagnostic_only=True, field_equations_validated=False,
        displacement_initial_guess="load_scaled_checked", seed_history_is_floor=False)
    root = args.output
    if not root.parent.is_dir():
        raise FileNotFoundError(root.parent)
    if args.resume:
        stored = json.loads((root/"policy.json").read_text())
        old_end = stored["last_load"]
        stored["last_load"] = policy["last_load"]
        if stored != policy or args.last_load < old_end:
            raise ValueError("warning resume differs from model/policy other than endpoint")
        if args.last_load > old_end:
            atomic_json(root/"policy.json", policy)
    else:
        root.mkdir()
        atomic_json(root/"policy.json", policy)
    print(json.dumps(dict(event="warning_model_ready", model_digest=model.model_digest,
        model_build_seconds=time.monotonic()-start, strict_parent_sha256=parent_sha)), flush=True)
    for index in range(args.fork_after_index+1, end+1):
        load = float(index*step)
        folder = root/f"state_{index:04d}"
        endpoint = folder/"warning.npz"
        inherited_warning = index > args.fork_after_index+1
        if endpoint.is_file() and (folder/"result.json").is_file():
            report = json.loads((folder/"result.json").read_text())
            digest = hashlib.sha256(endpoint.read_bytes()).hexdigest()
            if (report["accepted_reference"] or not report["diagnostic_only"] or not report["warning_usable"]
                    or report["parent_sha256"] != parent_sha or report["model_digest"] != model.model_digest
                    or report["warning_sha256"] != digest or report["inherited_warning"] != inherited_warning
                    or abs(report["load"]-load) > 1e-12 or report["state_index"] != index
                    or not all(report["hard_gates"].values())):
                raise ValueError("warning endpoint ancestry/status/digest differs")
            previous_floor = H_comm.copy()
            with np.load(endpoint, allow_pickle=False) as archive:
                q_comm, d_comm, H_comm = (archive[name] for name in ("q", "d", "H"))
            residual.validate_fields(q_comm, d_comm, H_comm, previous_floor, np.asarray(load))
            if min(H_comm-previous_floor) < 0 or min(d_comm) < -source_policy["range_atol"] or max(d_comm) > 1:
                raise ValueError("saved warning endpoint fails history/range hard checks")
            parent_sha = digest
            summaries.append(report)
            continue
        if (folder/"result.json").exists() or (folder/"failure.json").exists():
            raise ValueError("completed failed warning load requires an explicit retry root")
        load_start = time.monotonic()
        floor_d, floor_H = d_comm.copy(), H_comm.copy()
        d_iter = d_comm.copy()
        q_guess = q_comm*(index/(index-1))
        fixed = model.constraints.constrained_dofs
        q_guess[fixed] = model.constraints.prescribed_vector(load)[fixed]
        previous, consecutive, first = None, 0, 1
        progress, last_H = None, None
        if folder.exists():
            progress = json.loads((folder/"progress.json").read_text())
            path = trial_archive_path(folder, progress)
            if (progress["parent_sha256"] != parent_sha or progress["model_digest"] != model.model_digest
                    or hashlib.sha256(path.read_bytes()).hexdigest() != progress["trial_sha256"]):
                raise ValueError("warning in-progress context/digest differs")
            with np.load(path, allow_pickle=False) as archive:
                q_guess, d_iter, last_H = (archive[name] for name in ("q", "d", "H"))
            previous = q_guess.copy(), d_iter.copy(), last_H, progress["reaction"]
            first, consecutive = progress["sweep"]+1, progress["consecutive"]
        else:
            folder.mkdir()
            if args.seed_state is not None and not inherited_warning:
                q_guess, d_iter = read_trial_seed(args.seed_state, model_digest=model.model_digest,
                    parent_sha256=parent_sha, load=load, n_dofs=model.n_dofs, n_material=model.n_material)
                seed = json.loads((args.seed_state/"progress.json").read_text())
                if seed["trial_sha256"] != args.seed_sha256:
                    raise ValueError("warning numerical seed differs from explicit SHA")
        if previous is not None and (first > args.sweeps_per_load or consecutive >= 3
                                     or (progress is not None and not all(progress["hard_gates"].values()))):
            # Recover publication interrupted after the final progress snapshot.
            if progress is None or last_H is None:
                raise ValueError("warning final progress recovery lacks fields")
            residual.validate_fields(q_guess,d_iter,last_H,floor_H,np.asarray(load))
            report = {key:value for key,value in progress.items() if key not in ("trial_file","trial_sha256")}
            if all(progress["hard_gates"].values()):
                digest = publish_warning_endpoint(folder,q_guess,d_iter,last_H)
                report.update(warning_usable=True,warning_sha256=digest,strict_criteria_passed=consecutive>=3)
                q_comm,d_comm,H_comm,parent_sha = q_guess.copy(),d_iter.copy(),last_H.copy(),digest
            else:
                report["warning_usable"] = False
            atomic_json(folder/"result.json",report)
            summaries.append(report)
            atomic_json(root/"loading_summary.json",summaries)
            if not report["warning_usable"]:
                outcome = dict(complete=False,diagnostic_only=True,hard_stop_load=load,last=report)
                atomic_json(root/"result.json",outcome)
                return (outcome,model,backend) if return_model else outcome
            continue
        try:
            report, hard, q, d, H = None, None, None, None, None
            before = backend.solve_strict(d_iter, load, c.validation.cg_rtol,
                c.validation.cg_atol, c.validation.cg_maxiter, initial_guess=q_guess)
            for sweep in range(first, args.sweeps_per_load+1):
                tick = time.monotonic()
                H_used = np.asarray(residual.evaluate(before.displacement, jnp.asarray(d_iter), None,
                                                     jnp.asarray(floor_H)).trial_history)
                image, damage_info = solve_d(H_used, d_iter)
                d = d_iter+args.omega*(image-d_iter)
                after = backend.solve_strict(d, load, c.validation.cg_rtol,
                    c.validation.cg_atol, c.validation.cg_maxiter, initial_guess=np.asarray(before.displacement))
                q = np.asarray(after.displacement)
                fields = residual.evaluate(after.displacement, jnp.asarray(d), None, jnp.asarray(floor_H))
                H, reaction = np.asarray(fields.trial_history), float(fields.reaction)
                residual.validate_fields(q, d, H, floor_H, np.asarray(load))
                local, nonlocal_term, history = gc0*d, gc1*(D@d), 2*H*(d-1)
                rho = local+nonlocal_term+history
                scale = max(abs(local))+max(abs(nonlocal_term))+max(abs(history))
                if not np.isfinite(scale) or scale <= 0:
                    raise FloatingPointError("warning AT2 term scale is invalid")
                original, hard, changes, change, healing, ratios = gates(d,H,q,rho,
                    floor_d=floor_d,floor_H=floor_H,previous_d=d_iter,load=load,n_material=model.n_material,
                    scale=scale,previous=previous,reaction=reaction,limits=policy["original_limits"])
                consecutive = consecutive+1 if all(original.values()) else 0
                report = dict(state_index=index,load=load,sweep=sweep,model_digest=model.model_digest,
                    parent_sha256=parent_sha,inherited_warning=inherited_warning,
                    accepted_reference=False,diagnostic_only=True,strict_accepted=False,
                    field_equations_validated=False,reference_only=False,consecutive=consecutive,
                    reaction=reaction,damage_max=float(max(d)),damage_min=float(min(d)),
                    history_max=float(max(H)),healing=healing,damage_change_inf=change,changes=changes,
                    damage_inf=float(max(abs(rho))),damage_relative_inf=float(max(abs(rho))/scale),
                    max_displacement_ratio=float(max(ratios)),drift_count=int(sum(ratios>direct.DRIFT_ENVELOPE)),
                    gates={name:bool(value) for name,value in original.items()},hard_gates=hard,
                    warning_failures=[name for name,value in original.items() if not value]
                        + (["consecutive_confirmations"] if consecutive < 3 else []),
                    mechanics_iterations=int(after.cg_iterations),
                    mechanics_relative_residual=float(after.free_residual_relative_norm),
                    damage_solve=damage_info,timing=dict(sweep_seconds=time.monotonic()-tick,
                                                       load_seconds=time.monotonic()-load_start))
                print(json.dumps(report),flush=True)
                atomic_json(folder/f"sweep_{sweep:04d}.json",report)
                binding = write_trial_fields(folder,sweep,q=q,d=d,H=H)
                atomic_json(folder/"progress.json",{**report,**binding})
                if storage["trial_retention"] == "sparse":
                    prune_trial_fields(folder,binding,tail=storage["trial_tail"],stride=storage["trial_stride"])
                if not all(hard.values()):
                    report["warning_usable"] = False
                    break
                previous = q.copy(),d.copy(),H.copy(),reaction
                d_iter,before = d,after
                if consecutive >= 3:
                    break
            if report is None or hard is None or q is None or d is None or H is None:
                raise ValueError("warning load executed no complete sweep")
            if all(hard.values()):
                digest = publish_warning_endpoint(folder,q,d,H)
                report.update(warning_usable=True,warning_sha256=digest,strict_criteria_passed=consecutive>=3)
                q_comm,d_comm,H_comm,parent_sha = q.copy(),d.copy(),H.copy(),digest
            atomic_json(folder/"result.json",report)
            summaries.append(report)
            atomic_json(root/"loading_summary.json",summaries)
            if not report["warning_usable"]:
                outcome = dict(complete=False,diagnostic_only=True,hard_stop_load=load,last=report)
                atomic_json(root/"result.json",outcome)
                return (outcome,model,backend) if return_model else outcome
        except (RuntimeError,FloatingPointError) as error:
            failure = dict(complete=False,diagnostic_only=True,hard_stop_load=load,
                           kind=type(error).__name__,reason=str(error),last_usable_index=index-1)
            atomic_json(folder/"failure.json",failure)
            atomic_json(root/"result.json",failure)
            return (failure,model,backend) if return_model else failure
    outcome = dict(complete=True,diagnostic_only=True,strict_accepted=False,
                   field_equations_validated=False,last=summaries[-1],warning_loads=end-args.fork_after_index)
    atomic_json(root/"result.json",outcome)
    return (outcome,model,backend) if return_model else outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config","source-root","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--fork-after-index",type=int,required=True)
    parser.add_argument("--last-load",type=float,required=True)
    parser.add_argument("--sweeps-per-load",type=int,required=True)
    parser.add_argument("--omega",type=float,required=True)
    parser.add_argument("--seed-state",type=Path)
    parser.add_argument("--seed-sha256")
    parser.add_argument("--damage-backend",choices=("gpu","cpu"),default="gpu")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--trial-retention",choices=("sparse","all"),default=None)
    parser.add_argument("--trial-tail",type=int,default=None)
    parser.add_argument("--trial-stride",type=int,default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
