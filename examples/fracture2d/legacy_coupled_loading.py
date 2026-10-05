"""One old-kernel load from a SHA-bound, coupled-ready warning predecessor.

Only neural predecessor H/d become the next-load floor. Same-load diagnostic
seeds restore networks/Adam/d without committing their trial history.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from examples.fracture2d.continuation import _canonical, _digest
from examples.fracture2d.coupling_diagnostics import atomic_json, metrics, observer
from examples.fracture2d.diagnostic_continuation import _strict_parent, _trial_template
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.legacy_coupling_refine import readiness, stage_diagnostic
from examples.fracture2d.pinn_paths import physical_pinn_path
from examples.fracture2d.stagger import _validate_stagger, _warning_advancement_gates, run as run_stagger
from examples.fracture2d.stagger_checkpoint import SCHEMA_ACCEPTED, SCHEMA_DIAGNOSTIC, read_checkpoint
from examples.fracture2d.train import _build_grid, _reference


LIMITS = dict(at2_relative_inf=1e-4, damage_change_inf=1e-4,
              q_change_relative_inf=1e-4, H_change_relative_inf=1e-3,
              reaction_change_relative=1e-3, consecutive=3)


def require_ready_parent(config):
    """Legacy/default ancestry stays gated; warnings need an explicit policy."""
    if "allow_unready_parent" not in config.coupling:
        return True
    value = config.coupling.allow_unready_parent
    if not isinstance(value, bool):
        raise ValueError("coupling.allow_unready_parent must be boolean")
    return not value


def _preserved_settings(config, source):
    """Allow numerical controls, but require the original model/loss/budgets."""
    if "coupling" not in config:
        raise ValueError("diagnostic requires explicitly stored coupled-readiness limits")
    for name in ("arch", "physical", "nl", "loss"):
        if config[name] != source[name]:
            raise ValueError(f"diagnostic changes preserved {name} settings")
    for name in ("cloud", "cloud_sha256", "reference", "reference_sha256"):
        if config.data[name] != source.data[name]:
            raise ValueError(f"diagnostic changes preserved data.{name}")
    train = dict(config.train)
    for name in ("stagger_lr_policy", "stagger_tail_fraction", "stagger_tail_ratio"):
        train.pop(name, None)
    if train != dict(source.train):
        raise ValueError("diagnostic changes preserved training settings")
    stagger = dict(config.stagger)
    original = dict(source.stagger)
    for values in (stagger, original):
        for name in ("sweeps", "accept_on_first_passing_report", "coupling_omega"):
            values.pop(name, None)
    if stagger != original:
        raise ValueError("diagnostic changes preserved stage budgets/gates")
    if any(config.coupling[name] != value for name, value in LIMITS.items()):
        raise ValueError("diagnostic coupled-readiness limits differ")
    require_ready_parent(config)
    _validate_stagger(config)


def verify_confirmations(state_dir, config, state, *, require_ready):
    """Recompute readiness from recorded full fields; don't trust a ready flag."""
    state_dir = Path(state_dir)
    if len(state) != 7:
        raise ValueError("diagnostic checkpoint must contain networks/Adam and q/d/H")
    previous = None
    consecutive = 0
    final: dict | None = None
    fields = None
    for sweep in range(1, int(config.stagger.sweeps) + 1):
        report = json.loads((state_dir / f"sweep_{sweep:04d}.json").read_text())
        with np.load(state_dir / f"fields_{sweep:04d}.npz", allow_pickle=False) as archive:
            fields = {name: archive[name] for name in archive.files}
        if report["sweep"] != sweep or report["accepted"] or not report["diagnostic_only"]:
            raise ValueError("coupled parent sweep identity/status differs")
        gates = _warning_advancement_gates(config, report)
        if gates != report["gates"]:
            raise ValueError("coupled parent physical gates differ")
        diagnostic = metrics(report, fields, previous)
        stored = json.loads((state_dir / f"coupling_{sweep:04d}.json").read_text())
        if diagnostic != stored:
            raise ValueError("coupled parent diagnostics differ from recorded fields")
        checks = readiness(diagnostic, gates, config.coupling)
        consecutive = consecutive + 1 if all(checks.values()) else 0
        final = dict(checks=checks, consecutive=consecutive,
                     coupled_ready=consecutive >= config.coupling.consecutive,
                     diagnostic_only=True, strict_accepted=False)
        if final != json.loads((state_dir / f"readiness_{sweep:04d}.json").read_text()):
            raise ValueError("coupled parent readiness differs from recomputed checks")
        previous = fields, report
    if final is None or fields is None:
        raise ValueError("coupled parent has no completed sweep evidence")
    for name, value in zip(("q", "d", "H"), state[4:]):
        if not np.array_equal(np.asarray(value), fields[name]):
            raise ValueError(f"coupled parent checkpoint {name} differs from final full fields")
    if not final["checks"]["old_physical_and_mechanics"]:
        raise ValueError("diagnostic parent/seed must pass original physical/mechanics gates")
    if require_ready and not final["coupled_ready"]:
        raise ValueError("next load requires three consecutive coupled-ready sweeps")
    return final


def load_diagnostic(state_dir, digest, index, model, source, strict_state,
                    strict_digest, *, require_ready):
    """Validate descending parent ancestry, snapshot and field confirmations."""
    state_dir = physical_pinn_path(state_dir).resolve()
    path = state_dir / "diagnostic.ckpt.zip"
    if _digest(path) != digest:
        raise ValueError("diagnostic snapshot SHA256 differs from explicit parent/seed")
    config = ConfigDict(json.loads((state_dir / "config.json").read_text()))
    if (config.data.state_index != index or not 60 <= index <= 75
            or physical_pinn_path(config.output.dir).resolve() != state_dir):
        raise ValueError("diagnostic state index/path differs")
    _preserved_settings(config, source)
    if index == 60:
        floor, floor_digest, schema = strict_state, strict_digest, SCHEMA_ACCEPTED
    else:
        _, floor, floor_digest = load_diagnostic(
            config.output.coupled_parent_state, config.output.coupled_parent_sha256,
            index - 1, model, source, strict_state, strict_digest,
            require_ready=require_ready_parent(config))
        schema = SCHEMA_DIAGNOSTIC
    identity = dict(config_sha256=hashlib.sha256(_canonical(config).encode()).hexdigest(),
                    cloud_sha256=config.data.cloud_sha256,
                    reference_sha256=config.data.reference_sha256,
                    direct_model_digest=model.model_digest, precision="float64",
                    state_index=index, warning_diagnostic=True, parent_schema=schema)
    identity["parent_accepted_sha256" if index == 60 else "parent_diagnostic_sha256"] = floor_digest
    metadata, state = read_checkpoint(path, identity, _trial_template(config, model),
                                      schema=SCHEMA_DIAGNOSTIC)
    if (metadata["phase"] != "complete" or metadata["sweep"] != config.stagger.sweeps
            or not 1 <= metadata["step"] <= config.stagger.mechanics_steps):
        raise ValueError("diagnostic parent/seed must be a final complete checkpoint")
    q, d, H = map(np.asarray, state[4:])
    shapes = ((model.n_dofs,), (model.n_material,), (model.n_material,))
    if any(x.shape != shape or x.dtype != np.float64 or not np.all(np.isfinite(x))
           for x, shape in zip((q, d, H), shapes)):
        raise ValueError("diagnostic fields have invalid shape/dtype/values")
    load = _reference(config, model, diagnostic_warning=True)[0]
    fixed = model.constraints.constrained_dofs
    if not np.array_equal(q[fixed], model.constraints.prescribed_vector(load)[fixed]):
        raise ValueError("diagnostic fields violate load constraints")
    if (np.min(d) < -config.stagger.damage_range_atol or np.max(d) > 1
            or np.min(H) < 0 or np.min(H - np.asarray(floor[6])) < 0
            or np.max(np.asarray(floor[5]) - d) > config.stagger.healing_atol):
        raise ValueError("diagnostic fields violate damage/history floor")
    verify_confirmations(state_dir, config, state, require_ready=require_ready)
    # Independently evaluate the final physical state, not merely the stored
    # readiness booleans/AT2 constituents. The original model supplies all laws.
    from examples.fracture2d.fp64 import FP64A2Backend
    from examples.fracture2d.losses import FractureResidual
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    evaluated = residual.evaluate(state[4], state[5], None, floor[6])
    np.testing.assert_allclose(np.asarray(evaluated.trial_history), H, rtol=1e-11, atol=1e-12)
    points = jnp.asarray(model.geometry.coordinates)
    raw = eqx.filter_jit(lambda network: jax.vmap(network)(points))(state[0])
    base = jnp.stack((jnp.zeros_like(points[:, 1]), load * points[:, 1]), axis=1)
    rebuilt_q = residual.lift((base + load * raw).reshape(-1), load)
    np.testing.assert_allclose(np.asarray(rebuilt_q), q, rtol=1e-11, atol=1e-12)
    report = json.loads((state_dir / f"sweep_{config.stagger.sweeps:04d}.json").read_text())
    if report["load"] != load:
        raise ValueError("diagnostic report load differs from checkpoint state")
    np.testing.assert_allclose(float(evaluated.reaction), report["reaction"], rtol=1e-11, atol=1e-10)
    with np.load(state_dir / f"fields_{config.stagger.sweeps:04d}.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(np.asarray(evaluated.damage),
            archive["at2_local"] + archive["at2_nonlocal"] + archive["at2_history"],
            rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(archive["coordinates"], model.geometry.coordinates[:model.n_material])
    with np.load(config.data.reference, allow_pickle=False) as archive:
        first_load = float(archive["loads"][1])
    affine = jnp.stack((jnp.zeros_like(points[:, 1]), first_load * points[:, 1]), axis=1)
    affine_q = residual.lift(affine.reshape(-1), first_load)
    initial_force = backend._residual_from_factor(
        backend._edge_factor(jnp.zeros(model.n_material, jnp.float64)), affine_q)[backend.free_dofs]
    mechanics_scale = float(jnp.sqrt(jnp.mean(initial_force**2))) * load / first_load
    np.testing.assert_allclose(float(jnp.mean((evaluated.mechanics / mechanics_scale)**2)),
                               report["mechanics_loss"], rtol=1e-9, atol=1e-12)
    from examples.fracture2d import nonlocal_baseline as direct
    actual = dict(damage_min=float(np.min(d)), damage_max=float(np.max(d)),
        damage_healing_max=float(np.max(np.maximum(np.asarray(floor[5]) - d, 0))),
        history_decrease_max=float(np.max(np.maximum(np.asarray(floor[6]) - H, 0))),
        drift_count=int(np.count_nonzero(
            np.linalg.norm(q[:2*model.n_material].reshape(-1, 2), axis=1) / load > direct.DRIFT_ENVELOPE)))
    if any(report[name] != value for name, value in actual.items()):
        raise ValueError("diagnostic physical report differs from checkpoint fields")
    return config, state, digest


def coupling_callbacks(state_dir, limits):
    ordinary = observer(state_dir)

    def stop_after(sweep, fields):
        if sweep < 2:
            return False
        with np.load(state_dir / f"fields_{sweep-1:04d}.npz", allow_pickle=False) as archive:
            old = {name: archive[name] for name in ("q", "d", "H")}
        report = json.loads((state_dir / f"sweep_{sweep-1:04d}.json").read_text())
        return all(readiness(stage_diagnostic(fields, old, report),
                             {"core_checked": True}, limits).values())

    def observe(report, fields):
        ordinary(report, fields)
        sweep = report["sweep"]
        diagnostic = json.loads((state_dir / f"coupling_{sweep:04d}.json").read_text())
        checks = readiness(diagnostic, report["gates"], limits)
        consecutive = 0 if sweep == 1 else json.loads(
            (state_dir / f"readiness_{sweep-1:04d}.json").read_text())["consecutive"]
        consecutive = consecutive + 1 if all(checks.values()) else 0
        atomic_json(state_dir / f"readiness_{sweep:04d}.json", dict(
            checks=checks, consecutive=consecutive, coupled_ready=consecutive >= limits.consecutive,
            diagnostic_only=True, strict_accepted=False))

    return observe, stop_after


def run(args):
    if not 61 <= args.state_index <= 75 or args.sweeps < 4:
        raise ValueError("coupled loading requires state 61..75 and at least four sweeps")
    source_root = physical_pinn_path(args.source_root).resolve()
    root = physical_pinn_path(args.output_root).resolve()
    source = ConfigDict(json.loads((source_root / "config.json").read_text()))
    parent_config, _ = _experiment_configs(source, source_root, root, args.sweeps,
                                          tensorboard_root=args.tensorboard_root)
    model = _build_grid(parent_config)
    strict_state, strict_digest = _strict_parent(parent_config, model)
    parent_dir = physical_pinn_path(args.parent_state).resolve()
    parent_config, parent, digest = load_diagnostic(
        parent_dir, args.parent_sha256, args.state_index - 1, model, source,
        strict_state, strict_digest, require_ready=not args.warning_continue)
    if not require_ready_parent(parent_config) and not args.warning_continue:
        raise ValueError("warning ancestry requires explicit --warning-continue")
    parent_readiness = verify_confirmations(parent_dir, parent_config, parent, require_ready=False)
    c = ConfigDict(json.loads(parent_config.to_json_best_effort()))
    if args.warning_continue:
        c.coupling.allow_unready_parent = True
    for name in ("trial_seed_path", "trial_seed_sha256"):
        if name in c.output:
            del c.output[name]
    c.data.state_index = args.state_index
    c.data.end_state_index = args.state_index
    c.stagger.sweeps = args.sweeps
    c.stagger.accept_on_first_passing_report = False
    c.output.dir = str(root)
    c.output.tensorboard_root = str(Path(args.tensorboard_root).resolve())
    c.output.resume = False
    c.output.coupled_parent_state = str(parent_dir)
    c.output.coupled_parent_sha256 = digest
    seed = None
    if args.seed_state is not None:
        seed_dir = physical_pinn_path(args.seed_state).resolve()
        seed_digest = _digest(seed_dir / "diagnostic.ckpt.zip")
        seed_config, seed, _ = load_diagnostic(seed_dir, seed_digest, args.state_index,
            model, source, strict_state, strict_digest, require_ready=False)
        if seed_config.output.coupled_parent_sha256 != digest:
            raise ValueError("same-load seed has a different committed diagnostic parent")
        c.output.trial_seed_path = str(seed_dir / "diagnostic.ckpt.zip")
        c.output.trial_seed_sha256 = seed_digest
    _validate_stagger(c)
    if args.resume:
        if (root / "config.json").read_text() != _canonical(c):
            raise ValueError("coupled loading resume numerical identity differs")
        if (root / "result.json").exists():
            raise ValueError("completed coupled loading requires a new explicit trial")
    else:
        if not root.parent.is_dir():
            raise FileNotFoundError(root.parent)
        root.mkdir()
        (root / "config.json").write_text(_canonical(c))
    state_dir = root / f"state_{args.state_index:04d}"
    sc = ConfigDict(json.loads(c.to_json_best_effort()))
    sc.output.dir = str(state_dir)
    sc.output.tensorboard_root = str(Path(c.output.tensorboard_root) / root.name)
    observe, stop_after = coupling_callbacks(state_dir, c.coupling)
    print(json.dumps(dict(event="diagnostic_parent_verified" if args.warning_continue else "coupled_parent_verified",
        parent_coupled_ready=parent_readiness["coupled_ready"], warning_continue=args.warning_continue,
        state_index=args.state_index - 1,
        diagnostic_sha256=digest, strict_ancestor_sha256=strict_digest,
        history_floor_inf=float(np.max(np.asarray(parent[6]))))), flush=True)
    if args.resume and (state_dir / "diagnostic.ckpt.zip").is_file():
        # Recovery if the core completed atomically but publication of the root
        # result was interrupted. Never rerun a complete numerical checkpoint.
        _, state, _ = load_diagnostic(state_dir, _digest(state_dir / "diagnostic.ckpt.zip"),
            args.state_index, model, source, strict_state, strict_digest, require_ready=False)
        result = json.loads((state_dir / f"sweep_{sc.stagger.sweeps:04d}.json").read_text())
    else:
        result, state = run_stagger(sc, resume=args.resume and state_dir.exists(),
            initial_state=parent, parent_sha256=digest, parent_schema=SCHEMA_DIAGNOSTIC,
            return_state=True, warning_diagnostic=True, trial_initial_state=seed,
            sweep_observer=observe, after_stop=stop_after if c.coupling.after_stop else None)
    coupled = json.loads((state_dir / f"readiness_{result['sweep']:04d}.json").read_text())
    if state is not None:
        verify_confirmations(state_dir, sc, state, require_ready=False)
    warnings = [name for name, passed in coupled["checks"].items() if not passed]
    if coupled["consecutive"] < c.coupling.consecutive:
        warnings.append("three_consecutive_confirmations")
    result.update(coupled_readiness=coupled, diagnostic_parent_sha256=digest,
                  history_floor=f"{'warning diagnostic' if args.warning_continue else 'coupled-ready'} neural state {args.state_index - 1}",
                  warning_continue=args.warning_continue,
                  parent_coupled_ready=parent_readiness["coupled_ready"],
                  coupling_warnings=warnings,
                  strict_accepted=False, state_index=args.state_index)
    atomic_json(state_dir / "result.json", result)
    atomic_json(root / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "parent-state", "output-root", "tensorboard-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--seed-state", type=Path)
    parser.add_argument("--warning-continue", action="store_true",
                        help="Explicitly allow an uncoupled diagnostic parent; original physical/mechanics gates remain hard")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
