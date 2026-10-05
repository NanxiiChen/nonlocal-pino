"""Minimal numerical refinement of old fixed-load 60, keeping its accepted parent.

Reuses old stagger/operators and restored neural/Adam fields. A completed trial
is only a numerical seed; its history never replaces accepted state-59 history.
Coupled readiness is a new research report, not retroactive strict acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from examples.fracture2d.continuation import _canonical
from examples.fracture2d.coupling_diagnostics import atomic_json, observer
from examples.fracture2d.diagnostic_continuation import _strict_parent
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.inner_probe import load_snapshot
from examples.fracture2d.pinn_paths import physical_pinn_path
from examples.fracture2d.stagger import run as run_stagger
from examples.fracture2d.stagger_checkpoint import SCHEMA_ACCEPTED
from examples.fracture2d.train import _build_grid


def readiness(diagnostic, gates, limits):
    adjacent = diagnostic["adjacent_sweep"]
    checks = {"old_physical_and_mechanics": bool(all(gates.values())),
              "coupled_at2": diagnostic["at2_relative_inf"] <= limits.at2_relative_inf}
    names = {"damage_change_inf": limits.damage_change_inf,
             "q_change_relative_inf": limits.q_change_relative_inf,
             "H_change_relative_inf": limits.H_change_relative_inf,
             "reaction_change_relative": limits.reaction_change_relative}
    for key, bound in names.items():
        checks[key] = adjacent is not None and adjacent[key] <= bound
    return {key: bool(value) for key, value in checks.items()}


def stage_diagnostic(fields, old, previous_report):
    """Same normalization as completed-sweep diagnostics, at one logged state."""
    term_scale = sum(float(np.max(abs(fields[name]))) for name in ("at2_local", "at2_nonlocal", "at2_history"))
    rho = fields["at2_local"] + fields["at2_nonlocal"] + fields["at2_history"]
    q_scale = previous_report["load"]
    H_scale = max(float(np.max(abs(fields["H"]))), float(np.max(abs(old["H"]))))
    reaction_scale = max(abs(fields["reaction"]), abs(previous_report["reaction"]))
    if min(term_scale, q_scale, H_scale, reaction_scale) <= 0:
        raise ValueError("coupled stage-stop normalization must be positive")
    return {"at2_relative_inf": float(np.max(abs(rho))) / term_scale,
            "adjacent_sweep": {
                "damage_change_inf": float(np.max(abs(fields["d"] - old["d"]))),
                "q_change_relative_inf": float(np.max(abs(fields["q"] - old["q"]))) / q_scale,
                "H_change_relative_inf": float(np.max(abs(fields["H"] - old["H"]))) / H_scale,
                "reaction_change_relative": abs(fields["reaction"] - previous_report["reaction"]) / reaction_scale}}


def run(args):
    source = physical_pinn_path(args.source_root)
    seed_path = physical_pinn_path(args.seed_state)
    root = physical_pinn_path(args.output_root)
    source_config = ConfigDict(json.loads((source / "config.json").read_text()))
    parent_config, c = _experiment_configs(source_config, args.source_root, args.output_root,
                                          args.sweeps, tensorboard_root=args.tensorboard_root)
    model = _build_grid(parent_config)
    parent, parent_digest = _strict_parent(parent_config, model)
    seed_config, seed_model, seed = load_snapshot(seed_path, parent_digest)
    if (seed_model.model_digest != model.model_digest or seed_model.static_digest != model.static_digest
            or seed_config.arch != c.arch or seed_config.train.learning_rate != c.train.learning_rate):
        raise ValueError("numerical seed differs from preserved old kernel/network settings")
    c.train.stagger_lr_policy = "linear_tail"
    c.train.stagger_tail_fraction = args.tail_fraction
    c.train.stagger_tail_ratio = args.tail_ratio
    if args.coupling_omega is not None:
        c.stagger.coupling_omega = args.coupling_omega
    c.output.trial_seed_path = str(seed_path / "diagnostic.ckpt.zip")
    c.output.trial_seed_sha256 = hashlib.sha256((seed_path / "diagnostic.ckpt.zip").read_bytes()).hexdigest()
    c.coupling = ConfigDict()
    c.coupling.at2_relative_inf = 1e-4
    c.coupling.damage_change_inf = 1e-4
    c.coupling.q_change_relative_inf = 1e-4
    c.coupling.H_change_relative_inf = 1e-3
    c.coupling.reaction_change_relative = 1e-3
    c.coupling.consecutive = 3
    if args.after_stop:
        c.coupling.after_stop = True
    if args.resume:
        if (root / "config.json").read_text() != _canonical(c):
            raise ValueError("legacy refinement resume policy differs")
        if (root / "result.json").exists():
            raise ValueError("completed legacy refinement requires a new explicit trial")
    else:
        root.mkdir()
        (root / "config.json").write_text(_canonical(c))
    state_dir = root / "state_0060"
    sc = ConfigDict(json.loads(c.to_json_best_effort()))
    sc.data.state_index = 60
    sc.output.dir = str(state_dir)
    sc.output.tensorboard_root = str(Path(c.output.tensorboard_root) / root.name)
    ordinary_observer = observer(state_dir)
    def stop_after(sweep, fields):
        if sweep < 2:
            return False
        with np.load(state_dir / f"fields_{sweep-1:04d}.npz", allow_pickle=False) as archive:
            old = {name: archive[name] for name in ("q", "d", "H")}
        previous_report = json.loads((state_dir / f"sweep_{sweep-1:04d}.json").read_text())
        diagnostic = stage_diagnostic(fields, old, previous_report)
        return all(readiness(diagnostic, {"core_checked": True}, c.coupling).values())
    def observe(report, fields):
        ordinary_observer(report, fields)
        diagnostic = json.loads((state_dir / f"coupling_{report['sweep']:04d}.json").read_text())
        checks = readiness(diagnostic, report["gates"], c.coupling)
        consecutive = 0
        if report["sweep"] > 1:
            previous = json.loads((state_dir / f"readiness_{report['sweep']-1:04d}.json").read_text())
            consecutive = previous["consecutive"]
        consecutive = consecutive + 1 if all(checks.values()) else 0
        atomic_json(state_dir / f"readiness_{report['sweep']:04d}.json", {
            "checks": checks, "consecutive": consecutive,
            "coupled_ready": consecutive >= c.coupling.consecutive,
            "diagnostic_only": True, "strict_accepted": False})
    result, _ = run_stagger(sc, resume=args.resume and state_dir.exists(), initial_state=parent,
        parent_sha256=parent_digest, return_state=True, warning_diagnostic=True,
        parent_schema=SCHEMA_ACCEPTED, trial_initial_state=seed, sweep_observer=observe,
        after_stop=stop_after if args.after_stop else None)
    coupled = json.loads((state_dir / f"readiness_{result['sweep']:04d}.json").read_text())
    result.update({"coupled_readiness": coupled, "accepted_parent_sha256": parent_digest,
                   "numerical_seed_sha256": c.output.trial_seed_sha256,
                   "history_floor": "accepted neural state 59, not numerical trial"})
    atomic_json(root / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tensorboard-root", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--tail-fraction", type=float, required=True)
    parser.add_argument("--tail-ratio", type=float, required=True)
    parser.add_argument("--coupling-omega", type=float,
                        help="Explicit numerical mixing experiment; omitted retains the old kernel's 0.5 iteration")
    parser.add_argument("--after-stop", action="store_true", help="Stop mechanics-after only on same-pass coupled checks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not all(np.isfinite(value) and 0 < value <= 1 for value in (args.tail_fraction, args.tail_ratio)):
        raise ValueError("tail controls must lie in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
