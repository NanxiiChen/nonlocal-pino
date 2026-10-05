"""Fixed-load outer-iteration experiment from the verified strict state 59.

Only the sweep budget and first-passage stopping change. Training stages,
physical model, learning rate and parent network/optimizer/fields are held
fixed. Each sweep is reported; warning-reference fields are not targets.
"""

import argparse
import json
import os
from pathlib import Path

from ml_collections import ConfigDict

from examples.fracture2d.continuation import _canonical
from examples.fracture2d.pinn_paths import physical_pinn_path
from examples.fracture2d.diagnostic_continuation import _strict_parent
from examples.fracture2d.stagger import _validate_stagger, run as run_stagger
from examples.fracture2d.stagger_checkpoint import SCHEMA_ACCEPTED
from examples.fracture2d.train import _build_grid


def _summarize_sweeps(root, sweeps):
    """Render the predeclared completed budget, never select a passing sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root)
    final = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if final["sweep"] != sweeps or final["accepted"] or not final["diagnostic_only"]:
        raise ValueError("fixed-load summary requires a completed diagnostic budget")
    reports = []
    for sweep in range(1, sweeps + 1):
        report = json.loads((root / "state_0060" / f"sweep_{sweep:04d}.json").read_text(
            encoding="utf-8"
        ))
        if report["sweep"] != sweep or report["load"] != final["load"]:
            raise ValueError("fixed-load sweep report has mismatched load or order")
        reports.append(report)
    if reports[-1] != final:
        raise ValueError("fixed-load summary final sweep differs from final result")
    x = np.arange(1, sweeps + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label in (("damage_inf", "Current coupled AT2 residual"),
                       ("damage_solved_inf", "Fixed-history solved AT2 residual")):
        axes[0, 0].semilogy(x, [r[key] for r in reports], "o-", label=label)
    axes[0, 0].set_ylabel("Residual infinity norm (raw)")
    axes[0, 1].semilogy(x, [r["damage_change_inf"] for r in reports], "o-",
                        label="Damage change per sweep")
    axes[0, 1].semilogy(x, [r["history_change_inf"] for r in reports], "o-",
                        label="History change within sweep")
    axes[0, 1].set_ylabel("Field changes (raw; different units)")
    for key, label in (("d_rmse", "Damage RMSE"), ("d_max_abs_error", "Max absolute error")):
        axes[1, 0].semilogy(x, [r[key] for r in reports], "o-", label=label)
    axes[1, 0].set_ylabel("Damage error vs warning reference")
    axes[1, 1].plot(x, [r["reaction"] for r in reports], "o-", label="PINN reaction")
    axes[1, 1].axhline(final["reference_reaction"], color="black", linestyle="--",
                       label="Warning reference (comparator only)")
    axes[1, 1].set_ylabel("Top reaction")
    for axis in axes.flat:
        axis.set_xlabel("Outer stagger sweep")
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Fixed prescribed displacement {final['load']:.4f}; no first-passage stopping")
    figure.savefig(root / "sweep_summary.png", dpi=160)
    plt.close(figure)
    (root / "sweep_summary.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8"
    )
    return reports


def _experiment_configs(source, source_root, output_root, sweeps, *, tensorboard_root=None):
    """Separate original-parent validation settings from experimental settings."""
    if (type(sweeps) is not int or sweeps <= int(source.stagger.sweeps)
            or source.train.dtype != "float64"
            or int(source.data.end_state_index) != 59
            or int(source.data.state_index) != 1
            or source.stagger.mechanics_scale_policy != "first_load_times_load_ratio"
            or not source.stagger.accept_on_first_passing_report):
        raise ValueError("fixed-load experiment requires a strict-59 FP64 source and increased sweeps")
    if physical_pinn_path(source.output.dir).resolve() != physical_pinn_path(source_root).resolve():
        raise ValueError("fixed-load source config points outside its root")
    if Path(source_root).resolve() == Path(output_root).resolve():
        raise ValueError("fixed-load experiment requires a separate output root")
    parent_config = ConfigDict(json.loads(source.to_json_best_effort()))
    parent_config.output.fork_from = str(Path(source_root).resolve())
    parent_config.output.fork_after_state_index = 59
    parent_config.output.dir = str(Path(output_root).resolve())
    parent_config.output.resume = False
    if tensorboard_root is not None:
        parent_config.output.tensorboard_root = str(Path(tensorboard_root).resolve())
    parent_config.data.end_state_index = 60
    experiment = ConfigDict(json.loads(parent_config.to_json_best_effort()))
    experiment.stagger.sweeps = sweeps
    experiment.stagger.accept_on_first_passing_report = False
    _validate_stagger(experiment)
    return parent_config, experiment


def run(source_root, output_root, sweeps, *, resume=False, coupling_diagnostics=False,
        tensorboard_root=None):
    recorded_source = Path(source_root)
    recorded_output = Path(output_root)
    source_root = physical_pinn_path(recorded_source).resolve()
    root = physical_pinn_path(recorded_output).resolve()
    source = ConfigDict(json.loads((source_root / "config.json").read_text(encoding="utf-8")))
    parent_config, experiment = _experiment_configs(
        source, recorded_source, recorded_output, sweeps, tensorboard_root=tensorboard_root,
    )
    model = _build_grid(parent_config)
    parent, digest = _strict_parent(parent_config, model)
    print(json.dumps({"event": "fixed_load_parent_verified", "state_index": 59,
                      "accepted_sha256": digest, "sweeps": sweeps,
                      "first_passage": False}), flush=True)
    if resume:
        if (root / "config.json").read_text(encoding="utf-8") != _canonical(experiment):
            raise ValueError("fixed-load experiment resume configuration differs")
        if (root / "result.json").exists():
            raise ValueError("fixed-load experiment is already complete")
    else:
        if not root.parent.is_dir():
            raise FileNotFoundError(f"fixed-load output parent does not exist: {root.parent}")
        root.mkdir()
        (root / "config.json").write_text(_canonical(experiment), encoding="utf-8")
    from examples.fracture2d.coupling_diagnostics import POLICY, observer
    policy_path = root / "coupling_policy.json"
    if coupling_diagnostics:
        policy = json.dumps(POLICY, indent=2, sort_keys=True)
        if resume:
            if policy_path.read_text(encoding="utf-8") != policy:
                raise ValueError("coupling diagnostic policy differs on resume")
        else:
            policy_path.write_text(policy, encoding="utf-8")
    elif policy_path.exists():
        raise ValueError("resume requires --coupling-diagnostics for this experiment")
    state_config = ConfigDict(json.loads(experiment.to_json_best_effort()))
    state_config.data.state_index = 60
    state_config.output.dir = str(Path(experiment.output.dir) / "state_0060")
    state_config.output.tensorboard_root = str(Path(experiment.output.tensorboard_root) / root.name)
    # Root configuration can be written before state initialization succeeds.
    # An absent state directory contains no optimizer work to restore.
    state_resume = resume and physical_pinn_path(state_config.output.dir).exists()
    result, _ = run_stagger(
        state_config, resume=state_resume, initial_state=parent,
        parent_sha256=digest, parent_schema=SCHEMA_ACCEPTED,
        return_state=True, warning_diagnostic=True,
        sweep_observer=observer(root / "state_0060") if coupling_diagnostics else None,
    )
    report_path = root / "result.json"
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    os.replace(temporary, report_path)
    _summarize_sweeps(root, sweeps)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coupling-diagnostics", action="store_true")
    parser.add_argument("--tensorboard-root", type=Path)
    args = parser.parse_args()
    run(args.source_root, args.output_root, args.sweeps, resume=args.resume,
        coupling_diagnostics=args.coupling_diagnostics, tensorboard_root=args.tensorboard_root)


if __name__ == "__main__":
    main()
