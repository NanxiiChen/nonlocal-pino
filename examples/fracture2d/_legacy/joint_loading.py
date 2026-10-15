"""Checked research continuation of joint RBF-PINN states in fresh processes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from examples.fracture2d._legacy.rbf_stagger import atomic_json


def run(args):
    if min(args.max_evaluations, args.lsmr_maxiter) < 1 or args.max_reweights < 0 or args.extra_q_units < 0:
        raise ValueError("joint loading budgets must be positive and refinement counts nonnegative")
    parent = args.parent.resolve()
    parent_result = json.loads((parent / "result.json").read_text())
    if not parent_result["physical_checks_pass"] or not parent_result["optimizer_success"]:
        raise ValueError("loading parent did not pass joint physical/optimizer checks")
    parent_policy = json.loads((parent / "policy.json").read_text())
    if args.last_load <= parent_result["load"] or args.load_step <= 0:
        raise ValueError("joint loading requires a larger endpoint and positive increment")
    root = args.output.resolve()
    if not root.parent.is_dir():
        raise FileNotFoundError(root.parent)
    values = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    values["resume"] = False
    if args.resume:
        if json.loads((root / "experiment.json").read_text()) != values:
            raise ValueError("joint loading resume changes the experiment")
    else:
        root.mkdir()
        atomic_json(root / "experiment.json", values)
    first = parent_result["load"]
    count_float = (args.last_load - first) / args.load_step
    count = int(round(count_float))
    if abs(count - count_float) > 1e-8:
        raise ValueError("endpoint must be an integer number of explicit increments")
    reports = []
    for index in range(1, count + 1):
        load = first + index * args.load_step
        state = root / f"state_{index:04d}"
        if index == 1 and args.first_trial is not None:
            state = args.first_trial.resolve()
        if state.exists():
            if (not args.resume and not (index == 1 and args.first_trial is not None)) or not (state / "result.json").is_file():
                raise ValueError("incomplete joint state requires a separate explicit retry root")
            policy = json.loads((state / "policy.json").read_text())
            if abs(policy["load"] - load) > 1e-12 or policy["joint_parent"] != str(parent):
                raise ValueError("joint state load/parent identity differs on resume")
            if policy["source_field_sha256"] != hashlib.sha256((parent / "fields.npz").read_bytes()).hexdigest():
                raise ValueError("joint parent field digest changed")
        else:
            command = [sys.executable, "-m", "examples.fracture2d._legacy.joint_rbf",
                "--base", parent_policy["base"], "--source", parent_policy["source"],
                "--source-index", str(parent_policy["source_index"]), "--output", str(state),
                "--parent-digest", parent_policy["parent_digest"], "--load", repr(load),
                "--max-evaluations", str(args.max_evaluations), "--lsmr-maxiter", str(args.lsmr_maxiter),
                "--lsmr-tolerance", str(args.lsmr_tolerance), "--parameter-scaling", "column_norm",
                "--damage-weight", str(args.damage_weight), "--joint-parent", str(parent)]
            command.extend(("--extra-q-units", str(args.extra_q_units)))
            command.extend(("--mechanics-weighting", args.mechanics_weighting))
            command.extend(("--q-feature-source", "parent"))
            print(json.dumps({"event": "joint_load_start", "load": load, "parent": str(parent)}), flush=True)
            with (root / f"state_{index:04d}.log").open("x") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads((state / "result.json").read_text())
        for refinement in range(1, args.max_reweights + 1):
            if report["physical_checks_pass"] and report["optimizer_success"]:
                break
            refined = root / f"state_{index:04d}_reweight_{refinement:02d}"
            if refined.exists():
                if not args.resume or not (refined / "result.json").is_file():
                    raise ValueError("incomplete reweight trial requires an explicit retry root")
                policy = json.loads((refined / "policy.json").read_text())
                if abs(policy["load"] - load) > 1e-12 or policy["joint_parent"] != str(parent):
                    raise ValueError("cached reweight trial has a different load/parent")
                if policy["source_field_sha256"] != hashlib.sha256((parent / "fields.npz").read_bytes()).hexdigest():
                    raise ValueError("cached reweight committed field digest differs")
            else:
                command = [sys.executable, "-m", "examples.fracture2d._legacy.joint_rbf",
                    "--base", parent_policy["base"], "--source", parent_policy["source"],
                    "--source-index", str(parent_policy["source_index"]), "--output", str(refined),
                    "--parent-digest", parent_policy["parent_digest"], "--load", repr(load),
                    "--max-evaluations", str(args.max_evaluations), "--lsmr-maxiter", str(args.lsmr_maxiter),
                    "--lsmr-tolerance", str(args.lsmr_tolerance), "--parameter-scaling", "column_norm",
                    "--damage-weight", str(args.damage_weight), "--joint-parent", str(parent),
                    "--extra-q-units", str(args.extra_q_units),
                    "--mechanics-weighting", "initial_outliers", "--warm-trial", str(state),
                    "--q-feature-source", "parent"]
                print(json.dumps({"event": "joint_reweight_start", "load": load,
                                  "round": refinement, "committed_parent": str(parent)}), flush=True)
                with (root / f"state_{index:04d}_reweight_{refinement:02d}.log").open("x") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state = refined
            report = json.loads((state / "result.json").read_text())
            print(json.dumps({"event": "joint_reweight_result", "round": refinement, **report}), flush=True)
        reports.append(report)
        atomic_json(root / "loading_summary.json", reports)
        print(json.dumps({"event": "joint_load_result", **report}), flush=True)
        if not report["physical_checks_pass"] or not report["optimizer_success"]:
            atomic_json(root / "result.json", {"complete": False, "last_usable_parent": str(parent),
                                               "rejected_load": load, "last": report})
            return
        parent = state
        parent_policy = json.loads((parent / "policy.json").read_text())
    atomic_json(root / "result.json", {"complete": True, "last_usable_parent": str(parent), "last": reports[-1]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-load", type=float, required=True)
    parser.add_argument("--load-step", type=float, required=True)
    parser.add_argument("--max-evaluations", type=int, required=True)
    parser.add_argument("--lsmr-maxiter", type=int, required=True)
    parser.add_argument("--lsmr-tolerance", type=float, required=True)
    parser.add_argument("--damage-weight", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-q-units", type=int, required=True)
    parser.add_argument("--mechanics-weighting", choices=("none", "initial_outliers"), required=True)
    parser.add_argument("--max-reweights", type=int, required=True)
    parser.add_argument("--first-trial", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
