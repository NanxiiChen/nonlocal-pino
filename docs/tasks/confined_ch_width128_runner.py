"""Detached ordered spatial check -> GPU probe -> cold30k -> native audits."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main(attempt):
    root = Path(__file__).resolve().parents[2]
    base = Path("/root/autodl-tmp/tf-logs/confined_ch2d")
    dataset = Path("/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1")
    if attempt < 1:
        raise ValueError("Require a positive explicit attempt number")
    suffix = f"{attempt:02d}"
    prefix = "20261030-confined-width128-" + suffix
    refined = Path("/root/autodl-tmp/data/confined_ch2d") / ("operator_geometry_smallscale10_h030_check_" + suffix)
    run = base / ("20261030-confined-geometry-b16-t16-width128-30k-" + suffix)
    prior = base / "20261030-confined-geometry-b16-t16-wide96-soap-01"
    best = base / "20261030-confined-geometry-b16-t16-wide96-30k-retry-01"
    journal = base / (prefix + "-process.json")
    probe = base / (prefix + "-preflight.json")
    preset = "examples.confined_ch2d.configs.train_operator_smallscale10_512_width128"
    python = str(root / ".venv/bin/python")
    if any(p.exists() for p in (run, refined, journal, probe)):
        raise FileExistsError("Width128 pipeline output already exists; preserve it")
    if shutil.disk_usage(base).free < 4 * 2**30:
        raise RuntimeError("Need4GiB free for this bounded run; no automatic cleanup")
    record = dict(status="starting", runner_pid=os.getpid(), runner_session=os.getsid(0),
                  run=str(run), journal=str(journal), spatial_reference=str(refined),
                  preflight=str(probe), prior_cold_run=str(prior), best_width96_run=str(best),
                  requested_steps=30000, started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    def write():
        temporary = journal.with_name(journal.name + ".partial")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(journal)
    write()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(root))
    env.pop("VIRTUAL_ENV", None)
    env.pop("JAX_PLATFORMS", None)
    cpu = dict(env, JAX_PLATFORMS="cpu")
    def execute(stage, command, environment):
        entry = dict(stage=stage, command=command, started_utc=datetime.now(timezone.utc).isoformat())
        record["stages"].append(entry)
        record["status"] = stage
        logpath = base / f"{prefix}-{stage}.log"
        with logpath.open("x") as log:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            entry.update(pid=child.pid, session=os.getsid(child.pid), log=str(logpath))
            record["active_child_pid"] = child.pid
            write()
            print(json.dumps(entry), flush=True)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    record["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
                    write()
        entry.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
        record["active_child_pid"] = None
        write()
        if code:
            raise RuntimeError(f"{stage} exited {code}; see {logpath}")
    try:
        execute("spatial_refinement", [python, "-m", "docs.tasks.confined_ch_mesh_refinement",
                "--dataset", str(dataset), "--output", str(refined), "--indices", "514", "518",
                "--mesh-size", ".030"], cpu)
        spatial = json.loads((refined / "validation.json").read_text())
        assert spatial["status"] == "passed"
        (root / "docs/tasks/confined-ch-width128-spatial.json").write_text(json.dumps(spatial, indent=2) + "\n")
        execute("gpu_preflight", [python, "-m", "docs.tasks.confined_ch_width128_probe",
                "--config", preset, "--prior", str(prior), "--output", str(probe), "--updates", "12"], env)
        feasibility = json.loads(probe.read_text())
        if feasibility["status"] != "passed":
            raise RuntimeError("GPU preflight gate did not pass")
        shutil.copyfile(probe, root / "docs/tasks/confined-ch-width128-preflight.json")
        execute("training", [python, "examples/confined_ch2d/train.py",
                "--config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width128.py",
                "--config.log.run_name=" + run.name], env)
        summary = json.loads((run / "summary.json").read_text())
        assert summary["final_metrics"]["step"] == 30000 and summary["start_step"] == 0
        stages = [
            ("native_evaluation", [python, "-m", "examples.confined_ch2d.operator_evaluate", "--run", str(run), "--output", str(run / "evaluation")]),
            ("dense_evaluation", [python, "docs/tasks/confined-fourier-dense-evaluate.py", "--run", str(run), "--dense", "/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation", "--step", "30000", "--output", str(run / "dense_evaluation")]),
            ("geometry_evaluation", [python, "-m", "examples.confined_ch2d.operator_geometry_effect", "--run", str(run), "--dataset", str(dataset), "--output", str(run / "geometry_effect")]),
            ("training_report", [python, "docs/tasks/confined-fourier-training-report.py", "--run", str(run), "--output", str(run / "training_report")]),
            ("cpu_audit", [python, "docs/tasks/confined-fourier-audit.py", "--run", str(run), "--prior", str(prior), "--output", str(root / "docs/tasks/confined-ch-width128-audit.json"), "--config", preset]),
        ]
        for stage, command in stages:
            execute(stage, command, cpu)
        from examples.confined_ch2d.operator_history import training_records
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        comparison = []
        for width, path, color in ((96, best, "C0"), (128, run, "C1")):
            rows = training_records(path, "metrics.jsonl")
            evaluated = json.loads((path / "evaluation/metrics.json").read_text())
            timing = json.loads((path / "summary.json").read_text())
            dense = json.loads((path / "dense_evaluation/metrics.json").read_text())
            comparison.append(dict(width=width, run=str(path), native=evaluated["test"], dense61=dense,
                                   wall_seconds=timing["cumulative_wall_seconds"], warm_seconds=timing["mean_update_seconds_after_100"]))
            for axis, x in zip(axes, ([r["step"] for r in rows], [r["elapsed_seconds"] / 60 for r in rows]), strict=True):
                for split, style in (("test", "-"), ("train", "--")):
                    axis.plot(x, [100*r[split + "_phi_pooled"] for r in rows], style, color=color,
                              label=f"Width{width}, {'reserved dev' if split == 'test' else 'train16monitors'}")
        for axis, label in zip(axes, ("Updates", "Elapsed training minutes"), strict=True):
            axis.set(xlabel=label, ylabel="Phase relative L2 (%)", yscale="log")
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        fig.savefig(run / "width_comparison.png", dpi=190)
        fig.savefig(run / "width_comparison.pdf")
        plt.close(fig)
        result = dict(status="complete", controls=comparison, spatial_refinement=spatial,
                      preflight=feasibility, scope="Width-only numerical recipe, fixed30k; width96measured as exact-state stages, width128fresh cold. One common runtime seed/eight reused development geometries; native only. No default promotion.")
        (root / "docs/tasks/confined-ch-width128-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        record.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), results="docs/tasks/confined-ch-width128-results.json")
        write()
    except Exception as error:
        record.update(status="failed", error=repr(error), failed_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, required=True)
    main(parser.parse_args().attempt)
