"""Detached beta2=.9 cold run, initialization check and full native validation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    base = Path("/root/autodl-tmp/tf-logs/confined_ch2d")
    prefix = "20261030-confined-width128-beta09-01"
    run = base / "20261030-confined-geometry-b16-t16-width128-beta09-30k-01"
    prior = base / "20261030-confined-geometry-b16-t16-width128-30k-02"
    cold96 = base / "20261030-confined-geometry-b16-t16-wide96-soap-01"
    best = base / "20261030-confined-geometry-b16-t16-wide96-30k-retry-01"
    journal = base / (prefix + "-process.json")
    initial_receipt = root / "docs/tasks/confined-ch-beta09-initial-audit.json"
    python = str(root / ".venv/bin/python")
    preset = "examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_beta09"
    if any(p.exists() for p in (run, journal, initial_receipt)):
        raise FileExistsError("Beta2 control output exists; preserve it")
    stop = json.loads((root / "docs/tasks/confined-ch-width128-beta09-switch.json").read_text())
    assert stop["both_processes_terminated"] and stop["checkpoint"]["step"] >= 15000
    if shutil.disk_usage(base).free < 4 * 2**30:
        raise RuntimeError("Need4GiB free; no automatic cleanup")
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    if used > 1024:
        raise RuntimeError(f"GPU still in use ({used}MiB); refusing a concurrent training process")
    record = dict(status="starting", runner_pid=os.getpid(), runner_session=os.getsid(0), run=str(run),
                  prior_partial_run=str(prior), best_width96_run=str(best), requested_steps=30000,
                  configuration_change={"train.soap_b2": [.99, .9]}, effective_gram_ema=[.99, .9],
                  started_utc=datetime.now(timezone.utc).isoformat(), stages=[])
    def write():
        partial = journal.with_name(journal.name + ".partial")
        partial.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        partial.replace(journal)
    write()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(root))
    env.pop("JAX_PLATFORMS", None)
    env.pop("VIRTUAL_ENV", None)
    cpu = dict(env, JAX_PLATFORMS="cpu")
    def execute(stage, command, environment, monitor_initial):
        entry = dict(stage=stage, command=command, started_utc=datetime.now(timezone.utc).isoformat())
        record["stages"].append(entry)
        record["status"] = stage
        with (base / (prefix + "-" + stage + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            entry.update(pid=child.pid, log=str(base / (prefix + "-" + stage + ".log")))
            record["active_child_pid"] = child.pid
            write()
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if monitor_initial and not initial_receipt.exists() and (run / "metrics.jsonl").exists():
                        raw = (run / "metrics.jsonl").read_text()
                        if raw.endswith("\n"):
                            command = [python, "-m", "docs.tasks.confined_ch_beta09_initial_audit", "--run", str(run),
                                       "--prior", str(prior), "--output", str(initial_receipt)]
                            with (base / (prefix + "-initial-audit.log")).open("x") as auditlog:
                                result = subprocess.run(command, cwd=root, env=cpu, stdout=auditlog, stderr=subprocess.STDOUT)
                            if result.returncode:
                                child.terminate()
                                child.wait(timeout=60)
                                raise RuntimeError("Initialization audit failed; new training stopped")
                            record["initial_audit"] = str(initial_receipt)
                    if monitor_initial and (run / "metrics.jsonl").exists():
                        lines = [x for x in (run / "metrics.jsonl").read_text().splitlines(keepends=True) if x.endswith("\n")]
                        if lines:
                            metric = json.loads(lines[-1])
                            record["latest_evaluation"] = {k:v for k,v in metric.items() if k != "conditions"}
                    record["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
                    write()
        entry.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
        record["active_child_pid"] = None
        write()
        if code:
            raise RuntimeError(f"{stage} exited {code}")
    try:
        execute("training", [python, "examples/confined_ch2d/train.py",
            "--config=examples/confined_ch2d/configs/train_operator_smallscale10_512_width128_beta09.py",
            "--config.log.run_name=" + run.name], env, True)
        assert json.loads((run / "summary.json").read_text())["final_metrics"]["step"] == 30000
        assert json.loads(initial_receipt.read_text())["status"] == "passed"
        stages = [
            ("native", [python, "-m", "examples.confined_ch2d.operator_evaluate", "--run", str(run), "--output", str(run / "evaluation")]),
            ("dense61", [python, "docs/tasks/confined-fourier-dense-evaluate.py", "--run", str(run), "--dense", "/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation", "--step", "30000", "--output", str(run / "dense_evaluation")]),
            ("geometry", [python, "-m", "examples.confined_ch2d.operator_geometry_effect", "--run", str(run), "--dataset", "/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1", "--output", str(run / "geometry_effect")]),
            ("report", [python, "docs/tasks/confined-fourier-training-report.py", "--run", str(run), "--output", str(run / "training_report")]),
            ("cpu_audit", [python, "docs/tasks/confined-fourier-audit.py", "--run", str(run), "--prior", str(cold96), "--output", str(root / "docs/tasks/confined-ch-beta09-audit.json"), "--config", preset]),
        ]
        for stage, command in stages:
            execute(stage, command, cpu, False)
        from examples.confined_ch2d.operator_history import training_records
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        controls, histories = [], {}
        for label, path in (("Width128,beta2=.99 (partial)", prior), ("Width128,beta2=.9", run), ("Width96,beta2=.99", best)):
            rows = training_records(path, "metrics.jsonl")
            histories[label] = {row["step"]:row for row in rows}
            for axis, x in zip(axes, ([r["step"] for r in rows], [r["elapsed_seconds"] / 60 for r in rows]), strict=True):
                axis.plot(x, [100*r["test_phi_pooled"] for r in rows], label=label)
            controls.append(dict(label=label, run=str(path), evaluated_through=rows[-1]["step"],
                                 online_phase_pooled=rows[-1]["test_phi_pooled"], online_phase_terminal=rows[-1]["test_phi_terminal"]))
        matched = []
        a, b = histories["Width128,beta2=.99 (partial)"], histories["Width128,beta2=.9"]
        for step in sorted(set(a) & set(b)):
            matched.append(dict(step=step, beta099_phi=a[step]["test_phi_pooled"], beta09_phi=b[step]["test_phi_pooled"],
                                beta099_terminal=a[step]["test_phi_terminal"], beta09_terminal=b[step]["test_phi_terminal"]))
        for axis, xlabel in zip(axes, ("Updates", "Elapsed minutes"), strict=True):
            axis.set(xlabel=xlabel, ylabel="Reserved development phase L2 (%)", yscale="log")
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        fig.savefig(run / "beta_comparison.png", dpi=190)
        fig.savefig(run / "beta_comparison.pdf")
        plt.close(fig)
        result = dict(status="complete", controls=controls, matched_steps=matched,
                      native=json.loads((run / "evaluation/metrics.json").read_text()),
                      dense61=json.loads((run / "dense_evaluation/metrics.json").read_text()),
                      initial_audit=str(initial_receipt), scope="Cold beta2/Gram EMA .99->.9 control. Old128run intentionally stopped with a verified15kpair, not a completed30k result. Same512/8development cohort/common seed; native only, no default promotion.")
        (root / "docs/tasks/confined-ch-beta09-results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        record.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), results="docs/tasks/confined-ch-beta09-results.json")
        write()
    except Exception as error:
        record.update(status="failed", error=repr(error), failed_utc=datetime.now(timezone.utc).isoformat())
        write()
        raise


if __name__ == "__main__":
    main()
