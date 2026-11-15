"""Finite CPU-only rendering after the four existing experiment owners finish.

No training, inference, reference solve, README edit, asset promotion, Git action
or deployment. Each stage needs the original owner's successful final audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from docs.tasks.electro_threeway_queue import process_ticks, read, write

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path("/root/autodl-tmp/tf-logs")
JOBS = (
    ('channel', 'channel-sampled-ad-20261114-03', 'sampled-journal.json', 'comparison.json', 81883, 924156816),
    ('electropolish', 'electro-threeway-20261114-03', 'journal.json', 'comparison.json', 284429, 928607818),
    ('melting', 'melting-threeway-20261114-04', 'journal.json', 'comparison.json', 284431, 928607828),
    ('spinodal', 'spinodal-threeway-20261114-05', 'journal.json', 'final.json', 453994, 929829505),
)

SOURCES = (
    "scripts/pino_comparison_data.py", "scripts/pino_comparison_plot.py",
    "scripts/render_pino_three_method.py", "scripts/trajectory_trend.py",
    "scripts/local_trend.py", "docs/tasks/ceramic_fem_comparison_figures.py",
    "docs/tasks/pino_comparison_figure_queue.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs='+', choices=[job[0] for job in JOBS])
    args = parser.parse_args()
    if args.cases is not None and len(set(args.cases)) != len(args.cases):
        raise ValueError('Figure cases must be unique')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    state = dict(status="waiting", pid=os.getpid(), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat(),
                 training=False, inference=False, reference_solves=False,
                 publication=False, source_files=list(SOURCES))
    environment = dict(os.environ, JAX_PLATFORMS="cpu",
                       PYTHONPATH="/tmp/pino-pdf-audit",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def save():
        state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        write(output / "journal.json", state)

    save()
    try:
        for case, directory, journal_name, report_name, pid, ticks in JOBS:
            if args.cases is not None and case not in args.cases:
                continue
            journal = LOGS / directory / journal_name
            report = LOGS / directory / report_name
            state.update(status="waiting", waiting_for=dict(
                case=case, journal=str(journal), pid=pid, start_ticks=ticks,
                final_report=str(report)))
            save()
            while True:
                dependency = read(journal)
                if dependency["pid"] != pid:
                    raise RuntimeError("Original experiment owner changed: " + case)
                if dependency["status"] == "complete":
                    break
                if dependency["status"] not in ("waiting", "running"):
                    raise RuntimeError("Experiment stopped without successful completion: " + case)
                if process_ticks(pid) != ticks:
                    raise RuntimeError("Original experiment process is no longer live: " + case)
                save()
                time.sleep(30)
            if read(report)["status"] != "passed":
                raise ValueError("Final independent audit has not passed: " + case)
            for name in SOURCES:
                if (ROOT / name).read_bytes() != (output / "source" / name).read_bytes():
                    raise RuntimeError("Rendering source changed after qualification: " + name)
            if shutil.disk_usage(output).free < 512 * 1024**2:
                raise RuntimeError("Insufficient plot-output headroom; no files deleted")
            command = [sys.executable, "-m", "scripts.render_pino_three_method",
                       "--case", case, "--comparison", str(report),
                       "--output", str(output / case)]
            if case == "channel":
                command += ["--baseline-comparison", str(
                    LOGS / "channel-fem-full-20261114-01/comparison.json")]
            stage = dict(case=case, command=command, status="running")
            state["stages"].append(stage)
            state["status"] = "running"
            save()
            started = time.perf_counter()
            with (output / (case + ".log")).open("x") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                save()
                while True:
                    try:
                        code = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        save()
            stage.update(status="passed" if code == 0 else "failed",
                         exit_code=code, seconds=time.perf_counter() - started)
            save()
            if code:
                raise RuntimeError("Figure qualification failed: " + case)
            provenance = output / case / "provenance.json"
            if read(provenance)["status"] != "passed":
                raise ValueError("Missing passed figure record: " + case)
            stage["provenance"] = str(provenance)
            save()
        state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
