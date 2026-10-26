"""Four explicit matched cold starts; no overwrite or checkpoint resume."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/root/autodl-tmp/tf-logs/corrosion2d")
PLAN = [("phase", 0), ("phase", 1000), ("latent", 0), ("latent", 1000)]


def main():
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_COMPILATION_CACHE_DIR="/tmp/nonlocal-pinn-stagger-compilation-cache")
    records = []
    for formulation, stagger_steps in PLAN:
        name = f"20261025-stagger{stagger_steps}-every25-{formulation}-mapkks-ac1e-07-ch1.0-seed42"
        run, log = ROOT / name, Path("/tmp") / f"{name}.log"
        if run.exists():
            raise FileExistsError(run)
        command = [sys.executable, "-u", "-m", "examples.corrosion2d.residual_experiment", "train",
                   "--case", "corrosion2d", "--formulation", formulation,
                   "--stagger-steps", str(stagger_steps), "--stagger-switch-every", "25", "--seed", "42"]
        print("Starting", name, flush=True)
        with log.open("x") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        if run.is_dir():
            (run / "train.log").write_bytes(log.read_bytes())
        record = {"path": str(run), "command": command, "returncode": completed.returncode,
                  "formulation": formulation, "stagger_steps": stagger_steps, "log": str(log)}
        records.append(record)
        Path(__file__).with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
        if completed.returncode:
            print("FAILED", name, "returncode", completed.returncode, log.read_text(), flush=True)
        else:
            summary = json.loads((run / "summary.json").read_text())
            print("Finished", name, json.dumps(summary["final_metrics"]), flush=True)
    if any(r["returncode"] for r in records):
        raise SystemExit("Independent runs failed; see explicit logs and return codes")


if __name__ == "__main__":
    main()
