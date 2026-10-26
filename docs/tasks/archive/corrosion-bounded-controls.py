"""Explicit independent cold starts; no checkpoint resume or reference fitting."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/root/autodl-tmp/tf-logs/corrosion2d")
PLAN = [
    ("phase", 42, 1e-7, "20261025-residual-phase-control-seed42"),
    ("phase", 7, 1e-7, "20261025-residual-phase-control-seed7"),
    ("bounded", 7, 1e-7, "20261025-residual-bounded-eta0.1-mapkks-ac1e-07-ch1.0-seed7"),
    ("phase", 42, 1., "20261025-residual-phase-noscale-seed42"),
    ("bounded", 42, 1., "20261025-residual-bounded-eta0.1-mapkks-ac1.0-ch1.0-seed42"),
]
WORKER = """
import sys
from examples.corrosion2d.configs.residual_pilot import get_config
from examples.corrosion2d.residual_experiment import train
c=get_config('corrosion2d',sys.argv[1],int(sys.argv[2]))
c.loss.ac_scale=float(sys.argv[3])
c.loss.ch_scale=1.0
c.experiment.bounded_eta=0.1
c.log.run_name=sys.argv[4]
train(c)
"""


def main():
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_COMPILATION_CACHE_DIR="/tmp/nonlocal-pinn-bounded-compilation-cache")
    records = []
    for formulation, seed, ac_scale, name in PLAN:
        if (ROOT / name).exists():
            raise FileExistsError(ROOT / name)
        log = Path("/tmp") / f"{name}.log"
        print("Starting", name, flush=True)
        with log.open("x") as stream:
            completed = subprocess.run([sys.executable, "-u", "-c", WORKER,
                                        formulation, str(seed), str(ac_scale), name],
                                       stdout=stream, stderr=subprocess.STDOUT, env=env)
        run = ROOT / name
        if run.is_dir():
            (run / "train.log").write_bytes(log.read_bytes())
        record = {"path": str(run), "formulation": formulation, "seed": seed,
                  "ac_scale": ac_scale, "ch_scale": 1., "bounded_eta": .1,
                  "returncode": completed.returncode, "log": str(log)}
        records.append(record)
        Path(__file__).with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
        if completed.returncode:
            print("FAILED", name, "returncode", completed.returncode,
                  "full traceback:", log.read_text(), flush=True)
        else:
            summary = json.loads((run / "summary.json").read_text())
            metrics = summary["final_metrics"]
            print("Finished", name, metrics["terminal_rel_l2_phi"], metrics["terminal_rel_l2_c"], flush=True)
    if any(record["returncode"] for record in records):
        raise SystemExit("Some independent controls failed; see explicit records and logs")


if __name__ == "__main__":
    main()
