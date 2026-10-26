"""Predeclared matched corrosion objective controls; never overwrite a run."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/corrosion-robust-objective-20261026"
SPECS = (("phase", 2026), ("bounded_frozen", 7), ("bounded_frozen", 2026))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    specs = []
    for formulation, seed in SPECS:
        label = f"{formulation}-seed{seed}"
        name = "20261026-objective-" + label
        run = Path("/root/autodl-tmp/tf-logs/corrosion2d") / name
        if run.exists():
            raise FileExistsError(run)
        command = [str(ROOT / ".venv/bin/python"), "-u", "-m", "examples.corrosion2d.train",
                   f"--seed={seed}", f"--config.experiment.formulation={formulation}",
                   "--config.log.run_name=" + name]
        specs.append(dict(label=label, formulation=formulation, seed=seed,
                          run=str(run), command=command, stdout=str(OUT / (label + ".log"))))
    manifest = dict(status="running", maximum_workers=3, updates=5000,
                    selection="Every final5000 checkpoint and late window; no seed selection", runs=specs)
    (OUT / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_deterministic_ops=true", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="2", JAX_COMPILATION_CACHE_DIR="/tmp/corrosion-objective-cache")
    env.pop("JAX_PLATFORMS", None)

    def execute(spec):
        start = time.monotonic()
        with Path(spec["stdout"]).open("x") as stream:
            process = subprocess.Popen(spec["command"], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            print(f"Started {spec['label']}, pid={process.pid}", flush=True)
            rc = process.wait()
        run = Path(spec["run"])
        if run.is_dir():
            (run / "train.log").write_bytes(Path(spec["stdout"]).read_bytes())
        return dict(**spec, returncode=rc, wall_seconds=time.monotonic()-start)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(execute, s) for s in specs]):
            result = future.result()
            results.append(result)
            (OUT / "completed.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"Finished {result['label']}, code={result['returncode']}", flush=True)
    manifest["runs"] = sorted(results, key=lambda r: r["label"])
    manifest["status"] = "completed" if all(r["returncode"] == 0 for r in results) else "failed"
    (OUT / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "completed":
        raise RuntimeError("A predeclared control failed; inspect its retained log")


if __name__ == "__main__":
    main()
