"""Evaluate actual saved channel checkpoints on the qualified reserved cohort."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


STEPS = (0, 5000, 10000, 15000, 20000)
FIELDS = (("phi", slice(0, 1), 0), ("velocity", slice(2, 4), 1),
          ("pressure", slice(4, 5), 1), ("chemical", slice(1, 2), 1))


def read(path):
    return json.loads(Path(path).read_text())


def evaluate(comparison, output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.cohort import load
    from examples.channel_nsch2d.cohort_evaluate import field_metrics
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.geometry import with_nominal
    from examples.channel_nsch2d.network import build_network, encode, fields
    from examples.channel_nsch2d.problem import identity, walls
    from examples.channel_nsch2d.response_queries import validate_arrays, protocol

    report = read(comparison)
    if report["status"] != "passed" or report["paired"]["updates"] != 20000:
        raise ValueError("Require completed paired 20k comparison")
    output.mkdir(parents=True, exist_ok=False)
    arms = []
    started = time.monotonic()
    shared_references = None
    for method, run_name in (("nonlocal", report["historical"]), ("fem", report["fem_run"])):
        run = Path(run_name)
        c = ConfigDict(read(run / "config.json"))
        final = read(run / "unseen3/evaluation.json")
        manifest = Path(final["manifest"])
        cohort = load(manifest, c)
        gate = read(manifest.parent / "cohort-reference-validation.json")
        if (gate["status"] != "passed" or gate["cases"] != 3 or cohort["count"] != 3
                or gate["common_query_protocol"] != protocol(c, cohort["query_grid"])):
            raise ValueError("Require the same three qualified reserved references")
        configure_gpu_policy(c.train.gpu_policy)
        jax.config.update("jax_enable_x64", c.train.compute_dtype == "float64")
        jax.config.update("jax_default_matmul_precision", c.train.matmul_precision)
        dtype = np.dtype(c.train.compute_dtype)
        key, _ = jax.random.split(jax.random.PRNGKey(c.train.seed))
        template = build_network(c, key)
        predict = eqx.filter_jit(lambda model, data, x, t:
            fields(model, encode(model, data), x, t, data["amplitudes"], c))
        rows = {step: [] for step in STEPS}
        references = []
        final_checks = []
        for case in cohort["cases"]:
            index = case["index"]
            qualified = read(Path(case["root"]) / "reference-validation.json")
            a = np.asarray(case["amplitudes"])
            if (qualified["status"] != "passed" or qualified["identity"] !=
                    json.loads(json.dumps(identity(with_nominal(c, a))))):
                raise ValueError(f"Invalid reference identity: {index}")
            reference_path = Path(qualified["reference"]).resolve(strict=True)
            references.append(str(reference_path))
            with np.load(reference_path, allow_pickle=False) as f:
                ref = {k: f[k] for k in f.files}
            if validate_arrays(ref, c) != gate["common_query_protocol"]:
                raise ValueError("Reserved query protocol mismatch")
            np.testing.assert_array_equal(a, ref["amplitudes"])
            if ref["fields"].shape != (13, 3840, 5) or not np.isfinite(ref["fields"]).all():
                raise ValueError("Unexpected reserved field shape or nonfinite reference")
            raw, _ = geometry_data(a, c)
            data = {k: jnp.asarray(v, dtype=v.dtype if np.issubdtype(v.dtype, np.integer) else dtype)
                    for k, v in raw.items()}
            # Match the original cohort evaluator's full query ordering and shape.
            roots, _ = np.polynomial.legendre.leggauss(64)
            xx, ee = np.meshgrid(np.linspace(0, c.geometry.length, 17), (roots+1)/2, indexing="ij")
            b, t = walls(xx, a, c, xp=np)
            section = np.stack((xx, b+ee*(t-b)), axis=-1).reshape(-1, 2)
            query = np.concatenate((ref["points"], ref["common_points"], section, raw["boundary"][:, :2]))
            x = jnp.asarray(query, dtype=dtype)
            times = jnp.asarray(ref["time"], dtype=dtype)
            bottom, top = walls(ref["points"][:, 0], a, c, xp=np)
            for step in STEPS:
                name = "model_initial.eqx" if step == 0 else f"ckpt_{step:06d}.eqx"
                checkpoint = (run / name).resolve(strict=True)
                model = eqx.tree_deserialise_leaves(checkpoint, template)
                before = time.monotonic()
                value = np.asarray(predict(model, data, x, times), dtype=np.float64)[:, :3840]
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite {method}/{step}/{index}")
                metrics = field_metrics(value, ref["fields"], top-bottom)
                for field, channel, begin in FIELDS:
                    target = ref["fields"][begin:, :, channel]
                    independent = np.sqrt(np.sum(np.square(value[begin:, :, channel]-target)) /
                                          np.sum(np.square(target)))
                    np.testing.assert_allclose(independent, metrics[field], rtol=1e-12, atol=1e-14)
                destination = output / method / f"step_{step:06d}" / f"case_{index:03d}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, fields=value, points=ref["points"], time=ref["time"],
                                    amplitudes=a, reference_path=str(reference_path))
                rows[step].append(dict(index=index, errors={k: metrics[k] for k, _, _ in FIELDS},
                                       prediction=str(destination), reference=str(reference_path),
                                       checkpoint=str(checkpoint)))
                if step == 20000:
                    saved = run / "unseen3" / f"case_{index:03d}" / "prediction.npz"
                    with np.load(saved, allow_pickle=False) as f:
                        np.testing.assert_array_equal(f["reference"], ref["fields"])
                        gap = float(np.max(np.abs(value-f["fields"])))
                        np.testing.assert_allclose(value, f["fields"], rtol=0, atol=2e-5)
                    final_checks.append(dict(index=index, saved=str(saved), maximum_absolute_gap=gap))
                print(json.dumps(dict(method=method, case=index, step=step, seconds=time.monotonic()-before,
                                      errors=rows[step][-1]["errors"])), flush=True)
        if shared_references is not None and references != shared_references:
            raise ValueError("Methods evaluated on different references")
        shared_references = references
        convergence = []
        for step, samples in rows.items():
            if [x["index"] for x in samples] != [0, 1, 2]:
                raise ValueError("Incomplete reserved cohort")
            aggregate = {field: dict(mean=float(np.mean([s["errors"][field] for s in samples])),
                                    std=float(np.std([s["errors"][field] for s in samples], ddof=1)))
                         for field, _, _ in FIELDS}
            convergence.append(dict(step=step, samples=samples, aggregate=aggregate))
        arms.append(dict(method=method, run=str(run), config=str(run/"config.json"), manifest=str(manifest),
                         seed=c.train.seed, convergence=convergence, final_replay=final_checks))
    result = dict(schema="channel_fem_checkpoint_cohort_v1", status="passed", comparison=str(comparison),
                  steps=STEPS, samples=3, std_ddof=1, arms=arms, elapsed_seconds=time.monotonic()-started,
                  reference_paths=shared_references, evaluator=str(Path(__file__).resolve()),
                  device=jax.devices()[0].device_kind, jax_version=jax.__version__,
                  metric="Unweighted space-time field L2; phase all 13 times; joint velocity, pressure and mu positive 12 times")
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed", output=str(output), seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.comparison.resolve(strict=True), args.output.resolve())
