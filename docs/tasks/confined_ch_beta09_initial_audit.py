"""Compare actual initial checkpoints for the same-width beta2 control."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_training import build_network


def verify(run, prior, output):
    if output.exists():
        raise FileExistsError(output)
    current = json.loads((run / "config.json").read_text())
    previous = json.loads((prior / "config.json").read_text())
    assert current["train"]["soap_b2"] == .9 and previous["train"]["soap_b2"] == .99
    expected = json.loads(json.dumps(current))
    expected["train"]["soap_b2"] = .99
    expected["log"]["run_name"] = previous["log"]["run_name"]
    assert expected == previous
    assert "continuation" not in current["train"]
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise RuntimeError("Initial audit must use CPU during the GPU run")
    template = build_network(ConfigDict(current), jax.random.key(0))
    old = eqx.tree_deserialise_leaves(prior / "model_initial.eqx", template)
    new = eqx.tree_deserialise_leaves(run / "model_initial.eqx", template)
    gap, identical, count = 0., True, 0
    for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
        if eqx.is_array(a):
            a, b = np.asarray(a), np.asarray(b)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            gap = max(gap, float(np.max(np.abs(a-b))))
            identical &= bool(np.array_equal(a, b))
            count += 1
    sources = ("examples/confined_ch2d/operator_training.py", "src/nonlocal_pinn/nn/point_conditional.py",
               "src/nonlocal_pinn/training/optimizers.py", "src/nonlocal_pinn/training/runtime.py")
    for source in sources:
        if (run / "source" / source).read_bytes() != (prior / "source" / source).read_bytes():
            raise ValueError(f"Training or sampling source changed: {source}")
    def first_metric(path):
        lines = [line for line in (path / "metrics.jsonl").read_text().splitlines(keepends=True) if line.endswith("\n")]
        first = json.loads(lines[0])
        assert first["step"] == 0
        return first
    metrics = [first_metric(path) for path in (prior, run)]
    for key in ("test_phi_pooled", "test_phi_terminal", "train_phi_pooled", "test_mu_pooled"):
        np.testing.assert_allclose(metrics[0][key], metrics[1][key], rtol=0, atol=1e-10)
    result = dict(status="passed", run=str(run), prior=str(prior), configuration_change={"train.soap_b2": [.99, .9]},
                  effective_shampoo_beta=[.99, .9], initial_array_leaves=count,
                  initial_parameters_bytewise_equal=identical, initial_parameter_maximum_difference=gap,
                  initial_cohort_metrics_match=True, unchanged_training_and_sampling_sources=list(sources),
                  sampling="Same runtime seed/global keys and stateless sampler on identical bank. Old interrupted run lacks an exported actual-draw trace; new full trace audited at completion against global keys and saved width96 prefix.",
                  scope="Actual saved initialization and source/config identity; no continuation or prior optimizer import")
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.prior.resolve(), args.output.resolve())
