"""Full-graph checkpoint probes; no training, inverse reference or data fitting."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.corrosion2d.residual_experiment import build_model, load_reference, nondim, boundary_values
from examples.corrosion2d.residuals import compute_pde
from nonlocal_pinn.operators import build_D


def partition(original, latent, phase, target, free):
    regions = {
        "reference_bulk": ((target < .01) | (target > .99)) & free,
        "reference_interface": (target >= .01) & (target <= .99) & free,
        "correct_solid_bulk": (target > .99) & (phase > .99) & free,
        "correct_liquid_bulk": (target < .01) & (phase < .01) & free,
        "predicted_saturated": ((phase < 1e-5) | (phase > 1 - 1e-5)) & free,
    }
    result = {}
    for name, mask in regions.items():
        if not mask.any():
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": int(mask.sum()), "fraction_of_free_samples": float(mask.sum() / free.sum()),
            "mean_abs_phi_error": float(np.mean(np.abs(phase[mask] - target[mask]))),
            "original_abs_residual_fraction": float(np.abs(original[mask]).sum() / np.abs(original[free]).sum()),
            "latent_abs_residual_fraction": float(np.abs(latent[mask]).sum() / np.abs(latent[free]).sum()),
            "original_abs_residual_mean": float(np.abs(original[mask]).mean()),
            "latent_abs_residual_mean": float(np.abs(latent[mask]).mean()),
        }
    return result


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu")
    root = Path("/root/autodl-tmp/tf-logs/corrosion2d")
    runs = {
        "good_original_kks_1e-7": root / "20261024-residual-phase-seed42",
        "latent_kks_1e-7": root / "20261024-residual-latent-seed42",
        "latent_kks_1e-9": root / "20261024-residual-latent-mapkks-ac1e-09-ch1.0-seed42",
        "latent_direct_1e-9": root / "20261024-residual-latent-mapdirect-ac1e-09-ch1.0-seed42",
    }
    result = {"normalized_times": [.2, .5, 1.], "physical_times_seconds": [10, 25, 50], "runs": {}}
    for label, run in runs.items():
        c = ConfigDict(json.loads((run / "config.json").read_text()))
        ref = load_reference(c)
        points = np.stack([ref["X"].ravel(), ref["Y"].ravel()], axis=-1)
        zero, one, _ = boundary_values(c, points)
        free = np.broadcast_to((~(zero | one))[:, None], (len(points), 3))
        x = jnp.asarray(points, dtype=jnp.float32)
        n = len(points)
        row, col, weight = (np.asarray(ref[k]) for k in ("graph_row", "graph_col", "graph_weight"))
        diag = np.bincount(row, weights=weight, minlength=n).astype(np.float32)
        sub = tuple(jnp.asarray(a) for a in (row, col, -weight, diag))
        op = build_D(x, c.nl.delta, method=c.nl.method, kernel=c.nl.kernel)
        for actual, expected in zip((op._row_idx, op._col_idx, -op._vals, op._diag),
                                    (row, col, weight, diag)):
            np.testing.assert_array_equal(np.asarray(actual), expected)
        times = jnp.array([.2, .5, 1.], dtype=jnp.float32)
        model = build_model(c, jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
        model = eqx.tree_deserialise_leaves(run / "model_final.eqx", model)
        nd = nondim(c)

        @eqx.filter_jit
        def evaluate(net):
            def raw_values(ts):
                return jax.vmap(lambda t: jax.vmap(net.net, in_axes=(0, None))(x, t))(ts)
            raw, raw_t = jax.jvp(raw_values, (times,), (jnp.ones_like(times),))
            fields, fields_t = jax.jvp(net.constraint, (raw,), (raw_t,))
            original, ch = compute_pde(net, x, times, sub, nd, "phase", .5)
            latent, same_ch = compute_pde(net, x, times, sub, nd, "latent", .5)
            return raw, raw_t, fields, fields_t, original, latent, ch, same_ch

        raw, raw_t, fields, fields_t, original, latent, ch, same_ch = (np.asarray(a) for a in evaluate(model))
        np.testing.assert_array_equal(ch, same_ch)
        phase = fields[..., 0].T
        target = ref["phi"][[4, 10, 20]].reshape(3, n).T
        target_c = ref["c"][[4, 10, 20]].reshape(3, n).T
        n_ch, n1, n2, n3, dc, cle = nd
        z, zt = raw[..., 0].T.astype(np.float64), raw_t[..., 0].T.astype(np.float64)
        p = .5 * (1 + np.tanh(z))
        if c.arch.use_hard_constraint:
            ell = (1 - c.physical.c_se + cle) * .5 * (1 + np.tanh(raw[..., 1].T.astype(np.float64)))
            concentration = ell + dc * (3 * p**2 - 2 * p**3)
        else:
            concentration = raw[..., 1].T.astype(np.float64)
        log_cosh = np.logaddexp(z, -z) - np.log(2.)
        edge = weight[:, None] * np.sinh(z[row] - z[col]) * np.exp(log_cosh[row] - log_cosh[col])
        diffusion = np.zeros_like(z)
        np.add.at(diffusion, row, edge)
        terms = {
            "time": .5 * zt,
            "diffusion": .5 * n3 * diffusion,
            "double_well": .5 * n2 * (1 - 2 * p),
            "concentration_drive": -1.5 * n1 * dc * (concentration - dc * (3 * p**2 - 2 * p**3) - cle),
        }
        latent64 = sum(terms.values())
        difference = np.abs(latent64 - latent)
        coefficient = -1.5 * n1 * dc
        dc_error = fields[..., 1].T - target_c
        sensitivity = np.abs(coefficient * dc_error)
        record = {
            "path": str(run), "nd": list(nd), "ac_scale": c.loss.ac_scale,
            "native_graph_arrays_equal": True,
            "mean_abs_original_residual": float(np.abs(original[free]).mean()),
            "mean_abs_latent_residual": float(np.abs(latent[free]).mean()),
            "partitions": partition(original, latent, phase, target, free),
            "term_abs_mean": {key: float(np.abs(value[free]).mean()) for key, value in terms.items()},
            "latent_float64_reassembly_abs_difference_mean": float(difference[free].mean()),
            "latent_float64_reassembly_difference_over_residual_mae": float(difference[free].mean() / np.abs(latent64[free]).mean()),
            "phase_coupling_partial_derivative_wrt_c": coefficient,
            "concentration_error_contribution_mae_fixed_phi": float(sensitivity[free].mean()),
            "physical_phi_mean_absolute_error_free_samples": float(np.abs(phase[free] - target[free]).mean()),
            "terminal_full_grid_phi_relative_l2": float(np.linalg.norm(phase[:, -1] - target[:, -1]) / np.linalg.norm(target[:, -1])),
        }
        if label == "good_original_kks_1e-7":
            # Reference masks/gradients are diagnostics only, never training inputs.
            bulk = ((target < .01) | (target > .99)) & free
            interface = free & ~bulk
            denominator = jnp.asarray(free.sum(), dtype=jnp.float32)

            @eqx.filter_jit
            @eqx.filter_grad
            def residual_gradient(net, mask, formulation):
                ac, _ = compute_pde(net, x, times, sub, nd, formulation, .5)
                return jnp.sum(jnp.abs(ac) * mask) / denominator

            @eqx.filter_jit
            @eqx.filter_grad
            def phase_error_gradient(net):
                predictions = jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x, t))(times)[..., 0].T
                return jnp.sum((predictions - jnp.asarray(target))**2 * jnp.asarray(free)) / jnp.sum(jnp.asarray(target)**2 * jnp.asarray(free))

            def vector(tree):
                return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in jax.tree.leaves(tree) if eqx.is_array(a)])

            error_gradient = vector(phase_error_gradient(model))
            gradients = {}
            for formulation in ("phase", "latent"):
                region_vectors = {name: vector(residual_gradient(model, jnp.asarray(mask), formulation))
                                  for name, mask in (("bulk", bulk), ("interface", interface))}
                region_vectors["all"] = region_vectors["bulk"] + region_vectors["interface"]
                gradients[formulation] = {}
                for name, value in region_vectors.items():
                    gradients[formulation][name] = {
                        "unscaled_parameter_gradient_l1": float(np.abs(value).sum()),
                        "unscaled_parameter_gradient_l2": float(np.linalg.norm(value)),
                        "cosine_with_physical_phi_squared_relative_l2_gradient": float(np.dot(value, error_gradient) / (np.linalg.norm(value) * np.linalg.norm(error_gradient))),
                    }
            record["fixed_checkpoint_ac_parameter_gradients"] = gradients
            record["gradient_probe_scope"] = "Full graph, free nodes at 10/25/50 s, same global normalization for both regions. Unweighted AC only; not historical GradNorm/SOAP update directions. Reference used only to define diagnostic masks and error gradient."
        result["runs"][label] = record
        print(label, json.dumps(record), flush=True)
    # Check the physical-c sensitivity independently by finite differences.
    n_ch, n1, n2, n3, dc, cle = next(iter(result["runs"].values()))["nd"]
    p = np.array([.001, .01, .5, .99, .999])
    concentration = np.full_like(p, .4)
    def reaction(conc):
        q = conc - dc * (3 * p**2 - 2 * p**3) - cle
        return .5 * (n2 * (1 - 2 * p) - 3 * n1 * dc * q)
    eps = 1e-6
    numerical = (reaction(concentration + eps) - reaction(concentration - eps)) / (2 * eps)
    expected = -1.5 * n1 * dc
    np.testing.assert_allclose(numerical, expected, rtol=1e-8)
    result["concentration_sensitivity_finite_difference_relative_error"] = float(np.max(np.abs(numerical / expected - 1)))
    result["sensitivity_amplification_over_original"] = dict(zip(map(str, p), (1 / (4 * p * (1 - p))).tolist()))
    # Uniform, stationary near-solid test: Dphi=0, c=c_se=dc+c_le.
    # This is a local asymptotic diagnostic, not the full pit solution.
    epsilon = np.array([1e-3, 1e-6, 1e-9])
    p = 1 - epsilon
    q = dc + cle - dc * (3 * p**2 - 2 * p**3) - cle
    gp, hp = 2 * p * (1 - p) * (1 - 2 * p), 6 * p * (1 - p)
    original = n2 * gp - n1 * dc * q * hp
    transformed = .5 * (n2 * (1 - 2 * p) - 3 * n1 * dc * q)
    np.testing.assert_allclose(original, 4 * p * (1 - p) * transformed, rtol=1e-12)
    np.testing.assert_allclose(transformed[-1], -.5 * n2, rtol=1e-7)
    result["uniform_stationary_solid_limit"] = {
        "scope": "Dphi=0, phi_t=0, c=c_se. Reactions only; not a claim that diffusion/drive cannot cancel in the actual spatial solution.",
        "epsilon": epsilon.tolist(), "original": original.tolist(),
        "transformed": transformed.tolist(), "transformed_limit": -.5 * n2,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
