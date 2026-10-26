"""Checkpoint-only scaling audit; no training and no reference supervision."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.corrosion2d import train as geometry
from examples.corrosion2d.residual_experiment import (
    boundary_values, build_model, grid, initial_fields, nondim,
)
from examples.corrosion2d.residuals import make_loss_vector
from nonlocal_pinn.nn.constraints import kks_constraint
from nonlocal_pinn.operators import build_D


def proposed_balance(norms, cap):
    clipped = np.clip(norms, 1e-6, 1e6)
    uncapped = clipped.mean() / clipped
    weights = np.clip(uncapped, 1e-6, cap)
    return {
        "gradient_norms": norms.tolist(),
        "norm_floor_active": (norms < 1e-6).tolist(),
        "norm_ceiling_active": (norms > 1e6).tolist(),
        "weight_cap_active": (uncapped > cap).tolist(),
        "proposed_weights": weights.tolist(),
        "weighted_gradient_norms": (weights * norms).tolist(),
    }


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu for this audit")
    root = Path("/root/autodl-tmp/tf-logs/corrosion2d")
    runs = {kind: root / f"20261024-residual-{kind}-seed42" for kind in ("phase", "latent")}
    config = ConfigDict(json.loads((runs["phase"] / "config.json").read_text()))
    _, _, points = grid(config)
    x_all = jnp.asarray(points, dtype=jnp.float32)
    op = build_D(x_all, config.nl.delta, method=config.nl.method, kernel=config.nl.kernel)
    indices, sub_data, core_mask, _ = op.sub_sample(0, config.sample.K_core)
    sub_data = tuple(jnp.asarray(a) for a in sub_data)
    zero, one, _ = boundary_values(config, points)
    mask = jnp.asarray(core_mask & ~(zero | one)[indices])
    x_zero, x_one = geometry._bc_info(config.pits.entries, config.sample.N_bc_dir, config.sample.N_bc_solid)
    x_ic = geometry._sample_ic_points(config.pits.entries, config.sample.N_ic_global, config.sample.N_ic_local)
    initial = jnp.asarray(initial_fields(config, np.asarray(x_ic)), dtype=jnp.float32)
    key_net, key_time = jax.random.split(jax.random.PRNGKey(config.train.seed))
    times = jax.random.uniform(jax.random.fold_in(key_time, 0), (config.sample.Nt,),
                               minval=config.sample.t_min, maxval=1.0)
    template = build_model(config, key_net)
    models = {"initial": eqx.tree_deserialise_leaves(runs["phase"] / "model_initial.eqx", template),
              **{kind + "_final": eqx.tree_deserialise_leaves(run / "model_final.eqx", template)
                 for kind, run in runs.items()}}
    scales = np.array([config.loss.ac_scale, config.loss.ch_scale, 1, 1, 1])
    result = {"case": "corrosion2d", "loss_order": ["ac", "ch", "bc_zero", "bc_one", "ic"],
              "probe": "first deterministic core and time batch; recomputed proposals, not historical training weights",
              "ac_scale": config.loss.ac_scale, "ch_scale": config.loss.ch_scale,
              "nondimensional_coefficients": list(nondim(config)), "checkpoints": {}}
    for formulation in ("phase", "latent"):
        c = ConfigDict(config.to_dict())
        c.experiment.formulation = formulation
        loss = make_loss_vector(c, nondim(c), x_ic, initial, x_zero, x_one)

        @eqx.filter_jit
        def measure(net):
            values, pullback = eqx.filter_vjp(
                lambda n: loss(n, times, x_all[indices], sub_data, mask), net)
            norms = []
            for i in range(5):
                grad = pullback(jax.nn.one_hot(i, 5))[0]
                norms.append(sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            return values, jnp.stack(norms)

        for label, model in models.items():
            values, norms = (np.asarray(a, dtype=np.float64) for a in measure(model))
            entry = {"scaled_losses": values.tolist(),
                     "configured_scaling": proposed_balance(norms, c.loss.gradnorm_weight_cap),
                     "hypothetical_no_pre_scale": proposed_balance(norms / scales, c.loss.gradnorm_weight_cap)}
            result["checkpoints"].setdefault(label, {})[formulation] = entry
            print(label, formulation, json.dumps(entry), flush=True)

    # Independent AD check of the coupled output Jacobian and its inverse action.
    raw = jnp.array([0.7, -0.3])
    phi = 0.5 * (1 + jnp.tanh(raw[0]))
    dc = c.physical.c_se - c.physical.c_le
    dp = 2 * phi * (1 - phi)
    dl = (1 - c.physical.c_se + c.physical.c_le) * 0.5 * (1 - jnp.tanh(raw[1])**2)
    hp = 6 * phi * (1 - phi)
    jac = jax.jacfwd(lambda r: kks_constraint(r, c.physical.c_se, c.physical.c_le))(raw)
    expected = jnp.array([[dp, 0], [dc * hp * dp, dl]])
    np.testing.assert_allclose(jac, expected, rtol=1e-6, atol=1e-7)
    residual = jnp.array([0.3, -0.7])
    inverse_action = jnp.array([residual[0] / dp, (residual[1] - dc * hp * residual[0]) / dl])
    np.testing.assert_allclose(jac @ inverse_action, residual, rtol=1e-6, atol=1e-7)
    result["coupled_jacobian_ad_max_error"] = float(jnp.max(jnp.abs(jac - expected)))
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
