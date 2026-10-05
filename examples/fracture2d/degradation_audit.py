"""Energy-bound/source audit and synthetic crack-transmission diagnostics.

Prescribed damage strips are probes of the unchanged mechanics operator, not
AT2 solutions, load-path states or PINN training targets.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.inner_probe import load_snapshot
from examples.fracture2d.rbf_stagger import atomic_json


def plot_transmission(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    probes = json.loads((output / "strip_transmission.json").read_text())
    if not all(sample["explicit_converged"] for sample in probes):
        raise ValueError("transmission plot requires explicitly converged fixed-field probes")
    x = [sample["width_over_horizon"] for sample in probes]
    y = [sample["reaction_over_intact"] for sample in probes]
    if not np.all(np.isfinite(y)) or min(y) <= 0:
        raise ValueError("transmission log plot requires finite positive recorded ratios")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis in axes:
        axis.plot(x, y, "o-")
        axis.axvline(2, color="gray", linestyle="--", label="Band width = twice horizon")
        axis.set_xlabel("Prescribed failed-band width / mechanics horizon")
        axis.set_ylabel("Reaction / intact reaction")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[1].set_yscale("log")
    figure.suptitle("Fixed-support transmission audit; prescribed damage, not a PINN fracture solution")
    figure.savefig(output / "strip_transmission.png", dpi=160)
    plt.close(figure)


def run(args):
    if args.plot_only:
        plot_transmission(args.output)
        return
    config, model, _ = load_snapshot(args.base, args.parent_digest)
    with np.load(args.trial / "fields.npz", allow_pickle=False) as archive:
        damage = archive["d"]
    if damage.shape != (model.n_material,) or not np.all(np.isfinite(damage)) or min(damage) < 0 or max(damage) > 1:
        raise ValueError("energy audit requires the bounded material damage field")
    with np.load(args.modes, allow_pickle=False) as archive:
        free = archive["free_dofs"]
        modes = archive["vectors"]
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    np.testing.assert_array_equal(free, np.asarray(backend.free_dofs))
    if modes.shape != (len(free), 3):
        raise ValueError("energy audit expects three checked displacement modes")
    xy = np.asarray(model.geometry.coordinates)
    zero = jnp.zeros(model.n_material, jnp.float64)
    w0 = np.asarray(backend._center_weights(zero))
    wd = np.asarray(backend._center_weights(jnp.asarray(damage)))
    if min(w0) <= 0 or min(wd) <= 0:
        raise ValueError("center energy weights must be strictly positive")
    alpha = wd / w0
    theoretical_floor = config.physical.k_res / (1 + config.physical.k_res)
    if min(alpha) < theoretical_floor * (1 - 1e-12) or max(alpha) > 1 + 1e-12:
        raise ValueError("degradation center weights violate their physical bounds")
    factor0 = backend._edge_factor(zero)
    factord = backend._edge_factor(jnp.asarray(damage))

    @jax.jit
    def actions(vector):
        return backend._tangent_from_factor(factor0, vector), backend._tangent_from_factor(factord, vector)

    ratios = []
    rng = np.random.default_rng(0)
    test_vectors = [*modes.T, *(rng.normal(size=len(free)) for _ in range(12))]
    for values in test_vectors:
        vector = np.zeros(model.n_dofs)
        vector[free] = values / np.linalg.norm(values)
        intact, damaged = (np.asarray(value) for value in actions(jnp.asarray(vector)))
        E0, Ed = 0.5 * vector @ intact, 0.5 * vector @ damaged
        if E0 <= 0 or not np.isfinite(Ed) or not min(alpha) * E0 * (1 - 1e-8) <= Ed <= max(alpha) * E0 * (1 + 1e-8):
            raise ValueError("independent directional Loewner energy bound failed")
        ratios.append(float(Ed / E0))
    zero_F = np.asarray(backend._deformation_gradient(factor0, jnp.zeros(model.n_dofs, jnp.float64)))
    source = np.asarray(backend.source)
    eta = np.asarray(backend.eta)
    mode_reports = []
    for mode in modes.T:
        vector = np.zeros(model.n_dofs)
        vector[free] = mode
        delta_F = np.asarray(backend._deformation_gradient(factor0, jnp.asarray(vector))) - zero_F
        strain = 0.5 * (delta_F + np.swapaxes(delta_F, 1, 2))
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        density = (0.5 * float(backend._lame_lambda) * trace**2
                   + float(backend._shear_modulus) * np.sum(strain**2, axis=(1, 2)))
        center_density = np.bincount(source, weights=eta * density, minlength=model.n_dofs // 2)
        intact_energy = w0 * center_density
        damaged_energy = wd * center_density
        _, applied = actions(jnp.asarray(vector))
        np.testing.assert_allclose(sum(damaged_energy), 0.5 * vector @ np.asarray(applied), rtol=1e-8, atol=1e-10)
        peak = int(np.argmax(np.linalg.norm(vector.reshape(-1, 2), axis=1)))
        restored = damaged_energy.copy()
        restored[peak] = intact_energy[peak]
        order = np.argsort(damaged_energy)[-8:][::-1]
        mode_reports.append({"peak_node": peak, "peak_coordinate": xy[peak].tolist(),
            "peak_material_damage": float(damage[peak]) if peak < model.n_material else None,
            "intact_directional_energy": float(sum(intact_energy)),
            "damaged_directional_energy": float(sum(damaged_energy)),
            "restore_peak_center_energy_only": float(sum(restored)),
            "dominant_damaged_energy_sources": [{"node": int(i), "coordinate": xy[i].tolist(),
                "damage": float(damage[i]) if i < model.n_material else None,
                "fraction": float(damaged_energy[i] / sum(damaged_energy)),
                "degradation_ratio": float(alpha[i])} for i in order]})
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    args.output.mkdir()
    report = {"diagnostic_only": True, "training_targets": False,
        "direct_model_digest": model.model_digest, "k_res": config.physical.k_res,
        "actual_center_ratio_min": float(min(alpha)), "actual_center_ratio_max": float(max(alpha)),
        "theoretical_ratio_floor": theoretical_floor, "directional_energy_ratios": ratios,
        "modes": mode_reports, "strip_load": args.load, "strip_widths": args.widths,
        "strip_assumption": "prescribed fully damaged material strip centered at y=0.5; no AT2 solve",
        "cg_rtol": 1e-9, "cg_atol": 1e-10, "cg_maxiter": 10000}
    atomic_json(args.output / "energy_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    probes = []
    reference_reaction = None
    for width in (0.0, *args.widths):
        d = np.asarray(abs(xy[:model.n_material, 1] - 0.5) <= width / 2, np.float64) if width > 0 else np.zeros(model.n_material)
        solved = backend.solve(d, args.load, 1e-9, 1e-10, 10000)
        if reference_reaction is None:
            if not bool(solved.explicit_converged):
                raise ValueError("intact transmission control failed its explicit residual check")
            reference_reaction = float(solved.top_reaction)
        sample = {"width": width, "width_over_horizon": width / model.geometry.delta,
            "damaged_nodes": int(np.count_nonzero(d)), "diagnostic_only": True,
            "prescribed_damage_probe": True, "training_targets": False,
            "explicit_converged": bool(solved.explicit_converged),
            "cg_iterations": int(solved.cg_iterations),
            "cg_relative_residual": float(solved.free_residual_relative_norm),
            "reaction": float(solved.top_reaction),
            "reaction_over_intact": float(solved.top_reaction) / reference_reaction}
        probes.append(sample)
        print(json.dumps(sample), flush=True)
        atomic_json(args.output / "strip_transmission.json", probes)
        np.savez_compressed(args.output / f"strip_{len(probes)-1:02d}.npz", q=np.asarray(solved.displacement), d=d)
    plot_transmission(args.output)
    return report, probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--widths", type=float, nargs="+", required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if (not np.isfinite(args.load) or args.load <= 0 or not np.all(np.isfinite(args.widths))
            or args.widths != sorted(set(args.widths)) or min(args.widths) <= 0):
        raise ValueError("strip probe requires positive finite load and ordered unique widths")
    run(args)


if __name__ == "__main__":
    main()
