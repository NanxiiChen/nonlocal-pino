"""Plot saved electropolish fields; no training or reference solve.

Run from repository root:
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python <this file>
"""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ml_collections import ConfigDict
import numpy as np

from examples.corrosion2d.residual_experiment import build_model, load_reference


OUT = Path(__file__).with_suffix("")
ROOT = Path("/root/autodl-tmp/tf-logs/electropolish2d")
RUNS = {label: ROOT / f"20261024-residual-{kind}-seed42"
        for label, kind in (("Original", "phase"), ("Transformed", "latent"))}
TIMES = np.array([0., 1., 2.5, 5.])


def relative_l2(a, b):
    norm = np.linalg.norm(b)
    if norm <= 0:
        raise ValueError("Relative L2 needs a nonzero reference norm")
    return float(np.linalg.norm(a - b) / norm)


def crossing_histogram(phi):
    counts = ((phi[:-1] - .5) * (phi[1:] - .5) < 0).sum(axis=0)
    if np.any(phi == .5):
        raise ValueError("A node equals phi=0.5; adapt the crossing convention explicitly")
    return {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu")
    configs = {label: ConfigDict(json.loads((run / "config.json").read_text())) for label, run in RUNS.items()}
    cfg = configs["Original"]
    ref = load_reference(cfg)
    selected = []
    for time in TIMES:
        match = np.flatnonzero(np.isclose(ref["physical_time"], time, atol=1e-12, rtol=0))
        if len(match) != 1:
            raise ValueError(f"Missing/ambiguous time: {time}")
        selected.append(int(match[0]))
    X, Y = ref["X"], ref["Y"]
    fields = {"Reference": {name: ref[name][selected].astype(np.float64) for name in ("phi", "c")}}
    metrics = {"physical_times_seconds": TIMES.tolist(), "seed": 42, "updates": 5000,
               "reference_path": str(cfg.experiment.reference_path), "runs": {}}
    points = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=jnp.float32)
    times = jnp.asarray(ref["time"][selected], dtype=jnp.float32)
    initial = (RUNS["Original"] / "model_initial.eqx").read_bytes()
    for label, run in RUNS.items():
        c = configs[label]
        assert c.experiment.case == "electropolish2d" and c.train.seed == 42
        assert c.nondim.T_c == 5. and c.train.iterations == 5000
        assert c.physical.to_dict() == cfg.physical.to_dict()
        assert c.experiment.reference_path == cfg.experiment.reference_path
        assert (run / "model_initial.eqx").read_bytes() == initial
        summary = json.loads((run / "summary.json").read_text())
        assert summary["updates"] == 5000 and not summary["hard_initial_condition"]
        with np.load(run / "prediction.npz") as saved:
            for name in ("X", "Y", "time"):
                np.testing.assert_array_equal(saved[name], ref[name])
            fields[label] = {name: saved[name][selected].astype(np.float64) for name in ("phi", "c")}
        net = eqx.tree_deserialise_leaves(run / "model_final.eqx",
                build_model(c, jax.random.split(jax.random.PRNGKey(c.train.seed))[0]))

        @eqx.filter_jit
        def evaluate(model):
            return jax.vmap(lambda t: jax.vmap(model, in_axes=(0, None))(points, t))(times)

        replay = np.asarray(evaluate(net)).reshape(len(TIMES), *X.shape, 2)
        maximum = 0.
        for channel, name in enumerate(("phi", "c")):
            maximum = max(maximum, float(np.max(np.abs(replay[..., channel] - fields[label][name]))))
            np.testing.assert_allclose(replay[..., channel], fields[label][name], atol=5e-6, rtol=0)
        rows = []
        for j, time in enumerate(TIMES):
            row = {"seconds": float(time), "vertical_phi_05_crossing_histogram": crossing_histogram(fields[label]["phi"][j])}
            for name in ("phi", "c"):
                error = relative_l2(fields[label][name][j], fields["Reference"][name][j])
                row[name + "_relative_l2"] = error
                np.testing.assert_allclose(error, summary["final_metrics"]["rel_l2_by_time_" + name][selected[j]], rtol=1e-12)
            rows.append(row)
        metrics["runs"][label] = {"path": str(run), "checkpoint": str(run / "model_final.eqx"),
                                 "cpu_replay_max_abs_difference": maximum, "per_time": rows}
    metrics["reference_vertical_crossing_histograms"] = [crossing_histogram(p) for p in fields["Reference"]["phi"]]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    legend = [Line2D([0], [0], color="black", ls="--", label="Reference interface (phi = 0.5)"),
              Line2D([0], [0], color="#ee503f", label="PINN interface (phi = 0.5)")]

    def decorate(ax, label, j):
        if label != "Reference":
            ax.contour(X, Y, fields[label]["phi"][j], levels=[.5], colors="#ee503f", linewidths=1.)
        ax.contour(X, Y, fields["Reference"]["phi"][j], levels=[.5], colors="black", linestyles="--", linewidths=1.)
        ax.set(xlim=(0, 1), ylim=(0, .5), aspect="equal", xlabel="x / Lc", xticks=[0, .5, 1], yticks=[0, .25, .5])
        if j == 0:
            ax.set_ylabel("y / Lc")

    for name, title in (("phi", "Phase field phi: 0 = electrolyte, 1 = solid"), ("c", "Total concentration c (dimensionless)")):
        fig, axes = plt.subplots(3, 4, figsize=(14, 7.5), constrained_layout=True)
        for row, label in enumerate(fields):
            for j, time in enumerate(TIMES):
                ax = axes[row, j]
                im = ax.pcolormesh(X, Y, fields[label][name][j], cmap="viridis", shading="nearest", vmin=0, vmax=1, rasterized=True)
                decorate(ax, label, j)
                subtitle = "Matched graph reference" if label == "Reference" else f"Relative L2 = {metrics['runs'][label]['per_time'][j][name + '_relative_l2']:.3e}"
                ax.set_title(f"{label} | {time:g} s\n{subtitle}")
        fig.suptitle(f"Electropolish: {title}\nSeed42; final5000-update models; original vs transformed phase residual", fontsize=14)
        fig.colorbar(im, ax=axes, shrink=.8, label=name)
        fig.legend(handles=legend, loc="outside lower center", ncol=2)
        fig.savefig(str(OUT) + f"-{name}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(14, 9.5), constrained_layout=True)
    for row, (name, label) in enumerate((name, label) for name in ("phi", "c") for label in RUNS):
        for j, time in enumerate(TIMES):
            ax = axes[row, j]
            im = ax.pcolormesh(X, Y, fields[label][name][j] - fields["Reference"][name][j], cmap="RdBu_r", shading="nearest", vmin=-1, vmax=1, rasterized=True)
            decorate(ax, label, j)
            ax.set_title(f"{name}: {label} | {time:g} s")
    fig.suptitle("Electropolish signed errors: PINN - reference (same scale in all panels)", fontsize=14)
    fig.colorbar(im, ax=axes, shrink=.8, label="Signed field error")
    fig.legend(handles=legend, loc="outside lower center", ncol=2)
    fig.savefig(str(OUT) + "-errors.png", dpi=150)
    plt.close(fig)

    columns = np.flatnonzero(np.isclose(X[0], .5, rtol=0, atol=1e-12))
    assert len(columns) == 1
    column = int(columns[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, name in zip(axes, ("phi", "c")):
        for label, color, style in (("Reference", "black", "--"), ("Original", "#2467a6", "-"), ("Transformed", "#d86716", "-")):
            ax.plot(Y[:, column], fields[label][name][-1, :, column], color=color, ls=style, label=label)
        ax.set(xlabel="y / Lc", ylabel=name, title=f"{name} at x/Lc = 0.5, t = 5 s", ylim=(-.03, 1.03))
        if name == "phi":
            ax.axhline(.5, color="gray", lw=.8, ls=":")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(str(OUT) + "-profiles.png", dpi=150)
    plt.close(fig)
    metrics["plots"] = {name: str(OUT) + f"-{name}.png" for name in ("phi", "c", "errors", "profiles")}
    metrics["conventions"] = {"field_color_scales": [0, 1], "signed_error_scale": [-1, 1],
                              "L2_scope": "all spatial nodes, selected physical time, saved GPU fields",
                              "crossings": "strict sign changes of phi-0.5 along y in each x column",
                              "contours": "reference black dashed; PINN red solid"}
    OUT.with_suffix(".json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
