"""Render representative saved corrosion fields; no training/reference solve.

Run: JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python <this file>
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

from examples.corrosion2d.residual_experiment import boundary_values, build_model, load_reference


OUT = Path(__file__).with_suffix("")
ROOT = Path("/root/autodl-tmp/tf-logs/corrosion2d")
RUNS = {
    "Original": ROOT / "20261025-transport-physical_difference-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42",
    "Stable KKS/edges": ROOT / "20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42",
}
TIMES = np.array([0., 10., 25., 50.])


def rel(a, b):
    den = np.linalg.norm(b)
    if den <= 0:
        raise ValueError("Relative L2 requires a nonzero reference norm")
    return float(np.linalg.norm(a - b) / den)


def h(p):
    return 3 * p**2 - 2 * p**3


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Render with JAX_PLATFORMS=cpu")
    cfg = ConfigDict(json.loads((RUNS["Original"] / "config.json").read_text()))
    ref = load_reference(cfg)
    indices = []
    for time in TIMES:
        match = np.flatnonzero(np.isclose(ref["physical_time"], time, atol=1e-12, rtol=0))
        if len(match) != 1:
            raise ValueError(f"Missing/ambiguous physical time: {time}")
        indices.append(int(match[0]))
    X, Y = ref["X"], ref["Y"]
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    zero, one, _ = boundary_values(cfg, points)
    free = (~(zero | one)).reshape(X.shape)
    cle, dc = cfg.physical.c_le, cfg.physical.c_se - cfg.physical.c_le
    fields = {"Reference": {key: ref[key][indices].astype(np.float64) for key in ("phi", "c")}}
    # Dimensionless driving concentration: q/c_le = c_L/c_le - 1.
    fields["Reference"]["q_scaled"] = (fields["Reference"]["c"] - dc * h(fields["Reference"]["phi"]) - cle) / cle
    metrics = {"physical_times_seconds": TIMES.tolist(), "seed": 42, "updates": 5000,
               "reference_path": str(cfg.experiment.reference_path), "runs": {}}
    x = jnp.asarray(points, dtype=jnp.float32)
    ts = jnp.asarray(ref["time"][indices], dtype=jnp.float32)
    for label, run in RUNS.items():
        c = ConfigDict(json.loads((run / "config.json").read_text()))
        assert c.train.seed == 42 and c.train.iterations == 5000
        assert c.nondim.T_c == 50. and c.physical.to_dict() == cfg.physical.to_dict()
        with np.load(run / "prediction.npz") as saved:
            np.testing.assert_array_equal(saved["time"], ref["time"])
            np.testing.assert_array_equal(saved["X"], X)
            np.testing.assert_array_equal(saved["Y"], Y)
            fields[label] = {key: saved[key][indices].astype(np.float64) for key in ("phi", "c")}
        key = jax.random.split(jax.random.PRNGKey(c.train.seed))[0]
        net = eqx.tree_deserialise_leaves(run / "model_final.eqx", build_model(c, key))

        @eqx.filter_jit
        def evaluate(m):
            raw = jax.vmap(lambda t: jax.vmap(m.net, in_axes=(0, None))(x, t))(ts)
            return raw, m.constraint(raw)

        raw, replay = (np.asarray(v) for v in evaluate(net))
        for channel, name in enumerate(("phi", "c")):
            np.testing.assert_allclose(replay[..., channel].reshape(fields[label][name].shape),
                                       fields[label][name], atol=5e-6, rtol=0)
        fields[label]["q_scaled"] = (.5 * (np.tanh(raw[..., 1].astype(np.float64)) - 1)).reshape(fields[label]["phi"].shape)
        per_time = []
        for j, time in enumerate(TIMES):
            row = {"seconds": float(time)}
            for name in ("phi", "c"):
                row[name + "_relative_l2_all_nodes"] = rel(fields[label][name][j], fields["Reference"][name][j])
            row["q_relative_l2_free_nodes"] = rel(fields[label]["q_scaled"][j][free], fields["Reference"]["q_scaled"][j][free])
            per_time.append(row)
        summary = json.loads((run / "summary.json").read_text())
        for name in ("phi", "c"):
            np.testing.assert_allclose(per_time[-1][name + "_relative_l2_all_nodes"],
                                       summary["final_metrics"]["terminal_rel_l2_" + name], rtol=1e-12)
        metrics["runs"][label] = {"path": str(run), "checkpoint": str(run / "model_final.eqx"), "per_time": per_time}

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})

    def axes_base(ax, j):
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, .5)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .25, .5])
        if j == 0:
            ax.set_ylabel("y / Lc")
        ax.set_xlabel("x / Lc")

    def overlays(ax, j, label):
        if label != "Reference":
            ax.contour(X, Y, fields[label]["phi"][j], levels=[.5], colors="#ee503f", linewidths=1.1)
        ax.contour(X, Y, fields["Reference"]["phi"][j], levels=[.5], colors="black", linestyles="--", linewidths=1.1)

    legend = [Line2D([0], [0], color="black", ls="--", label="Reference interface (phi = 0.5)"),
              Line2D([0], [0], color="#ee503f", label="PINN interface (phi = 0.5)")]
    for name, title, vmin, vmax in (("phi", "Phase field phi: 0 = electrolyte, 1 = solid", 0, 1),
                                     ("c", "Total concentration c (dimensionless)", 0, 1),
                                     ("q_scaled", "Driving concentration q / c_le = c_L / c_le - 1", -1, 0)):
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), constrained_layout=True)
        for row, label in enumerate(fields):
            for j, time in enumerate(TIMES):
                ax = axes[row, j]
                im = ax.pcolormesh(X, Y, fields[label][name][j], shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
                overlays(ax, j, label)
                axes_base(ax, j)
                suffix = ""
                if label != "Reference":
                    metric_key = "q_relative_l2_free_nodes" if name == "q_scaled" else name + "_relative_l2_all_nodes"
                    error = metrics["runs"][label]["per_time"][j][metric_key]
                    suffix = f" | L2 = {error:.2e}"
                display = "Stable" if label == "Stable KKS/edges" else label
                ax.set_title(f"{display} | {time:g} s{suffix}")
        fig.colorbar(im, ax=axes, shrink=.82, label=name)
        fig.suptitle(title + "\nSeed42; final5000-update models; same matched-graph reference", fontsize=14)
        fig.legend(handles=legend, loc="outside lower center", ncol=2)
        fig.savefig(str(OUT) + f"-{name}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(14, 9), constrained_layout=True)
    for row, (name, label) in enumerate((name, label) for name in ("phi", "c") for label in RUNS):
        for j, time in enumerate(TIMES):
            ax = axes[row, j]
            err = fields[label][name][j] - fields["Reference"][name][j]
            im = ax.pcolormesh(X, Y, err, shading="nearest", cmap="RdBu_r", vmin=-.05, vmax=.05, rasterized=True)
            overlays(ax, j, label)
            axes_base(ax, j)
            display = "Stable" if label == "Stable KKS/edges" else label
            ax.set_title(f"{name}: {display} | {time:g} s")
    fig.colorbar(im, ax=axes, shrink=.85, label="PINN - reference (scale clipped at +/-0.05)", extend="both")
    fig.suptitle("Signed physical-field errors; identical color scale for all panels", fontsize=14)
    fig.legend(handles=legend, loc="outside lower center", ncol=2)
    fig.savefig(str(OUT) + "-errors.png", dpi=150)
    plt.close(fig)
    metrics["plots"] = {name: str(OUT) + f"-{name}.png" for name in ("phi", "c", "q_scaled", "errors")}
    metrics["plot_conventions"] = {"field_scales": {"phi": [0, 1], "c": [0, 1], "q_scaled": [-1, 0]},
                                    "signed_error_scale": [-.05, .05], "contours": "reference black dashed; PINN red solid",
                                    "q_evaluation": "raw KKS network output in float64; free-node norm"}
    OUT.with_suffix(".json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
