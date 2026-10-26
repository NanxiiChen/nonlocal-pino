"""Labelled phase-field L2 monitoring; references never enter the training loss."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _relative(pred, truth):
    denominator = np.linalg.norm(truth)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Relative L2 requires a finite nonzero reference norm")
    return float(np.linalg.norm(pred - truth) / denominator)


class MeltingEvaluator:
    def __init__(self, config, x_base, x_eval, t_eval, run_dir, writer):
        self.config = config
        self.x_base = x_base
        self.x_eval = x_eval
        self.times = np.asarray(t_eval, dtype=np.float64)
        self.writer = writer
        self.path = Path(run_dir) / "metrics.jsonl"
        self.reference = None
        self.reference_indices = None
        self.reference_info = {"analytic": "approximate linear-radius tanh profile"}
        self.sample_analytic = self.analytic(np.asarray(x_eval), self.times)
        self.sample_graph = None
        path = config.eval.graph_reference_path
        if path:
            if config.nl.method != "sparse":
                raise ValueError("The accuracy.py graph reference applies only to nl.method='sparse'")
            with np.load(path, allow_pickle=False) as data:
                ref = {name: data[name] for name in data.files}
            ident = json.loads(str(ref["identity"]))
            expected = {"physical": config.physical.to_dict(), "nl": config.nl.to_dict(),
                        "grid": [config.sample.Nx, config.sample.Ny, config.sample.Nz]}
            for field, value in expected.items():
                if ident[field] != value:
                    raise ValueError(f"Graph reference {field} mismatch: {ident[field]} != {value}")
            if ident["nondim"]["L_c"] != config.nondim.L_c or tuple(ident["nondim"]["xc"]) != tuple(config.nondim.xc):
                raise ValueError("Graph reference length scale/centre mismatch")
            n = config.sample.Nx
            if not (n == config.sample.Ny == config.sample.Nz and n % 2 == 0):
                raise ValueError("Octant reference requires an even cubic grid")
            if tuple(config.nondim.xc) != (.5, .5, .5):
                raise ValueError("Octant reference requires the centred sphere")
            if ref["phi"].shape != (len(ref["time"]), n//2, n//2, n//2):
                raise ValueError("Invalid reference field shape")
            if not np.all(np.isfinite(ref["phi"])) or not np.all(np.isfinite(ref["time"])):
                raise ValueError("Reference contains nonfinite values")
            if np.any(np.diff(ref["time"]) <= 0) or ident["nondim"]["T_c"] <= 0:
                raise ValueError("Reference times must increase and T_c must be positive")
            if not np.allclose(ref["axis"], (np.arange(n//2)+.5)/(n-1), rtol=0, atol=1e-12):
                raise ValueError("Invalid octant coordinate axis")
            physical = ref["time"] * ident["nondim"]["T_c"]
            indices = []
            for t in self.times * config.nondim.T_c:
                match = np.flatnonzero(np.isclose(physical, t, rtol=0, atol=1e-7))
                if len(match) != 1:
                    raise ValueError(f"Reference must contain exactly one snapshot at physical {t} s")
                indices.append(int(match[0]))
            self.reference = ref
            self.reference_indices = indices
            self.sample_graph = np.stack([self.graph(np.asarray(x_eval), i) for i in indices])
            self.reference_info["graph"] = {
                "path": str(Path(path).resolve()),
                "indices": indices, "stored_T_c": ident["nondim"]["T_c"],
                "comparison": "identical dimensional parameters and spatial graph, matched physical snapshot times",
            }
        self.predict_sample = eqx.filter_jit(
            lambda net: jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x_eval, t)[:, 0])(t_eval))
        self.predict_terminal = eqx.filter_jit(
            lambda net, x: jax.vmap(net, in_axes=(0, None))(x, jnp.float32(self.times[-1]))[:, 0])
        (Path(run_dir)/"evaluation.json").write_text(json.dumps(self.reference_info, indent=2)+"\n")

    def analytic(self, points, times):
        c = self.config
        radius = np.linalg.norm(points.astype(np.float64) - np.asarray(c.nondim.xc), axis=1)
        R = c.physical.R0 / c.nondim.L_c
        speed = c.physical.lam * c.nondim.T_c / c.nondim.L_c
        return np.tanh((R - speed * times[:, None] - radius[None, :]) / (np.sqrt(2.) * c.physical.eps_star))

    def graph(self, points, index):
        n = self.config.sample.Nx
        integer = np.rint(points * (n-1)).astype(int)
        if np.any(integer < 0) or np.any(integer >= n) or not np.allclose(points, integer/(n-1), atol=1e-7, rtol=0):
            raise ValueError("Graph reference evaluation requires points on its grid")
        folded = (np.abs(integer - (n-1)/2) - .5).astype(int)
        return self.reference["phi"][index, folded[:, 0], folded[:, 1], folded[:, 2]]

    def evaluate(self, net, step, full_grid):
        pred = np.asarray(self.predict_sample(net), dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            raise FloatingPointError(f"Nonfinite evaluation field at update {step}")
        record = {"step": step, "physical_terminal_s": float(self.times[-1] * self.config.nondim.T_c),
                  "sample_points": len(self.x_eval), "full_grid_points": len(self.x_base) if full_grid else None,
                  "analytic_pooled_rel_l2": _relative(pred, self.sample_analytic),
                  "analytic_terminal_rel_l2": _relative(pred[-1], self.sample_analytic[-1])}
        if self.sample_graph is not None:
            record["graph_pooled_rel_l2"] = _relative(pred, self.sample_graph)
            record["graph_terminal_rel_l2"] = _relative(pred[-1], self.sample_graph[-1])
        if full_grid:
            sums = {"analytic": [0., 0.]}
            if self.reference is not None:
                sums["graph"] = [0., 0.]
            chunk = self.config.eval.chunk_size
            for start in range(0, len(self.x_base), chunk):
                points = self.x_base[start:start+chunk]
                actual = len(points)
                if actual < chunk:
                    points = jnp.pad(points, ((0, chunk-actual), (0, 0)), mode="edge")
                p = np.asarray(self.predict_terminal(net, points), dtype=np.float64)[:actual]
                xx = np.asarray(points)[:actual]
                targets = {"analytic": self.analytic(xx, self.times[-1:])[0]}
                if self.reference is not None:
                    targets["graph"] = self.graph(xx, self.reference_indices[-1])
                if not np.all(np.isfinite(p)):
                    raise FloatingPointError(f"Nonfinite full-grid field at update {step}")
                for name, target in targets.items():
                    sums[name][0] += float(np.sum((p-target)**2))
                    sums[name][1] += float(np.sum(target**2))
            for name, (error, denominator) in sums.items():
                if denominator <= 0:
                    raise ValueError("Zero reference norm in full-grid evaluation")
                record[f"{name}_terminal_full_rel_l2"] = float(np.sqrt(error/denominator))
        self.writer.add_scalar("eval/rel_L2", record["analytic_pooled_rel_l2"], step)
        for key, value in record.items():
            if key.endswith("rel_l2"):
                self.writer.add_scalar(f"eval/{key}", value, step)
        with self.path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")
        self.writer.flush()
        return record
