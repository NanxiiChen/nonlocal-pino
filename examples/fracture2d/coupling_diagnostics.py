"""Read-only, full-sweep diagnostics for the fixed-load A2/Amor/AT2 run."""

import json
import os
from pathlib import Path

import numpy as np


POLICY = {
    "schema": "fracture2d-fixed-load-coupling-diagnostics-v1",
    "at2_normalization": "sum of infinity norms of local/nonlocal/history terms",
    "q_change_normalization": "prescribed displacement",
    "H_change_normalization": "max of previous/current history infinity norms",
    "reaction_change_normalization": "max of previous/current reaction magnitudes",
    "first_sweep_adjacent_changes": "unavailable; no same-load predecessor",
    "early_stopping": False,
    "acceptance_gate": False,
}


def atomic_json(path, value):
    """Atomically publish a numerical report without solver-side imports."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _inf(values):
    return float(np.max(np.abs(values)))


def metrics(report, fields, previous):
    """Compute dimensionless defects without changing any training fields."""
    if not all(np.all(np.isfinite(value)) for value in fields.values()):
        raise FloatingPointError("coupling observer received nonfinite fields")
    local, nonlocal_term, history = (
        fields[name] for name in ("at2_local", "at2_nonlocal", "at2_history")
    )
    rho = local + nonlocal_term + history
    scale = _inf(local) + _inf(nonlocal_term) + _inf(history)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("AT2 equation-term scale must be positive and finite")
    history_shift = 2.0 * (fields["H"] - fields["H_used"]) * (fields["d"] - 1.0)
    location = int(np.argmax(np.abs(rho)))
    result = {
        "sweep": report["sweep"], "load": report["load"],
        "at2_inf": _inf(rho), "at2_rms": float(np.sqrt(np.mean(rho**2))),
        "at2_term_scale": scale, "at2_relative_inf": _inf(rho) / scale,
        "at2_fixed_solved_inf": _inf(fields["at2_fixed_solved"]),
        "at2_fixed_relaxed_inf": _inf(fields["at2_fixed_relaxed"]),
        "at2_history_shift_inf": _inf(history_shift),
        "at2_decomposition_defect_inf": _inf(
            rho - fields["at2_fixed_relaxed"] - history_shift),
        "at2_max_node": location,
        "at2_max_coordinates": fields["coordinates"][location].tolist(),
        "reaction": report["reaction"],
        "mechanics_loss": report["mechanics_loss"],
        "physical_gates_pass": all(report["gates"].values()),
        "adjacent_sweep": None,
    }
    if not np.isclose(result["at2_inf"], report["damage_inf"], rtol=1e-10, atol=1e-12):
        raise ValueError("coupling observer AT2 residual differs from same-pass report")
    if previous is not None:
        old_fields, old_report = previous
        if old_report["sweep"] + 1 != report["sweep"] or old_report["load"] != report["load"]:
            raise ValueError("coupling predecessor must be the adjacent sweep at the same load")
        H_scale = max(_inf(old_fields["H"]), _inf(fields["H"]))
        reaction_scale = max(abs(old_report["reaction"]), abs(report["reaction"]))
        if min(H_scale, reaction_scale, report["load"]) <= 0:
            raise ValueError("adjacent-sweep normalization scales must be positive")
        result["adjacent_sweep"] = {
            "damage_change_inf": _inf(fields["d"] - old_fields["d"]),
            "q_change_relative_inf": _inf(fields["q"] - old_fields["q"]) / report["load"],
            "H_change_relative_inf": _inf(fields["H"] - old_fields["H"]) / H_scale,
            "reaction_change_relative": abs(report["reaction"] - old_report["reaction"]) / reaction_scale,
        }
    return result


def observer(root):
    """Persist per-sweep arrays; previous files make interrupted runs auditable."""
    root = Path(root)

    def observe(report, fields):
        sweep = report["sweep"]
        previous = None
        if sweep > 1:
            with np.load(root / f"fields_{sweep-1:04d}.npz", allow_pickle=False) as archive:
                old_fields = {name: archive[name] for name in archive.files}
            old_report = json.loads((root / f"sweep_{sweep-1:04d}.json").read_text())
            previous = old_fields, old_report
        result = metrics(report, fields, previous)
        path = root / f"fields_{sweep:04d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **fields)
        os.replace(temporary, path)
        path = root / f"coupling_{sweep:04d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        print(json.dumps({"event": "coupling_diagnostic", **result}), flush=True)

    return observe
