"""Explicit readiness gates for frozen-field hybrid stagger iterations."""

import math


def validate_controls(c):
    if type(c.enabled) is not bool:
        raise ValueError("outer_stop.enabled must be boolean")
    if not c.enabled:
        return
    for name in ("min_sweeps", "max_sweeps", "consecutive_passes"):
        value = c[name]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"outer_stop.{name} must be a positive integer")
    if (c.min_sweeps < 2 or c.max_sweeps < c.min_sweeps
            or c.max_sweeps < c.consecutive_passes + 1):
        raise ValueError("outer_stop budget cannot support its minimum/stability confirmations")
    for name in ("inner_AT2_relative_limit", "q_change_limit", "H_change_limit",
                 "reaction_change_limit", "scale_floor"):
        value = c[name]
        if not math.isfinite(value) or value < 0 or name == "scale_floor" and value == 0:
            raise ValueError(f"outer_stop.{name} must be finite nonnegative (scale_floor positive)")


def evaluate(report, c, limits, *, sweep, previous_consecutive):
    """Small updates alone never certify residual accuracy or physical validity."""
    if (type(sweep) is not int or not 1 <= sweep <= c.max_sweeps
            or type(previous_consecutive) is not int or not 0 <= previous_consecutive <= c.max_sweeps):
        raise ValueError("outer stopping sweep/confirmation counter is invalid")
    for name in ("damage_min", "damage_max", "fixed_H_image_damage_min", "fixed_H_image_damage_max"):
        if not math.isfinite(report[name]):
            raise FloatingPointError(f"outer stopping {name} is nonfinite")
    values = {
        "fixed_H_image_AT2": (report["fixed_H_image_AT2_relative"], c.inner_AT2_relative_limit),
        "current_AT2": (report["coupled_AT2_relative"], limits.coupled_AT2_relative_limit),
        "damage_change": (report["damage_change"], limits.damage_change_limit),
        "q_change": (report["q_change_over_load"], c.q_change_limit),
        "H_change": (report["H_change_relative"], c.H_change_limit),
        "reaction_change": (report["reaction_change_relative"], c.reaction_change_limit),
        "healing": (report["healing"], limits.healing_limit),
        "drift": (report["drift_count"], 0),
        "history_decrease": (report["history_decrease"], 0),
        "damage_range": (max(0., -report["damage_min"], report["damage_max"] - 1.), limits.range_atol),
        "image_range": (max(0., -report["fixed_H_image_damage_min"], report["fixed_H_image_damage_max"] - 1.), limits.range_atol),
    }
    checks = {}
    for name, (value, limit) in values.items():
        if value is not None and not math.isfinite(value):
            raise FloatingPointError(f"outer stopping {name} is nonfinite")
        if value is not None and value < 0:
            raise ValueError(f"outer stopping {name} must be nonnegative")
        checks[name] = {"value": value, "limit": float(limit), "pass": value is not None and value <= limit}
    mechanical = report["mechanics"]
    if (type(mechanical["converged"]) is not bool
            or not math.isfinite(mechanical["free_residual_norm"])
            or not math.isfinite(mechanical["convergence_limit"])
            or mechanical["free_residual_norm"] < 0 or mechanical["convergence_limit"] < 0):
        raise ValueError("outer stopping mechanical diagnostics are invalid")
    checks["mechanics"] = {"value": float(mechanical["free_residual_norm"]),
        "limit": float(mechanical["convergence_limit"]),
        "pass": mechanical["converged"] and mechanical["free_residual_norm"] <= mechanical["convergence_limit"]}
    passed = all(check["pass"] for check in checks.values())
    consecutive = previous_consecutive + 1 if passed else 0
    ready = sweep >= c.min_sweeps and consecutive >= c.consecutive_passes
    reason = "converged" if ready else "budget_exhausted" if sweep == c.max_sweeps else "iterating"
    return {"mode": "residual_gated", "checks": checks, "pass_this_sweep": passed,
            "consecutive": consecutive, "converged": ready, "reason": reason}
