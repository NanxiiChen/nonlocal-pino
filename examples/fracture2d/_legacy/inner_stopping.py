"""Explicit engineering convergence/stagnation windows for fixed subproblems."""

import math


def validate_controls(c, cap):
    for key in ("check_every", "min_updates", "first_load_min_updates", "stable_checks"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"inner_stop.{key} must be a positive integer")
    if max(c["min_updates"], c["first_load_min_updates"]) > cap:
        raise ValueError("inner-stop minimum exceeds a training block budget")
    for key in ("u_change_relative", "d_change_atol", "reaction_atol", "reaction_rtol",
                "objective_atol", "objective_rtol", "residual_atol", "residual_rtol",
                "mechanics_limit", "damage_limit"):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"inner_stop.{key} must be finite nonnegative")
    if type(c["allow_stagnation_warning"]) is not bool:
        raise ValueError("inner_stop.allow_stagnation_warning must be boolean")


class StopWindow:
    """Scalar checkpoint history; fields are evaluated on GPU outside this object."""

    def __init__(self, controls, phase, minimum):
        if phase not in ("before", "damage", "after"):
            raise ValueError("unknown inner training phase")
        self.c, self.phase, self.minimum = controls, phase, minimum
        self.rows = []

    def observe(self, step, *, objective, residual, field_change, reaction, physical_pass):
        if not all(math.isfinite(x) for x in (objective, residual, reaction)):
            raise FloatingPointError("inner stopping diagnostics are nonfinite")
        if field_change is not None and not math.isfinite(field_change):
            raise FloatingPointError("inner stopping field change is nonfinite")
        if residual < 0 or field_change is not None and field_change < 0:
            raise ValueError("inner residual/change must be nonnegative")
        if self.rows and step <= self.rows[-1]["step"]:
            raise ValueError("inner stopping checkpoints must advance")
        self.rows.append(dict(step=step, objective=objective, residual=residual,
                              change=field_change, reaction=reaction, physical=physical_pass))
        self.rows = self.rows[-(self.c["stable_checks"] + 1):]
        if step < self.minimum or len(self.rows) <= self.c["stable_checks"]:
            return None
        if not all(r["physical"] for r in self.rows):
            return None
        limit = self.c["d_change_atol"] if self.phase == "damage" else self.c["u_change_relative"]
        if not all(r["change"] is not None and r["change"] <= limit for r in self.rows[1:]):
            return None
        for name in ("objective", "residual"):
            values = [r[name] for r in self.rows]
            if max(values) - min(values) > self.c[name + "_atol"] + self.c[name + "_rtol"] * max(abs(x) for x in values):
                return None
        if self.phase != "damage":
            for a, b in zip(self.rows, self.rows[1:]):
                if abs(b["reaction"] - a["reaction"]) > self.c["reaction_atol"] + self.c["reaction_rtol"] * max(abs(a["reaction"]), abs(b["reaction"])):
                    return None
        accuracy = self.c["damage_limit"] if self.phase == "damage" else self.c["mechanics_limit"]
        if residual <= accuracy:
            return "converged"
        if self.c["allow_stagnation_warning"]:
            return "stagnation_warning"
        return None
