"""Explicit PINN target ordering on the existing direct-reference load grid."""

import math
import numbers


def reference_indices(data):
    end = data["end_state_index"]
    step = data["load_step"]
    if isinstance(end, bool) or not isinstance(end, numbers.Integral) or end < 1:
        raise ValueError("end_state_index must be a positive neural ordinal")
    end = int(end)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("reference load_step must be positive finite")
    if "reference_indices" not in data:
        return list(range(1, int(end) + 1))  # Legacy uniform schema.
    values = list(data["reference_indices"])
    if len(values) < end:
        raise ValueError("explicit loading sequence is shorter than end_state_index")
    if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) or i < 1 for i in values):
        raise ValueError("reference_indices require positive integer reference states")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("reference_indices must be strictly increasing")
    return [int(i) for i in values[:end]]


def check_fork_loading(stored, current, index):
    """A numerical fork can change future targets, never inherited loading history."""
    if stored["load_step"] != current["load_step"]:
        raise ValueError("split neural fork load step differs")
    if "reference_indices" in stored or "reference_indices" in current:
        old = list(stored["reference_indices"]) if "reference_indices" in stored else list(range(1, index + 1))
        new = list(current["reference_indices"]) if "reference_indices" in current else list(range(1, index + 1))
        if len(old) < index or len(new) < index or old[:index] != new[:index]:
            raise ValueError("split neural fork inherited loading sequence differs")
