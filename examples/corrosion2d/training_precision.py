"""Explicit arithmetic precision with common FP32 experiment inputs."""
import equinox as eqx
import jax
import numpy as np


def computation_dtype(config):
    name = config.train.compute_dtype
    if name not in ("float32", "float64"):
        raise ValueError(f"train.compute_dtype must be float32 or float64, got {name}")
    return np.dtype(name)


def cast_floating(tree, dtype):
    """Promote stored values without regenerating data or mutating modules."""
    return jax.tree.map(
        lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


def audit_floating(tree, dtype):
    leaves = [a for a in jax.tree.leaves(tree) if eqx.is_inexact_array(a)]
    if not leaves or any(np.dtype(a.dtype) != dtype for a in leaves):
        raise TypeError(f"Expected {dtype} floating arrays: {[str(a.dtype) for a in leaves]}")
    return {"dtype": str(dtype), "leaves": len(leaves),
            "bytes": sum(a.size * a.dtype.itemsize for a in leaves)}
