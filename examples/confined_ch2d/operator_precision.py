"""Explicit operator compute precision; host reference construction stays FP64."""
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.training.optimizers import make_optimizer


def compute_dtype(c):
    name = c.train.compute_dtype
    if name not in ('float32', 'float64'):
        raise ValueError('Geometry operator train.compute_dtype must be float32 or float64')
    return np.dtype(name)


def compute_arrays(tree, c):
    dtype = compute_dtype(c)
    return jax.tree.map(lambda a:a.astype(dtype) if eqx.is_inexact_array(a) else a, tree)


def make_compute_optimizer(c):
    """Type SOAP beta scalars so bias correction cannot promote FP32 updates."""
    dtype=compute_dtype(c)
    if dtype==np.dtype('float32') and c.train.optimizer=='soap':
        typed=ConfigDict(c.to_dict(),type_safe=False)
        typed.train.soap_b1=np.float32(c.train.soap_b1)
        typed.train.soap_b2=np.float32(c.train.soap_b2)
        return make_optimizer(typed)
    return make_optimizer(c)
