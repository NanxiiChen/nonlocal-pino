"""Bounded AT2 ansatzes; output reconstruction leaves the equation unchanged.

D=diag(D_ii)(I-P), C=gc1*D_ii, a=gc0+2H+C. For a learned neighbor field
sigma, d=(2H+C*sigma)/a and r=C*(sigma-P*d). No damage iteration is applied.
"""
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.train import _nonnegative_damage


MODES = ('nonnegative', 'sigmoid', 'exp_square',
         'lifted_sigmoid', 'lifted_exp_square')


def validate_graph(D, H, *, gc0, gc1, roundoff_factor):
    if (D.shape != (len(H), len(H)) or D.dtype != np.float64
            or H.dtype != np.float64 or H.ndim != 1
            or not np.isfinite(D.data).all() or not np.isfinite(H).all()
            or np.min(H) < 0 or np.max(H) <= 0
            or not np.isfinite([gc0, gc1, roundoff_factor]).all()
            or min(gc0, gc1) <= 0 or roundoff_factor < 1):
        raise ValueError('bounded damage requires a finite FP64 graph and nonzero nonnegative H')
    diagonal = D.diagonal()
    off = D.copy()
    off.setdiag(0.)
    off.eliminate_zeros()
    if np.min(diagonal) <= 0 or np.any(off.data > 0):
        raise ValueError('bounded reconstruction requires positive diagonal/nonpositive offdiagonal')
    defect = np.max(abs(np.asarray(D.sum(axis=1)).ravel())/diagonal)
    if defect > roundoff_factor*np.finfo(float).eps:
        raise ValueError('bounded reconstruction requires a row-stochastic neighbor average')
    return diagonal, dict(row_sum_defect=float(defect))


def head(z, mode, *, floor):
    if mode not in MODES or not np.isfinite(floor) or floor <= 0:
        raise ValueError('explicit recognized output mode and positive damage floor required')
    if mode == 'nonnegative':
        return _nonnegative_damage(z, floor)
    if mode in ('sigmoid', 'lifted_sigmoid'):
        return jax.nn.sigmoid(z)
    return jnp.exp(-z*z)


def inverse_head(value, mode, *, floor):
    """Caller validates0<value<1; exponential chooses its positive branch."""
    if mode not in MODES:
        raise ValueError('unrecognized output mode')
    if mode == 'nonnegative':
        return value-floor*floor/(4*value)
    if mode in ('sigmoid', 'lifted_sigmoid'):
        return jnp.log(value)-jnp.log1p(-value)
    return jnp.sqrt(-jnp.log(value))


def reconstruct(sigma, H, diagonal, *, gc0, gc1):
    C = gc1*diagonal
    return (2*H+C*sigma)/(gc0+2*H+C)


def make_image(frozen_image, latent_increment, diagonal, *, mode, gc0, gc1,
               floor, initialization_epsilon):
    """Same initial physical field for all modes; changing H/load are explicit.

    Frozen image is an own-neural predictor, not a direct target. Auxiliary
    initialization epsilon makes inverses finite without clipping predictions.
    """
    if mode not in MODES or not 0 < initialization_epsilon < .5:
        raise ValueError('invalid explicit head/initialization epsilon')
    lifted = mode.startswith('lifted_')

    def anchors(H, load):
        own = frozen_image(H, load)
        sigma = initialization_epsilon+(1-2*initialization_epsilon)*own
        damage = reconstruct(sigma, H, diagonal, gc0=gc0, gc1=gc1)
        initial = sigma if lifted else damage
        return inverse_head(initial, mode, floor=floor), damage

    def image(net, H, load):
        z, _ = anchors(H, load)
        value = head(z+latent_increment(net, H, load), mode, floor=floor)
        return reconstruct(value, H, diagonal, gc0=gc0, gc1=gc1) if lifted else value

    return image, anchors
