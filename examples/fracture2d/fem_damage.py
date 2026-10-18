"""Material-P1 scalar AT2 operator and frozen-history lumped FE energy.

Natural zero-flux damage BCs follow from the element energy. Grip triangles
are excluded. The H-only constant is dropped; no damage labels are used.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags


class P1DamageOperator:
    """D=M_lumped^-1 S; identical weak system for direct and neural damage."""

    def __init__(self, mechanics):
        n, nc = mechanics.n_material, mechanics.n_material_cells
        self.triangles = mechanics.triangles[:nc].copy()
        self.areas = mechanics.areas[:nc].copy()
        self.gradients = np.stack((mechanics.B[:nc, 0, 0::2],
                                   mechanics.B[:nc, 1, 1::2]), axis=2)
        if self.triangles.min() < 0 or self.triangles.max() >= n:
            raise ValueError('material-P1 damage triangles include grip/invalid nodes')
        volumes = np.bincount(self.triangles.ravel(),
            weights=np.repeat(self.areas/3, 3), minlength=n)
        if not np.allclose(volumes, mechanics.volumes, rtol=1e-10, atol=1e-20):
            raise ValueError('P1 damage requires material triangle-lumped nodal volumes')
        elements = self.areas[:, None, None]*np.einsum(
            'eia,eja->eij', self.gradients, self.gradients)
        rows = np.broadcast_to(self.triangles[:, :, None], elements.shape)
        columns = np.broadcast_to(self.triangles[:, None, :], elements.shape)
        self.stiffness = coo_matrix((elements.ravel(), (rows.ravel(), columns.ravel())),
                                   shape=(n, n)).tocsr()
        self.stiffness.eliminate_zeros()
        scale = max(1., np.max(abs(self.stiffness.data)))
        skew = self.stiffness-self.stiffness.T
        if ((skew.nnz and np.max(abs(skew.data)) > 1e-12*scale)
                or np.max(abs(self.stiffness@np.ones(n))) > 1e-10*scale
                or not np.isfinite(self.stiffness.data).all()
                or np.any(self.stiffness.diagonal() <= 0)):
            raise ValueError('invalid P1 scalar stiffness symmetry/constant mode/diagonal')
        self.matrix = (diags(1/mechanics.volumes)@self.stiffness).tocsr()
        self.shape = self.matrix.shape

    def to_scipy(self):
        return self.matrix.copy()


def make_energy(operator, volumes, *, gc0, gc1, floor):
    """Normalized shifted J_H, differentiated only through nodal damage.

    Divide by reaction-only release sum V*2H^2/(gc0+2H), floored by
    gc0*sum(V)*floor. This is a detached positive scalar for each fixed H.
    d=0 has energy 0; the minimizer normally has NEGATIVE energy.
    """
    if (not isinstance(operator, P1DamageOperator) or min(gc0, gc1) <= 0
            or not np.isfinite([gc0, gc1, floor]).all() or not 0 < floor < 1
            or volumes.shape != (operator.shape[0],)
            or not np.isfinite(volumes).all() or np.any(volumes <= 0)):
        raise ValueError('invalid P1 energy operator/material/volume/normalization')
    tri = jnp.asarray(operator.triangles, jnp.int32)
    gradients, areas, V = map(jnp.asarray, (operator.gradients, operator.areas, volumes))

    def energy(damage, history):
        if damage.shape != V.shape or history.shape != V.shape:
            raise ValueError('P1 energy requires material-node fields in mesh order')
        H = jax.lax.stop_gradient(history)
        gradient = jnp.einsum('ei,eia->ea', damage[tri], gradients)
        reaction = jnp.sum(V*(.5*(gc0+2*H)*damage**2-2*H*damage))
        diffusion = .5*gc1*jnp.sum(areas*jnp.sum(gradient**2, axis=1))
        release = jnp.sum(V*2*H**2/(gc0+2*H))
        normalization = jnp.maximum(release, gc0*jnp.sum(V)*floor)
        return (reaction+diffusion)/normalization
    return energy


def make_energy_loss(*, coarse, fine, points, operator, volumes, gc0, gc1, floor):
    energy = make_energy(operator, volumes, gc0=gc0, gc1=gc1, floor=floor)

    def image(net, context):
        return net(context, points, coarse, fine, gc0=gc0)

    def objective(net, batch):
        batch = jax.tree.map(jax.lax.stop_gradient, batch)
        def body(total, task):
            H, ids, weight = task
            value = jax.lax.cond(weight > 0,
                lambda _: energy(image(net, dict(H=H, fine_ids=ids)), H),
                lambda _: jnp.zeros((), jnp.float64), None)
            return total+weight*value, None
        return jax.lax.scan(body, jnp.zeros((), jnp.float64),
            (batch['H'], batch['fine_ids'], batch['weights']))[0]
    return image, objective
