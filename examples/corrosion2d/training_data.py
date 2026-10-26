"""Fixed analytic IC/BC samplers shared by canonical and research training.

Keep RNG order and sample construction identical to the validated saved runs.
These functions do not import CLI modules or use evolved reference fields.
"""
import jax.numpy as jnp
import numpy as np


def corrosion_boundary_points(pits, N_dir=1000, N_solid=500):
    """Sample BC points for 2D corrosion geometry.

    Dirichlet (pit mouth): y=0, |x-cx| < R*0.5 → φ=0, c=0.
    Solid boundaries (left/right/top): x=0, x=1, y=0.5 → φ=1, c=1.
    Pure random sampling — no grid points needed (boundaries are now separate).
    """
    rng = np.random.RandomState(42)
    pit_mouth_x = [(float(cx) - float(R)*0.5, float(cx) + float(R)*0.5) for cx, cy, R in pits]

    # Pit mouth: sample uniformly from each pit's mouth range on y=0
    pit_points = []
    n_each = N_dir // len(pits) + 1
    for lo, hi in pit_mouth_x:
        xs = rng.uniform(lo, hi, n_each)
        pit_points.append(np.column_stack([xs, np.zeros_like(xs)]))
    x_bc_pit = jnp.array(np.vstack(pit_points)[:N_dir])

    # Solid edges: sample uniformly from left (x=0), right (x=1), top (y=0.5)
    solid_points = []
    n_each_edge = N_solid // 3 + 1
    for edge in range(3):
        if edge == 0:   # left: x=0, y in [0, 0.5]
            xs = np.zeros(n_each_edge)
            ys = rng.uniform(0.0, 0.5, n_each_edge)
        elif edge == 1: # right: x=1, y in [0, 0.5]
            xs = np.ones(n_each_edge)
            ys = rng.uniform(0.0, 0.5, n_each_edge)
        else:           # top: y=0.5, x in [0, 1]
            xs = rng.uniform(0.0, 1.0, n_each_edge)
            ys = np.full(n_each_edge, 0.5)
        solid_points.append(np.column_stack([xs, ys]))
    x_bc_solid = jnp.array(np.vstack(solid_points)[:N_solid])

    return x_bc_pit, x_bc_solid


def corrosion_initial_points(pits, N_global=2000, N_local=2000):
    """Sample IC points: global uniform + local oversampling near pit interface.

    Global: N_global uniform random points in [0,1]×[0,0.5].
    Local: N_local uniform random points in 2R×2R rectangle around each pit.
    """
    rng = np.random.RandomState(42)

    # Global uniform samples
    xs_g = rng.uniform(0.0, 1.0, N_global)
    ys_g = rng.uniform(0.0, 0.5, N_global)
    points = [np.column_stack([xs_g, ys_g])]

    # Local oversampling near each pit
    n_each = N_local // len(pits) + 1
    for cx, cy, R in pits:
        x_lo = max(float(cx) - 2.0 * float(R), 0.0)
        x_hi = min(float(cx) + 2.0 * float(R), 1.0)
        y_lo = max(float(cy) - 2.0 * float(R), 0.0)
        y_hi = min(float(cy) + 2.0 * float(R), 0.5)
        xs = rng.uniform(x_lo, x_hi, n_each)
        ys = rng.uniform(y_lo, y_hi, n_each)
        points.append(np.column_stack([xs, ys]))

    return jnp.array(np.vstack(points)[:N_global + N_local])


def electropolish_boundary_points(config):
    """Sample BC points.

    Top (y=0.5): liquid — φ=0, c=0.
    Bottom (y=0): solid — φ=1, c=1.
    Left/Right: Neumann — no BC points (key ablation point!).
    """
    rng = np.random.RandomState(42)

    # Top edge: y=0.5, x ∈ [0, 1]
    xs_top = rng.uniform(0.0, 1.0, config.sample.N_bc_top)
    x_bc_top = jnp.column_stack([xs_top, np.full_like(xs_top, 0.5)])

    # Bottom edge: y=0, x ∈ [0, 1]
    xs_bot = rng.uniform(0.0, 1.0, config.sample.N_bc_bottom)
    x_bc_bottom = jnp.column_stack([xs_bot, np.zeros_like(xs_bot)])

    return jnp.array(x_bc_top), jnp.array(x_bc_bottom)


def electropolish_initial_points(config):
    """Sample IC points: global uniform + local oversampling near interface."""
    rng = np.random.RandomState(42)
    N_global = config.sample.N_ic_global
    N_local = config.sample.N_ic_local

    # Global uniform samples
    xs_g = rng.uniform(0.0, 1.0, N_global)
    ys_g = rng.uniform(0.0, 0.5, N_global)
    points = [np.column_stack([xs_g, ys_g])]

    # Local oversampling near rough interface: y ∈ [y0-2*A, y0+2*A]
    y_lo = max(config.surface.y0 - 2.0 * config.surface.amplitude, 0.0)
    y_hi = min(config.surface.y0 + 2.0 * config.surface.amplitude, 0.5)
    xs_l = rng.uniform(0.0, 1.0, N_local)
    ys_l = rng.uniform(y_lo, y_hi, N_local)
    points.append(np.column_stack([xs_l, ys_l]))

    return jnp.array(np.vstack(points)[:N_global + N_local])

