"""Independent NumPy periodic first-derivative feasibility/limitation check."""
from pathlib import Path
import json
import numpy as np


def operators(n, delta):
    h = 1 / n
    offsets = []
    for dy in range(-int(np.ceil(delta/h)), int(np.ceil(delta/h))+1):
        for dx in range(-int(np.ceil(delta/h)), int(np.ceil(delta/h))+1):
            r = h*np.hypot(dx, dy)
            if 0 < r < delta:
                offsets.append((dx, dy, (1-r/delta)**2*h*h))
    xi = h*np.asarray([(dx, dy) for dx, dy, _ in offsets])
    w = np.asarray([weight for _, _, weight in offsets])
    K = np.einsum('e,ei,ej->ij', w, xi, xi)
    coeff = (w[:, None]*xi) @ np.linalg.inv(K)
    yy, xx = np.indices((n, n))
    row = np.arange(n*n)
    gx, gy, lap = (np.zeros((n*n, n*n)) for _ in range(3))
    for (dx, dy, _), c, weight in zip(offsets, coeff, w, strict=True):
        col = (((yy+dy) % n)*n + (xx+dx) % n).ravel()
        for g, a in zip((gx, gy), c, strict=True):
            g[row, col] += a
            g[row, row] -= a
        a = 2*weight/K[0, 0]
        lap[row, row] += a
        lap[row, col] -= a
    return gx, gy, lap, K, coeff, xi


def main():
    n = 16
    gx, gy, lap, K, coeff, xi = operators(n, .2)
    G = np.vstack((gx, gy))
    D = -G.T
    ones = np.ones(n*n)
    rng = np.random.default_rng(42)
    q = rng.normal(size=n*n)
    phi = rng.normal(size=n*n)
    velocity = rng.normal(size=(2, n*n))
    flux = phi[None]*velocity
    lhs = np.dot(q, D @ flux.ravel())
    rhs = -np.dot(G @ q, flux.ravel())
    phase_work = -lhs
    capillary_work = np.sum(velocity*(-phi[None]*(G @ q).reshape(2, -1)))
    checkerboard = (-1.)**np.indices((n, n)).sum(axis=0).ravel()
    result = {
        'scope': 'Independent nodal periodic candidate only, not a validated NS/CH discretization or production operator.',
        'moment_identity_max_error': float(np.max(np.abs(coeff.T @ xi-np.eye(2)))),
        'gradient_constant_max': float(np.max(np.abs(G @ ones))),
        'gradient_skew_max': float(max(np.max(np.abs(gx+gx.T)), np.max(np.abs(gy+gy.T)))),
        'divergence_global_sum': float(abs(np.sum(D @ flux.ravel()))),
        'adjoint_inner_product_gap': float(abs(lhs-rhs)),
        'phase_capillary_work_cancellation_gap': float(abs(phase_work+capillary_work)),
        'checkerboard_gradient_max': float(np.max(np.abs(G @ checkerboard))),
        'checkerboard_pair_laplacian_rms': float(np.sqrt(np.mean((lap @ checkerboard)**2))),
        'composition_is_not_pair_laplacian_relative_frobenius': float(np.linalg.norm(G.T @ G-lap)/np.linalg.norm(lap)),
        'sin_mode_relative_errors': [],
    }
    for size in (16, 32):
        a, b, _, _, _, _ = operators(size, 3.2/size)
        y, x = np.indices((size, size))/size
        f = np.sin(2*np.pi*x).ravel()
        expected = 2*np.pi*np.cos(2*np.pi*x).ravel()
        error = np.linalg.norm(a @ f-expected)/np.linalg.norm(expected)
        result['sin_mode_relative_errors'].append({'n': size, 'delta': 3.2/size, 'relative': float(error)})
    assert result['moment_identity_max_error'] < 1e-13
    assert result['gradient_constant_max'] < 1e-13
    assert result['gradient_skew_max'] < 1e-13
    assert result['divergence_global_sum'] < 1e-11
    assert result['adjoint_inner_product_gap'] < 1e-11
    assert result['phase_capillary_work_cancellation_gap'] < 1e-11
    assert result['checkerboard_gradient_max'] < 1e-12
    assert result['checkerboard_pair_laplacian_rms'] > 1.
    errors = result['sin_mode_relative_errors']
    assert errors[1]['relative'] < errors[0]['relative']/3
    result['status'] = 'passed'
    Path(__file__).with_name('fully-nonlocal-flow-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
