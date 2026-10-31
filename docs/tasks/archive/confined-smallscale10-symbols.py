"""Infinite-medium kernel dispersion guide, not a perforated-domain eigenanalysis.

For A=-D_delta with radial K=120/(pi*delta^4)*(1-r/delta)^2,
A exp(ik.x)=lambda exp(ik.x), lambda=2pi int_0^delta K(r)r(1-J0(kr))dr.
The angular identity int cos(kr cos(theta))dtheta=2pi J0(kr) is independently
checked by tensor Gauss-Legendre quadrature below. The second moment gives
lambda/k^2 -> 1 as k*delta -> 0. No boundary or mesh approximation is included.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import j0


def analyze(config, output):
    c = json.loads(config.read_text())
    delta = c['nl']['delta']
    p = c['physical']
    modes = np.asarray(c['initial']['waves'])[:, :2]
    k = np.pi*modes/np.asarray(c['geometry']['axes'])
    norm = np.linalg.norm(k, axis=1)
    def symbol(q):
        return 240/delta**2*quad(lambda s:s*(1-s)**2*(1-j0(q*delta*s)), 0, 1,
                                 epsabs=1e-13, epsrel=1e-13)[0]
    native = np.asarray([symbol(q) for q in norm])
    # Independent two-dimensional angular/radial integration, no Bessel function.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    r, wr = delta*(nodes+1)/2, delta*weights/2
    theta, wt = np.pi*(nodes+1), np.pi*weights
    direct = np.asarray([np.sum(wr*r*120/(np.pi*delta**4)*(1-r/delta)**2*
        np.sum(wt[None, :]*(1-np.cos(q*r[:, None]*np.cos(theta)[None, :])), axis=1)) for q in norm])
    np.testing.assert_allclose(native, direct, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(symbol(.01)/.01**2, 1., rtol=1e-6)
    def growth(a):return p['mobility']*a*(p['beta']-p['epsilon']**2*a)
    rows = [dict(mode=mode.tolist(), k_delta=float(q*delta), local_symbol=float(q*q),
        nonlocal_symbol=float(a), symbol_relative_deficit=float(1-a/q**2),
        local_linear_growth=float(growth(q*q)), nonlocal_linear_growth=float(growth(a)))
        for mode, q, a in zip(modes, norm, native, strict=True)]
    result = dict(config=str(config), scope='Infinite homogeneous medium only; no holes, wall truncation, mesh, nonlinear mode coupling or neural optimization.',
        formula='lambda=240/delta^2 int_0^1 s(1-s)^2 (1-J0(k*delta*s)) ds',
        independent_angular_check_maximum_error=float(abs(direct-native).max()), modes=rows)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.config, args.output)
