"""Read-only NumPy diagnosis of completed native/local Spinodal references.

No JAX/FEniCS imports, time integration, or training-data changes. The interior
Fourier symbol is a diagnostic of the implemented kernel, not the spectrum of
the bounded, perforated domain.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import j0


def relative(a, b, weights):
    if a.shape != b.shape or a.shape[-1] != len(weights):
        raise ValueError('Matching field shapes and spatial weights required')
    if not all(np.isfinite(x).all() for x in (a, b, weights)):
        raise ValueError('Nonfinite diagnostic input')
    denominator = float(np.sum(b*b*weights))
    if denominator <= 0 or np.any(weights <= 0):
        raise ValueError('Positive norm and quadrature required')
    return float(np.sqrt(np.sum((a-b)**2*weights)/denominator))


def symbol(k, delta):
    # Angular integration of K(z)[1-cos(k.z)] gives 2*pi*(1-J0(k*r)).
    integral, error = quad(
        lambda r: r*(1-r)**2*(1-j0(k*delta*r)), 0., 1.,
        epsabs=1e-13, epsrel=1e-13)
    if error*240/delta**2 > 1e-8:
        raise ValueError('Fourier-symbol integration did not resolve tolerance')
    return 240/delta**2*integral


def diagnose(native_root, local_root, indices):
    bank = json.loads((native_root/'manifest.json').read_text())
    local = json.loads((local_root/'manifest.json').read_text())
    spec = bank['specification']
    if bank['status'] != 'passed' or spec != local['specification']:
        raise ValueError('Reference specifications must match exactly')
    if spec['initial']['kind'] != 'fixed_ambient' or spec['nl']['kernel'] != 'quadratic_visible':
        raise ValueError('This diagnostic requires the explicit ambient cosine IC and quadratic kernel')
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Explicit unique completed case indices required')
    if not set(indices).issubset(bank['test_indices']):
        raise ValueError('Requested indices must belong to the existing test cohort')
    accepted = {row['index']: row for row in local['cases'] if row['status'] == 'passed'}
    if not set(indices).issubset(accepted):
        raise ValueError('Only completed, passed local references may be diagnosed')
    center = np.asarray(spec['geometry']['center'])
    axes = np.asarray(spec['geometry']['axes'])
    waves = np.asarray(spec['initial']['waves'])
    kvec = np.pi*waves[:, :2]/axes
    k2 = np.sum(kvec*kvec, axis=1)
    delta = spec['nl']['delta']
    beta = spec['physical']['beta']
    epsilon = spec['physical']['epsilon']
    mobility = spec['physical']['mobility']
    amplitude = spec['initial']['amplitude']*waves[:, 2]
    eigenvalue = np.array([symbol(float(k), delta) for k in np.sqrt(k2)])
    # Check the Bessel reduction independently against direct polar cubature.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    radius, radial_weight = .5*(nodes+1), .5*weights
    angle = 2*np.pi*(np.arange(256)+.5)/256
    polar_eigenvalue = np.array([
        120/(np.pi*delta**2)*np.sum(
            radial_weight[:, None]*radius[:, None]*(1-radius[:, None])**2
            *(1-np.cos(k*delta*radius[:, None]*np.cos(angle))))*2*np.pi/256
        for k in np.sqrt(k2)])
    np.testing.assert_allclose(eigenvalue, polar_eigenvalue, atol=1e-9, rtol=1e-12)
    # Independent low-frequency normalization: 1/2 integral K(z) z_x^2 dz = 1.
    normalization = 60*quad(lambda r: r**3*(1-r)**2, 0., 1.)[0]
    np.testing.assert_allclose(normalization, 1., atol=1e-13, rtol=0.)
    rows = []
    for index in indices:
        record = bank['conditions'][index]
        if record['index'] != index:
            raise ValueError('Native reference index/order mismatch')
        native_path = native_root/record['file']
        local_path = local_root/accepted[index]['file']
        with np.load(native_path, allow_pickle=False) as f:
            native = {key: f[key] for key in f.files}
        with np.load(local_path, allow_pickle=False) as f:
            fine = {key: f[key] for key in f.files}
        for key in ('points', 'volumes', 'time', 'holes'):
            np.testing.assert_array_equal(native[key], fine[key])
        points, volumes = native['points'], native['volumes']
        angles = (points-center)@kvec.T+waves[:, 3]
        modes = np.cos(angles)*amplitude
        phi0 = modes.sum(axis=1)
        minus_laplacian = modes@k2
        exact_mu0 = beta*(phi0**3-phi0)+epsilon**2*minus_laplacian
        np.testing.assert_allclose(native['phi'][0], phi0, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(fine['mu'][0], exact_mu0, atol=1e-13, rtol=1e-12)
        graph_action = np.sum(native['coefficients']*(
            phi0[:, None]-phi0[native['neighbors']]), axis=1)
        np.testing.assert_allclose(native['mu'][0],
            beta*(phi0**3-phi0)+epsilon**2*graph_action, atol=1e-13, rtol=1e-12)
        # Conservative ellipse-distance lower bound; only the resulting safe
        # interior subset is used. It is not claimed to be the exact wall band.
        clearance = axes.min()*(1-np.linalg.norm((points-center)/axes, axis=1))
        for x, y, radius in native['holes']:
            clearance = np.minimum(clearance, np.linalg.norm(points-[x, y], axis=1)-radius)
        interior = clearance > delta
        if not interior.any():
            raise ValueError('No full-horizon interior points for operator diagnosis')
        continuum_nonlocal = modes@eigenvalue
        errors = {}
        for name, weights in (('equal_node', np.ones(len(points))), ('volume', volumes)):
            errors[name] = dict(
                phi_pooled=relative(native['phi'], fine['phi'], weights),
                phi_by_time=[relative(a, b, weights) for a, b in zip(native['phi'], fine['phi'], strict=True)],
                mu_pooled=relative(native['mu'], fine['mu'], weights),
                mu_positive=relative(native['mu'][1:], fine['mu'][1:], weights),
                fine_phi0_vs_exact=relative(fine['phi'][0], phi0, weights))
        difference = (native['phi'][1:]-fine['phi'][1:])**2*volumes
        rows.append(dict(index=index, native=str(native_path), local=str(local_path),
            points=len(points), times=native['time'].tolist(), errors=errors,
            interior=dict(count=int(interior.sum()),
                volume_fraction=float(volumes[interior].sum()/volumes.sum()),
                positive_phase_squared_error_fraction=float(difference[:, interior].sum()/difference.sum()),
                graph_vs_continuum_nonlocal=relative(graph_action[interior], continuum_nonlocal[interior], volumes[interior]),
                continuum_nonlocal_vs_local=relative(continuum_nonlocal[interior], minus_laplacian[interior], volumes[interior]),
                graph_vs_local=relative(graph_action[interior], minus_laplacian[interior], volumes[interior])),
            local_refinement=accepted[index]['checks']))
    return dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Completed subset diagnostic; no training, method ranking or change to the live queue',
        native_bank=str(native_root), local_bank=str(local_root),
        specification=spec, normalization=normalization,
        symbol_polar_verification_max_abs=float(np.max(np.abs(eigenvalue-polar_eigenvalue))),
        derivation=dict(
            positive_operator='A_delta f = integral K_delta(z) [f(x)-f(x+z)] dz',
            full_space_symbol='lambda_delta(k) = 240/delta^2 integral_0^1 r(1-r)^2 [1-J0(|k|delta*r)] dr',
            local_linear_rate='M [beta |k|^2 - epsilon^2 |k|^4]',
            nonlocal_linear_rate='M [beta lambda_delta - epsilon^2 lambda_delta^2]',
            limitation='Whole-space linearization around phi=0; does not isolate bounded nonlinear trajectory error'),
        modes=[dict(mode=wave[:2].tolist(), wavelength=float(2*np.pi/np.sqrt(a)),
            horizon_over_wavelength=float(delta*np.sqrt(a)/(2*np.pi)),
            local_eigenvalue=float(a), nonlocal_eigenvalue=float(b), ratio=float(b/a),
            local_linear_rate=float(mobility*(beta*a-epsilon**2*a*a)),
            nonlocal_linear_rate=float(mobility*(beta*b-epsilon**2*b*b)))
            for wave, a, b in zip(waves, k2, eigenvalue, strict=True)], cases=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.native, args.local, args.indices)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], output=str(args.output),
        cases=[dict(index=row['index'], errors=row['errors'], interior=row['interior'])
               for row in report['cases']], modes=report['modes']), indent=2))
