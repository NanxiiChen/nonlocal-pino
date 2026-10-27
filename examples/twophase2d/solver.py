"""Independent CPU Fourier IMEX reference; imports no PINN/JAX operator."""
import argparse
import json
import importlib
from pathlib import Path
import numpy as np
from scipy.fft import fftn, ifftn

from examples.twophase2d.problem import (identity, initial_field, validate_config,
                                       is_constant_flow, velocity_field)


def grid(n, length):
    axis = np.arange(n)*length/n
    X, Y = np.meshgrid(axis, axis, indexing='xy')
    return np.stack((X.ravel(), Y.ravel()), axis=-1)


def first_derivative_symbols(n, config, method):
    """Independent Fourier multipliers for the selected reference advection.

    The common local-PDE target always uses local spectral first derivatives.
    Only the native nonlocal model switches to finite-horizon first derivatives.
    """
    if method not in ('nonlocal', 'local_ad'):
        raise ValueError(f'Unknown first-derivative reference method: {method}')
    freq = 2*np.pi*np.fft.fftfreq(n, d=config.geometry.length/n)
    freq[n//2] = 0
    ax, ay = np.meshgrid(freq, freq, indexing='xy')
    if method != 'nonlocal' or 'advection' not in config or config.advection != 'nonlocal_flux':
        return 1j*ax, 1j*ay
    h, delta = config.geometry.length/n, config.nl.delta
    if not h < delta < config.geometry.length/2:
        raise ValueError('Nonlocal reference derivative requires h<delta<L/2')
    limit = int(np.ceil(delta/h))
    ys, xs = np.mgrid[-limit:limit+1, -limit:limit+1]
    xx, yy = h*xs.ravel(), h*ys.ravel()
    distance = np.hypot(xx, yy)
    valid = (distance > 0) & (distance < delta)
    xx, yy, distance = xx[valid], yy[valid], distance[valid]
    weights = (1-distance/delta)**2*h*h
    moment = np.sum(weights*xx*xx)
    if not np.isfinite(moment) or moment <= 0:
        raise ValueError('Nonlocal reference gradient moment must be positive finite')
    # Use actual FFT wave numbers in the sine symbol; unlike local first
    # derivatives, zeroing a single coordinate's Nyquist before this sum would
    # alter oblique neighbor contributions at mixed Nyquist modes.
    raw_freq = 2*np.pi*np.fft.fftfreq(n, d=config.geometry.length/n)
    kx, ky = np.meshgrid(raw_freq, raw_freq, indexing='xy')
    sine = np.sin(xx[:, None, None]*kx+yy[:, None, None]*ky)
    gx = 1j*np.sum((weights*xx/moment)[:, None, None]*sine, axis=0)
    gy = 1j*np.sum((weights*yy/moment)[:, None, None]*sine, axis=0)
    return gx, gy


def symbols(n, config, method):
    """Independent circulant-kernel symbol, using no training operator values."""
    length = config.geometry.length
    freq = 2*np.pi*np.fft.fftfreq(n, d=length/n)
    kx, ky = np.meshgrid(freq, freq, indexing='xy')
    if method == 'local_ad':
        lam = kx*kx+ky*ky
    elif method == 'nonlocal':
        h, delta = length/n, config.nl.delta
        if not h < delta < length/2:
            raise ValueError('Reference graph requires h<delta<L/2 at its actual grid size')
        limit = int(np.ceil(delta/h))
        ys, xs = np.mgrid[-limit:limit+1, -limit:limit+1]
        xx, yy = h*xs.ravel(), h*ys.ravel()
        distance = np.hypot(xx, yy)
        valid = (distance > 0) & (distance < delta)
        xx, yy, distance = xx[valid], yy[valid], distance[valid]
        w = 120/(np.pi*delta**4)*(1-distance/delta)**2*h*h
        moment = np.sum(w*xx*xx)
        if not np.isfinite(moment) or moment <= 0:
            raise ValueError('Reference kernel has no positive finite second moment')
        w *= 2/moment
        lam = np.sum(w[:, None, None]*(1-np.cos(xx[:, None, None]*kx+yy[:, None, None]*ky)), axis=0)
    else:
        raise ValueError(f'Unknown spatial reference method: {method}')
    gx, gy = first_derivative_symbols(n, config, method)
    adv = ((config.physical.velocity[0]*gx+config.physical.velocity[1]*gy)
           if is_constant_flow(config) else np.zeros_like(kx, dtype=complex))
    return lam, adv


def solve_reference(config, n, method, dt):
    """BDF2: implicit biharmonic, explicit CH reaction; Euler starter.

    Constant advection is implicit. Variable advection uses explicit conservative
    flux divergence at each extrapolated time level, evaluated with independent
    Fourier multipliers (local spectral or finite-horizon symbols).

    Stability/accuracy are checked by actual step/grid refinement; no unconditional
    energy stability is asserted for this un-stabilized pilot time integrator.
    """
    duration = config.physical.terminal_time
    steps = round(duration/dt)
    if not np.isclose(steps*dt, duration, atol=1e-12, rtol=0):
        raise ValueError('Reference dt must exactly divide the duration')
    intervals = config.reference.snapshots-1
    if steps % intervals:
        raise ValueError('Reference dt must align with all requested snapshots')
    lam, adv = symbols(n, config, method)
    points = grid(n, config.geometry.length)
    initial = initial_field(points, config, xp=np).reshape(n, n)
    gx, gy = first_derivative_symbols(n, config, method)
    steady_velocity = velocity_field(points, 0., config, xp=np).reshape(n, n, 2)
    current = fftn(initial)
    linear = -adv-config.physical.mobility*config.physical.epsilon**2*lam**2

    def nonlinear(hat, t):
        phi = ifftn(hat).real
        result = -config.physical.mobility*lam*fftn(phi**3-phi)
        if not is_constant_flow(config):
            velocity = steady_velocity*np.cos(np.pi*t/config.physical.velocity.period)
            result -= gx*fftn(velocity[..., 0]*phi)+gy*fftn(velocity[..., 1]*phi)
        return result

    def snapshot(hat):
        value = ifftn(hat)
        if not np.isfinite(value).all() or np.max(np.abs(value.imag)) > 1e-9:
            raise FloatingPointError('Reference lost finite real-valued symmetry')
        phi = value.real
        if np.max(np.abs(phi)) > 3:
            raise FloatingPointError('Reference phase range diverged; refine dt')
        lap = ifftn(lam*hat).real
        q = phi**3-phi+config.physical.epsilon**2*lap
        energy = np.mean(.25*(phi**2-1)**2+.5*config.physical.epsilon**2*phi*lap)
        return phi.copy(), q, float(energy)

    fields, chemicals, energies = [], [], []
    p, q, e = snapshot(current)
    fields.append(p); chemicals.append(q); energies.append(e)
    previous = current.copy()
    previous_nonlin = nonlinear(current, 0.)
    for step in range(1, steps+1):
        if step == 1:
            after = (current+dt*previous_nonlin)/(1-dt*linear)
        else:
            now_nonlin = nonlinear(current, (step-1)*dt)
            after = (4*current-previous+2*dt*(2*now_nonlin-previous_nonlin))/(3-2*dt*linear)
            previous_nonlin = now_nonlin
        previous, current = current, after
        if step % (steps//intervals) == 0:
            p, q, e = snapshot(current)
            fields.append(p); chemicals.append(q); energies.append(e)
    return dict(phi=np.asarray(fields), q=np.asarray(chemicals), energy=np.asarray(energies))


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples.twophase2d.configs.train_base',
                        help='Explicit importable configuration module')
    parser.add_argument('--output', help='Override the explicit default reference path')
    args = parser.parse_args()
    c = importlib.import_module(args.config).get_config()
    validate_config(c)
    path = Path(c.reference.path if args.output is None else args.output)
    if path.exists():raise FileExistsError(path)
    results = {}
    for name, n, method, dt in [
        ('graph_coarse_time', c.sample.Nx, 'nonlocal', c.reference.dt),
        ('graph', c.sample.Nx, 'nonlocal', c.reference.refined_dt),
        ('local_coarse', c.reference.local_grid, 'local_ad', c.reference.dt),
        ('local_fine_time', c.reference.local_grid, 'local_ad', c.reference.refined_dt),
        ('local', c.reference.refined_local_grid, 'local_ad', c.reference.refined_dt)]:
        print(f'Reference {name}: {n}x{n}, dt={dt}', flush=True)
        results[name] = solve_reference(c, n, method, dt)
    check = {}
    for name, a, b in [
        ('graph_time', results['graph_coarse_time']['phi'], results['graph']['phi']),
        ('local_time', results['local_coarse']['phi'], results['local_fine_time']['phi']),
        ('local_space', results['local_fine_time']['phi'], results['local']['phi'][:,
            ::c.reference.refined_local_grid//c.reference.local_grid,
            ::c.reference.refined_local_grid//c.reference.local_grid])]:
        check[name] = dict(pooled_phi=relative(a, b), terminal_phi=relative(a[-1], b[-1]),
            max_snapshot_phi=max(relative(aa, bb) for aa, bb in zip(a, b)))
        if max(check[name].values()) > c.reference.error_tolerance:
            raise ValueError(f'Reference refinement failed: {name}: {check[name]}')
    stride = c.reference.refined_local_grid//c.sample.Nx
    local = {k: results['local'][k][:, ::stride, ::stride] for k in ('phi', 'q')}
    graph = results['graph']
    check['graph_vs_local'] = dict(pooled_phi=relative(graph['phi'], local['phi']),
                                  terminal_phi=relative(graph['phi'][-1], local['phi'][-1]))
    for name, data in [('local', results['local']), ('nonlocal', graph)]:
        mass = np.mean(data['phi'], axis=(1, 2))
        check[name+'_mass_max_drift'] = float(np.max(np.abs(mass-mass[0])))
        if check[name+'_mass_max_drift'] > 1e-10:
            raise ValueError('Reference zero mode/mass was not conserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, identity=json.dumps(identity(c), sort_keys=True),
        points=grid(c.sample.Nx, c.geometry.length),
        time=np.linspace(0, c.physical.terminal_time, c.reference.snapshots),
        phi_local=local['phi'], q_local=local['q'],
        phi_nonlocal=graph['phi'], q_nonlocal=graph['q'],
        energy_local=results['local']['energy'], energy_nonlocal=graph['energy'])
    path.with_suffix('.json').write_text(json.dumps(dict(status='passed',
        identity=identity(c), checks=check, method='Independent Fourier IMEX-BDF2; no PINN operator import',
        configuration=c.reference.to_dict()), indent=2, allow_nan=False)+'\n')
    print(json.dumps(check, indent=2), flush=True)


if __name__ == '__main__':main()
