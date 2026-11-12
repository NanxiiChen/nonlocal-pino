"""Finite cube/radial qualification for the extreme-interface reference.

This comparison does not train a network or prescribe a shrinking trajectory.
Its gates are declared before the reference runs. Only small provenance and
comparison records are retained; full cube states remain transient.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import jax
import numpy as np
from scipy.interpolate import PchipInterpolator

from examples.icemelting3d.configs.train_extreme_interface import get_config
from examples.icemelting3d.extreme_reference import sample_solution, solve_radial
from examples.icemelting3d.thin_interface_reference import evaluation_points, sample_reference
from nonlocal_pinn.training.runtime import configure_gpu_policy


EPSILON = .00375
GRIDS = (512, 768)
DT = .00125
GATES = dict(fine_radial_pooled=2e-3, cube_radial_refinement_ratio=.8,
             cube_time_pooled=1e-5)
FLUX_WIDTHS = (.0075, .00375, .001875, .0009375)


def write_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metrics(prediction, truth):
    if (prediction.shape != truth.shape or prediction.ndim != 2
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError('Comparison requires matching finite snapshot fields')
    p, y = prediction[1:], truth[1:]
    band = np.abs(y) < .9
    if not band.any():
        raise ValueError('Comparison queries contain no interface samples')
    relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    return dict(pooled=relative(p, y), terminal=relative(p[-1], y[-1]),
                interface=relative(p[band], y[band]),
                max_absolute=float(np.max(np.abs(p-y))))


def solve(c):
    spec = c.reference
    return solve_radial(c, nodes=spec.grids[1], outer_radius=spec.outer_radius,
                        max_step=spec.max_step, rtol=spec.rtol, atol=spec.atol)


def face_flux(solution):
    """Physical normal derivative at positive cube faces, using latent PCHIP.

    Radial symmetry makes all six faces equivalent. On x=0.5 relative to the
    sphere center, dphi/dn=sech(u)^2*u_r*(0.5/r). The range r in[0.5,sqrt(3)/2]
    covers every face radius. Stable sech^2 avoids saturated tanh subtraction.
    """
    radius = np.linspace(.5, np.sqrt(3)/2, 8193)
    maximum, positive = 0., 0.
    per_time = []
    for index, state in enumerate(solution.latent):
        interpolant = PchipInterpolator(solution.radius, state, extrapolate=False)
        latent = interpolant(radius)
        slope = interpolant.derivative()(radius)
        tail = np.exp(-2*np.abs(latent))
        jacobian = 4*tail/(1+tail)**2
        flux = slope*jacobian*.5/radius
        if not np.isfinite(flux).all():
            raise FloatingPointError('Radial cube-face physical flux must remain finite')
        value = float(np.max(np.abs(flux)))
        per_time.append(value)
        maximum = max(maximum, value)
        if index:
            positive = max(positive, value)
    return dict(max_absolute=maximum, positive_time_max_absolute=positive,
        by_time=per_time, face_radius_samples=len(radius),
        formula='sech(u)^2 * du/dr * (0.5/r)',
        scope='all six cube faces by spherical symmetry; physical phi flux, not latent flux')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    c = get_config()
    c.physical.eps_star = EPSILON
    if not jax.config.x64_enabled:
        raise ValueError('Cube qualification requires JAX_ENABLE_X64=1')
    configure_gpu_policy(c.train.gpu_policy)
    write_json(output/'config.json', json.loads(c.to_json_best_effort()))
    report = dict(status='running', pid=os.getpid(), epsilon=EPSILON,
        grids=list(GRIDS), dt=DT, gates=GATES,
        started_utc=datetime.now(timezone.utc).isoformat(),
        gpu=jax.devices()[0].device_kind,
        definition='same local PDE and initial field; refined Neumann cube versus far-field radial approximation',
        saved_dense_cube_fields=False, positive_time_metrics=True)
    def update(stage):
        report['stage'] = stage
        report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output/'comparison.json', report)
        print('CUBE_CHECK', json.dumps(dict(stage=stage, status=report['status'])), flush=True)
    started = time.perf_counter()
    try:
        points, rays, _ = evaluation_points(c)
        queries = np.concatenate((points, rays.reshape(-1, 3)))
        update('radial_reference')
        radial = solve(c)
        truth = sample_solution(radial, queries)
        report['radial'] = radial.statistics
        report['queries'] = dict(cube_axis=c.experiment.evaluation_axis,
            cube_points=len(points), ray_directions=c.experiment.radial_directions,
            ray_points=c.experiment.radial_points)
        update('cube512')
        coarse, coarse_seconds = sample_reference(c, GRIDS[0], DT, local=True)
        report['coarse_radial'] = dict(field=metrics(coarse[:, :len(points)], truth[:, :len(points)]),
            rays=metrics(coarse[:, len(points):], truth[:, len(points):]), seconds=coarse_seconds)
        update('cube768')
        fine, fine_seconds = sample_reference(c, GRIDS[1], DT, local=True)
        report['fine_radial'] = dict(field=metrics(fine[:, :len(points)], truth[:, :len(points)]),
            rays=metrics(fine[:, len(points):], truth[:, len(points):]), seconds=fine_seconds)
        report['cube_space'] = dict(field=metrics(coarse[:, :len(points)], fine[:, :len(points)]),
            rays=metrics(coarse[:, len(points):], fine[:, len(points):]))
        report['cube_radial_refinement_ratio'] = (report['fine_radial']['field']['pooled']
            /report['coarse_radial']['field']['pooled'])
        update('cube768_half_dt')
        half, half_seconds = sample_reference(c, GRIDS[1], DT/2, local=True)
        report['cube_time'] = dict(field=metrics(half[:, :len(points)], fine[:, :len(points)]),
            rays=metrics(half[:, len(points):], fine[:, len(points):]), seconds=half_seconds)
        # All gates are evaluated and saved together, without changing thresholds.
        failures = []
        if report['fine_radial']['field']['pooled'] > GATES['fine_radial_pooled']:
            failures.append('fine radial/cube pooled error')
        if report['cube_radial_refinement_ratio'] >= GATES['cube_radial_refinement_ratio']:
            failures.append('cube/radial spatial refinement ratio')
        if report['cube_time']['field']['pooled'] > GATES['cube_time_pooled']:
            failures.append('cube temporal refinement')
        report['gate_failures'] = failures
        update('cube_gates_checked')
        if failures:
            raise ValueError(f'Predeclared cube qualification failed: {failures}')
        report['face_flux'] = []
        for epsilon in FLUX_WIDTHS:
            update(f'physical_face_flux_{epsilon}')
            current = c.copy_and_resolve_references()
            current.physical.eps_star = epsilon
            solution = radial if epsilon == EPSILON else solve(current)
            report['face_flux'].append(dict(epsilon=epsilon, **face_flux(solution),
                radial_statistics=solution.statistics))
        report['status'] = 'passed'
        report['seconds'] = time.perf_counter()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        update('complete')
    except Exception as error:
        report['status'] = 'failed'
        report['failure'] = f'{type(error).__name__}: {error}'
        report['seconds'] = time.perf_counter()-started
        update('failed')
        raise


if __name__ == '__main__':
    main()
