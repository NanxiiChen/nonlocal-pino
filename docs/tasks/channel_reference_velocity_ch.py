"""CH control driven by stored reference velocities; interpolation uncertainty explicit."""
import argparse
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, RegularGridInterpolator
from ml_collections import ConfigDict
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.problem import walls
from docs.tasks.channel_frozen_ch import FrozenCH


def main(run, numerical, kind, output):
    run, numerical, output = Path(run), Path(numerical), Path(output)
    output.mkdir(exist_ok=False)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    c.sample.Nx, c.sample.Ny = 96, 40
    with np.load(run/'evaluation_inputs.npz', allow_pickle=False) as z:
        amplitudes = z['amplitudes']
    raw, _ = geometry_data(amplitudes.astype(np.float64), c)
    scheme = FrozenCH(c, raw, (96, 40))
    with np.load(numerical/'solution.npz', allow_pickle=False) as z:
        fields, times = z['nodal_fields'], z['time']
    if fields.shape != (13, 15633, 5):
        raise ValueError('Explicit fine192x80/13snapshot reference required')
    velocity = fields.reshape(13, 193, 81, 5)[:, ::2, ::2, 2:4].reshape(13, scheme.n, 2)
    interpolate = interp1d(times, velocity, axis=0, bounds_error=True) if kind == 'linear' else PchipInterpolator(times, velocity, axis=0, extrapolate=False)
    with np.load(Path(c.reference.root)/'bdf-fine.npz', allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    query = reference['points']
    bottom, top = walls(query[:, 0], amplitudes, c, xp=np)
    mapped = np.column_stack((query[:, 0], (query[:, 1] - bottom)/(top - bottom)))
    axes = (np.linspace(0, c.geometry.length, 97), np.linspace(0, 1, 41))
    def export(z):
        values = np.column_stack((z[:scheme.n], z[scheme.n:])).reshape(97, 41, 2)
        return RegularGridInterpolator(axes, values)(mapped)
    old = scheme.initial(); older = old.copy()
    snapshots, rows = [export(old)], []
    start = time.perf_counter()
    for step in range(1, 241):
        t = step*c.time_discretization.dt
        u = np.asarray(interpolate(t))
        history = old[:scheme.n] if step == 1 else 2*old[:scheme.n] - .5*older[:scheme.n]
        new, info = scheme.step(old, history, u, np.zeros((scheme.n, 2)),
                               1. if step == 1 else 1.5, c.time_discretization.dt, 1e-9, 12)
        older, old = old, new
        rows.append(dict(step=step, time=t, **info))
        if step % 20 == 0:
            snapshots.append(export(old))
            print(json.dumps({'kind': kind, 'step': step, 'seconds': time.perf_counter() - start, **info}), flush=True)
    values = np.asarray(snapshots); target = reference['fields'][..., 0]
    relative = lambda a, b: float(np.linalg.norm(a - b)/np.linalg.norm(b))
    report = dict(status='reference_velocity_CH_control_completed', interpolation=kind,
                  seconds=time.perf_counter() - start, phi_L2=relative(values[..., 0], target),
                  terminal_phi_L2=relative(values[-1, :, 0], target[-1]),
                  terminal_minority_L2=relative((1 - values[-1, :, 0])/2, (1 - target[-1])/2),
                  maximum_scaled_residual=max(r['maximum_scaled_residual'] for r in rows),
                  scope='Same96x40/dt.025 CH solver and zero natural BCs; velocities interpolated from13fineGMLS snapshots. This introduces temporal interpolation uncertainty and is a diagnostic, not a new fully coupled reference.')
    np.savez_compressed(output/'solution.npz', fields=values, reference=reference['fields'][..., :2],
                        points=query, time=reference['time'], amplitudes=amplitudes)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--numerical', required=True)
    p.add_argument('--kind', choices=('linear', 'pchip'), required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.run, a.numerical, a.kind, a.output)
