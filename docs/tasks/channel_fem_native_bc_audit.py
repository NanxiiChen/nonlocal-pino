"""Read native FEM boundary traces, preserving the reference's original equations."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['PKG_CONFIG_PATH'] = str(Path(sys.prefix)/'lib/pkgconfig')
import dolfin as df
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.solver import fem


def main(config, output):
    c = ConfigDict(json.loads(Path(config).read_text()))
    output = Path(output); output.mkdir(exist_ok=False)
    df.set_log_level(40)
    # Short callback/no-callback parity is independent of the later full solve.
    short = ConfigDict(c.to_dict()); short.physical.terminal_time = .1; short.reference.snapshots = 3
    query = np.array([[.5, .5], [1.5, .5], [3.5, .5]])
    ordinary, _ = fem(short, (.12, .08), (16, 8), .025, query, 'parity-none', output/'parity-none.npz')
    seen = []
    observed, _ = fem(short, (.12, .08), (16, 8), .025, query, 'parity-hook', output/'parity-hook.npz',
                       diagnostic_callback=lambda time, fields, mesh, tags: seen.append(float(time)))
    gap = float(np.max(np.abs(ordinary['fields'] - observed['fields'])))
    if gap != 0 or seen != [0., .05, .1]:
        raise ValueError(f'Read-only FEM callback changed results: {gap}, {seen}')
    with np.load(Path(c.reference.root)/'bdf-fine.npz', allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    rows = []

    def inspect(time, fields, mesh, tags):
        u, pressure, phi, mu = fields
        ds = df.Measure('ds', domain=mesh, subdomain_data=tags, metadata={'quadrature_degree': 8})
        n = df.FacetNormal(mesh)
        traction = (2*c.physical.viscosity*df.sym(df.grad(u)) - pressure*df.Identity(2))*n
        entries = {}
        for label, tag in [('wall', 1), ('inlet', 2), ('outlet', 3)]:
            length = float(df.assemble(df.Constant(1.)*ds(tag)))
            entries[label] = dict(
                phi_normal_rms=float(np.sqrt(df.assemble(df.dot(df.grad(phi), n)**2*ds(tag))/length)),
                mu_normal_rms=float(np.sqrt(df.assemble(df.dot(df.grad(mu), n)**2*ds(tag))/length)),
                phi_normal_integral=float(df.assemble(df.dot(df.grad(phi), n)*ds(tag))),
                mu_normal_integral=float(df.assemble(df.dot(df.grad(mu), n)*ds(tag))),
                advective_phase_flux=float(df.assemble(phi*df.dot(u, n)*ds(tag))))
        entries['outlet']['traction_rms'] = float(np.sqrt(df.assemble(df.inner(traction, traction)*ds(3))))
        rows.append(dict(time=float(time), boundaries=entries))
        (output/'native-progress.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')

    actual, report = fem(c, (.12, .08), (64, 24), .025, reference['points'], 'native-boundary-audit',
                         output/'solution.npz', diagnostic_callback=inspect)
    replay = float(np.max(np.abs(actual['fields'] - reference['fields'])))
    if replay > 1e-8:
        raise ValueError(f'Full FEM rerun changed the stored reference: {replay}')
    result = dict(status='native_boundary_traces_and_reference_reproduction_completed',
                  callback_parity_maximum_gap=gap, full_reference_maximum_gap=replay, rows=rows,
                  maximum_newton_iterations=report['maximum_newton_iterations'],
                  scope='Original64x24/dt.025 NSCH equations; native FE traces, weak natural BCs need not vanish pointwise. No new training or changed physical BC.')
    (output/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.config, a.output)
