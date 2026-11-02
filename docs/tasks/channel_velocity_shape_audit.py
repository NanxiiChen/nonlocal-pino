"""Compare velocity gradients and shape-changing velocity, using a common droplet mask."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.operators import first_derivative_stencils
from examples.channel_nsch2d.problem import walls
from docs.tasks.channel_frozen_ch import difference_matrix


def main(run, numerical, output):
    run, numerical, output = Path(run), Path(numerical), Path(output)
    if output.exists():
        raise FileExistsError(output)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    with np.load(run/'diagnostic_fields.npz', allow_pickle=False) as z:
        learned = {k: z[k] for k in z.files}
    with np.load(numerical/'solution.npz', allow_pickle=False) as z:
        target = {k: z[k] for k in z.files}
    np.testing.assert_allclose(learned['points'], target['nodes'], atol=3e-7, rtol=0)
    np.testing.assert_array_equal(learned['time'], target['time'])
    raw, _ = geometry_data(target['amplitudes'], c)
    n = len(raw['points'])
    Dx, Dy = [difference_matrix(raw['neighbors'], raw[name], n) for name in ('gradient_x', 'gradient_y')]
    rows = []
    for i, time in enumerate(target['time']):
        if time == 0:
            continue
        actual, ref = learned['fields'][i], target['nodal_fields'][i]
        gradients = []
        for value in (actual, ref):
            vx, vy = Dx@value[:, 2:4], Dy@value[:, 2:4]
            gradients.append(np.column_stack((vx[:, 0], vy[:, 1], (vy[:, 0] + vx[:, 1])/np.sqrt(2))))
        groups = {}
        for name, mask in [('droplet', ref[:, 0] < 0), ('interface', np.abs(ref[:, 0]) < .9)]:
            w = raw['volumes']*mask
            if w.sum() <= 0:
                raise ValueError('Reference droplet/interface missing')
            mean = [np.sum(value[:, 2:4]*w[:, None], axis=0)/w.sum() for value in (actual, ref)]
            fluctuation = [value[:, 2:4] - m for value, m in zip((actual, ref), mean, strict=True)]
            norm = lambda x: float(np.sqrt(np.sum(x*x*w[:, None])/w.sum()))
            groups[name] = dict(velocity_relative_L2=norm(actual[:, 2:4] - ref[:, 2:4])/norm(ref[:, 2:4]),
                centroid_velocity_gap=float(np.linalg.norm(mean[0] - mean[1])),
                reference_fluctuating_velocity_rms=norm(fluctuation[1]),
                fluctuating_velocity_relative_L2=norm(fluctuation[0] - fluctuation[1])/norm(fluctuation[1]),
                strain_rate_relative_L2=norm(gradients[0] - gradients[1])/norm(gradients[1]))
        rows.append(dict(time=float(time), regions=groups))
    with np.load(run/'prediction.npz', allow_pickle=False) as z:
        common = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(common['points'], target['points'])
    ids, gx, gy, _ = first_derivative_stencils(common['points'], common['points'], c)
    Bx = difference_matrix(ids, gx, len(ids)); By = difference_matrix(ids, gy, len(ids))
    bottom, top = walls(common['points'][:, 0], target['amplitudes'], c, xp=np)
    area = (top - bottom)*c.geometry.length/len(ids)
    reference_checks = []
    for i, time in enumerate(common['time']):
        if time == 0:
            continue
        fields = [common['reference'][i], target['fields'][i], common['fields'][i]]
        E = []
        for f in fields:
            dx, dy = Bx@f[:, 2:4], By@f[:, 2:4]
            E.append(np.column_stack((dx[:, 0], dy[:, 1], (dy[:, 0] + dx[:, 1])/np.sqrt(2))))
        record = {}
        for label, mask in [('droplet', fields[0][:, 0] < 0), ('interface', np.abs(fields[0][:, 0]) < .9)]:
            w = area*mask
            norm = lambda a: float(np.sqrt(np.sum(a*a*w[:, None])/w.sum()))
            record[label] = dict(GMLS_reference_vs_FEM_strain_L2=norm(E[1] - E[0])/norm(E[0]),
                                 PINO_vs_FEM_strain_L2=norm(E[2] - E[0])/norm(E[0]))
        reference_checks.append(dict(time=float(time), regions=record))
    report = dict(status='completed', rows=rows, common_query_reference_strain_check=reference_checks,
                  scope='Same fine grid and FP64 GMLS derivatives on both native saved fields; masks from direct reference only for diagnosis. Reference strain independently compared with FEM on common96x40midpoint queries using identical derivative stencils. Remove each field\'s local mean velocity to expose deformation components; not a causal ablation or a full droplet-moment evolution law.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows[-1], indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--numerical', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.run, a.numerical, a.output)
