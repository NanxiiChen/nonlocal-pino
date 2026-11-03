"""Render the four compact README galleries from existing saved fields/meshes.

No inference, training or reference solve is performed. Run with --case all.
The direct paths deliberately describe these selected historical checkpoints.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path('/root/autodl-tmp/data')
LOGS = Path('/root/autodl-tmp/tf-logs')
DPI = 150
CASES = ('icemelting3d', 'electropolish2d', 'confined_ch2d', 'channel_nsch2d')
MAX_IMAGE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024


def read_json(path):
    return json.loads(Path(path).read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def relative_l2(prediction, reference, *, weights):
    if prediction.shape != reference.shape:
        raise ValueError('Prediction/reference shapes differ')
    numerator = np.sum(weights * (prediction - reference)**2)
    denominator = np.sum(weights * reference**2)
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0:
        raise ValueError('Invalid relative L2 inputs')
    return float(np.sqrt(numerator / denominator))


def time_indices(saved, selected):
    result = []
    for value in selected:
        found = np.flatnonzero(np.isclose(saved, value, rtol=0, atol=1e-7))
        if len(found) != 1:
            raise ValueError(f'Expected one actual snapshot at {value}, got {found}')
        result.append(int(found[0]))
    return result


def save(fig, case, record):
    image = REPO / 'assets' / 'pino' / (case + '.png')
    image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image, dpi=DPI, facecolor='white', pil_kwargs={'optimize': True})
    plt.close(fig)
    size = image.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f'Documentation image exceeds 1 MiB: {image} ({size})')
    record.update(image=str(image.relative_to(REPO)), bytes=size, dpi=DPI,
                  field='phi', numerical_fields_changed=False,
                  weights_or_raw_data_copied=False)
    return record


def melting():
    case = 'icemelting3d'
    root = DATA / case / 'operator_render3d_v1'
    run = LOGS / case / '20261029-melting-operator-online-b16-01'
    provenance = read_json(root / 'prediction-provenance.json')
    manifest = read_json(DATA / case / 'operator_ellipsoids_v1/manifest.json')
    samples, frames, times = (5, 9, 2), (0, 15, 30, 45, 60), (0, 1.25, 2.5, 3.75, 5)
    if not set(samples).issubset(manifest['test_indices']):
        raise ValueError('Melting samples are outside the declared test bank')
    if not (run / 'model_final.eqx').is_file():
        raise ValueError('Selected melting checkpoint is missing')
    elev, azim = np.deg2rad([23, -57])
    view = np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)])
    light = np.array([-1., -1., 2.]); light /= np.linalg.norm(light)
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.8), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(left=.065, right=.96, top=.865, bottom=.045, wspace=.18, hspace=.15)
    sources = []
    for row, sample in enumerate(samples):
        for col, (frame, seconds) in enumerate(zip(frames, times, strict=True)):
            path = root / 'meshes' / f'case{sample:03d}-frame{frame:03d}.npz'
            mesh = read_arrays(path)
            vertices, faces = mesh['vertices'].astype(float), mesh['faces']
            np.testing.assert_allclose(float(mesh['physical_time']), seconds, atol=2e-7, rtol=0)
            if not np.isfinite(vertices).all() or vertices.shape[1:] != (3,) or faces.shape[1:] != (3,):
                raise ValueError('Invalid saved interface mesh')
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths <= 0):
                raise ValueError('Degenerate saved interface triangle')
            front = normals @ view > 0
            colors = to_rgb('#6CA6C7') * (.6 + .4*np.maximum(normals @ light / lengths, 0))[:, None]
            ax = axes[row, col]
            ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none',
                                                antialiased=False, zsort='average'))
            edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
            owners = np.tile(np.arange(len(faces)), 3)
            order = np.lexsort((edges[:, 1], edges[:, 0]))
            edges, owners = edges[order], owners[order]
            np.testing.assert_array_equal(edges[::2], edges[1::2])
            contour = vertices[edges[::2][front[owners[::2]] != front[owners[1::2]]]]
            ax.add_collection3d(Line3DCollection(contour, colors='#17384b', linewidths=.45))
            ax.set_proj_type('ortho'); ax.view_init(elev=23, azim=-57)
            ax.set(xlim=(.08, .92), ylim=(.08, .92), zlim=(.08, .92))
            ax.set_box_aspect((1, 1, 1))
            for key in ('x', 'y', 'z'):
                getattr(ax, 'set_'+key+'ticks')([.2, .8])
                getattr(ax, 'set_'+key+'label')(key+'*' if col == 0 else '', labelpad=-5)
                getattr(ax, key+'axis').pane.fill = False
            ax.tick_params(labelsize=6, pad=-2)
            if row == 0: ax.set_title(f't = {seconds:g} s', fontsize=10)
            if col == 0: ax.text2D(-.15, .85, f'Input {sample}', transform=ax.transAxes, weight='bold')
            sources.append(str(path))
    fig.suptitle('3D melting: three held-out initial ellipsoids\nPINO phase interfaces (phi = 0), fixed camera and spatial scale', fontsize=13)
    return save(fig, case, dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        samples=list(samples), physical_times=list(times), source_files=sources,
        prediction_provenance=str(root/'prediction-provenance.json'),
        prior_prediction_checks=provenance,
        selection='The existing randomly selected held-out trio; unchanged IDs',
        display='Actual saved marching-cubes meshes; no shape smoothing/rescaling; orthographic camera',
        reference='Prediction-only 3D surfaces; eight-input error uses native nonlocal references'))


def electropolish():
    case = 'electropolish2d'
    run = LOGS / case / '20261029-electro-operator-match-nonlocal-01'
    source = run / 'evaluation64.npz'
    arrays = read_arrays(source); report = read_json(run/'evaluation64.json')
    config = read_json(run/'config.json')
    samples, times = (0, 21, 42, 63), (0, .25, .5, .75, 1)
    ids = time_indices(arrays['time'], times)
    regular = report['regular_time_indices']
    fig, axes = plt.subplots(4, 5, figsize=(13.5, 7.3), layout='constrained')
    records = []
    for row, sample in enumerate(samples):
        path = DATA/case/'operator_local_test64_20261029'/f'condition_{sample:03d}'/'local101.npz'
        ref = read_arrays(path)
        for key in ('time', 'X', 'Y'): np.testing.assert_array_equal(arrays[key], ref[key])
        pred = arrays['prediction'][sample]
        np.testing.assert_allclose(arrays['initial'][sample].reshape(51, 101, 2), ref['fields'][0], rtol=0, atol=3e-7)
        error = relative_l2(pred[regular, ..., 0], ref['fields'][regular, ..., 0], weights=1)
        logged = next(r for r in report['records'] if r['index'] == sample)['local101_64']['pooled_phi']
        np.testing.assert_allclose(error, logged, rtol=0, atol=2e-10)
        for col, index in enumerate(ids):
            ax = axes[row, col]
            field = pred[index, ..., 0]
            if not np.isfinite(field).all(): raise ValueError('Nonfinite saved electropolish field')
            image = ax.pcolormesh(arrays['X'], arrays['Y'], field, shading='gouraud', cmap='viridis', vmin=0, vmax=1)
            ax.contour(ref['X'], ref['Y'], ref['fields'][index, ..., 0], levels=[.5], colors='black', linewidths=.8, linestyles='--')
            ax.set(xlim=(0, 1), ylim=(0, .5), aspect='equal')
            ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .25, .5]); ax.tick_params(labelsize=7)
            if row == 0: ax.set_title(f't = {times[col]*config["nondim"]["T_c"]:g} s')
            if col == 0: ax.set_ylabel(f'Input {sample}\nL2 = {100*error:.2f}%', fontsize=9)
        records.append(dict(index=sample, reference=str(path), pooled_phi=error))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='PINO phi')
    fig.suptitle('Electropolishing: four unseen initial interfaces\nColor: PINO phase; dashed: local-FD reference phi = 0.5', fontsize=13)
    return save(fig, case, dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        source_files=[str(source), str(run/'evaluation64.json')], samples=list(samples),
        physical_times=[t*config['nondim']['T_c'] for t in times], records=records,
        selection='All four previously declared spatial-refinement inputs (0,21,42,63), not selected by model score',
        reference='Independent local FD101; metrics use the original21uniform times',
        display='Saved physical phase, continuous display interpolation only; no phase clipping or metric interpolation'))


def cell_to_vertex(values, cells, volumes, vertices):
    weight = np.bincount(cells.ravel(), weights=np.repeat(volumes, 3), minlength=vertices)
    if np.any(weight <= 0): raise ValueError('Uncovered mesh vertex')
    return np.stack([np.bincount(cells.ravel(), weights=np.repeat(v*volumes, 3), minlength=vertices)/weight for v in values])


def spinodal():
    case = 'confined_ch2d'
    run = LOGS/case/'20261031-confined-width256-online-cold50k-01-train'
    gallery = read_json(run/'final_fields_02/gallery.json')
    samples = tuple(gallery['selected'])
    if samples != (1027, 1028, 1026): raise ValueError('Unexpected historical low/median/worst selection')
    fig, axes = plt.subplots(3, 5, figsize=(13, 8), layout='constrained')
    records = []
    for row, sample in enumerate(samples):
        path = run/f'prediction_{sample:03d}.npz'; a = read_arrays(path)
        refpath = DATA/case/'operator_geometry_smallscale10_1024_v1'/f'case_{sample:03d}'/'reference.npz'
        ref = read_arrays(refpath)
        for key in ('points', 'volumes', 'mesh_points', 'mesh_cells', 'holes', 'time'):
            np.testing.assert_array_equal(a[key], ref[key])
        np.testing.assert_array_equal(a['phi_reference'], ref['phi'])
        error = relative_l2(a['phi'], ref['phi'], weights=a['volumes'])
        expected = next(v for v in gallery['verified_all8'] if v['index'] == sample)['phi_pooled']
        np.testing.assert_allclose(error, expected, atol=2e-12, rtol=0)
        vertices, cells = a['mesh_points'], a['mesh_cells']
        tri = Triangulation(vertices[:, 0], vertices[:, 1], cells)
        p = cell_to_vertex(a['phi'], cells, a['volumes'], len(vertices))
        r = cell_to_vertex(ref['phi'], cells, a['volumes'], len(vertices))
        for col, time in enumerate(a['time']):
            ax = axes[row, col]
            image = ax.tripcolor(tri, p[col], shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.tricontour(tri, r[col], levels=[0], colors='black', linewidths=.65, linestyles='--')
            for x, y, radius in a['holes']:
                ax.add_patch(Circle((x, y), radius, facecolor='white', edgecolor='#555555', linewidth=.7))
            ax.add_patch(Ellipse((.5, .5), 1, .8, fill=False, edgecolor='#555555', linewidth=.7))
            ax.set(xlim=(-.02, 1.02), ylim=(.08, .92), aspect='equal')
            ax.set_xticks([0, .5, 1]); ax.set_yticks([.1, .5, .9]); ax.tick_params(labelsize=7)
            if row == 0: ax.set_title(f't = {time:g}')
            if col == 0: ax.set_ylabel(f'Geometry {sample}\nL2 = {100*error:.2f}%', fontsize=9)
        records.append(dict(index=sample, source=str(path), reference=str(refpath), pooled_phi=error))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='PINO phi')
    fig.suptitle('Spinodal decomposition: fixed initial field, three hole geometries\nColor: PINO phase; dashed: native nonlocal reference phi = 0', fontsize=13)
    return save(fig, case, dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        samples=list(samples), times=a['time'].tolist(), records=records,
        source_files=[str(run/'final_fields_02/gallery.json')],
        selection='Existing low/upper-median/worst pooled development-error examples, explicitly stratified',
        reference='Time-refined native regional nonlocal; not local FEM',
        display='Area-weighted incident-cell projection and linear original-mesh interpolation; analytic hole masks; original cell L2 unchanged'))


def channel():
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.problem import walls
    case = 'channel_nsch2d'
    run = LOGS/case/'20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04'
    config = ConfigDict(read_json(run/'config.json'))
    report = read_json(run/'unseen3/evaluation.json')
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 4.8), layout='constrained')
    records = []; times = (0, 1.5, 3, 4.5, 6)
    for row, sample in enumerate(range(3)):
        path = run/'unseen3'/f'case_{sample:03d}'/'prediction.npz'; a = read_arrays(path)
        case_record = report['per_geometry'][sample]
        refpath = Path(case_record['reference']); ref = read_arrays(refpath)
        np.testing.assert_array_equal(a['reference'], ref['fields'])
        np.testing.assert_array_equal(a['points'], ref['points'])
        np.testing.assert_array_equal(a['time'], ref['time'])
        error = relative_l2(a['fields'][..., 0], a['reference'][..., 0], weights=1)
        np.testing.assert_allclose(error, case_record['metrics']['phi'], atol=2e-12, rtol=0)
        ids = time_indices(a['time'], times)
        x, y = a['points'].reshape(96, 40, 2).transpose(2, 0, 1)
        xx = np.linspace(0, 4, 1001); bottom, top = walls(xx, a['amplitudes'], config, xp=np)
        for col, index in enumerate(ids):
            ax = axes[row, col]
            image = ax.pcolormesh(x, y, a['fields'][index, :, 0].reshape(96, 40), shading='gouraud', cmap='RdBu_r', vmin=-1, vmax=1)
            ax.contour(x, y, a['reference'][index, :, 0].reshape(96, 40), levels=[0], colors='black', linewidths=.9, linestyles='--')
            ax.fill_between(xx, 0, bottom, color='#e6e6e6'); ax.fill_between(xx, top, 1, color='#e6e6e6')
            ax.plot(xx, bottom, color='#444444', lw=.8); ax.plot(xx, top, color='#444444', lw=.8)
            ax.set(xlim=(0, 4), ylim=(0, 1), aspect='equal'); ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
            if row == 0: ax.set_title(f't = {times[col]:g}')
            if col == 0: ax.set_ylabel(f'Geometry {sample}\nL2 = {100*error:.2f}%', fontsize=9)
        records.append(dict(index=sample, source=str(path), reference=str(refpath), pooled_phi=error))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.8, label='PINO phi')
    fig.suptitle('Two-phase channel flow: three reserved random-wall geometries\nColor: PINO phase; dashed: local-FEM reference phi = 0', fontsize=13)
    return save(fig, case, dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        samples=[0, 1, 2], times=list(times), records=records,
        source_files=[str(run/'unseen3/evaluation.json')],
        selection='All three predeclared reserved reference geometries; reused validation cohort',
        reference='Independent spatial/time-refined local mixed FEM',
        display='Physical mapped coordinates; fixed physical domain scale; no droplet position correction'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=('all', *CASES))
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})
    manifest = REPO/'artifacts/pino_readme_figures.json'
    report = read_json(manifest) if manifest.exists() else dict(schema=1, figures={})
    if report['schema'] != 1: raise ValueError('Unsupported README figure schema')
    renderers = dict(zip(CASES, (melting, electropolish, spinodal, channel), strict=True))
    selected = CASES if args.case == 'all' else (args.case,)
    for case in selected:
        report['figures'][case] = renderers[case]()
    report.update(generator='scripts/render_pino_readme_figures.py',
                  command='.venv/bin/python scripts/render_pino_readme_figures.py --case all',
                  source_loading='Direct paths, no hashes', inference_runs=0, reference_solves=0,
                  training_updates=0, total_image_bytes=sum(r['bytes'] for r in report['figures'].values()))
    if report['total_image_bytes'] > MAX_TOTAL_BYTES: raise ValueError('Four README images exceed 2 MiB budget')
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(images={k:v['bytes'] for k,v in report['figures'].items()},
                          total_bytes=report['total_image_bytes']), indent=2))


if __name__ == '__main__':
    main()
