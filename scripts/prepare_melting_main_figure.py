"""Prepare current-checkpoint Melting fields and actual zero-level meshes.

Run inference in the JAX environment; run mesh extraction in the optional
rendering environment. No training, reference integration or surface smoothing.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def infer(spec):
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.icemelting3d.operator_data import build_network, identity
    from examples.icemelting3d.operator_training import make_predictor
    from examples.icemelting3d.ellipsoids import signed_distance_numpy

    c = ConfigDict(read(spec['config']))
    manifest = read(spec['reference_manifest'])
    if (identity(c) != manifest['identity'] or c.train.iterations != spec['checkpoint_step']
            or c.train.seed != spec['runtime_seed'] or c.sample.spatial.mode != 'rolling_pool'
            or c.nondim.T_c != spec['physical_time_scale_seconds']):
        raise ValueError('Model/configuration/reference identity mismatch')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This declared dense inference requires the GPU backend')
    checkpoint = Path(spec['checkpoint'])
    original = checkpoint.read_bytes()
    net = eqx.tree_deserialise_leaves(checkpoint, build_network(c, jax.random.key(0)))
    if any(not np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(net) if eqx.is_array(a)):
        raise ValueError('Nonfinite checkpoint parameters')
    output = Path(spec['prepared_root'])
    output.mkdir(parents=True, exist_ok=False)
    axis = np.linspace(0, 1, spec['surface_axis_points'], dtype=np.float32)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    times = np.asarray(spec['times'], dtype=np.float32)
    predict = make_predictor(c)
    records = []
    for record in spec['records']:
        index = record['sample']
        if index not in manifest['test_indices']:
            raise ValueError('Figure sample is not in the declared test bank')
        with np.load(record['reference'], allow_pickle=False) as a:
            ref = {k: a[k] for k in a.files}
        with np.load(record['saved_prediction'], allow_pickle=False) as a:
            old = {k: a[k] for k in a.files}
        np.testing.assert_array_equal(ref['parameters'], manifest['parameters'][index])
        for key in ('parameters', 'points', 'time'):
            np.testing.assert_array_equal(old[key], ref[key])
        np.testing.assert_array_equal(old['target'], ref['phi'])
        np.testing.assert_array_equal(times, ref['time'])
        np.testing.assert_allclose(ref['slice_points'][:, 2], spec['slice_z'], rtol=0, atol=1e-14)
        start = time.perf_counter()
        replay = predict(net, ref['parameters'], ref['points'], times)
        gap = float(abs(replay-old['prediction']).max())
        if gap >= spec['replay_tolerance']:
            raise ValueError(f'Checkpoint replay failed for {index}: {gap}')
        dense = predict(net, ref['parameters'], points, times)
        sliced = predict(net, ref['parameters'], ref['slice_points'], times)
        if (dense.shape != (len(times), len(points)) or sliced.shape != ref['slice_phi'].shape
                or not np.isfinite(dense).all() or not np.isfinite(sliced).all()
                or min(dense.min(), sliced.min()) < -1 or max(dense.max(), sliced.max()) > 1):
            raise ValueError('Invalid dense or slice phase fields')
        ids = np.linspace(0, len(points)-1, 1024, dtype=int)
        independent_ic = np.tanh(signed_distance_numpy(points[ids], ref['parameters'],
            c.initial.reference_sdf_iterations)/(np.sqrt(2)*c.physical.eps_star))
        ic_gap = float(abs(dense[0, ids]-independent_ic).max())
        small = predict(net, ref['parameters'], points[ids], times)
        chunk_gap = float(abs(dense[:, ids]-small).max())
        # The dense native plane and the separately queried reference plane
        # must agree, despite float32 conversion of reference coordinates.
        cube = dense.reshape(len(times), len(axis), len(axis), len(axis))
        plane_gap = float(abs(cube[:, :, :, spec['slice_index']].reshape(len(times), -1)-sliced).max())
        if ic_gap >= spec['initial_tolerance'] or max(chunk_gap, plane_gap) >= spec['replay_tolerance']:
            raise ValueError(f'IC/chunk/slice replay failed: {ic_gap}, {chunk_gap}, {plane_gap}')
        target = output/f'case{index:03d}-fields.npz'
        np.savez_compressed(target, phi=cube, axis=axis, time=times, parameters=ref['parameters'],
            slice_prediction=sliced, slice_reference=ref['slice_phi'], slice_points=ref['slice_points'],
            metric_prediction=old['prediction'], metric_reference=ref['phi'], metric_points=ref['points'])
        result = dict(sample=index, file=str(target), seconds=time.perf_counter()-start,
            replay_gap=gap, initial_gap=ic_gap, chunk_gap=chunk_gap, slice_gap=plane_gap,
            phase_range=[float(dense.min()), float(dense.max())], output_bytes=target.stat().st_size)
        records.append(result)
        print(json.dumps(result), flush=True)
    if checkpoint.read_bytes() != original:
        raise RuntimeError('Source checkpoint changed during rendering inference')
    receipt = dict(status='passed', checkpoint=spec['checkpoint'], config=spec['config'],
        reference_manifest=spec['reference_manifest'], samples=spec['samples'], times=spec['times'],
        backend=jax.default_backend(), jax_version=jax.__version__, numpy_version=np.__version__,
        matmul_precision=c.train.matmul_precision, dtype='float32', axis_order='time,x,y,z',
        surface_axis_points=spec['surface_axis_points'], slice_z=spec['slice_z'],
        checkpoint_unchanged=True, training=False, reference_solve=False, records=records)
    (REPO/spec['inference_receipt']).write_text(json.dumps(receipt, indent=2)+'\n')


def meshes(spec):
    import skimage
    from skimage.measure import marching_cubes
    from scipy.ndimage import map_coordinates

    receipt = read(REPO/spec['inference_receipt'])
    if receipt['checkpoint'] != spec['checkpoint'] or receipt['samples'] != spec['samples']:
        raise ValueError('Inference provenance mismatch')
    root = Path(spec['prepared_root'])/'meshes'
    root.mkdir(exist_ok=False)
    records = []
    for sample in spec['samples']:
        with np.load(Path(spec['prepared_root'])/f'case{sample:03d}-fields.npz', allow_pickle=False) as data:
            phi, axis = data['phi'], data['axis']
        spacing = float(axis[1]-axis[0])
        for frame, time_value in enumerate(spec['times']):
            if not phi[frame].min() < 0 < phi[frame].max():
                raise ValueError(f'No zero-level surface: sample={sample}, time={time_value}')
            vertices, faces, _, _ = marching_cubes(phi[frame], level=0, spacing=(spacing,)*3,
                allow_degenerate=False, gradient_direction='descent', method='lewiner')
            triangles = vertices[faces].astype(np.float64)
            volume = np.einsum('ij,ij->i', triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
            if volume < 0:
                faces = faces[:, ::-1]
                volume = -volume
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            if np.any(counts != 2) or volume <= 0:
                raise ValueError('Zero-level surface is not a closed manifold')
            residual = float(abs(map_coordinates(phi[frame], (vertices/spacing).T, order=1,
                mode='nearest', prefilter=False)).max())
            if residual > spec['mesh_tolerance']:
                raise ValueError(f'Isosurface interpolation mismatch: {residual}')
            path = root/f'case{sample:03d}-frame{frame:03d}.npz'
            np.savez_compressed(path, vertices=vertices, faces=faces)
            records.append(dict(sample=sample, frame=frame, time=time_value, file=str(path),
                vertices=len(vertices), faces=len(faces), volume=float(volume),
                phi_residual=residual, closed=True, bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()]))
    report = dict(status='passed', source=spec['inference_receipt'], skimage_version=skimage.__version__,
        numpy_version=np.__version__, level=0, smoothing=False, records=records)
    (REPO/spec['mesh_receipt']).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(meshes=len(records), maximum_phi_residual=max(r['phi_residual'] for r in records))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--stage', choices=('infer', 'meshes'), required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    if spec['schema'] != 'melting_main_figure_v1':
        raise ValueError('Unsupported Melting figure specification')
    {'infer': infer, 'meshes': meshes}[args.stage](spec)
