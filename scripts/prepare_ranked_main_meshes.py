"""Extract actual zero isosurfaces for the ranked Melting preview, without smoothing."""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates


def prepare(root):
    ranking = json.loads((root/'melting_ranking.json').read_text())
    records = []
    for item in ranking['selected']:
        i = item['index']
        with np.load(root/f'melting_dense_{i:03d}.npz') as a:
            fields, axis, times = a['phi'], a['axis'], a['time']
        h = float(axis[1]-axis[0])
        for k, field in enumerate(fields):
            if not np.isfinite(field).all():
                raise ValueError(f'Nonfinite phase in sample {i}, frame {k}')
            path = root/f'melting_mesh_{i:03d}_{k}.npz'
            if field.max() <= 0:
                # A completely liquid field genuinely has no ice surface.
                # Preserve the frame, score and explicit disappearance receipt.
                np.savez_compressed(path, vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32))
                records.append(dict(index=i, frame=k, time=float(times[k]), file=str(path), volume=0.,
                    phi_residual=0., vertices=0, faces=0, closed=None, state='no_positive_phase',
                    phase_min=float(field.min()), phase_max=float(field.max())))
                continue
            if field.min() >= 0:
                raise ValueError(f'Unexpected all-solid domain: {i}, frame {k}')
            vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(h, h, h),
                allow_degenerate=False, gradient_direction='descent', method='lorensen')
            triangles = vertices[faces].astype(float)
            volume = float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
            if volume < 0:
                faces = faces[:, ::-1]; volume = -volume
            edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            gap = float(np.max(np.abs(map_coordinates(field, (vertices/h).T, order=1, mode='nearest', prefilter=False))))
            if np.any(counts != 2) or volume <= 0 or gap > 1e-5:
                raise ValueError(f'Invalid mesh: sample={i}, frame={k}, edge_counts={np.unique(counts)}, volume={volume}, phi_gap={gap}')
            np.savez_compressed(path, vertices=vertices, faces=faces)
            records.append(dict(index=i, frame=k, time=float(times[k]), file=str(path), volume=volume,
                                phi_residual=gap, vertices=len(vertices), faces=len(faces), closed=True))
    (root/'melting_meshes.json').write_text(json.dumps(dict(records=records, smoothing=False,
        method='lorensen', reason='Edge-interpolated zero vertices; every surface independently passes closed-edge and zero-residual checks'), indent=2)+'\n')
    print(json.dumps(dict(meshes=len(records), max_phi_residual=max(r['phi_residual'] for r in records))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root)
