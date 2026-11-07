"""Reproducible independent local/native trajectories on prescribed geometries.

This evaluation-only entry never constructs a training target dataset. The
training runner reads these files solely inside its diagnostic evaluator.
"""
import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .evaluate import render_gallery
from .inputs import conditioning
from .mesh import build_mesh
from .operators import build_operator
from .problem import validate
from .solver import solve
from .dimensional import has_dimensional_inputs


def qualify(c,preview,output,backend,spacing,boundary_points):
    validate(c)
    if backend not in ('local','nonlocal'):raise ValueError('Select local or nonlocal explicitly')
    manifest=json.loads(Path(preview).read_text())
    for name in ('geometry','particles'):
        if manifest['config'][name]!=json.loads(c.to_json())[name]:
            raise ValueError(f'Prescribed geometry {name} differs from physical configuration')
    original_config=json.loads(c.to_json())
    c.mesh.spacing=spacing;c.mesh.boundary_points=boundary_points;c.nl.visibility_points=boundary_points
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    times=np.linspace(0,c.sample.t_max,c.reference.snapshots)
    records=[];references=[];started=time.perf_counter()
    for i,entry in enumerate(manifest['samples']):
        record=entry['geometry'];mesh=build_mesh(c,record,spacing)
        matrix=(sparse.diags(1/mesh['volumes'])@mesh['stiffness'] if backend=='local'
                else build_operator(c,record,mesh['points'],mesh['volumes']))
        fields,info=solve(c,matrix,mesh['points'],mesh['mouth'],times,
            c.reference.rtol,c.reference.atol,c.reference.max_step)
        data=dict(points=mesh['points'],triangles=mesh['triangles'],volumes=mesh['volumes'],
                  times=times,fields=fields,**conditioning(c,record))
        if has_dimensional_inputs(c):
            data['physical_time'] = times*c.nondim.T_c
        file=f'case_{i:03d}.npz';np.savez_compressed(output/file,**data)
        records.append(dict(file=file,geometry=record,mesh=mesh['mesh_info'],solver=info))
        references.append(data)
        print(json.dumps(dict(case=i,backend=backend,nodes=len(mesh['points']),**info)),flush=True)
    result=dict(config=original_config,cases=records,
        backend='local_p1_lumped_fem' if backend=='local' else 'native_visible_regional_integral',
        actual_spacing=spacing,actual_boundary_points=boundary_points,
        preview=str(Path(preview).resolve()),seconds=time.perf_counter()-started,
        use='held-out evaluation only; not a training label or checkpoint-selection target')
    (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    render_gallery(output/'reference_fields.png',references,[r['fields'] for r in references],
                   f'{backend} reference: prescribed held-out geometries')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples.particle_corrosion2d.configs.train_default')
    parser.add_argument('--preview',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--backend',choices=('local','nonlocal'),required=True)
    parser.add_argument('--spacing',type=float,required=True)
    parser.add_argument('--boundary-points',type=int,required=True)
    args=parser.parse_args()
    qualify(importlib.import_module(args.config).get_config(),args.preview,args.output,
            args.backend,args.spacing,args.boundary_points)


if __name__=='__main__':main()
