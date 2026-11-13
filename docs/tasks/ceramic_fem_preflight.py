"""Real64geometry input-only audit of the new signed weak spatial arm."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np


def run(inputs, reference, output):
    from examples.particle_corrosion2d.configs.train_fem_comparison_weak import get_config
    from examples.particle_corrosion2d.train import load_bank
    from examples.particle_corrosion2d.fem_weak import prepare_initial_bank
    from examples.particle_corrosion2d.method_validation import compare_banks
    c=get_config()
    output.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter()
    bank,manifest=load_bank(inputs,c)
    result=prepare_initial_bank(bank,c,manifest,4)
    compare_banks(bank,result,'64initial geometries',True)
    active=np.count_nonzero(result['coefficients'],axis=-1)
    halos=[len(np.unique(points,axis=0)) for geometry in result['graph_points'] for points in geometry]
    # This is a consistency/refinement receipt, not training supervision.
    validation=json.loads((reference/'validation.json').read_text())
    cohort=json.loads((reference/'manifest.json').read_text())
    if validation['status']!='passed' or len(cohort['cases'])!=64 or cohort['backend']!='local_p1_lumped_fem':
        raise ValueError('Qualified64geometry local reference required')
    from examples.particle_corrosion2d.online import geometry_identity
    train_ids={geometry_identity(g) for g in manifest['geometries']}
    if any(geometry_identity(e['geometry']) in train_ids for e in cohort['cases']):
        raise ValueError('Initial training pool overlaps the reserved reference cohort')
    report=dict(status='passed',geometries=len(manifest['geometries']),sets=len(halos),
        identical_fixed_data_and_actual_cores=True,maximum_halo=max(halos),median_halo=float(np.median(halos)),
        maximum_neighbors=int(active.max()),negative_signed_coefficients=int(np.sum(result['coefficients']<0)),
        inputs=str(inputs),reference=str(reference),reference_validation=str(reference/'validation.json'),
        future_training_fields=0,seconds=time.perf_counter()-t)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    shutil.copyfile(__file__,output/'source.py')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.inputs,a.reference,a.output)
