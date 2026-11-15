"""Independent local reference monitoring; never added to a loss payload."""
import json
from pathlib import Path

import numpy as np

from examples.confined_ch2d.operator_problem import specification


def load(c, bank, refs):
    settings=c.operator.comparison_reference
    if set(settings)!={'kind','path'} or settings.kind!='local_fem_mixed_p2':
        raise ValueError('Explicit local mixed-P2 reference kind/path required')
    root=Path(settings.path)
    manifest=json.loads((root/'manifest.json').read_text())
    if (manifest['status']!='passed' or manifest['kind']!='independent_mixed_p2_bdf2'
            or manifest['specification']!=specification(c) or manifest['indices']!=bank['test_indices']):
        raise ValueError('Local reference physics, cohort or refinement does not match')
    result={}
    if [row['index'] for row in manifest['cases']]!=bank['test_indices']:
        raise ValueError('Local reference cohort is incomplete/reordered')
    for row in manifest['cases']:
        if row['status']!='passed':raise ValueError('Unqualified local reference')
        path=Path(row['file'])
        if path.is_absolute() or '..' in path.parts:raise ValueError('Reference file must stay inside explicit root')
        with np.load(root/path) as stored:
            ref={key:stored[key] for key in stored.files}
        for key in ('points','volumes','time','holes'):
            np.testing.assert_array_equal(ref[key],refs[row['index']][key])
        if (ref['phi'].shape!=refs[row['index']]['phi'].shape or ref['mu'].shape!=ref['phi'].shape
                or not np.isfinite(ref['phi']).all() or not np.isfinite(ref['mu']).all()):
            raise ValueError('Invalid local physical fields')
        result[row['index']]=ref
    return manifest,result


def metrics(prediction, reference):
    target=np.stack((reference['phi'],reference['mu']),axis=-1).astype(np.float64)
    actual=np.asarray(prediction,dtype=np.float64)
    if actual.shape!=target.shape or not np.isfinite(actual).all():
        raise ValueError('Finite matching physical fields required')
    norm=np.sum(target**2,axis=(0,1))
    positive=np.sum(target[1:]**2,axis=(0,1))
    if np.any(norm<=0) or np.any(positive<=0):raise ValueError('Zero reference channel norm')
    values=np.sqrt(np.sum((actual-target)**2,axis=(0,1))/norm)
    later=np.sqrt(np.sum((actual[1:]-target[1:])**2,axis=(0,1))/positive)
    return dict(phi=float(values[0]),chemical=float(values[1]),channel_mean=float(values.mean()),
                chemical_positive_times=float(later[1]))
