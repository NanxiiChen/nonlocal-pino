"""Select the reviewed Electropolish-only15k library figure and direct inputs.

The old two-panel10k/20k artifact remains archived; it is no longer selected for
the example or website. This command copies no raw arrays and performs no Git
operations or deployment.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def publish(source, visual_review):
    source = source.resolve(strict=True)
    result = json.loads(source.read_text())
    data, figure = result['data'], result['figure']
    if (result['status'] != 'passed' or data['status'] != 'passed'
            or data['schema'] != 'electropolish_library15k_v1' or data['example'] != 'electropolish2d'
            or data['budget'] != 15000 or data['seed'] != 42
            or data['counts'] != [32,64,128,256,512] or data['test_ids'] != list(range(64))
            or [arm['count'] for arm in data['fixed']] != data['counts']
            or not visual_review.strip()):
        raise ValueError('Require the reviewed complete Electropolish-only15k study')
    for arm in [*data['fixed'], data['online']]:
        values = np.asarray(arm['channel_errors'], float)
        if (arm['budget'] != 15000 or arm['sample_count'] != 64 or values.shape != (64,2)
                or not np.isfinite(values).all() or np.any(values <= 0)
                or not np.isfinite(arm['cold_process_seconds']) or arm['cold_process_seconds'] <= 0):
            raise ValueError('All six final64case endpoints and cold costs are required')
        np.testing.assert_allclose(values.mean(1), arm['sample_errors'], rtol=2e-13)
        np.testing.assert_allclose([values.mean(),values.mean(1).std(ddof=1)],
                                  [arm['mean'],arm['sample_std']], rtol=2e-13)
        for key in ('run','configuration','checkpoint','audit','reference'):
            if not Path(arm[key]).is_absolute() or not Path(arm[key]).exists():
                raise ValueError('Missing direct source: ' + key)
        audit = json.loads(Path(arm['audit']).read_text())
        if audit['status'] != 'passed' or audit['steps'] != 15000:
            raise ValueError('Final independent audit missing')
        audited = audit['final'] if 'final' in audit else audit
        np.testing.assert_array_equal(values, audited['channel_errors'])
    publication = dict(maximum_file_bytes=192*1024, maximum_collection_bytes=256*1024,
                       bytes={}, size_mm=figure['size_mm'], source_provenance=str(source),
                       visual_review=visual_review, selected_examples=['electropolish2d'])
    copies = []
    for extension in ('pdf','png'):
        original = Path(figure[extension]).resolve(strict=True)
        if original.name != 'electropolish2d_library_size_error.'+extension:
            raise ValueError('Unexpected figure filename')
        size = original.stat().st_size
        if size != figure['bytes'][extension] or not 0 < size < publication['maximum_file_bytes']:
            raise ValueError('Figure differs from its audit or exceeds the size budget')
        target = ROOT/'assets/pino-library-size'/original.name
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise ValueError('A different selected figure already exists')
        copies.append((original,target))
        publication[extension] = str(target.relative_to(ROOT))
        publication['bytes'][extension] = size
    if sum(publication['bytes'].values()) >= publication['maximum_collection_bytes']:
        raise ValueError('Selected pair exceeds its size budget')
    publication['png_dimensions'] = list(struct.unpack('>II',Path(figure['png']).read_bytes()[16:24]))
    artifact = dict(data, publication=publication, figure_validation=figure,
                    supersedes='artifacts/fixed_condition_library_results.json',
                    history='Old two-panel assets/data are retained for reproduction; Ceramic is not plotted in this selected study',
                    reproduction=dict(cwd=str(ROOT), command=[str(ROOT/'.venv/bin/python'),'-m','scripts.electro_15k_results',
                        '--bundle',str(Path(data['journals'][1]).parent),'--action','library',
                        '--output','/path/to/new/electropolish-library-figures'],
                        environment=dict(JAX_PLATFORMS='cpu',PYTHONPATH='/tmp/pino-pdf-audit',OPENBLAS_NUM_THREADS='1')))
    target = ROOT/'artifacts/electropolish_library15k_results.json'
    if target.exists() and json.loads(target.read_text()) != artifact:
        raise ValueError('A different library selection already exists')
    for original,destination in copies:
        shutil.copyfile(original,destination)
        if destination.read_bytes() != original.read_bytes():
            raise ValueError('Figure copy differs')
    target.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    return dict(status='selected_locally',manifest=str(target),bytes=sum(publication['bytes'].values()),deployment=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--visual-review',required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.provenance,args.visual_review),indent=2))
