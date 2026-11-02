"""Build an offline Chinese explainer from actual SOTA geometry input arrays."""
import json
from pathlib import Path
import shutil

import numpy as np

from examples.confined_ch2d.configs.train_operator import get_config
from examples.confined_ch2d.operator_problem import condition, fixed_initial, valid_holes

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO/'docs/tasks/geometry-operator-explainer.html.in'
OUTPUT = REPO/'docs/tasks/geometry-operator-explainer.html'
DISPLAY = Path('/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/geometry-operator-explainer.html')


def main():
    c = get_config()
    if c.arch.model != 'heterogeneous' or tuple(c.arch.input_dims) != (5, 3) or 'phase' in c:
        raise ValueError('Explainer describes the current two-stream, linear-time SOTA only')
    manifest = json.loads((Path(c.operator.dataset_path)/'manifest.json').read_text())
    cases = []
    for index in (0, 256, 768):
        record = manifest['conditions'][index]
        source = Path(c.operator.dataset_path)/record['file']
        with np.load(source, allow_pickle=False) as f:
            points, volumes, holes = f['points'], f['volumes'], f['holes']
            features, measures = condition(holes, c)
            np.testing.assert_array_equal(features, f['condition'])
            np.testing.assert_array_equal(measures, f['condition_weights'])
            initial = fixed_initial(points, c, xp=np)
            neighbors, coefficients = f['neighbors'], f['coefficients']
            adjacency = [nb[co > 0].astype(int).tolist() for nb, co in zip(neighbors, coefficients, strict=True)]
        if not valid_holes(holes, c) or len(points) > c.operator.online.max_points:
            raise ValueError('Geometry/padding outside the described SOTA range')
        if features.shape != (192, 5) or len(initial) != len(volumes):
            raise ValueError('Unexpected point-stream dimensions')
        np.testing.assert_allclose(np.linalg.norm(features[:, 2:4], axis=1), 1., atol=1e-14)
        for h, offset in zip(holes, (96, 144), strict=True):
            unit = (features[offset:offset+48, :2]-h[:2])/h[2]
            np.testing.assert_allclose(features[offset:offset+48, 2:4], -unit, atol=2e-15)
        domain = np.column_stack([points, initial])
        cases.append(dict(index=index, source=str(source), holes=holes.tolist(),
            boundary=features.tolist(), ds=measures.tolist(), domain=domain.tolist(), area=volumes.tolist(),
            adjacency=[a[:24] for a in adjacency], degree=[len(a) for a in adjacency],
            nodes=len(points), discrete_area=float(volumes.sum()),
            initial_mean=float(volumes@initial/volumes.sum())))
    data = dict(cases=cases, configuration=c.to_dict(), revision='0dc54bc',
        scope='Actual original geometry-bank input arrays; no predicted fields or learned attention scores are displayed')
    template = TEMPLATE.read_text()
    if template.count('__GEOMETRY_INPUT_DATA__') != 1:
        raise ValueError('Require exactly one inline data slot')
    rendered = template.replace('__GEOMETRY_INPUT_DATA__', json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    OUTPUT.write_text(rendered)
    DISPLAY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT, DISPLAY)
    receipt = dict(status='data_and_source_checks_passed', html=str(OUTPUT), display_html=str(DISPLAY),
        template=str(TEMPLATE), source_revision='0dc54bc',
        geometry_inputs=[{k: row[k] for k in ('index','source','nodes','discrete_area','initial_mean')} for row in cases],
        boundary_shape=[192, 5], domain_features=3, padding=1536,
        coordinate_features=2+2*c.arch.spatial_features, width=c.arch.width, heads=c.arch.heads,
        condition_blocks=c.arch.condition_depth, decoder_layers=c.arch.decoder_depth,
        image_patches=False, learned_latent_bottleneck=False, direct_hole_parameter_input=False,
        physical_prediction_displayed=False, browser_validation='pending',
        runtime_or_default_changes=False, new_training=False)
    (REPO/'docs/tasks/geometry-operator-explainer-validation.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: receipt[k] for k in ('status','html','display_html','geometry_inputs')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
