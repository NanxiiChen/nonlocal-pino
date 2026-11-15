"""Select the reviewed Ceramic temporal figures with audited publication metadata.

Reuses the original PDF/PNG bytes; never renders, trains or changes defaults.
The original renderer provenance remains intact alongside the publication copy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from docs.tasks.ceramic_temporal_figures import comparison_data
from scripts.publish_pino_three_method import ROOT, publish


def read(path):
    return json.loads(Path(path).read_text())


def prepare(bundle, reviewed):
    bundle = Path(bundle).resolve(strict=True)
    reviewed = Path(reviewed).resolve(strict=True)
    report = read(reviewed)
    current = comparison_data(bundle)
    if (report['schema'] != 'pino_time_fusion_figure_v1'
            or report['status'] != 'passed' or report['data'] != current):
        raise ValueError('Reviewed figure data differs from the audited completed pair')
    receipt = read(ROOT/'artifacts/ceramic_time_fusion_validation.json')
    audit = read(bundle/'comparison.json')
    journal = read(bundle/'journal.json')
    if (receipt['status'] != 'passed' or Path(receipt['bundle']) != bundle
            or not audit['common_parameters_identical']
            or not audit['actual_input_streams_identical']):
        raise ValueError('Missing common initialization/sampling audit')
    source = Path(receipt['frozen_source']).resolve(strict=True)
    for name in receipt['numerical_source_files_verified']:
        if (ROOT/name).read_bytes() != (source/name).read_bytes():
            raise ValueError('Numerical source differs from completed run: '+name)
    checks = []
    clocks = []
    for arm, name in zip(current['arms'], ('film', 'joint'), strict=True):
        expected = audit['methods'][name]
        np.testing.assert_allclose(np.array(arm['final_errors'])[:, :2],
                                   expected['final_field_norms'], rtol=2e-12, atol=2e-14)
        if (arm['run'] != expected['run'] or arm['timing']['cold_process_seconds']
                != expected['full_cold_seconds']):
            raise ValueError('Mismatched run or cold-clock endpoint')
        for index, values in enumerate(expected['final_field_norms']):
            for field, value in zip(('phi', 'c'), values, strict=True):
                checks.append(dict(method=arm['method'], sample=index, field=field,
                                   relative_l2=value, evidence=str(bundle/'comparison.json')))
        stage = next(s for s in journal['stages'] if s['name'] == 'train-'+name)
        clocks.append(dict(method=arm['method'], run=arm['run'],
            parent_journal=stage['origin_journal'] if 'origin_journal' in stage else str(bundle/'journal.json'),
            parent_stage=stage['name'], monitor=arm['monitor'],
            elapsed_seconds=arm['elapsed_seconds'], cold_process_seconds=stage['seconds'],
            origin='Direct monotonic parent timestamp before child creation',
            endpoint='Final observation includes final saves and child exit'))
    if len(checks) != receipt['saved_field_channel_norms_checked']:
        raise ValueError('Incomplete saved-field validation')
    metric = ('Equal-point L2 over all space-time nodes per sample and channel; '
              'arithmetic channel mean within each sample, then equal sample mean')
    current.update(convergence_metric=metric, final_metric=metric,
        checkpoint_selection='Final20k weights; no best-checkpoint selection',
        uncertainty=receipt['uncertainty'], saved_field_audit=checks,
        source_reports=[str(bundle/'comparison.json'), str(bundle/'journal.json'),
                        str(ROOT/'artifacts/ceramic_time_fusion_validation.json')],
        pairing=dict(shared_initial_backbone_equal=True, all_actual_sampling_arrays_equal=True,
            source_bytes_verified=True, complete_initial_trees_equal=False,
            stored_entries=[receipt['methods'][key]['model_array_entries'] for key in ('film', 'joint')],
            parameter_count_scope='Stored entries; different temporal-module capacities',
            optimizer_trees='Different temporal shapes; same recipe and fresh initial states'),
        independent_local_single_case=[],
        limitation='One initialization; unequal temporal capacity; all64cases monitored during training')
    report['data'] = current
    report['original_render_provenance'] = str(reviewed)
    report['reproduction'] = dict(
        command=receipt['reproduction'], cwd=str(ROOT),
        publication_command=['.venv/bin/python', '-m', 'scripts.publish_ceramic_time_fusion',
            '--bundle', str(bundle), '--provenance', str(reviewed),
            '--visual-review', receipt['visual_review']],
        note='Use a fresh output directory to rerender; this adapter preserves reviewed image bytes')
    return report, dict(example='particle_corrosion2d', comparison=str(bundle/'comparison.json'),
                        steps=current['steps'], arms=clocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--visual-review', required=True)
    args = parser.parse_args()
    report, clock = prepare(args.bundle, args.provenance)
    target = args.provenance.with_name('publication-provenance.json')
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if target.exists() and target.read_text() != text:
        raise ValueError('Refusing to overwrite different publication provenance')
    clocks_path = ROOT/'artifacts/pino_time_fusion_wall_clock.json'
    clocks = read(clocks_path)
    matches = [c for c in clocks['cases'] if c['example'] == clock['example']]
    if matches and matches != [clock]:
        raise ValueError('Refusing to replace different selected clock evidence')
    target.write_text(text)
    result = publish(target, args.visual_review)
    if not matches:
        clocks['cases'].append(clock)
        clocks_path.write_text(json.dumps(clocks, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
