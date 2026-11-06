"""Compare full-grid and rolling core/halo endpoints from saved physical fields."""
import argparse
import json
from pathlib import Path

from ml_collections import ConfigDict

from .compare import _history, _plot_convergence, saved_predictions
from .compare_sampling import (_costs, _online_receipt, _saved_evaluation,
                               _verify_metrics, TABLE_METRICS)
from .evaluate import load_references
from .full_grid_validation import validate_full_result


def compare_full_grid(run, control, full_evaluation, core_evaluation, output):
    run, control = Path(run).resolve(strict=True), Path(control).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Comparison requires a new output directory: {output}')
    paired = validate_full_result(run, control)
    records, evaluations = [], []
    for label, path, directory in (('rolling core/halo', control, core_evaluation),
                                    ('full grid', run, full_evaluation)):
        c = ConfigDict(json.loads((path/'config.json').read_text()))
        audit = _online_receipt(path, c)
        runtime = json.loads((path/'runtime.json').read_text())
        summary = json.loads((path/'summary.json').read_text())
        refs, manifest = load_references(runtime['reference'], c)
        if manifest['backend'] != 'local_p1_lumped_fem':
            raise ValueError('The common independent reference must be local FEM')
        _, measured = saved_predictions(path, refs)
        history = _history(path, c.train.iterations)
        _verify_metrics(measured, summary['final'], str(path)+'/summary')
        _verify_metrics(measured, history[-1], str(path)+'/last_history')
        full = 'full_grid' in c.sample
        shape = dict(graph_nodes=c.sample.full_grid.nodes if full else c.operator.online.halo,
            pde_rows=c.sample.full_grid.nodes if full else c.sample.K_core,
            neighbors=c.sample.full_grid.neighbors if full else c.operator.online.neighbors,
            graph_copies_per_geometry=1 if full else c.sample.K_pool,
            padding_weights='zero for padded full-grid rows' if full else 'all sampled core weights are one')
        records.append(dict(run=str(path), method=label, history=history, metrics=measured,
            reference=runtime['reference'], online_validation=audit,
            costs=_costs(path, c, True), execution_shape=shape))
        evaluations.append(_saved_evaluation(directory, path, c))
    if records[0]['reference'] != records[1]['reference'] or evaluations[0]['reference'] != evaluations[1]['reference']:
        raise ValueError('Both methods must share the exact development and reused-eight reference paths')
    table = [dict(condition=index, core_halo={name:left[name] for name in TABLE_METRICS},
                  full_grid={name:right[name] for name in TABLE_METRICS})
        for index,(left,right) in enumerate(zip(evaluations[0]['metrics']['conditions'],
                                               evaluations[1]['metrics']['conditions'], strict=True))]
    result = dict(validation=paired, runs=records,
        reused8=dict(core_halo=evaluations[0], full_grid=evaluations[1], table=table),
        metric_definition='Mean per-geometry relative L2 with physical P1 volumes and trapezoidal time weights.',
        limitations=[
            'One paired cold initialization and eight previously inspected comparison geometries.',
            'Whole-mesh volume quadrature targets the same discrete PDE spatial mean as mass-proportional core draws.',
            'Full-grid PDE rows exclude the Dirichlet mouth only; mouth predictions remain available as nonlocal neighbors.',
            'Same geometry, initial/boundary inputs and time draws; full-grid pool IDs are unused provenance.',
            'Complete-method timings include different graph shapes, setup and compilation; producer wait is already included in online overhead.'],
        model_inference_performed=False, training_performed=False)
    output.mkdir(parents=True, exist_ok=False)
    _plot_convergence(output, records)
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), metrics=[item['metrics'] for item in evaluations])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'control', 'full-evaluation', 'core-evaluation', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    compare_full_grid(args.run, args.control, args.full_evaluation, args.core_evaluation, args.output)


if __name__ == '__main__':
    main()
