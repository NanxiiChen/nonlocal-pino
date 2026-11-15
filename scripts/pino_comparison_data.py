"""Case-specific, saved-field-audited inputs for the five PINO comparison figures.

All external artifacts are loaded by explicit paths. No inference or trajectory
solver runs here; incomplete declared experiments are rejected, not substituted.
"""
import json
from pathlib import Path

import numpy as np

METHODS=('nonlocal','local_ad','fem')
LABELS=('Nonlocal','Local AD','FEM PINO')
SPINODAL_OMISSION='local_ad: omitted by user after resource measurement'
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())


def lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def require_report(path):
    path=Path(path).resolve(strict=True);report=read(path)
    if report['status']!='passed':raise ValueError('A completed independent audit is required: '+str(path))
    return path,report


def norm(prediction,target):
    a,b=np.asarray(prediction,float),np.asarray(target,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite physical fields required')
    numerator=float(np.sum((a-b)**2));denominator=float(np.sum(b*b))
    if denominator<=0:raise ValueError('No relative norm is defined for this zero reference')
    return dict(error=float(np.sqrt(numerator/denominator)),numerator=numerator,denominator=denominator)


def base_arm(method,run,budget):
    run=Path(run).resolve(strict=True);config=read(run/'config.json')
    if config['train']['iterations']!=budget or config['train']['seed']!=42:
        raise ValueError('Use the predeclared complete seed42/fixed-budget model')
    return dict(method=method,run=str(run),configuration=str(run/'config.json'),
        checkpoint=str((run/'model_final.eqx').resolve(strict=True))),config


def history(path,expected_steps,ids,fields,aggregate_keys):
    records=lines(path)
    if [row['step'] for row in records]!=expected_steps:raise ValueError('Missing actual convergence records')
    values=[]
    for row in records:
        if [r['index'] for r in row['conditions']]!=ids:raise ValueError('Monitoring cohort/order changed')
        item=np.array([[r[key] for key in fields] for r in row['conditions']])
        if not np.isfinite(item).all() or np.any(item<0):raise ValueError('Finite nonnegative monitoring errors required')
        np.testing.assert_allclose(item.mean(axis=0),[row[k] for k in aggregate_keys],rtol=2e-12,atol=1e-14)
        values.append(item.tolist())
    return values


def spec(example,fields,budget,steps,monitor_ids,test_ids,arms,reference,convergence_metric,interval,fraction):
    smoothing=dict(kind=interval,fraction=fraction,replicates=2000,seed=20261114,grid_points=401)
    if interval=='circular_residual':smoothing['block_size']=4
    return dict(status='passed',example=example,fields=[dict(key=key,label=label) for key,label in fields],
        budget=budget,seed=42,steps=steps,monitor_ids=monitor_ids,test_ids=test_ids,arms=arms,
        reference=reference,convergence_metric=convergence_metric,convergence_ylabel='Rel. L² error',
        smoothing=smoothing,
        uncertainty=('Pointwise95% empirical whole-condition-trajectory bootstrap of log-LOWESS arithmetic case means; '
            'exact enumeration for up to4cases,2000draws for larger cohorts. Conditional on the reused cohort and one training trajectory; '
            'no cross-seed uncertainty, simultaneous/population coverage guarantee or smoothing/model-selection bias correction.'
            if interval=='whole_trajectory' else
            'Approximate pointwise95% basic circular-block residual-bootstrap trend interval. Assumes smooth log trend '
            'and approximately stationary short-memory residuals; not cross-seed/cohort uncertainty or simultaneous coverage.'),
        final_metric='Unweighted space-time norm per declared field and sample; no smoothing or field averaging in component panels',
        checkpoint_selection='Final fixed-budget weights; no best-checkpoint or best-sample selection')


def ceramic():
    from docs.tasks.ceramic_fem_comparison_figures import load,REFERENCE
    from scripts.render_ceramic_component_comparison import audit_saved_fields
    original=load();audit=audit_saved_fields(original);arms=[]
    for method,old in zip(METHODS,original,strict=True):
        arms.append(dict(method=method,run=old['run'],configuration=old['configuration'],checkpoint=old['checkpoint'],
            monitor=old['monitor'],monitor_errors=old['monitor_case_values'],
            final_errors=np.asarray(old['final_values'])[:,:2].tolist(),evaluation=old['evaluation']))
    monitor_manifest=Path(original[0]['monitor_reference'])/'manifest.json'
    ids=[r['index'] for r in read(monitor_manifest)['cases']]
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Ceramic reference manifest must identify its four ordered monitors')
    result=spec('particle_corrosion2d',[('phi','Phase'),('concentration','Concentration')],20000,
        original[0]['steps'],ids,list(range(64)),arms,str(REFERENCE),
        'Actual4case volume/time-weighted development errors; differs from final64equal-node norms',
        'whole_trajectory',.30)
    result['convergence_ylabel']='Weighted rel. L² error';result['saved_field_audit']=audit
    result['source_reports']=[str(ROOT/'artifacts/ceramic_component_comparison_figures.json'),str(monitor_manifest)]
    return result


def electro(path):
    path,report=require_report(path);arms=[];steps=list(range(0,10001,250));audit=[];common=None
    for method,key in zip(METHODS,('nonlocal','local_ad','fem_weak'),strict=True):
        selected=report['cases'][key];arm,c=base_arm(method,selected['run'],10000);run=Path(arm['run'])
        monitor=read(run/'local_monitor.json')
        if monitor['indices']!=list(range(64)) or monitor['split']!='test' or monitor['count']!=64:
            raise ValueError('Complete64test monitoring required for new Electro controls')
        identity={key:monitor[key] for key in ('indices','references','times','inputs')}
        if common is None:common=identity
        elif identity!=common:raise ValueError('Local monitoring input/reference changed between methods')
        arm['monitor']=str(run/'local_metrics.jsonl')
        arm['monitor_errors']=history(arm['monitor'],steps,monitor['indices'],('phi','c'),('phi','c'))
        errors=[];predpath=Path(selected['predictions']);summary=read(selected['evaluation'])
        if summary['status']!='passed' or summary['count']!=64:raise ValueError('Final64evaluation incomplete')
        with np.load(predpath) as p:
            take=np.asarray(summary['regular_time_indices']);times=p['time'][take]
            np.testing.assert_array_equal(times,np.linspace(0,1,21))
            for i in range(64):
                refpath=Path(c['operator']['comparison_monitor']['reference_root'])/f'condition_{i:03d}/local101.npz'
                with np.load(refpath) as r:
                    for name in ('time','X','Y'):np.testing.assert_array_equal(p[name],r[name])
                    values=[norm(p['prediction'][i,take,...,j],r['fields'][take,...,j]) for j in range(2)]
                scores=[v['error'] for v in values]
                np.testing.assert_allclose(scores,selected['errors_phi_c'][i],rtol=2e-13,atol=1e-14)
                errors.append(scores);audit.append(dict(method=method,sample=i,prediction=str(predpath),
                    reference=str(refpath),regular_time_indices=take.tolist(),channels=values))
        arm['final_errors']=errors;arms.append(arm)
    result=spec('electropolish2d',[('phi','Phase'),('c','Concentration')],10000,steps,
        common['indices'],list(range(64)),arms,common['references'],
        'All64local-FD test inputs at every record;21uniform times, full physical query grid; per-channel norm then case mean',
        'whole_trajectory',.22)
    result['source_reports']=[str(path)];result['saved_field_audit']=audit
    return result


def melting(path):
    path,report=require_report(path);steps=list(range(0,10001,250));arms=[];audit=[];local=[]
    reference=Path(report['reference']);manifest=read(reference/'manifest.json')
    test_ids=manifest['test_indices'];monitor_ids=test_ids
    if len(test_ids)!=64 or len(set(test_ids))!=64:raise ValueError('Complete64native test cohort required')
    for method in METHODS:
        selected=report['results'][method];arm,c=base_arm(method,selected['run'],10000);run=Path(arm['run'])
        monitoring=read(run/'monitoring.json')
        if (c['operator']['monitor_split']!='test' or monitoring['split']!='test'
                or monitoring['indices']!=test_ids or monitoring['count']!=len(test_ids)
                or monitoring['reference']!=str(reference)):
            raise ValueError('Complete64test monitoring required for new Melting controls')
        arm['monitor']=str(run/'metrics.jsonl');records=[r for r in lines(arm['monitor']) if r['split']=='monitor']
        if [r['step'] for r in records]!=steps:raise ValueError('Require41actual native monitoring records')
        values=[]
        for row in records:
            if [r['index'] for r in row['conditions']]!=monitor_ids:raise ValueError('Melting monitor identities changed')
            case=[[r['pooled_phi']] for r in row['conditions']];values.append(case)
            if not np.isfinite(case).all() or np.any(np.asarray(case)<0):raise ValueError('Finite nonnegative monitoring errors required')
            np.testing.assert_allclose(np.mean(case),row['pooled_phi'],rtol=2e-13)
        errors=[]
        for i,case in zip(test_ids,selected['native_records'],strict=True):
            if i!=case['index']:raise ValueError('Native test ordering mismatch')
            predpath=run/f'prediction_{i:03d}.npz';refpath=reference/f'condition_{i:03d}.npz'
            with np.load(predpath) as p,np.load(refpath) as r:
                for key in ('points','time','parameters'):np.testing.assert_array_equal(p[key],r[key])
                np.testing.assert_array_equal(p['target'],r['phi']);value=norm(p['prediction'],r['phi'])
                np.testing.assert_allclose(value['error'],case['pooled_phi'],rtol=2e-13)
            errors.append([value['error']]);audit.append(dict(method=method,sample=i,prediction=str(predpath),reference=str(refpath),channels=[value]))
        with np.load(run/'prediction_005.npz') as p,np.load(report['local_reference']) as r:
            for key in ('points','time','parameters'):np.testing.assert_array_equal(p[key],r[key])
            score=norm(p['prediction'],r['phi'])
            np.testing.assert_allclose(score['error'],selected['local_case005']['pooled_phi'],rtol=2e-13)
            local.append(dict(method=method,index=5,reference=report['local_reference'],**score))
        arm.update(monitor_errors=values,final_errors=errors);arms.append(arm)
    result=spec('icemelting3d',[('phi','Phase')],10000,steps,monitor_ids,test_ids,arms,str(reference),
        'All64native-reference test ellipsoids at every record;full5time volume-query norm then case mean',
        'whole_trajectory',.22)
    result.update(source_reports=[str(path)],saved_field_audit=audit,independent_local_single_case=local,
        local_refinement_passed=report['strict_local_refinement_passed'],
        limitation='Native64violin can favor the native discretization; accepted localCG2case005 is separate, not a distribution/cohort')
    return result


def spinodal(path):
    path,report=require_report(path);steps=list(range(0,50001,500));arms=[];audit=[];common=None
    methods=METHODS
    if 'methods' in report:
        if (report['methods']!=['nonlocal','fem']
                or 'omitted_method' not in report or report['omitted_method']!=SPINODAL_OMISSION):
            raise ValueError('Spinodal two-method figures require the explicit user-authorized AD omission')
        methods=('nonlocal','fem')
    if set(report['results'])!=set(methods):
        raise ValueError('Incomplete or unexpected declared Spinodal methods')
    for method in methods:
        selected=report['results'][method];arm,c=base_arm(method,selected['run'],50000);run=Path(arm['run'])
        reference=Path(c['operator']['comparison_reference']['path']);manifest=read(reference/'manifest.json')
        if manifest['status']!='passed':raise ValueError('Independent local reference not qualified')
        if common is None:common=(reference,manifest['indices'])
        elif (reference,manifest['indices'])!=common:raise ValueError('Spinodal reference cohort changed')
        arm['monitor']=str(run/'local_metrics.jsonl')
        arm['monitor_errors']=history(arm['monitor'],steps,manifest['indices'],('phi','chemical'),('phase_mean','chemical_mean'))
        values=[]
        for row,expected in zip(manifest['cases'],selected['cases'],strict=True):
            i=row['index']
            if expected['index']!=i or row['status']!='passed':raise ValueError('Unqualified or reordered local cohort')
            predpath=run/f'prediction_{i:03d}.npz';refpath=reference/row['file']
            with np.load(predpath) as p,np.load(refpath) as r:
                for key in ('points','time','volumes','holes'):np.testing.assert_array_equal(p[key],r[key])
                scores=[norm(p[key],r[key]) for key in ('phi','mu')]
                np.testing.assert_allclose([v['error'] for v in scores],[expected['phi'],expected['chemical']],rtol=2e-13)
            values.append([v['error'] for v in scores]);audit.append(dict(method=method,sample=i,
                prediction=str(predpath),reference=str(refpath),channels=scores))
        arm['final_errors']=values;arms.append(arm)
    result=spec('confined_ch2d',[('phi','Phase'),('chemical','Chemical potential')],50000,steps,
        common[1],common[1],arms,str(common[0]),
        'Eight reused reserved geometries;unweighted full5time centroid-query norm, then case mean',
        'whole_trajectory',.12)
    result.update(source_reports=[str(path)],saved_field_audit=audit,
        limitation='Reused eight-case development cohort;primarychemical includes the declared incompatible-wall initial trace; positive-time diagnostic reported separately')
    if methods!=METHODS:
        result.update(comparison_kind='spinodal_nonlocal_fem',omitted_method=report['omitted_method'])
    return result


def channel(path,baseline):
    path,report=require_report(path);baseline,prior=require_report(baseline)
    runs=(prior['historical'],report['local_ad_run'],prior['fem_run']);arms=[];audit=[];reference=None;monitor_ref=None
    fields=(('phi','Phase',slice(0,1),0),('velocity','Velocity',slice(2,4),1),
            ('pressure','Pressure',slice(4,5),1),('chemical','Chemical potential',slice(1,2),1))
    steps=list(range(0,20001,500))
    for method,run in zip(METHODS,runs,strict=True):
        arm,_=base_arm(method,run,20000);run=Path(arm['run']);arm['monitor']=str(run/'metrics.jsonl')
        records=lines(arm['monitor'])
        if [r['step'] for r in records]!=steps:raise ValueError('Require41actual single-case records')
        if any(r['reference']!=records[0]['reference'] for r in records):raise ValueError('Monitoring reference changed')
        if monitor_ref is None:monitor_ref=records[0]['reference']
        elif monitor_ref!=records[0]['reference']:raise ValueError('Method monitoring reference mismatch')
        arm['monitor_errors']=[[[r[key] for key,*_ in fields]] for r in records]
        cohort=read(run/'unseen3/evaluation.json');cases=cohort['per_geometry'];paths=[r['reference'] for r in cases]
        if [r['index'] for r in cases]!=[0,1,2]:raise ValueError('Require all3reserved geometries')
        if reference is None:reference=paths
        elif paths!=reference:raise ValueError('Final evaluation references differ')
        values=[]
        for row in cases:
            predpath=run/'unseen3'/f"case_{row['index']:03d}/prediction.npz"
            with np.load(predpath) as p,np.load(row['reference']) as r:
                for key in ('points','time','amplitudes'):np.testing.assert_array_equal(p[key],r[key])
                np.testing.assert_array_equal(p['reference'],r['fields']);scores=[]
                for key,_,take,begin in fields:
                    score=norm(p['fields'][begin:,:,take],r['fields'][begin:,:,take]);scores.append(score)
                    np.testing.assert_allclose(score['error'],row['metrics'][key],rtol=0,atol=1e-12)
                values.append([v['error'] for v in scores]);audit.append(dict(method=method,sample=row['index'],
                    prediction=str(predpath),reference=row['reference'],channels=scores))
        for j,(key,*_) in enumerate(fields):np.testing.assert_allclose(np.mean(values,axis=0)[j],cohort['aggregate'][key]['mean'],atol=1e-12,rtol=0)
        arm['final_errors']=values;arms.append(arm)
    result=spec('channel_nsch2d',[(key,label) for key,label,*_ in fields],20000,steps,[monitor_ref],[0,1,2],arms,reference,
        'One local-FEM monitoring geometry;phase13times,velocity/pressure/mu12positive times;velocity uses joint vector norm',
        'circular_residual',.22)
    result.update(source_reports=[str(path),str(baseline)],saved_field_audit=audit,
        final_metric='Per-geometry unweighted space-time norm;phase13times,velocity/pressure/mu12positive times;joint vector velocity norm',
        limitation='One initialization and3reused reserved cases;KDE is descriptive only. Historical nonlocal long run uses a different physical GPU UUID.')
    return result
