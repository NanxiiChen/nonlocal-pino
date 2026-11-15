"""Independent norms and rejection of corrupted final comparison artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from docs.tasks.pino_channel_final_audit import norms,audit_fields,paired_inputs


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fixture(root):
    # Generate a declared two-case reference cohort. Production evaluator
    # formulas write the reports; the auditor uses independent arithmetic.
    from examples.channel_nsch2d.cohort_evaluate import field_metrics,geometry_responses
    manifest=root/'manifest.json';runs=[root/'full',root/'compact'];cases=[]
    for i in range(2):
        folder=root/f'ref{i}';folder.mkdir()
        cases.append(dict(index=i,root=str(folder),amplitudes=[float(i)]))
    family=dict(formulation='matched-nsch-open-traction-v1',geometry=dict(kind='fixture'),
                physical=dict(mobility=.1,terminal_time=1.),initial=dict(radius=.2))
    write(manifest,dict(count=2,cases=cases,physical_family=family,query_grid=[2,2]))
    for arm,run in enumerate(runs):
        run.mkdir();evaluation=run/'reserved';evaluation.mkdir()
        write(run/'config.json',dict(train=dict(iterations=12),log=dict(run_name=run.name),
            sample=dict(spatial=dict(mode='full' if arm==0 else 'stratified_tiles')),
            geometry=family['geometry'],physical=family['physical'],initial=family['initial'],
            reference=dict(snapshots=3)))
        write(run/'summary.json',dict(status='completed',iterations=12,supervised_evolved_field_samples=0,
            device=dict(kind='fixture',platform='cpu',jax_version='fixture'),wall_seconds=5.,
            spatial=dict(warm_step_with_batch_median_seconds=.21),device_memory_bytes=dict(peak_bytes_in_use=100)))
        write(run/'online-validation.json',dict(status='actual_rolling_inputs_reconstructed'))
        for name in ('model_initial.eqx','optimizer_initial.eqx'):(run/name).write_bytes(b'identical fixture state')
        source=run/'source/example.py';source.parent.mkdir();source.write_text('VALUE=1\n')
        np.savez(run/'sampling.npz',indices=np.zeros((12,2),dtype=int),times=np.ones((12,3)),
                 draw_ids=np.zeros((12,2),dtype=int),amplitudes=np.ones((12,2,1)))
        for name in ('known_inputs.npz','known_inputs_final.npz','evaluation_inputs.npz','reserved_geometries.npz'):
            np.savez(run/name,points=np.arange(8).reshape(4,2),volumes=np.ones(4))
        np.savez(run/'spatial_timing.npz',update=np.full(12,.2),batch_preparation=np.full(12,.01))
        for name,value in (
            ('online_inputs.jsonl',dict(draw_id=0,amplitudes=[0.],training_amplitudes=[0.],pde_solves=0,fem_solves=0)),
            ('online_refresh.jsonl',dict(completed_updates=1,batch=0,slots=[0],draw_ids=[0]))):
            write(run/name,value)
        rows=[];predictions=[];targets=[]
        for i,case in enumerate(cases):
            time=np.array([0.,.5,1.]);points=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
            target=np.linspace(.2,1.3,60).reshape(3,4,5)+i*.08
            prediction=target*(1+.01*(arm+1));prediction[0,:,0]=target[0,:,0]
            fields=dict(points=points,time=time,amplitudes=np.array([float(i)]),common_points=points,
                        common_query_identity=np.array('fixture-v1'))
            reference=Path(case['root'])/'reference.npz'
            np.savez(reference,**fields,fields=target,common_fields=target)
            write(Path(case['root'])/'reference-validation.json',dict(status='passed',reference=str(reference)))
            folder=evaluation/f'case_{i:03d}';folder.mkdir()
            np.savez(folder/'prediction.npz',**fields,fields=prediction,reference=target,
                     common_fields=prediction,common_reference=target)
            values=field_metrics(prediction,target,np.ones(4))
            rows.append(dict(index=i,amplitudes=[float(i)],metrics=values,common_metrics=values,reference=str(reference)))
            predictions.append(prediction);targets.append(target)
        aggregate={key:dict(mean=float(np.mean([r['metrics'][key] for r in rows])),
                           p90=float(np.quantile([r['metrics'][key] for r in rows],.9)),
                           worst=float(np.max([r['metrics'][key] for r in rows]))) for key in rows[0]['metrics']}
        p,q=np.stack(predictions),np.stack(targets)
        physical,mapped=geometry_responses(p,q,p,q)
        write(evaluation/'evaluation.json',dict(status='frozen_unseen_geometry_evaluation_completed',
            run=str(run),manifest=str(manifest),geometries=2,per_geometry=rows,aggregate=aggregate,
            geometry_response=physical,mapped_coordinate_response=mapped))
    return runs,manifest


class FinalAuditTests(unittest.TestCase):
    def test_norms_have_declared_time_and_vector_reduction(self):
        target=np.ones((3,4,5));target[:,:,3]=100.
        prediction=target*np.array([1.1,1.3,1.1,1.2,.8])
        prediction[0,:,1:]=1.e6
        # Avoid a zero dispersed-phase reference while keeping closed-form
        # proportional field errors. Initial flow/chemical errors are excluded.
        target[:,:,0]=.5;prediction[:,:,0]=.55
        result=norms(prediction,target)
        self.assertAlmostEqual(result['phi'],.1)
        self.assertAlmostEqual(result['chemical'],.3)
        self.assertAlmostEqual(result['pressure'],.2)
        self.assertAlmostEqual(result['velocity'],np.sqrt((.1**2+20**2)/(1+100**2)))
        self.assertAlmostEqual(result['terminal_phi'],.1)
        self.assertAlmostEqual(result['dispersed_fraction'],.1)
        prediction[0,0,1]=np.nan
        with self.assertRaises(ValueError):norms(prediction,target)

    def test_rejects_scalar_reference_and_cohort_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            runs,manifest=fixture(Path(temp));run=runs[0];evaluation=run/'reserved'
            result=audit_fields(run,evaluation,manifest)
            self.assertEqual(result['geometries'],2)
            cfg=run/'config.json';original_config=json.loads(cfg.read_text())
            broken_config=copy.deepcopy(original_config);broken_config['physical']['mobility']*=2
            write(cfg,broken_config)
            with self.assertRaises(ValueError):audit_fields(run,evaluation,manifest)
            write(cfg,original_config)
            report=evaluation/'evaluation.json';original=json.loads(report.read_text())
            broken=copy.deepcopy(original);broken['aggregate']['phi']['mean']+=.1;write(report,broken)
            with self.assertRaises(AssertionError):audit_fields(run,evaluation,manifest)
            broken=copy.deepcopy(original);broken['per_geometry'].reverse();write(report,broken)
            with self.assertRaises(ValueError):audit_fields(run,evaluation,manifest)
            write(report,original)
            path=evaluation/'case_000/prediction.npz'
            with np.load(path) as data:arrays={key:data[key] for key in data.files}
            arrays['reference'][1,0,0]+=.2;np.savez(path,**arrays)
            with self.assertRaises(AssertionError):audit_fields(run,evaluation,manifest)

    def test_pair_rejects_uncontrolled_config_source_and_actual_draws(self):
        with tempfile.TemporaryDirectory() as temp:
            runs,_=fixture(Path(temp));self.assertEqual(paired_inputs(runs)['status'],'passed')
            path=runs[1]/'config.json';original=json.loads(path.read_text())
            broken=copy.deepcopy(original);broken['physical']['mobility']*=2;write(path,broken)
            with self.assertRaises(ValueError):paired_inputs(runs)
            write(path,original)
            source=runs[1]/'source/example.py';source.write_text('VALUE=2\n')
            with self.assertRaises(ValueError):paired_inputs(runs)
            source.write_text('VALUE=1\n')
            path=runs[1]/'sampling.npz'
            with np.load(path) as data:arrays={key:data[key] for key in data.files}
            arrays['times'][0,0]+=.01;np.savez(path,**arrays)
            with self.assertRaises(AssertionError):paired_inputs(runs)


if __name__=='__main__':unittest.main()
