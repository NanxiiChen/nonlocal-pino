"""Synthetic I/O fixtures only: verify metric reductions and cohort rejection.

These are not numerical-reference solutions or experimental training results.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scripts import pino_comparison_data as loader


def write(path,data):path.write_text(json.dumps(data)+'\n')
def logs(path,rows):path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
def run(root,name,budget,extra):
    p=root/name;p.mkdir();c=dict(train=dict(seed=42,iterations=budget),**extra)
    write(p/'config.json',c);(p/'model_final.eqx').write_text('Synthetic path fixture; not a checkpoint')
    return p

def independent(a,b):return float(np.linalg.norm(np.ravel(a-b))/np.linalg.norm(np.ravel(b)))


class ComparisonDataTests(unittest.TestCase):
    def test_electro_full_space_time_channel_norms_and_incomplete_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);refs=root/'references';refs.mkdir();times=np.linspace(0,1,21)
            x,y=np.meshgrid([.2,.8],[.1,.4]);target=np.empty((21,2,2,2))
            target[...,0]=np.arange(1,22)[:,None,None];target[...,1]=.01*np.arange(2,23)[:,None,None]
            for i in range(64):
                p=refs/f'condition_{i:03d}';p.mkdir();np.savez(p/'local101.npz',time=times,X=x,Y=y,fields=target)
            cases={};steps=list(range(0,10001,250))
            for method in ('nonlocal','local_ad','fem_weak'):
                p=run(root,method,10000,dict(operator=dict(comparison_monitor=dict(reference_root=str(refs)))))
                actual=np.stack([target*(1+np.array([.1+i/1000,.3+i/1000])) for i in range(64)])
                errors=[[independent(a[...,j],target[...,j]) for j in range(2)] for a in actual]
                pred=p/'all.npz';np.savez(pred,prediction=actual,time=times,X=x,Y=y)
                write(p/'evaluation.json',dict(status='passed',count=64,regular_time_indices=list(range(21))))
                write(p/'local_monitor.json',dict(indices=list(range(64)),split='test',count=64,references=[str(refs/f'condition_{i:03d}/local101.npz') for i in range(64)],
                    times=times.tolist(),inputs=str(root/'known.npz')))
                conditions=[dict(index=i,phi=errors[i][0],c=errors[i][1]) for i in range(64)]
                logs(p/'local_metrics.jsonl',[dict(step=s,conditions=conditions,phi=np.mean(errors,axis=0)[0],c=np.mean(errors,axis=0)[1]) for s in steps])
                cases[method]=dict(run=str(p),predictions=str(pred),evaluation=str(p/'evaluation.json'),errors_phi_c=errors)
            report=root/'comparison.json';write(report,dict(status='passed',cases=cases))
            data=loader.electro(report)
            self.assertEqual(len(data['saved_field_audit']),192)
            np.testing.assert_allclose(data['arms'][0]['final_errors'][0],[.1,.3],rtol=1e-13)
            self.assertEqual(np.asarray(data['arms'][0]['monitor_errors']).shape,(41,64,2))
            self.assertEqual(data['monitor_ids'], data['test_ids'])
            p=root/'local_ad/local_metrics.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
            with self.assertRaises(ValueError):loader.electro(report)

    def test_melting_native_cohort_stays_separate_from_single_local_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);refs=root/'native';refs.mkdir();ids=list(range(2,66));times=np.linspace(0,1,5)
            write(refs/'manifest.json',dict(test_indices=ids,monitor_indices=[0,1]))
            points=np.array([[.2,.3,.4],[.5,.6,.7]]);params=np.arange(3.);target=np.linspace(-1,1,10).reshape(5,2)
            for i in ids:np.savez(refs/f'condition_{i:03d}.npz',phi=target,points=points,time=times,parameters=params)
            local=root/'local005.npz';np.savez(local,phi=target*1.01,points=points,time=times,parameters=params)
            results={}
            for method in loader.METHODS:
                p=run(root,method,10000,dict(operator=dict(dataset_path=str(refs),monitor_split='test')));rows=[]
                write(p/'monitoring.json',dict(split='test',indices=ids,count=64,reference=str(refs)))
                for i in ids:
                    prediction=target*(1+.01*i);np.savez(p/f'prediction_{i:03d}.npz',prediction=prediction,target=target,points=points,time=times,parameters=params)
                    rows.append(dict(index=i,pooled_phi=independent(prediction,target)))
                logs(p/'metrics.jsonl',[dict(step=s,split='monitor',conditions=rows,pooled_phi=np.mean([r['pooled_phi'] for r in rows])) for s in range(0,10001,250)])
                results[method]=dict(run=str(p),native_records=rows,local_case005=dict(pooled_phi=independent(target*1.05,target*1.01)))
            report=root/'comparison.json';write(report,dict(status='passed',reference=str(refs),local_reference=str(local),
                strict_local_refinement_passed=False,results=results))
            data=loader.melting(report)
            self.assertEqual(data['test_ids'],ids);self.assertEqual(len(data['independent_local_single_case']),3)
            self.assertFalse(data['local_refinement_passed']);self.assertEqual(len(data['saved_field_audit']),192)
            self.assertEqual(np.asarray(data['arms'][0]['final_errors']).shape,(64,1))
            self.assertEqual(np.asarray(data['arms'][0]['monitor_errors']).shape,(41,64,1))
            self.assertEqual(data['monitor_ids'], data['test_ids'])
            self.assertNotAlmostEqual(data['arms'][0]['final_errors'][3][0],data['independent_local_single_case'][0]['error'])
            write(root/'nonlocal/monitoring.json',dict(split='monitor',indices=[0,1],count=2,reference=str(refs)))
            with self.assertRaisesRegex(ValueError,'Complete64test monitoring'):loader.melting(report)

    def test_spinodal_eight_case_monitor_and_saved_field_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);refs=root/'local';refs.mkdir();ids=list(range(1024,1032))
            phi=np.linspace(.1,.8,10).reshape(5,2);mu=np.linspace(.04,.14,10).reshape(5,2)
            common=dict(points=np.array([[.2,.3],[.6,.7]]),time=np.linspace(0,3,5),volumes=np.array([.1,.2]),holes=np.array([[.4,.5,.05],[.6,.5,.05]]))
            cases=[]
            for i in ids:
                filename=f'case_{i}.npz';np.savez(refs/filename,phi=phi,mu=mu,**common)
                cases.append(dict(index=i,file=filename,status='passed'))
            write(refs/'manifest.json',dict(status='passed',indices=ids,cases=cases));results={}
            for method in loader.METHODS:
                p=run(root,method,50000,dict(operator=dict(comparison_reference=dict(path=str(refs)))));errors=[]
                for i in ids:
                    a=phi*(1+.001*(i-1000));b=mu*(1+.002*(i-1000));np.savez(p/f'prediction_{i:03d}.npz',phi=a,mu=b,**common)
                    errors.append(dict(index=i,phi=independent(a,phi),chemical=independent(b,mu)))
                average=np.mean([[r['phi'],r['chemical']] for r in errors],axis=0)
                logs(p/'local_metrics.jsonl',[dict(step=s,conditions=errors,phase_mean=average[0],chemical_mean=average[1]) for s in range(0,50001,500)])
                results[method]=dict(run=str(p),cases=errors)
            report=root/'comparison.json';write(report,dict(status='passed',results=results))
            data=loader.spinodal(report)
            self.assertEqual(data['test_ids'],ids);self.assertEqual(len(data['saved_field_audit']),24)
            self.assertEqual(np.asarray(data['arms'][0]['monitor_errors']).shape,(101,8,2))
            two={method:results[method] for method in ('nonlocal','fem')}
            write(report,dict(status='passed',results=two))
            with self.assertRaisesRegex(ValueError,'declared Spinodal methods'):loader.spinodal(report)
            declared=dict(status='passed',results=two,methods=['nonlocal','fem'],
                          omitted_method=loader.SPINODAL_OMISSION)
            write(report,declared);two_data=loader.spinodal(report)
            self.assertEqual(len(two_data['saved_field_audit']),16)
            self.assertEqual(two_data['comparison_kind'],'spinodal_nonlocal_fem')
            self.assertEqual(two_data['omitted_method'],loader.SPINODAL_OMISSION)
            for arm,original in zip(two_data['arms'],(data['arms'][0],data['arms'][2]),strict=True):
                np.testing.assert_array_equal(arm['final_errors'],original['final_errors'])
                np.testing.assert_array_equal(arm['monitor_errors'],original['monitor_errors'])
            write(report,dict(declared,omitted_method='unexplained missing run'))
            with self.assertRaisesRegex(ValueError,'explicit user-authorized'):loader.spinodal(report)
            write(report,declared)
            monitor=root/'fem/local_metrics.jsonl';complete=monitor.read_text()
            monitor.write_text('\n'.join(complete.splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'Missing actual convergence records'):loader.spinodal(report)
            monitor.write_text(complete)
            bad=cases.copy();bad[0]=dict(bad[0],status='failed');write(refs/'manifest.json',dict(status='passed',indices=ids,cases=bad))
            with self.assertRaises(ValueError):loader.spinodal(report)

    def test_channel_joint_velocity_and_declared_positive_time_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=np.ones((13,3,5));target[...,3]=2.;target[0,:,1:]=0.
            common=dict(points=np.array([[.2,.2],[.4,.2],[.6,.2]]),time=np.linspace(0,6,13),amplitudes=np.arange(4.))
            refs=[]
            for i in range(3):
                path=root/f'reference{i}.npz';np.savez(path,fields=target,**common);refs.append(str(path))
            paths={}
            for method in loader.METHODS:
                p=run(root,method,20000,{});paths[method]=str(p);directory=p/'unseen3';directory.mkdir();cases=[]
                for i,reference in enumerate(refs):
                    output=target*(1+np.array([.1,.3,.1,.2,.4]));output[0,:,1:]=100.
                    sub=directory/f'case_{i:03d}';sub.mkdir();np.savez(sub/'prediction.npz',fields=output,reference=target,**common)
                    errors=dict(phi=independent(output[...,0],target[...,0]),velocity=independent(output[1:,:,2:4],target[1:,:,2:4]),
                        pressure=independent(output[1:,:,4],target[1:,:,4]),chemical=independent(output[1:,:,1],target[1:,:,1]))
                    cases.append(dict(index=i,reference=reference,metrics=errors))
                write(directory/'evaluation.json',dict(per_geometry=cases,aggregate={k:dict(mean=v) for k,v in errors.items()}))
                logs(p/'metrics.jsonl',[dict(step=s,reference=refs[0],**errors) for s in range(0,20001,500)])
            before=root/'prior.json';report=root/'comparison.json'
            write(before,dict(status='passed',historical=paths['nonlocal'],fem_run=paths['fem']))
            write(report,dict(status='passed',local_ad_run=paths['local_ad']))
            data=loader.channel(report,before);values=data['arms'][0]['final_errors'][0]
            np.testing.assert_allclose(values,[.1,np.sqrt((.1**2+.4**2)/5),.4,.3],rtol=2e-13)
            self.assertEqual(len(data['saved_field_audit']),9)
            write(report,dict(status='running',local_ad_run=paths['local_ad']))
            with self.assertRaises(ValueError):loader.channel(report,before)


if __name__=='__main__':unittest.main()
