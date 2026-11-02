"""Frozen physics-trained operator on every predeclared unseen FEM geometry."""
from pathlib import Path
import argparse
import json
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.cohort import load
from examples.channel_nsch2d.problem import identity,walls,flow_rate,initial_field
from examples.channel_nsch2d.response_queries import validate_arrays,protocol


def relative(a,b):
    denominator=np.linalg.norm(b)
    if denominator==0:raise ValueError('Relative field error needs a nonzero reference norm')
    return float(np.linalg.norm(a-b)/denominator)


def field_metrics(prediction,target,measure):
    result={}
    for name,sl,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),
                         ('pressure',slice(4,5),1),('chemical',slice(1,2),1)):
        a,b=prediction[begin:,...,sl],target[begin:,...,sl]
        result[name]=relative(a,b)
        weight=np.sqrt(measure)[None,:,None]
        result[name+'_area_weighted']=relative(a*weight,b*weight)
    result['terminal_phi']=relative(prediction[-1,:,0],target[-1,:,0])
    result['dispersed_fraction']=relative((1-prediction[...,0])/2,(1-target[...,0])/2)
    return result


def response_metric(prediction,target):
    a=prediction-prediction.mean(axis=0,keepdims=True)
    b=target-target.mean(axis=0,keepdims=True)
    norm=np.linalg.norm(b)
    if norm==0:raise ValueError('Reserved reference cohort has no geometry response')
    product=np.linalg.norm(a)*norm
    return dict(relative_error=float(np.linalg.norm(a-b)/norm),
        amplitude_ratio=float(np.linalg.norm(a)/norm),
        correlation=float(np.sum(a*b)/product) if product>0 else None,
        scope='Centered across geometries on common normalized(x,eta) query indices; includes coordinate-map effects')


def geometry_responses(prediction,target,common_prediction,common_target):
    if (prediction.ndim!=4 or prediction.shape!=target.shape
            or common_prediction.ndim!=4 or common_prediction.shape!=common_target.shape
            or prediction.shape[:2]!=common_prediction.shape[:2]
            or prediction.shape[-1]!=5 or common_prediction.shape[-1]!=5
            or prediction.shape[0]<2 or prediction.shape[1]<2):
        raise ValueError('Geometry responses need aligned case/time/query/five-field arrays in both coordinate frames')
    physical={};mapped={}
    for name,sl,begin in (('phi',slice(0,1),0),('velocity',slice(2,4),1),('pressure',slice(4,5),1)):
        physical[name]=response_metric(common_prediction[:,begin:,...,sl],common_target[:,begin:,...,sl])
        physical[name]['scope']='Identical physical(x,y)points; excludes coordinate-map effects; flow uses positive times'
        mapped[name]=response_metric(prediction[:,begin:,...,sl],target[:,begin:,...,sl])
    return physical,mapped


def evaluate(run,manifest,output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.network import build_network,encode,fields,field_jets
    run=Path(run);output=Path(output);c=ConfigDict(json.loads((run/'config.json').read_text()))
    cohort=load(manifest,c)
    if json.loads((run/'summary.json').read_text())['status']!='completed':raise ValueError('A complete frozen model is required')
    if json.loads((run/'online-validation.json').read_text())['status']!='actual_rolling_inputs_reconstructed':
        raise ValueError('Actual online training input stream must be independently audited first')
    reference_gate=json.loads((Path(manifest).parent/'cohort-reference-validation.json').read_text())
    if reference_gate['status']!='passed' or reference_gate['cases']!=cohort['count']:
        raise ValueError('Every predeclared geometry needs passed independent FEM refinement')
    if reference_gate['common_query_protocol']!=protocol(c,cohort['query_grid']):
        raise ValueError('Common-physical geometry-response protocol mismatch')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',c.train.compute_dtype=='float64')
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    dtype=np.dtype(c.train.compute_dtype)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    predict=eqx.filter_jit(lambda model,d,x,t:fields(model,encode(model,d),x,t,d['amplitudes'],c))
    jets=eqx.filter_jit(lambda model,d,x,t:field_jets(model,encode(model,d),x,t,d['amplitudes'],c))
    output.mkdir(parents=True,exist_ok=False);results=[];predictions=[];targets=[]
    common_predictions=[];common_targets=[];common_points=None
    for case in cohort['cases']:
        a=np.asarray(case['amplitudes']);root=Path(case['root'])
        gate=json.loads((root/'reference-validation.json').read_text())
        from examples.channel_nsch2d.geometry import with_nominal,envelope_parameters
        cc=with_nominal(c,a)
        if gate['status']!='passed' or gate['identity']!=json.loads(json.dumps(identity(cc))):
            raise ValueError(f'Unvalidated/mismatched reserved reference: {case["index"]}')
        with np.load(gate['reference'],allow_pickle=False) as f:ref={k:f[k] for k in f.files}
        if json.loads(str(ref['identity_json']))!=gate['identity'] or not np.isfinite(ref['fields']).all():
            raise ValueError('Current reserved FEM payload differs from the accepted physical family')
        spec=validate_arrays(ref,c)
        if gate['common_queries']['status']!='passed' or spec!=reference_gate['common_query_protocol']:
            raise ValueError('Every common-physical reference must independently pass refinement')
        if common_points is None:common_points=ref['common_points']
        else:np.testing.assert_array_equal(ref['common_points'],common_points)
        np.testing.assert_array_equal(ref['amplitudes'],a)
        np.testing.assert_array_equal(ref['time'],np.linspace(0,c.physical.terminal_time,c.reference.snapshots))
        if ref['points'].shape!=(int(np.prod(cohort['query_grid'])),2) or ref['fields'].shape!=(c.reference.snapshots,len(ref['points']),5):
            raise ValueError('Reserved reference shapes do not match the frozen query protocol')
        nx,ny=cohort['query_grid'];xx,ee=np.meshgrid((np.arange(nx)+.5)*c.geometry.length/nx,
            (np.arange(ny)+.5)/ny,indexing='ij')
        lower,upper=walls(xx,a,c,xp=np)
        np.testing.assert_allclose(ref['points'],np.stack((xx,lower+ee*(upper-lower)),axis=-1).reshape(-1,2),atol=1e-13,rtol=0)
        raw,_=geometry_data(a,c)
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else dtype) for k,v in raw.items()}
        roots,weight=np.polynomial.legendre.leggauss(64);eta=(roots+1)/2
        section=np.linspace(0,c.geometry.length,17);xx,ee=np.meshgrid(section,eta,indexing='ij')
        b,t=walls(xx,a,c,xp=np);section_points=np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2)
        p=len(ref['points']);m=len(common_points);n=len(section_points);section_start=p+m
        query=np.concatenate((ref['points'],common_points,section_points,raw['boundary'][:,:2]))
        times=jnp.asarray(ref['time'],dtype=dtype)
        values=np.asarray(predict(net,data,jnp.asarray(query,dtype=dtype),times),dtype=np.float64)
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite unseen-domain prediction')
        prediction=values[:,:p];target=ref['fields']
        common_prediction=values[:,p:section_start];common_target=ref['common_fields']
        tp=np.array([1,len(times)//2,len(times)-1]);probe_ids=np.linspace(0,p-1,24,dtype=int)
        common_ids=np.linspace(0,m-1,24,dtype=int)
        probe_points=np.concatenate((ref['points'][probe_ids],common_points[common_ids]))
        derivatives=[np.asarray(v) for v in jets(net,data,jnp.asarray(probe_points,dtype=dtype),times[tp])]
        if any(not np.isfinite(v).all() for v in derivatives):raise FloatingPointError('Nonfinite unseen field derivatives')
        primal,_,dx,dy=derivatives
        expected_primal=np.concatenate((prediction[tp][:,probe_ids],common_prediction[tp][:,common_ids]),axis=1)
        primal_gap=float(np.max(np.abs(primal-expected_primal)))
        divergence=float(np.max(np.abs(dx[...,2]+dy[...,3])))
        flux=np.sum(values[:,section_start:section_start+n,2].reshape(len(times),17,64)*(t-b)[None]*weight/2,axis=-1)
        expected=np.asarray(flow_rate(ref['time'],c,xp=np))[:,None]
        flux_gap=float(np.max(np.abs(flux-expected)))
        bc=values[:,section_start+n:];wall_speed=float(np.max(np.abs(bc[:,raw['boundary'][:,4]==0,2:4])))
        initial_gap=float(np.max(np.abs(prediction[0,:,0]-initial_field(ref['points'],c,xp=np))))
        initial_gap=max(initial_gap,float(np.max(np.abs(common_prediction[0,:,0]-initial_field(common_points,c,xp=np)))))
        if max(primal_gap,divergence,flux_gap,wall_speed,initial_gap)>2e-5:
            raise ValueError(f'Unseen physical/replay constraint failed: {case["index"]}')
        bottom,top=walls(ref['points'][:,0],a,c,xp=np)
        metrics=field_metrics(prediction,target,top-bottom)
        cb,ct=walls(common_points[:,0],envelope_parameters(c),c,xp=np)
        common_metrics=field_metrics(common_prediction,common_target,ct-cb)
        folder=output/f'case_{case["index"]:03d}';folder.mkdir()
        (folder/'config.json').write_text(cc.to_json(indent=2)+'\n')
        np.savez_compressed(folder/'prediction.npz',fields=prediction,reference=target,
            points=ref['points'],time=ref['time'],amplitudes=a,
            common_points=common_points,common_fields=common_prediction,common_reference=common_target,
            common_query_identity=ref['common_query_identity'])
        result=dict(index=case['index'],amplitudes=a.tolist(),metrics=metrics,common_metrics=common_metrics,
            maximum_flux_error=flux_gap,maximum_wall_speed=wall_speed,maximum_divergence=divergence,
            primal_jvp_max_gap=primal_gap,initial_phi_max_gap=initial_gap,reference=gate['reference'])
        results.append(result);predictions.append(prediction);targets.append(target)
        common_predictions.append(common_prediction);common_targets.append(common_target)
        print(json.dumps(result,allow_nan=False),flush=True)
    aggregate={name:dict(mean=float(np.mean([r['metrics'][name] for r in results])),
        p90=float(np.quantile([r['metrics'][name] for r in results],.9)),
        worst=float(np.max([r['metrics'][name] for r in results]))) for name in results[0]['metrics']}
    prediction=np.stack(predictions);target=np.stack(targets)
    common_prediction=np.stack(common_predictions);common_target=np.stack(common_targets)
    response,mapped_response=geometry_responses(prediction,target,common_prediction,common_target)
    report=dict(status='frozen_unseen_geometry_evaluation_completed',run=str(run),manifest=str(manifest),
        device=dict(platform=jax.default_backend(),kind=jax.devices()[0].device_kind,jax_version=jax.__version__,dtype=c.train.compute_dtype),
        geometries=len(results),aggregate=aggregate,per_geometry=results,geometry_response=response,
        common_query_protocol=reference_gate['common_query_protocol'],mapped_coordinate_response=mapped_response,
        mean_principal_target_met=all(aggregate[k]['mean']<.1 for k in ('phi','terminal_phi','velocity','pressure')),
        geometry_response_beats_mean_predictor=all(response[k]['relative_error']<1 for k in ('phi','velocity','pressure')),
        all_principal_cases_below10percent=all(aggregate[k]['worst']<.1 for k in ('phi','terminal_phi','velocity','pressure')),
        physical_checks='Flux/wall/initial at13times; divergence/primal-JVP at3times/24own+24common queries per geometry',
        scope=f'{len(results)}predeclared unseen geometries from the declared prior; finite-cohort estimate, not a uniform continuous-prior guarantee')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(aggregate=aggregate,target=report['mean_principal_target_met']),indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();evaluate(a.run,a.manifest,a.output)
