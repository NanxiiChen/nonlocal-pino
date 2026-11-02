"""Frozen interior-mu condensation against the independently solved native grid."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.chemical_projection import project_state,scalar_laplacian
from examples.channel_nsch2d.losses import residual_components,gmls_phase_advection
from examples.channel_nsch2d.problem import identity,material


def probe(run,reference,output):
    run,reference,output=Path(run),Path(reference),Path(output)
    if any(not p.is_absolute()for p in (run,reference,output)):raise ValueError('Absolute artifact paths required')
    if output.exists():raise FileExistsError(output)
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    chemical_scale=c.loss.chemical_residual_scale if 'chemical_residual_scale'in c.loss else c.loss.chemical_scale
    if 'derivatives'not in c or 'time_derivative'not in c:raise ValueError('Original GMLS/JVP model required')
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False)as f:
        data={k:f[k].astype(np.float64)if f[k].dtype.kind=='f'else f[k]for k in f.files}
    with np.load(run/'diagnostic_fields.npz',allow_pickle=False)as f:
        times=f['time'];value=f['fields'];points=f['points']
    with np.load(reference,allow_pickle=False)as f:
        native=f['nodal_fields'];nodes=f['nodes'];nt=f['time'];amps=f['amplitudes'];reference_identity=json.loads(str(f['identity_json']))
    if reference_identity!=json.loads(json.dumps(identity(c))):raise ValueError('Native/PINO physicalidentity differs')
    np.testing.assert_array_equal(times,nt)
    np.testing.assert_allclose(points,data['points'],rtol=0,atol=0)
    np.testing.assert_allclose(nodes,data['points'],rtol=0,atol=2e-7)
    np.testing.assert_allclose(amps,data['amplitudes'],rtol=0,atol=1e-7)
    if value.shape!=native.shape or value.shape!=(len(times),len(points),5):raise ValueError('Aligned native nodal fields required')
    if not all(np.isfinite(a).all()for a in (value,native)):raise ValueError('Nonfinite saved states')
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name)
    shutil.copyfile(Path(__file__).resolve().parents[2]/'examples/channel_nsch2d/chemical_projection.py',output/'chemical_projection.py')
    projected=project_state(value,data,c,xp=np);native_projected=project_state(native,data,c,xp=np)
    inside=data['interior']>0;weight=data['transport_measure']*inside
    rms=lambda a,w:np.sqrt(np.sum(a*a*w,axis=-1)/w.sum())
    relative=lambda a,b,w:rms(a-b,w)/rms(b,w)
    if np.any(rms(native[...,1],weight)<=0):raise ValueError('Native chemicalnorm is zero')
    native_gap=relative(native_projected[...,1],native[...,1],weight)
    if np.max(native_gap)>5e-4:raise ValueError(f'Native interiorlaw not aligned with current stored operator: {native_gap}')
    np.testing.assert_array_equal(projected[:,~inside,1],value[:,~inside,1])
    np.testing.assert_array_equal(projected[...,[0,2,3,4]],value[...,[0,2,3,4]])
    A,kappa=material(c);chemical=projected[...,1]-A*(projected[...,0]**3-projected[...,0])+kappa*scalar_laplacian(projected[...,0],data,xp=np)
    chemical_gap=float(np.max(np.abs(chemical[:,inside])))
    if chemical_gap>2e-10:raise ValueError('Interior constitutive projection failed')
    def gradient(a):
        delta=a[...,data['neighbors']]-a[...,None]
        return (np.sum(data['gradient_x']*delta,axis=-1),np.sum(data['gradient_y']*delta,axis=-1))
    def jets(a):
        dx=[];dy=[];lap=[]
        for i in range(5):
            x,y=gradient(a[...,i]);dx.append(x);dy.append(y);lap.append(scalar_laplacian(a[...,i],data,xp=np))
        return np.stack(dx,axis=-1),np.stack(dy,axis=-1),np.stack(lap,axis=-1)
    x0,y0,l0=jets(value);x1,y1,l1=jets(projected);dm=projected[...,1]-value[...,1];gx,gy=gradient(dm)
    jd={k:jnp.asarray(v)for k,v in data.items()};advection=gmls_phase_advection(jnp.asarray(value),jd)
    # Identical time derivatives cancel in this DIFFERENCE. No zero-time-rate
    # absolute residual is reported as a physical or training diagnostic.
    r0=np.asarray(residual_components(value,np.zeros_like(value),x0,y0,l0,c,phase_advection=advection))
    r1=np.asarray(residual_components(projected,np.zeros_like(value),x1,y1,l1,c,phase_advection=advection))
    expected=np.stack((-c.physical.mobility*scalar_laplacian(dm,data,xp=np),dm,
        value[...,0]*gx,value[...,0]*gy,np.zeros_like(dm)),axis=-1)
    delta_gap=float(np.max(np.abs(r1-r0-expected)))
    if delta_gap>1e-8:raise ValueError(f'Canonical PDE residual-change identity failed: {delta_gap}')
    rows=[]
    for i,t in enumerate(times):
        band=np.abs(native[i,:,0])<.9;wband=weight*band
        if wband.sum()<=0:raise ValueError('Native interface absent')
        row=dict(time=float(t),original_mu_native_relative=float(relative(value[i,:,1],native[i,:,1],weight)),
            projected_mu_native_relative=float(relative(projected[i,:,1],native[i,:,1],weight)),
            native_mu_RMS=float(rms(native[i,:,1],weight)),native_projection_relative_gap=float(native_gap[i]),
            original_interface_mu_native_relative=float(relative(value[i,:,1],native[i,:,1],wband)),
            projected_interface_mu_native_relative=float(relative(projected[i,:,1],native[i,:,1],wband)),
            projected_CH_residual_change_RMS=float(rms(expected[i,:,0],weight)),
            projected_momentum_residual_change_RMS=float(np.sqrt(np.sum(weight*np.sum(expected[i,:,2:4]**2,axis=-1))/weight.sum())))
        # dm is exactly minus the original interior constitutive residual.
        # Compare its physical size to mu, not only to the loss normalizer.
        mismatch=float(rms(dm[i],weight));interface_mismatch=float(rms(dm[i],wband))
        row.update(constitutive_residual_RMS=mismatch,
            loss_normalized_constitutive_residual_RMS=mismatch/chemical_scale,
            constitutive_residual_over_native_mu_RMS=mismatch/float(rms(native[i,:,1],weight)),
            interface_constitutive_residual_RMS=interface_mismatch,
            interface_constitutive_residual_over_native_mu_RMS=interface_mismatch/float(rms(native[i,:,1],wband)))
        rows.append(row)
    report=dict(status='frozen_interior_chemical_projection_consistency_completed',run=str(run),native_reference=str(reference),
        native_maximum_time_relative_projection_gap=float(native_gap.max()),maximum_projected_interior_chemical_residual=chemical_gap,
        maximum_canonical_PDE_change_identity_gap=delta_gap,boundary_mu_values_exactly_retained=True,other_physical_fields_exactly_retained=True,
        rows=rows,training_updates=0,reference_solves=0,
        configured_chemical_scale=c.loss.chemical_scale,
        configured_chemical_residual_scale=chemical_scale,
        chemical_scale_role=('Explicit independent residual normalizer; legacy chemical_scale still controls mu output amplitude'
            if 'chemical_residual_scale'in c.loss else
            'Legacy config value controls BOTH decoder mu amplitude and chemical-row loss normalization; changing it is not a loss-only ablation'),
        saved_state_precision='Stored FP32 fields/operators promoted to hostFP64 for this diagnostic; not an exact replay of training-dtype MSE',
        scope='Frozen nodal equation-level projection only; native reference is the validated directGMLS trajectory, not a new FEM muaccuracy certificate. No trainedmodel, changedBC, projectedboundarylaw or prospectiveaccuracy gain. Time derivatives/velocity/pressure held fixed when reporting residual CH/NS CHANGES.')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();probe(a.run,a.reference,a.output)
