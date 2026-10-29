"""Independent exhaustive-polyline IC distance comparison; no physics changes."""
import json
from pathlib import Path
import numpy as np
from examples.electropolish2d.losses import _rough_surface, _sdf_to_curve

out=Path('docs/tasks/archive')
run=Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01')
c=json.loads((run/'config.json').read_text())
with np.load(Path(c['operator']['dataset_path'])/'conditions.npz') as f:
    data={k:f[k] for k in f.files}
points=data['points'].astype(float)
curve=np.linspace(0,1,2000)
k=np.sqrt(c['physical']['omega_phi']/(2*c['physical']['alpha_phi']))*c['nondim']['L_c']


def exhaustive_distance(points,curve,ycurve,ysurface):
    start=np.column_stack([curve[:-1],ycurve[:-1]])
    delta=np.column_stack([np.diff(curve),np.diff(ycurve)])
    length2=np.sum(delta*delta,axis=1)
    distance=np.empty(len(points))
    for left in range(0,len(points),128):
        q=points[left:left+128,None,:]-start[None,:,:]
        projection=np.clip(np.sum(q*delta[None,:,:],axis=-1)/length2[None,:],0,1)
        displacement=q-projection[...,None]*delta[None,:,:]
        distance[left:left+128]=np.sqrt(np.min(np.sum(displacement**2,axis=-1),axis=1))
    return np.where(points[:,1]<ysurface,1.,-1.)*distance


profiles=[]
for index in (0,1,16,17,18,19):
    rng=np.random.RandomState(data['surface_ids'][index])
    coeff=np.array([(rng.uniform(.3,1)*c['surface']['amplitude'],
        rng.uniform(c['surface']['freq_lo'],c['surface']['freq_hi']),rng.uniform(0,2*np.pi))
        for _ in range(c['surface']['n_modes'])])
    profiles.append(('bank_'+str(index),coeff,index))
coefficients=np.load(run/'online_coefficients.npy')
profiles += [('online_step1_item'+str(i),coefficients[0,i].astype(float),None) for i in range(4)]
records=[];arrays={}
for name,coeff,index in profiles:
    yc=_rough_surface(curve,c['surface']['y0'],coeff)
    ys=_rough_surface(points[:,0],c['surface']['y0'],coeff)
    window=_sdf_to_curve(points[:,0],points[:,1],curve,yc,ys,window=16)
    exact=exhaustive_distance(points,curve,yc,ys)
    vertical=ys-points[:,1]
    assert np.all(abs(window)>=abs(exact)-2e-14)
    phi_window=.5*(1+np.tanh(k*window))
    phi_exact=.5*(1+np.tanh(k*exact))
    phi_vertical=.5*(1+np.tanh(k*vertical))
    boundary=np.isclose(points[:,1],0)|np.isclose(points[:,1],.5)
    for phase in (phi_window,phi_exact,phi_vertical):
        phase[boundary]=np.where(points[boundary,1]==0,1.,0.)
    if index is not None:
        np.testing.assert_allclose(phi_window,data['initial'][index,:,0],rtol=0,atol=3e-7)
    band=(phi_exact>.05)&(phi_exact<.95)&~boundary
    error=abs(phi_window-phi_exact)
    records.append(dict(profile=name,
        max_curve_slope=float(np.max(abs(np.diff(yc)/np.diff(curve)))),
        window_vs_global_phi_relative_l2=float(np.linalg.norm(phi_window-phi_exact)/np.linalg.norm(phi_exact)),
        window_vs_global_phi_max_abs=float(error.max()),
        exact_interface_band_mean_abs_phi_error=float(error[band].mean()),
        exact_interface_band_max_abs_phi_error=float(error[band].max()),
        exact_interface_band_fraction_affected=float(np.mean(error[band]>1e-5)),
        vertical_vs_global_phi_relative_l2=float(np.linalg.norm(phi_vertical-phi_exact)/np.linalg.norm(phi_exact)),
        max_distance_overestimate=float(np.max(abs(window)-abs(exact)))))
    if name=='bank_16':
        arrays=dict(points=points,curve_x=curve,curve_y=yc,phi_window=phi_window,phi_global=phi_exact,
            phi_vertical=phi_vertical,distance_window=window,distance_global=exact,distance_vertical=vertical)
    print(json.dumps(records[-1]),flush=True)
np.savez_compressed(out/'electro-training-distance.npz',**arrays)
result=dict(k=k,phi_10_to_90_normal_width=float(2*np.arctanh(.8)/k),
    max_x_window_half_width=16/1999,profiles=records,
    method='Exhaustive distance to all1999segments, in chunks of128queries; same analytic sign and BC overrides.',
    scope='Global distance is a diagnostic only, not replacement IC/reference. PINN and NO share finite-window construction.')
(out/'electro-training-distance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
