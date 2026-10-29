"""Actual deterministic feature wavelengths, mesh resolution and map range checks."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config
from examples.confined_ch2d.problem import load_reference
c=get_config();r=load_reference(c);v=r['volumes'];distance=cKDTree(r['points']).query(r['points'],k=2)[0][:,1]
norm=lambda a:np.sqrt(np.sum(v*a*a,axis=-1).sum())
report=dict(feature_formula='sin/cos(pi*k*(x-center)/axes), k=1..m, deterministic and fixed',
    normalized_coordinate_range='approximately[-1,1] across each ellipse diameter',
    spatial={},initial_basic_coordinate_maximum=float(np.max(abs(np.asarray(c.initial.waves)[:,:2]))),
    temporal='NO temporal Fourier embedding in global_basis static spatial MLP; rank24 Chebyshev in log time',
    nearest_neighbor_quantiles=np.quantile(distance,[.05,.5,.95]).tolist(),reference_ranges={})
for m in (8,5):
    wavelengths=2*np.asarray(c.geometry.axes)/m
    report['spatial'][str(m)]=dict(max_angular_wave_number=(np.pi*m/np.asarray(c.geometry.axes)).tolist(),minimum_wavelength=wavelengths.tolist(),wavelength_over_median_nearest_spacing=(wavelengths/np.median(distance)).tolist(),feature_count=2+4*m+int(c.arch.initial_feature))
for kind in ('local','nonlocal'):
    phi=r['phi_'+kind];clipped=np.clip(phi,-1,1)
    report['reference_ranges'][kind]=dict(minimum=float(phi.min()),maximum=float(phi.max()),terminal_volume_outside_unit=float(v[abs(phi[-1])>1].sum()/v.sum()),strict_unit_range_pooled_error_lower_bound=float(norm(clipped-phi)/norm(phi)),strict_unit_range_terminal_error_lower_bound=float(norm(clipped[-1]-phi[-1])/norm(phi[-1])))
from examples.confined_ch2d.native_reference import native_matrix
A=native_matrix(r['points'],v,c);z=(r['points']-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes)
rms=lambda a:float(np.sqrt(np.sum(v*a*a)/v.sum()))
report['native_feature_operator_gains']={}
for k in (5,8):
    rows=[]
    for axis in (0,1):
        for name,fn in [('sin',np.sin),('cos',np.cos)]:
            f=fn(np.pi*k*z[:,axis]);f-=v@f/v.sum();norm=rms(f)
            rows.append(dict(axis=axis,function=name,A_gain=rms(A@f)/norm,A_squared_gain=rms(A@(A@f))/norm))
    report['native_feature_operator_gains'][str(k)]=rows
report['scope']='Nearest spacing ratios are descriptive, not a Nyquist theorem for an irregular graph. IC tanh creates higher harmonics. Lowering modes changes feature dimension/initialization; comparison is not parameter matched. Range lower bound ignores additional mass constraint.'
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
