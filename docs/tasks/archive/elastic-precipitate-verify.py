"""Independent saved-field energy/equilibrium/morphology audit; direct paths."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from examples.precipitate2d.problem import load_reference
from examples.precipitate2d.solver import coefficients,elastic_fields,relative


def audit(run):
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    ref=load_reference(c)
    with np.load(run/'replayed_prediction.npz') as f:pred={k:f[k].copy() for k in f.files}
    n=c.sample.Nx
    coef=coefficients(c,n,'nonlocal')
    L=coef[0]
    energy=[]
    force=[]
    for phi in pred['phi']:
        qel,u,stress,ee=elastic_fields(phi,c,coef)
        lap=np.fft.ifft2(L*np.fft.fft2(phi)).real
        energy.append(float(np.mean(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*lap)+ee))
        sx,sy,ss=np.moveaxis(stress,-1,0)
        kx,ky=coef[3:5]
        f=np.stack((np.fft.ifft2(1j*(kx*np.fft.fft2(sx)+ky*np.fft.fft2(ss))).real,
                    np.fft.ifft2(1j*(kx*np.fft.fft2(ss)+ky*np.fft.fft2(sy))).real),axis=-1)
        force.append(float(np.linalg.norm(f)/(np.linalg.norm(stress)/c.geometry.length)))
    np.testing.assert_allclose(pred['u'],np.asarray([elastic_fields(p,c,coef)[1] for p in pred['phi']]),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(pred['stress'],np.asarray([elastic_fields(p,c,coef)[2] for p in pred['phi']]),atol=1e-10,rtol=1e-12)
    band=np.abs(ref['phi_nonlocal'])<.9
    report=dict(status='passed',run=str(run),
        native_pooled_interface_band_phi=relative(pred['phi'][band],ref['phi_nonlocal'][band]),
        native_terminal_interface_band_phi=relative(pred['phi'][-1][band[-1]],ref['phi_nonlocal'][-1][band[-1]]),
        maximum_scaled_equilibrium_force=max(force),
        predicted_max_energy_increase=max(0.,float(np.max(np.diff(energy)))),
        predicted_energy=energy,reference_energy=ref['energy_nonlocal'].tolist(),
        common_local_terminal_phi=relative(pred['phi'][-1],ref['phi_local'][-1]),
        max_q_per_time_relative_error_at_initial=float(np.linalg.norm(pred['q'][0]-ref['q_nonlocal'][0])/np.linalg.norm(ref['q_nonlocal'][0])),
        stress_norm='Euclidean norm of stored xx,yy,xy components; not tensor Frobenius norm')
    if report['maximum_scaled_equilibrium_force']>1e-10 or report['predicted_max_energy_increase']>1e-8:
        raise AssertionError(report)
    shapes={}
    coords=np.arange(n)*c.geometry.length/n
    fig,ax=plt.subplots()
    for name,p in [('initial',ref['phi_nonlocal'][0]),('uncoupled',ref['phi_uncoupled'][-1]),
                   ('coupled_reference',ref['phi_nonlocal'][-1]),('pinn',pred['phi'][-1])]:
        contour=ax.contour(coords,coords,p,levels=[0])
        vertices=max(contour.allsegs[0],key=len)
        axes=(vertices.max(axis=0)-vertices.min(axis=0))/2
        shapes[name]=dict(semiaxes=axes.tolist(),x_to_y_ratio=float(axes[0]/axes[1]))
    plt.close(fig)
    report['zero_contour_geometry']=shapes
    report['reference_checks']=json.loads(Path(c.reference.path).with_suffix('.json').read_text())['checks']
    return report


if __name__=='__main__':
    run=Path('/root/autodl-tmp/tf-logs/precipitate2d/20261027-periodic-elastic-pilot-01')
    result=audit(run)
    target=Path(__file__).with_name('elastic-precipitate-validation.json')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
