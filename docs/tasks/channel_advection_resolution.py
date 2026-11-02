"""Passive-transport consistency only; no CH/NS solution or training labels."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(output):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.sparse import csr_matrix,diags,kron
    from scipy.sparse.linalg import expm_multiply
    from examples.channel_nsch2d.configs.train_operator import get_config
    from examples.channel_nsch2d.problem import quadrature,walls,initial_field
    from examples.channel_nsch2d.transport import build_transport
    c=get_config();amplitudes=np.array([c.geometry.upper,c.geometry.lower])
    # Integrate h(x)=H-(upper+lower)*bump(x) exactly in x.
    def primitive(x):
        s=np.clip((x-c.geometry.center)/c.geometry.half_length,-1.,1.)
        P=lambda z:z-4*z**3/3+6*z**5/5-4*z**7/7+z**9/9
        return c.geometry.height*x-amplitudes.sum()*c.geometry.half_length*(P(s)-P(-1.))
    dense=np.linspace(0,c.geometry.length,40001)
    inverse=PchipInterpolator(primitive(dense),dense,extrapolate=False)
    # Integral_0^T Q(t)dt, using the declared quintic ramp and T>=ramp_time.
    T=c.physical.terminal_time
    if T<c.physical.ramp_time:raise ValueError('This probe requires a complete driving ramp')
    integrated_Q=c.physical.mean_velocity*c.geometry.height*(T-c.physical.ramp_time/2)
    rows=[]
    for nx,ny in ((48,20),(96,40),(192,80)):
        c.sample.Nx=nx;c.sample.Ny=ny;points,_=quadrature(amplitudes,c,nx,ny)
        data=build_transport(points,amplitudes,c);x=np.linspace(0,c.geometry.length,nx+1)
        eta=np.linspace(0,1,ny+1);profile=3*eta**2-2*eta**3
        discrete_speed=data['transport_deta']@profile
        # For psi=Q(t)*profile(eta), q_eta=0 and transport separates in eta.
        D=data['transport_dx'].copy();D[0,:]=0.  # prescribed zero dispersed inlet
        generator=kron(diags(-1/data['transport_jacobian'])@csr_matrix(D),diags(discrete_speed),format='csr')
        initial=(1-initial_field(points,c,xp=np))/2
        prediction=expm_multiply(integrated_Q*generator,initial,traceA=float(integrated_Q*generator.diagonal().sum()))
        coordinate=primitive(x)[:,None]-integrated_Q*(6*eta*(1-eta))[None,:]
        valid=coordinate>=0;x0=np.zeros_like(coordinate);x0[valid]=inverse(coordinate[valid])
        inverse_gap=float(np.max(np.abs(primitive(x0[valid])-coordinate[valid])))
        if not np.isfinite(x0).all() or inverse_gap>1e-10:raise ValueError('Characteristic inversion is unresolved')
        bottom,top=walls(x0,amplitudes,c,xp=np);y0=bottom+eta[None,:]*(top-bottom)
        backtracked=np.stack((x0,y0),axis=-1).reshape(-1,2)
        exact=((1-initial_field(backtracked,c,xp=np))/2).reshape(nx+1,ny+1)
        exact[~valid]=0.;exact=exact.ravel();measure=data['transport_measure']
        def relative(a,b):return float(np.sqrt(np.sum((a-b)**2*measure)/np.sum(b*b*measure)))
        rows.append(dict(grid=[nx,ny],points=len(points),dispersed_relative_L2=relative(prediction,exact),
            phi_relative_L2=relative(1-2*prediction,1-2*exact),
            initial_dispersed_area=float(initial@measure),final_dispersed_area=float(prediction@measure),
            exact_final_area=float(exact@measure),characteristic_inversion_gap=inverse_gap))
    report=dict(status='passive_advection_probe_completed_not_coupled_solution_accuracy',time=T,rows=rows,
        method='Sparse exponential of SBP transport; exact mapped characteristics; independent of PINN',
        scope='Known base stream and M=0 passive scalar diagnostic; not the full NS--CH reference, no labels passed to training')
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(p.parse_args().output)
