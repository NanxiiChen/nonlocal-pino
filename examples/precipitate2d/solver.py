"""Independent NumPy/SciPy reference; no JAX or PINN operator imports."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.signal import resample
from examples.precipitate2d.problem import grid, identity, initial_field, validate_config


def coefficients(c,n,method):
    k = 2*np.pi*np.fft.fftfreq(n,d=c.geometry.length/n)
    kx,ky = np.meshgrid(k,k,indexing='xy')
    kv = np.stack((kx,ky),axis=-1)
    k2 = kx*kx+ky*ky
    lam,mu = c.physical.lame_lambda,c.physical.shear_modulus
    exx,eyy,exy = c.physical.eigenstrain
    E = np.array([[exx,exy],[exy,eyy]])
    S = lam*np.trace(E)*np.eye(2)+2*mu*E
    A = mu*k2[...,None,None]*np.eye(2)+(lam+mu)*kv[...,None]*kv[...,None,:]
    b = np.einsum('ij,...j->...i',S,kv)
    A[0,0] = np.eye(2)  # Explicit translational gauge, b[0,0]=0.
    v = np.linalg.solve(A,b[...,None])[...,0]
    B = np.sum(S*E)-np.sum(b*v,axis=-1)
    if method == 'local_spectral':
        L = k2
    elif method == 'nonlocal':
        h,delta = c.geometry.length/n,c.nl.delta
        if not h<delta<c.geometry.length/2:
            raise ValueError('Reference finite horizon must contain neighbors')
        radius = int(np.ceil(delta/h))
        yy,xx = np.mgrid[-radius:radius+1,-radius:radius+1]
        xx,yy = xx.ravel()*h,yy.ravel()*h
        r = np.hypot(xx,yy)
        mask = (r>0)&(r<delta)
        xx,yy,r = xx[mask],yy[mask],r[mask]
        w = 120*h*h/(np.pi*delta**4)*(1-r/delta)**2
        w *= 2/np.sum(w*xx*xx)
        L = np.sum(w[:,None,None]*(1-np.cos(xx[:,None,None]*kx+yy[:,None,None]*ky)),axis=0)
    else:
        raise ValueError(f'Unknown reference method {method}')
    return L,B,v,kx,ky,E


def elastic_fields(phi,c,coef):
    _,B,v,kx,ky,E = coef
    h = (1+phi)/2
    hh = fft2(h)
    uh = -1j*v*hh[...,None]
    u = ifft2(uh,axes=(0,1)).real
    strain = np.stack((ifft2(1j*kx*uh[...,0]).real,ifft2(1j*ky*uh[...,1]).real,
        .5*ifft2(1j*(ky*uh[...,0]+kx*uh[...,1])).real),axis=-1)
    elastic = strain-h[...,None]*np.array([E[0,0],E[1,1],E[0,1]])
    stress = 2*c.physical.shear_modulus*elastic+c.physical.lame_lambda*(elastic[...,0]+elastic[...,1])[...,None]*np.array([1.,1.,0.])
    chemical = -.5*(stress[...,0]*E[0,0]+stress[...,1]*E[1,1]+2*stress[...,2]*E[0,1])
    energy = .5*np.mean(stress[...,0]*elastic[...,0]+stress[...,1]*elastic[...,1]+2*stress[...,2]*elastic[...,2])
    return chemical,u,stress,float(energy)


def solve_reference(c,n,method,dt):
    if n%2 != 1:
        raise ValueError('Reference requires an odd grid')
    steps = round(c.physical.terminal_time/dt)
    if not np.isclose(steps*dt,c.physical.terminal_time,atol=1e-12,rtol=0) or steps%(c.reference.snapshots-1):
        raise ValueError('Reference dt must align duration and snapshots')
    coef = coefficients(c,n,method)
    L,B,*_ = coef
    p0 = initial_field(grid(n,c.geometry.length),c,xp=np).reshape(n,n)
    current = fft2(p0)
    M,beta,eps = c.physical.mobility,c.physical.beta,c.physical.epsilon
    # Exact homogeneous elastic feedback is linear and treated implicitly.
    linear = -M*L*(eps**2*L+B/4)
    def nonlinear(hat):
        p = ifft2(hat).real
        return -M*L*fft2(beta*(p**3-p))
    data = {k:[] for k in ('phi','q','u','stress','energy','elastic_energy')}
    def save():
        p = ifft2(current).real
        if not np.isfinite(p).all() or np.max(np.abs(p))>3:
            raise FloatingPointError('Reference phase divergence; reduce dt')
        qel,u,s,ee = elastic_fields(p,c,coef)
        lap = ifft2(L*current).real
        q = beta*(p**3-p)+eps**2*lap+qel
        energy = np.mean(.25*beta*(p*p-1)**2+.5*eps**2*p*lap)+ee
        for k,a in zip(data,(p.copy(),q,u,s,energy,ee),strict=True):data[k].append(a)
    save()
    previous = current.copy()
    oldnon = nonlinear(current)
    for step in range(1,steps+1):
        if step==1:
            after = (current+dt*oldnon)/(1-dt*linear)
        else:
            newnon = nonlinear(current)
            after = (4*current-previous+2*dt*(2*newnon-oldnon))/(3-2*dt*linear)
            oldnon = newnon
        previous,current = current,after
        if step%(steps//(c.reference.snapshots-1))==0:save()
    return {k:np.asarray(v) for k,v in data.items()}


def transfer(a,n):
    return resample(resample(a,n,axis=1),n,axis=2)


def relative(a,b):
    denom = np.linalg.norm(b)
    if denom == 0:
        raise ValueError('Relative L2 undefined for zero reference; use absolute norm')
    return float(np.linalg.norm(a-b)/denom)


def build_reference(c):
    validate_config(c)
    path = Path(c.reference.path)
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    jobs = [('nl_coarse',c.sample.Nx,'nonlocal',c.reference.dt),
            ('nonlocal',c.sample.Nx,'nonlocal',c.reference.refined_dt),
            ('local_coarse',c.reference.local_grid,'local_spectral',c.reference.dt),
            ('local_time',c.reference.local_grid,'local_spectral',c.reference.refined_dt),
            ('local',c.reference.refined_local_grid,'local_spectral',c.reference.refined_dt)]
    results = {}
    for name,n,method,dt in jobs:
        print(f'Reference {name}: {n}x{n}, dt={dt}',flush=True)
        results[name] = solve_reference(c,n,method,dt)
    checks = {}
    for name,a,b in [('nonlocal_time',results['nl_coarse'],results['nonlocal']),
                     ('local_time',results['local_coarse'],results['local_time']),
                     ('local_space',results['local_time'],{k:transfer(results['local'][k],c.reference.local_grid) for k in ('phi','q','u','stress')})]:
        checks[name] = {}
        for field in ('phi','q','u','stress'):
            checks[name][field] = max(relative(aa,bb) for aa,bb in zip(a[field],b[field],strict=True))
        if max(checks[name].values())>c.reference.error_tolerance:
            raise ValueError(f'Reference refinement failed: {name}: {checks[name]}')
    output = dict(identity=json.dumps(identity(c),sort_keys=True),points=grid(c.sample.Nx,c.geometry.length),
                  time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots))
    for name in ('nonlocal','local'):
        d = results[name]
        for field in ('phi','q','u','stress'):
            output[field+'_'+name] = d[field] if name=='nonlocal' else transfer(d[field],c.sample.Nx)
        output['energy_'+name] = d['energy']
        mass = np.mean(d['phi'],axis=(1,2))
        checks[name+'_mass_drift'] = float(np.max(np.abs(mass-mass[0])))
        checks[name+'_max_energy_increase'] = float(max(0,np.max(np.diff(d['energy']))))
        if checks[name+'_mass_drift']>1e-10 or checks[name+'_max_energy_increase']>1e-8:
            raise ValueError(f'Mass/energy validation failed: {name}')
    # Control uses the same graph/time integrator, with no elastic coupling.
    uncoupled = type(c)(c.to_dict())
    uncoupled.physical.eigenstrain = (0.,0.,0.)
    control = solve_reference(uncoupled,c.sample.Nx,'nonlocal',c.reference.refined_dt)
    output['phi_uncoupled'] = control['phi']
    checks['elastic_effect_terminal_phi'] = relative(output['phi_nonlocal'][-1],control['phi'][-1])
    checks['nonlocal_vs_local_pooled_phi'] = relative(output['phi_nonlocal'],output['phi_local'])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**output)
    path.with_suffix('.json').write_text(json.dumps(dict(status='passed',checks=checks,
        identity=identity(c),settings=c.reference.to_dict(),
        method='independent NumPy acoustic tensor solve + SciPy Fourier IMEX-BDF2'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    return checks


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples.precipitate2d.configs.train_base')
    args = parser.parse_args()
    build_reference(importlib.import_module(args.config).get_config())
