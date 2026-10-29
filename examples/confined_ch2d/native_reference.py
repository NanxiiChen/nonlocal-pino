"""Independent SciPy graph assembly/time integration; no training/JAX imports."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.integrate import solve_ivp
import time
from examples.confined_ch2d.problem import initial_field


def native_matrix(points,volumes,c):
    """Independent ordered-row assembly and geometric line obstruction check."""
    rows=[];cols=[];weights=[];delta=c.nl.delta
    tree=cKDTree(points)
    for i,x in enumerate(points):
        for j in tree.query_ball_point(x,delta):
            if i==j:continue
            direction=points[j]-x;r=np.linalg.norm(direction)
            unobstructed=True
            for cx,cy,radius in c.geometry.holes:
                projection=np.clip(np.dot(np.array([cx,cy])-x,direction)/(r*r),0,1)
                if np.sum((x+projection*direction-np.array([cx,cy]))**2)<radius*radius:
                    unobstructed=False;break
            if unobstructed:
                rows.append(i);cols.append(j)
                weights.append(volumes[j]*120/(np.pi*delta**4)*(1-r/delta)**2)
    off=sparse.coo_matrix((weights,(rows,cols)),shape=(len(points),len(points))).tocsr()
    A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
    if np.any(np.asarray(off.sum(axis=1)).ravel()<=0):raise ValueError('Native reference has isolated points')
    return A


def native(c,points,volumes,A,dt,rtol):
    p=c.physical;phi0=initial_field(points,c,xp=np)
    A2=A@A
    def rhs(t,phi):return -p.mobility*(A@(p.beta*(phi**3-phi))+p.epsilon**2*(A2@phi))
    def jac(t,phi):return -p.mobility*(A@sparse.diags(p.beta*(3*phi**2-1))+p.epsilon**2*A2)
    start=time.perf_counter()
    sol=solve_ivp(rhs,(0,p.terminal_time),phi0,method='BDF',jac=jac,
        rtol=rtol,atol=rtol*.01,max_step=dt,t_eval=np.linspace(0,p.terminal_time,c.reference.snapshots))
    if not sol.success:raise RuntimeError(sol.message)
    phi=sol.y.T;mu=p.beta*(phi**3-phi)+p.epsilon**2*(A@phi.T).T
    energies=np.sum(volumes*(.25*p.beta*(phi**2-1)**2+.5*p.epsilon**2*phi*(A@phi.T).T),axis=1)
    return dict(phi=phi,mu=mu,mass=phi@volumes,energy=energies,seconds=time.perf_counter()-start)
