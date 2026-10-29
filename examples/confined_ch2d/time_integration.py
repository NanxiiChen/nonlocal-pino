"""Global polynomial quadrature in a logarithmic time coordinate.

This constructs a Volterra integration matrix, not a time-stepping solution.
All field values are outputs of one simultaneously trained global network.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss,legvander
from scipy.linalg import solve


def integration_nodes(*,count,duration,time_scale):
    if isinstance(count,bool) or not isinstance(count,int) or count<4:
        raise ValueError('Global integration requires integer count>=4')
    if not np.isfinite([duration,time_scale]).all() or min(duration,time_scale)<=0:
        raise ValueError('Global integration requires positive duration/time_scale')
    s=.5*(1-np.cos(np.arange(count+1)*np.pi/count))
    times=time_scale*np.expm1(np.log1p(duration/time_scale)*s)
    times[0]=0.;times[-1]=duration
    return times


def integration_matrix(times,*,time_scale):
    times=np.asarray(times,dtype=np.float64)
    if times.ndim!=1 or len(times)<5 or times[0]!=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Ordered global integration nodes must start at zero')
    if not np.isfinite(time_scale) or time_scale<=0:raise ValueError('Positive integration time_scale required')
    L=np.log1p(times[-1]/time_scale);u=2*np.log1p(times/time_scale)/L-1
    V=legvander(u,len(times)-1)
    if np.linalg.cond(V)>1e6:raise ValueError('Ill-conditioned global interpolation nodes')
    q,w=leggauss(2*len(times));rows=[np.zeros(len(times))]
    for endpoint in u[1:]:
        mapped=-1+.5*(endpoint+1)*(q+1)
        physical_weights=w*.5*(endpoint+1)*.5*L*time_scale*np.exp(.5*L*(mapped+1))
        moments=physical_weights@legvander(mapped,len(times)-1)
        rows.append(solve(V.T,moments,assume_a='gen',check_finite=True))
    W=np.stack(rows)
    np.testing.assert_allclose(W.sum(axis=1),times,rtol=1e-12,atol=1e-14)
    if np.any(W[-1]<=0):raise ValueError('Global physical-time quadrature requires positive endpoint weights')
    return W
