"""Whole-condition bootstrap of log-LOWESS means; never resample time points."""
from itertools import product
import numpy as np

from scripts.local_trend import local_linear_matrix


def trajectory_interval(steps, values, *, fraction, replicates, seed, grid_points):
    steps,values=np.asarray(steps,float),np.asarray(values,float)
    if (values.ndim!=2 or values.shape[0]!=len(steps) or values.shape[1]<2
            or not np.isfinite(values).all() or np.any(values<=0)
            or type(replicates) is not int or replicates<100 or grid_points<len(steps)):
        raise ValueError('Finite positive per-case histories and an explicit bootstrap budget are required')
    count=values.shape[1];grid=np.linspace(steps[0],steps[-1],grid_points)
    matrix=local_linear_matrix(steps,grid,fraction)
    if count<=4:
        indices=np.asarray(list(product(range(count),repeat=count)),dtype=np.int32)
        scheme='Exact enumeration of all equally likely ordered condition resamples'
    else:
        indices=np.random.default_rng(seed).integers(0,count,size=(replicates,count),dtype=np.int32)
        scheme='Fixed-seed bootstrap of complete condition trajectories'
    means=values[:,indices].mean(axis=-1).T
    raw=values.mean(axis=1)
    if count<=4:
        np.testing.assert_allclose(means.mean(axis=0),raw,rtol=2e-14,atol=2e-16)
        np.testing.assert_allclose(means.var(axis=0),values.var(axis=1)/count,rtol=2e-12,atol=2e-18)
    fitted=10**(np.log10(means)@matrix.T)
    lower,upper=np.quantile(fitted,[.025,.975],axis=0)
    smooth=10**(matrix@np.log10(raw))
    if not np.isfinite(fitted).all() or np.any(lower>upper):raise ValueError('Invalid bootstrap interval')
    return dict(x=grid,smooth=smooth,lower=lower,upper=upper,raw_steps=steps,raw_mean=raw,
        raw_case_errors=values,bootstrap_geometry_indices=indices,scheme=scheme)
