"""Optional known initial tangent field for the point-function encoder.

For A=-Delta_delta and s=t/T, dphi/ds at0 is
-T*M*A[beta*(phi0**3-phi0)+epsilon**2*A*phi0]. This is computed once from
prescribed inputs; it is neither a future label nor an output constraint.
"""
import numpy as np


def initial_rhs_enabled(c):
    return 'initial_rhs_condition' in c.operator and c.operator.initial_rhs_condition


def initial_operator_fields(data,c):
    """Return (N,2) columns phi0 and its horizon-scaled physical initial RHS."""
    initial=np.asarray(data['initial']);neighbors=np.asarray(data['neighbors'])
    coefficients=np.asarray(data['coefficients'])
    if initial.ndim!=1 or len(initial)==0:raise ValueError('Initial RHS encoding requires a nonempty initial vector')
    n=len(initial)
    if (neighbors.ndim!=2 or neighbors.shape[0]!=n
            or coefficients.shape!=neighbors.shape or not np.issubdtype(neighbors.dtype,np.integer)
            or np.any(neighbors<0) or np.any(neighbors>=n) or np.any(coefficients<0)
            or not np.isfinite(initial).all() or not np.isfinite(coefficients).all()):
        raise ValueError('Initial RHS encoding requires aligned finite initial/graph arrays')
    def action(values):
        return np.sum(coefficients*(values[:,None]-values[neighbors]),axis=1)
    p=c.physical
    mu0=p.beta*(initial**3-initial)+p.epsilon**2*action(initial)
    tangent=-p.terminal_time*p.mobility*action(mu0)
    result=np.column_stack((initial,tangent))
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite known initial tangent field')
    return result
