"""Read-only structural checks of the current reciprocal corrosion reference graph."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def main():
    path=Path('/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz')
    with np.load(path) as ref:
        row,col=(ref[k] for k in ('graph_row','graph_col'))
        w=ref['graph_weight'].astype(np.float64)
        x=ref['X'].ravel();n=x.size
    if np.any(w<0) or np.any(row==col):raise ValueError('Positive pair graph required')
    W=sparse.csr_matrix((w,(row,col)),shape=(n,n))
    asym=W-W.T
    asymmax=float(np.max(np.abs(asym.data))) if asym.nnz else 0.
    if asymmax!=0:raise ValueError('Graph must be exactly reciprocal')
    degree=np.bincount(row,weights=w,minlength=n)
    D=sparse.diags(degree)-W
    u=np.random.default_rng(42).standard_normal(n)
    du=D@u
    pair=.5*np.sum(w*(u[row]-u[col])**2)
    quadratic=float(u@du)
    np.testing.assert_allclose(quadratic,pair,rtol=2e-14,atol=0)
    flux_relative=float(abs(du.sum())/np.sum(np.abs(du)))
    assert flux_relative<1e-14
    np.testing.assert_allclose(D@np.ones(n),0,rtol=0,atol=1e-8)
    perturb=np.random.default_rng(7).uniform(-1,1,n)
    bound=2*float(degree.max())*float(np.max(np.abs(perturb)))
    amplification=float(np.max(np.abs(D@perturb)))
    assert amplification<=bound
    jump=(x>.5).astype(float)
    assert np.isfinite(D@jump).all()
    stored_defect=degree.astype(np.float32).astype(np.float64)-degree
    result={'status':'passed','reference':str(path),'nodes':n,'directed_edges':len(w),
            'weight_min':float(w.min()),'weight_asymmetry_max':asymmax,
            'constant_action_max':float(np.max(np.abs(D@np.ones(n)))),
            'quadratic_form':quadratic,'pair_energy':float(pair),
            'energy_identity_relative_error':float(abs(quadratic-pair)/pair),
            'full_graph_internal_flux_relative_cancellation_error':flux_relative,
            'positive_kernel_infinity_bound':bound,'measured_perturbation_action_max':amplification,
            'discontinuous_step_action_finite':True,
            'stored_FP32_diagonal_row_sum_defect_max':float(np.max(np.abs(stored_defect))),
            'limits':'Exact-edge/reference graph checks; current AC stored-diagonal roundoff, Dirichlet reservoirs and soft PINN residuals prevent a claim of exact full-network conservation/dissipation.'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
