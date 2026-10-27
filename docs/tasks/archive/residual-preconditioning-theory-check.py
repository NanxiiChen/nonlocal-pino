"""Independent NumPy checks of local preconditioning identities; no PDE training."""
from pathlib import Path
import json
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/residual-preconditioning-theory';OUT.mkdir(parents=True,exist_ok=True)


def G(z):return np.array([np.tanh(z[0])+.1*z[0],z[1]])
def B(z):return np.diag([1-np.tanh(z[0])**2+.1,1.])
def W(z):return np.linalg.inv(B(z))
star=np.array([.4,-.7]);ystar=G(star);A=np.array([[2.,3.],[-1.,4.]])


def R(y):
    e=y-ystar
    return A@e+np.array([.2*e[0]**2,.1*(np.sin(e[1])-e[1])])


def DR(y):
    e=y-ystar
    return A+np.diag([.4*e[0],.1*(np.cos(e[1])-1)])


def S(z):return W(z)@R(G(z))


def derivative(f,x,h):
    return np.stack([(f(x+h*np.eye(len(x))[k])-f(x-h*np.eye(len(x))[k]))/(2*h) for k in range(len(x))],axis=1)


actual=derivative(S,star,1e-6);expected=W(star)@A@B(star)
np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=0)
z=np.array([.8,.1]);value=np.tanh(z[0]);gp=1-value**2+.1;gpp=-2*value*(1-value**2)
extra=np.zeros((2,2));extra[0,0]=-gpp/gp**2*R(G(z))[0]
full=W(z)@DR(G(z))@B(z)+extra
np.testing.assert_allclose(derivative(S,z,1e-6),full,atol=2e-9,rtol=0)
np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(expected)),np.sort_complex(np.linalg.eigvals(A)),atol=1e-14)
fixed=np.diag([3.,.2]);J=DR(G(z));r=R(G(z))
np.testing.assert_allclose(np.linalg.solve(fixed@J,-fixed@r),np.linalg.solve(J,-r),atol=1e-14)
# No promise that inverse-coordinate scaling improves a square-system condition.
non_normal=np.array([[1.,1000.],[0.,2.]])
good=np.diag([1000.,1.]);bad=np.diag([.001,1.])
conds=[float(np.linalg.cond(non_normal)),float(np.linalg.cond(np.linalg.solve(good,non_normal@good))),float(np.linalg.cond(np.linalg.solve(bad,non_normal@bad)))]
if not conds[1]<conds[0]<conds[2]:raise ValueError('Conditioning counterexample failed')
diagonal=np.diag([1.,1e6]);scale=np.diag([.001,3.])
np.testing.assert_array_equal(np.linalg.solve(scale,diagonal@scale),diagonal)
# Residual weighting changes a nonzero minimum for a restricted representation.
design=np.ones((2,1));target=np.array([1.,-1.]);row_scale=np.diag([10.,1.])
raw_minimum=float(np.linalg.lstsq(design,target,rcond=None)[0][0])
weighted_minimum=float(np.linalg.lstsq(row_scale@design,row_scale@target,rcond=None)[0][0])
np.testing.assert_allclose([raw_minimum,weighted_minimum],[0.,99/101],atol=1e-14)
# Target physical-state error metric for a well-posed invertible linearization.
e=np.array([.1,-.03]);np.testing.assert_allclose(np.linalg.solve(A,A@e),e,atol=1e-15)
result=dict(status='passed',root_W_A_B_derivative_max_error=float(np.max(np.abs(actual-expected))),
    away_root_full_weight_derivative_max_error=float(np.max(np.abs(derivative(S,z,1e-6)-full))),
    similarity_eigenvalues_unchanged=True,condition_numbers_original_improved_worsened=conds,
    diagonal_stiffness_unchanged=True,constant_left_scaling_exact_Newton_step_unchanged=True,
    restricted_least_squares_minimizers=dict(raw=raw_minimum,weighted=weighted_minimum),
    ideal_linearized_error_metric_identity=True,
    scope='Algebraic illustrations, not measured conditioning or new convergence claims for actual PINNs')
(OUT/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
