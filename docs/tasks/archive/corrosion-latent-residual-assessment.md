# Corrosion latent residual and cancellable reactions

## Goal and acceptance criteria

Classify reaction terms that remain bounded under inverse-Jacobian residual
scaling; assess the repository's corrosion1d/2d models in their original 0-1
phase range. Exclude hard initial conditions from the theory. Trace equations
to code and references, derive local and discrete-graph forms, and independently
check their algebra. Distinguish applicability from unmeasured L2 improvements.

## Context

The user asks whether the melting residual transformation generalizes to
corrosion, particularly whether the reaction/chemical driving terms cancel
the denominator. Both corrosion losses use h=-2*phi^3+3*phi^2 and
g'=4*phi^3-6*phi^2+2*phi. Existing validated melting defaults remain unchanged.

## Result: sigmoid inherits the tanh identities

No hard IC is assumed. The neural latent field is unrestricted by initial data
in this assessment. The existing corrosion output map is already

$$
\phi=m(z)=\frac{1+\tanh z}{2}=\sigma(2z),\qquad
\sigma(u)=\frac{1}{1+e^{-u}}=\frac{1+\tanh(u/2)}{2}.
$$

Thus sigma'(u)=phi*(1-phi), while m'(z)=2*phi*(1-phi). The change between these
coordinates is u=2z. Existing soft IC/BC losses are separate from this identity.
The shared `src/nonlocal_pinn/nn/constraints.py:kks_constraint` also constrains
c through the KKS mixture rule. Do not confuse this output constraint with
hard initial conditions.

## Cancellable reaction classes

For sigmoid coordinates, a total reaction B(phi,fields) remains bounded after
division if B/[phi*(1-phi)] has a bounded extension to both endpoints, for the
physical coupled-field range. A useful form is B=phi*(1-phi)*Q, with Q bounded.
Polynomial reactions with roots0 and1 factor this way. For a C1 reaction on the
closed phase interval, endpoint zeros imply a bounded quotient (with uniform
derivative control over coupled fields). Merely continuous endpoint zeros do
not suffice: sqrt(phi*(1-phi)) divided by phi*(1-phi) diverges. More generally,
B=phi^alpha*(1-phi)^beta*Q with nonzero bounded endpoint Q has a bounded ratio
when alpha,beta>=1. For tanh coordinates on -1..1 the corresponding factor is
1-phi^2. This is a criterion for the total reaction; requiring every individual
term to factor is a convenient sufficient condition, not logically necessary.

Common examples, with J=phi*(1-phi):

| Term | Analytic term divided by J |
| --- | --- |
| g'(phi), g=phi^2*(1-phi)^2 | 2*(1-2phi) |
| h3'(phi), h3=3phi^2-2phi^3 | 6 |
| h5'(phi), h5=6phi^5-15phi^4+10phi^3 | 30*J |
| h3'(phi)*bounded driving field | 6*driving field |
| J^k*Q, k>=1 | J^(k-1)*Q |
| Constant nonzero forcing | forcing/J, unbounded |
| Forcing*(1-phi) | forcing/phi, unbounded at0 |

Smooth double-well potentials with stationary pure phases and interpolation
functions with zero endpoint slopes naturally generate compatible reactions.
The assertion requires checking the actual constitutive terms, not replacing
an incompatible physical source to force cancellation.

## Trace to the implemented corrosion model

Both `examples/corrosion1d/losses.py` and `examples/corrosion2d/losses.py` use

$$
h=3\phi^2-2\phi^3,\quad h'=6\phi(1-\phi),\quad
g'=2\phi(1-\phi)(1-2\phi),\quad
q=c-h(\phi)\Delta c_0-c_{Le}.
$$

Their unscaled phase residual is

$$
r_\phi=\phi_t+N_3D_\delta\phi+N_2g'(\phi)-N_1q\Delta c_0h'(\phi).
$$

Here N1,N2,N3 denote the code's N_AC1,N_AC2,N_AC3; signs follow D_delta~minus
Laplacian. In the existing z coordinate, division by m'(z)=2J gives

$$
r_z=z_t+N_3\frac{D_\delta\phi}{2\phi(1-\phi)}
+N_2(1-2\phi)-3N_1q\Delta c_0,
\qquad r_\phi=2\phi(1-\phi)r_z.
$$

Both reaction contributions have no phase denominator. In sigmoid u=2z
coordinates, the corresponding residual is r_u=2*r_z; its reaction coefficients
are2*N2 and-6*N1. The physical field/PDE is the same.

The local form (constant N3) is

$$
r_z=z_t-N_3\Delta z+2N_3(2\phi-1)|\nabla z|^2
+N_2(1-2\phi)-3N_1q\Delta c_0.
$$

For the current KKS constraint c=c_L+h(phi)*dc0, hence q=c_L-c_Le. It is bounded
by the concentration map; this gives no hidden phi denominator. Preserve this
mixture rule during any future ablation.

## Nonlocal diffusion: same identity as melting

For D_delta(phi)_i=sum_j w_ij*(phi_i-phi_j), w_ij>=0 and zero row sums,

$$
\frac{\phi_i-\phi_j}{2\phi_i(1-\phi_i)}
=\frac{\tanh z_i-\tanh z_j}{\operatorname{sech}^2z_i}
=\sinh(z_i-z_j)\frac{\cosh z_i}{\cosh z_j}.
$$

This is exactly the melting implementation's `_tanh_difference_ratio`.
Log-cosh arithmetic avoids saturated subtraction and division; large neighbour
latent differences can still produce large ratios. The continuum chain rule
is not a replacement for transforming the actual discrete graph operator.

## Coupling, constraints and interpretation

The concentration residual remains r_c=c_t+N_CH*D_delta(q). Do not divide it by
the phase Jacobian. Keeping the original c equation with the transformed phase
equation preserves their exact common zeros, but finite training errors need
not preserve mass/energy balance. Existing Dirichlet phi0/1 values cannot be
represented exactly by finite z; physical BC losses can approach them without
taking atanh of endpoint data. No IC/BC hard constraint is introduced here.

Cancellation removes an endpoint denominator from the reaction, not the large
physical coefficients, phase-concentration stiffness or graph ratios. For the
diffusion config, N_AC1=1.07e10, N_AC2=1.76e9 and N_AC3=1.03e6 before existing
ac_scale=1e-7. Phase/concentration loss balance must be checked separately.

For a future comparison, preserve the same current output map, soft IC/BC
handling and concentration residual and change only the phase residual. Note
that z/u residuals differ by a constant and couple to other loss terms: a
clearly declared normalization is needed. An interface-normalized option is
r_phi/[4*phi*(1-phi)]=r_z/2; its weight equals1 at phi=1/2. This is a proposed
loss normalization, not a change to the physical equation or a tested choice.
Assess both phi/c full-field relative L2, interface/pit position, boundary
errors and flux balances against a suitable independent reference. No
corrosion training gain has been measured in this task.

## Primary reference checked

Cui, Ma and Martinez-Paneda, *A phase field formulation for dissolution-driven
stress corrosion cracking*, JMPS147,104254(2021), arXiv:2011.12068:
https://arxiv.org/html/2011.12068 . Equations18,28,32,34 and38 give the matching
interpolation, KKS chemical energy, phase reaction and concentration diffusion.
These support model provenance; the residual scaling above is our derivation.

## Verification and completion

Complete; no blocker. CPU AD checks used the existing KKS output wrapper and
actual1D/2D production loss functions, without hard ICs. The local chain rule
passed. Graph phase identities passed for DenseD/SparseD float64 with maximum
relative discrepancy1.7423e-15 and ConvD float32 with maximum9.5005e-7. Verified
the concentration residual is unchanged. Direct230-digit Decimal arithmetic
checked saturated graph ratios at latent magnitudes40/100 (including opposite
signs), max relative discrepancy4.20e-15. The first ConvD check with float64
inputs encountered its fixed float32 kernel; verification was rerun explicitly
in float32 without changing production code. These are algebra checks, not
PDE convergence or PINN-training experiments.

Changed memory files: this archived task, DECISIONS and PROGRESS. No production source,
default, checkpoint, external artifact or dependency changes. Diff check passed.
Next action: none required for this assessment; a controlled corrosion training
comparison remains future work.
