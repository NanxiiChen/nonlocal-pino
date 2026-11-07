# More corrosion within the normalized unit-time interval

Date: 2026-11-06. Status: complete; parameter proposal only.

## Goal and scope

The user asks how to adjust parameters to cover more corrosion area while
retaining t_max=1. Give a concrete parameter proposal with its physical meaning.
Check autonomous coupled-equation time scaling against the implemented RHS;
do not change the default or start training merely to answer this question.
No Git operation authorized this turn.

## Plan and acceptance

Verify uniform speed factor a applied once to n_ac1,n_ac2,n_ac3,n_ch with kappa
unchanged. Show that U_new(t)=U_old(a*t) for the fixed initial/boundary data,
while n_ac2/(2*n_ac3),interface thickness and reaction/transport balance stay
fixed. Distinguish this from raising kappa alone at fixed time units. Propose
a=2 as an untested next candidate, not a measured accuracy/area improvement.

Blocker: none. No pending computation or selected configuration change.

## Outcome

Recommend evaluating a common factor2 on n_ac1,n_ac2,n_ac3,n_ch while retaining
kappa2 and t_max1: coefficients48.6363636364,8,0.00444444444444,160. A milder
factor1.5 yields36.4772727273,6,0.00333333333333,120. These represent T0_new=a*T0
with the same material ratios; in the autonomous coupled PDE,
U_new(x,t*)=U_old(x,a*t*). Fixed initial/mouth/wall data preserve this identity.
Both AC energy ratios and phase-versus-transport balance are unchanged.
Only increasing kappa is instead a changed reaction-versus-transport regime;
it need not produce the same shape or a proportional area increase.

An in-memory check using the actual solver.make_system verified both factors:
RHS_new=a*RHS_old, Jacobian_new=a*Jacobian_old, identical initial arrays and
n_ac2/(2*n_ac3)=900. No integration, meshing, inference or training was performed.
This is an algebra/implementation check, not a new corrosion-area result.
References beyond the old horizon and a corresponding model would be required
to evaluate the proposed case; old t<=1 fields are not its full reference.

Default source, README and saved artifacts are unchanged. Only memory files
updated; documentation diff/link checks pass. No commit/push this turn.
