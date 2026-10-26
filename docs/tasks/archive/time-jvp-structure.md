# Time-JVP structure clarification — completed 2026-10-26

## Goal and conclusion

Verify current space/time sampling and explain time JVP versus pointwise grad.
Within an update, all16 times share the same selected spatial subgraph (including
halo); updates cycle precomputed pool entries. Full-grid evaluation shares the
full grid across all times. Current corrosion matched residuals already compute
one batched raw-network time JVP, then a constraint-map JVP. Electropolish and
melting nonlocal paths already use time JVP as well. No production change.

## Identity and performance limits

For fixed spatial points, F(T)[j,i,a]=f_a(x_i,T[j]). Its time Jacobian has zero
cross-time blocks. Therefore J_F(T) times an all-ones vector equals the complete
array of pointwise partial time derivatives. Time-independent evaluation of each
point is the essential condition; identical spatial coordinates across times
are convenient but not required for this AD identity. Cross-time coupling would
need a new derivation. JVP avoids materializing the full time Jacobian and can
return values plus time derivatives together. It still processes Nx*Nt outputs;
one API call is not constant work independent of batch size.

Python point loops can be much slower. Properly jitted vmapped pointwise grad
also batches all points: no Nx*Nt-factor speedup follows against that baseline.
One scalar time direction produces both phi_t and c_t through forward AD;
reverse AD field-by-field can need separate output seeds. For scalar melting,
the distinction need not produce a large speedup. Loss gradients with respect
to all parameters still use reverse AD (current shared step uses filter_vjp);
SOAP, GradNorm and graph operations remain training costs. No timing measured.

## Evidence and verification

Source: examples/corrosion2d/residual_experiment.py:616 (shared x per update),
examples/corrosion2d/residuals.py:30 (raw and constraint JVP),
examples/icemelting3d/losses.py:38 and examples/electropolish2d/losses.py:80.

Independent CPU float64 toy f(x,t)=[sin(x0+t*x1), exp(-t)*(x0**2+x1)] on
3 spatial points and4 times: batched all-ones JVP exactly matches vmapped
per-field time grad (max error0); also matches pointwise jacfwd and full time
Jacobian sum. All cross-time Jacobian blocks exactly zero. Fields/JVP shape
(4,3,2), full Jacobian shape(4,3,2,4). Evidence: time-jvp-structure.json.
No numerical-core edits, training or performance claim. git diff --check passes.

## Status and next action

Complete; no blocker or remaining requested work. An actual synchronized
benchmark of vmapped grad versus JVP and parameter-gradient steps would be a
separate performance investigation, not an inferred speedup here.
