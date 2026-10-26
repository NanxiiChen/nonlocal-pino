# Ice melting late-time accuracy exploration — 2026-10-24

> Follow-up qualification (2026-10-24): the user rejects the strong embedded
> motion prior. These numbers remain valid conditional on that prior; they do
> not demonstrate autonomous evolution learning. The production recommendation
> below is superseded by [completed initial-condition-only controls](icemelting-weak-prior.md),
> which remove the motion term and separately test learned evolution.
> The commands below reflect the former default; select
> `--config examples/icemelting3d/configs/accuracy_pilot.py` explicitly to
> reproduce these strong-prior controls after the runner default changes.


## Status and outcome

Completed exploration requested after the late-time diagnosis. A concrete,
reproducible improvement is supported by eight short GPU pilots: predict a
finite latent variable, embed the given initial condition and known forcing,
and minimise the analytically transformed **nonlocal** residual. Three seeds
at physical 5 s give full-grid relative L2 between 6.31e-5 and 1.02e-4, with
six-direction near-axis radius errors below 0.057% in absolute value.
These are errors against the same fixed spatial graph, not continuous-PDE
accuracy claims. Production train.py/losses.py/default config remain unchanged.

[Evidence and run provenance](icemelting-late-time-accuracy.json),
[figure](icemelting-late-time-accuracy.png),
[prior diagnosis](icemelting-late-time-diagnosis.md).

## Scope, acceptance and time interpretation

User asks to explore better late-time accuracy and expects physical 5 s to be
harder than the current 4 s. Existing training uses Tc=4 s, t* in [0,1]. Its
normalised endpoint is **1**, not 4. This pilot uses Tc=5 s, still t* in [0,1];
physical 4 s is t*=0.8. Coefficients are recomputed consistently:
N1=5e-5, N2=0.09876543209876544, N3=7.856742013183862, lambda*=0.25.
Physical mobility, forcing, initial radius and eps* are retained.

Acceptance met: matched direct 4/5 s controls, independently checked exact
reparameterisation, isolated short ablations, three-seed replication and
full-grid geometry/field/residual evaluation. No reference targets in training,
historical checkpoint resume, external artifact overwrite or Git history change.
True multi-slab time marching was considered but not experimentally validated;
the implemented temporal variant is explicitly an expanding-horizon curriculum.

The homogeneous ice-phase linear amplification is ~245920 by 4 s and ~5476371
by 5 s. The additional second contributes a factor ~22.27. These are local
linear estimates near phi=+1, not a global nonlinear error theorem.

## Exact change of variables and why it helps

Let d=sqrt(2)*eps*, phi=tanh(u), and let the declared graph have positive weights
w_ij, so Dphi_i=sum_j w_ij*(phi_i-phi_j). For finite latent values:

$$
r_\phi=(1-\phi_i^2)r_u,
$$

$$
r_u=(u_i)_t+N_1\sum_j w_{ij}
\frac{\sinh(u_i-u_j)\cosh(u_i)}{\cosh(u_j)}-N_2\tanh(u_i)+N_3.
$$

Derivation: phi_t=(1-phi²)u_t; phi³-phi=-phi*(1-phi²); and
(tanh(ui)-tanh(uj))/(1-tanh(ui)²)=sinh(ui-uj)*cosh(ui)/cosh(uj).
Thus the pointwise equation is unchanged. The optimisation objective changes:
MAE(r_u) is not MAE(r_phi); it retains constraints on bulk latent evolution
that the small factor 1-phi² suppresses in the original phase residual.

Compute cosh(ui)/cosh(uj) as
exp(logaddexp(ui,-ui)-logaddexp(uj,-uj)). This avoids subtracting saturated tanh
values or dividing by a rounded zero. No arbitrary denominator floor, residual
clipping, Eikonal penalty, imposed later radius or extra physical law is used.

The trial function is

$$
u_\theta(x,t^*)=\frac{R_0^*-|x-x_c|}{d}-N_3t^*+t^*N_\theta(x,t^*).
$$

This exactly embeds the given IC. The known constant latent forcing is
integrated explicitly as -N3*t*. Initialising the correction network's final
layer to zero gives the nominal planar motion. This is an **explicit prior**,
not evidence the network learned the solution: untrained error is measured
separately. Corrections depend on full x,y,z,t; radial symmetry or fixed
interface width is not enforced on the network.

For intuition only, a spatially uniform latent state obeys
u_t=N2*tanh(u)-N3, whose derivative with respect to u is N2*sech(u)², rather
than the large positive phi-space derivative near +1. This explains the
conditioning benefit in the bulk; it is not a stability proof for the full
graph-coupled system.

## Implemented experiment and reproduction

New isolated entry point: examples/icemelting3d/accuracy.py.
Explicit config: examples/icemelting3d/configs/accuracy_pilot.py.
Example README corrects the exact-analytic claim, explains the time scales,
derivation, changed loss objective and reproduction commands.

From the project root with its .venv:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m examples.icemelting3d.accuracy reference
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m examples.icemelting3d.accuracy check
.venv/bin/python -m examples.icemelting3d.accuracy train
```

Commands refuse to overwrite existing reference/method outputs. For another run,
copy the config, set a new experiment.output_dir, then pass --config PATH.
--method latent_balanced selects only that method. Seeds 7/123 used copied
configs with only seed, output directory and method list changed. Their
reference.npz is an identical independent copy of the seed42 reference, with
strict physical/grid identity checks and recorded SHA256.

Output roots (ignored reports/, all within this workspace):

- reports/icemelting-accuracy-20261024
- reports/icemelting-accuracy-20261024-seed7
- reports/icemelting-accuracy-20261024-seed123

Each method stores config.json, initial_metrics.json, metrics.json,
model_final.eqx, the experiment source and the complete shared Python package
plus example-config source snapshot. No old identity guard is bypassed.
Final validation additions only strengthen fail-fast checks; the actual training
source at run time is preserved per method.

Training: eight runs of 2000 SOAP updates, MLP 64x4 with Fourier embeddings,
highest matmul precision, float32 parameters, LR=1e-3 with the configured decay,
256 uniform grid core samples x16 uniform times per step. The implicit graph is
fixed; integer-index arithmetic gathers each sampled core's six neighbours with
face masks, fixed tensor shapes and precomputed scalar weights. Raw control
also uses 4096 fixed IC points with GradNorm. All methods use the same samples
within each seed; latent methods share identical zero-correction initial fields.
This batch/IC count and budget differ from the old 20000-step production run;
raw-control failure is not a fair converged-performance claim about that old run.

Evaluation streams all 128³ grid points at physical times 0,1,2,3,4,5 s in fixed
32768-point chunks. It reports original phase PDE residuals, reference-interface
MAE, full-grid relative L2, max phase error, positive-node volume and interpolated
near-axis radius/width. Additional checkpoint inference checks all six coordinate
directions at the same neighbour-centre transverse offsets. The geometry audit
uses the same highest matmul precision as training. A first default-precision
audit is excluded from the retained summary.

## Independent control and focused verification

Keep the original cubic graph: h=1/127, delta=0.01, six axial neighbours.
The second-moment coefficient ~0.68585 is deliberately retained, not silently
normalised to the local Laplacian. A direct 64³ even-coordinate octant reduction
of the full 128³ cube uses mirrored centre planes and missing-neighbour exterior
faces. Weights are recomputed in float64 from the declared kernel. RK4 with
normalised dt=0.001 and 0.0005 yields max phase difference 3.46e-9.
This is a **time-convergence check at fixed spatial discretisation**.

At 4/5 s, reference near-axis radii are 0.1502045834 / 0.1000490353.
At 5 s the reference nearest-centre phase is ~0.99168487, so it has not vanished.
The near-axis line is y=z=h/2 relative to the centre; its coordinates and
interpolation are identical for predictions and reference. Reported sub-grid
radius precision refers to this interpolation, not a proven continuum radius.

Focused checks:

- Residual equivalence on 100 independent random finite states: max abs 3.49e-13.
- Stable ratio at latent values -30,-20,0,20,30: max abs error 1.00e-15 against
  independent 80-digit Decimal arithmetic.
- Hard IC with a nonzero correction network: max abs error exactly zero.
- Manual difference stencil vs actual SparseD on every face/interior row of
  an 8³ complete unit cube: max abs error 3.41e-13 in float64.
- An initial small-patch comparison failed a strict absolute tolerance because
  SparseD infers dV from rounded patch endpoints. Replacing the test setup with
  a complete unit cube resolves that test artifact; no operator code changed.
- Original CPU float32 matvec on a constant field gives exactly zero on the
  checked patch; no evidence here supports blaming a constant-field roundoff
  source for the old drift.
- Full-grid acceptance checks completed for every pilot, plus seed repetitions,
  six-direction geometry and visual inspection of the summary figure.

## Results

Seed42, errors against the matched direct discrete reference:

| Method | 4 s relative L2 | 5 s relative L2 | 5 s +x near-axis radius error |
|---|---:|---:|---:|
| Untrained structured prior | 1.016e-3 | 4.248e-4 | -0.1892% |
| Hard IC + tanh, original phase residual | 2.459e-4 | 5.471e-4 | +0.2347% |
| Hard IC + tanh, transformed residual | 1.471e-5 | 6.307e-5 | -0.02522% |
| Transformed residual + expanding horizon | 2.147e-5 | 7.987e-5 | -0.02636% |

Three-seed replication:

| Objective | 4 s relative L2 range | 5 s relative L2 range | Largest absolute 5 s radius error across 18 direction/seed checks |
|---|---:|---:|---:|
| Original phase residual in hard-IC tanh model | 1.44e-4 to 2.46e-4 | 4.36e-4 to 5.47e-4 | 0.2347% |
| Exact transformed residual | 1.47e-5 to 3.88e-5 | 6.31e-5 to 1.02e-4 | 0.0563% |

Transformed 5 s max pointwise phase error is ~0.0024–0.00275, larger than the
spatial-average L2 number; error remains near the inner ice as well as the
interface. Reference-interface MAE is 0.000317–0.000534. Near-axis widths differ
by about 0.14–0.26%. Same-grid positive-node counts agree for the latent methods,
but zero count-based volume error is quantised, not exact continuous volume.

Raw soft-IC pilot ended in a spurious nearly all-ice field at late times:
terminal L2~2, positive fraction 1, original PDE MAE~0.011. Its IC MSE was
~4.6e-6. This demonstrates why residual/IC scalars alone cannot accept an
unconverged short pilot; it does not prove all sufficiently trained raw models
fail. Its result is preserved rather than silently omitted from the evidence.

## Recommendation and rejected overclaims

Prioritise an **opt-in hard-IC latent representation plus the exactly transformed
residual** for integration into production training. Merely applying tanh or
increasing IC weights is insufficient: all three original-phase latent pilots
worsen the terminal field error relative to the untrained prior. The transformed
objective improves over both that prior and the matched latent-phase control.

Expanding the time horizon has no demonstrated extra benefit for this case at
this budget; do not turn on causal weights indiscriminately. True time slabs
remain a possible follow-up for longer evolution or more difficult geometry.
If implemented, hand off finite latent values, preserving information that
rounded +/-1 phase arrays lose; never recover them by clipped arctanh without
an explicit numerical derivation. Existing CausalWeightor core/halo
normalisation still needs its own fix before a sparse causal production path.

After integration, use a matched full-budget run with the intended sampling and
several seeds, keep radius/interface metrics as acceptance criteria, and test
other geometries/initial conditions. The current trial function is tailored to
the stated tanh IC and constant forcing. No claim about general shapes or
arbitrary ICs follows from a centred-sphere benchmark. Separately address
spatial quadrature consistency and matched FEM BC/parameter convergence; this
work has not fixed that independent error source.

Primary literature provides context, not validation of this new forced-nonlocal
pilot: [causal PINNs](https://arxiv.org/abs/2203.07404),
[temporal sweeping/decomposition](https://arxiv.org/abs/2302.14227), and
[tanh-bounded phase-field networks](https://arxiv.org/abs/2405.06119).
The latter studies a different gradient-flow/minimising-movement formulation;
its energy-stability result is **not** asserted for this forced graph equation.

## Session changes, blocker and exact next action

Added isolated accuracy.py and accuracy_pilot.py; updated the example README;
updated docs/DECISIONS.md and docs/PROGRESS.md; archived this task, JSON evidence
and PNG figure. No production default or legacy artifact was changed. All
required focused numerical/training checks and diff checks are complete. No
unrelated full suite or FEM/MPI job was run.

No blocker for this exploration. Next implementation task: port the validated
latent field and stable graph residual into an opt-in production train/loss
path with explicit config and matched 5 s acceptance, then validate the intended
full-batch/subgraph training budget. Reuse the preserved controls and replay
metadata; choose a new output directory and do not resume historical runs.
