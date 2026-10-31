# Development Guide

This document contains repository conventions that are broader than one
example but more detailed than the mandatory policy in `AGENTS.md`.

## Example Layout

Every example under `examples/<name>/` should use:

```text
examples/<name>/
├── README.md                  # physics, BCs, formulation, usage
├── train.py                   # grid, network, operator, training loop
├── losses.py                  # residuals, BC losses, loss factory
├── solver.py                  # independent FEM/reference solver if applicable
├── mesh.geo                   # geometry if applicable
└── configs/
    ├── train_base.py          # PINN configuration
    └── solver_base.py         # reference-solver configuration
```

Keep reference solvers independent from PINN operators when they are intended
to measure model or discretisation error.

## APIs And Naming

Use the shared loss-factory shape:

```python
def make_loss_fn(x_all, D, ...physical_params...,
                 ...bc_data..., alpha=0.9, gradnorm_cap=1000.0):
    """Return loss_fn(net, w_ema, bias, update_w, x_points=None)."""
```

- `loss_fn` receives `x_points` explicitly, never `batch_idx`.
- Capture fixed BC data in the factory; pass changing collocation data per step.
- `D is None` means local AD; `D is not None` means nonlocal mode.
- Hard constraints handle Dirichlet data; do not duplicate it as a loss term.
- Use `make_loss_fn`, `_make_step`, `_build_grid`, `_build_network`,
  `_build_D`, `_build_optimizer`, and `_eval_fem` where those roles apply.
- Use names such as `c.arch.hidden_dim`, `c.nl.delta`, `c.sample.Nx`, and
  `c.train.lr_init`.

## Configuration

- Use `ml_collections.ConfigDict`.
- Put material data in `c.physical`, geometry in `c.geometry`, nonlocal data in
  `c.nl`, architecture in `c.arch`, and training data in `c.train`.
- Declare every required field explicitly. Do not use `.get()` fallbacks or
  hidden function-argument defaults for required settings.
- Validate physical ranges, array shapes, grid ordering, operator assumptions,
  and required input files at startup.
- Fail loudly with context. Never catch an arbitrary exception and silently
  switch to another numerical method.

Confined CH optionally enables `operator.online` for a bounded continuously
refreshed geometry pool. Its explicit fields are workers, refresh_size,
refresh_every, prefetch_batches, max_points, max_neighbors and origin_step.
All quadrature/graph/stable-metric construction stays outside parameter AD;
FIFO batches at fixed global counters determine replacement, not completion
timing. Separate immutable evaluation payloads from training slots, preserve
fixed shape/dtype signatures, donate replaced pool buffers and fail on overflow.
New inputs carry exactly eight known-input arrays and no evolved labels.
For fresh online training explicitly set origin_step=0 and omit
train.continuation. Positive origins require continuation; imported online
states retain their original origin and global refresh counters. Save both fresh
initial model and SOAP state, assert zero optimizer counters, and check the first
25 actual production updates in the same process. The width256 cold50k preset
uses checkpoint_every=5000 to bound snapshot storage without deleting snapshots.

`train.continuation.checkpoint` extends the strict complete-parent import with
an explicit absolute paired intermediate marker inside parent_run. Validate the
marker/model/optimizer/config/global step and one complete parent evaluation;
allow only the explicitly declared online-input addition, not changed physics,
architecture or optimizer settings. Preserve full SOAP/Optax schedule state and
check serialized initial bytes plus physical-field metrics. Online checkpoint
markers additionally name a bounded pool-state file; reconstruct input identity
before consuming new global RNG draws. Existing complete-parent imports and
old marker formats remain valid. Never edit parent artifacts or synthesize a
completed summary for an interrupted run.

Canonical phase-field training starts in each example's `train.py`, usually
with `configs/train_base.py`. Confined CH explicitly defaults to
`configs/train_coarsening_windows.py` (ten-wave/T1.5/30windows); its train_base
remains the legacy simple foundation so dependent research presets do not
change. Its reference solver defaults to the same complex physical config.
Corrosion/electropolish matched configs dispatch to
the shared KKS engine through `training_entry.py`; legacy configs explicitly
retain their original loop. Pure IC/BC samplers live in `training_data.py`,
so the shared engine never imports CLI modules or registers duplicate flags.
Canonical strategy configs omit `train.seed`; the runtime `--seed` is resolved
by `training.runtime.with_run_seed` into a copied config for serialized run
provenance. The common run default42 is not tuned per case; existing serialized
seed values remain loadable. Keep default selection criteria, seed limitations and direct checkpoint paths
in the example README and `artifacts/index.yaml`.

Matched KKS configs declare `train.compute_dtype` explicitly: canonical corrosion
uses `float64`, electropolish `float32`, selected from full ordinary-policy controls.
The shared trainer generates common FP32 initialization,
geometry/operator storage, IC/BC and temporal samples before promoting floating
arrays through `corrosion2d/training_precision.py`. Preserve these common inputs
when comparing arithmetic precision; rebuilding graph weights/diagonals or using
native FP64 random draws is a separate control. Each run writes `precision.json`.
Historical checkpoint templates must match their saved parameter dtype; load an
FP64 checkpoint into a functionally promoted template, with JAX x64 enabled.
Old FP32 checkpoints remain unchanged and load through their original templates.

Corrosion research presets `train_local_tensor.py` and `train_local_joint.py`
retain the canonical `train.py` entry, dispatching to `local_training.py` with
AD spatial Hessians and a separate refined local reference. They add a sixth
GradNorm loss for natural bottom no-flux conditions, while graph training keeps
its five losses. `train_graph_ordinary.py` explicitly selects ordinary GPU
FP32 compilation as a historical control; local presets explicitly retain FP32
and deterministic policy. Canonical corrosion/electropolish/melting now explicitly
select `train.gpu_policy=ordinary`. The shared `training.runtime.configure_gpu_policy`
runs before backend initialization, preserves unrelated flags and rejects conflicting
inherited settings. Deterministic policy remains an explicit override. The full
selection evidence and limitations are in the example READMEs and archived
ordinary-policy-defaults task. No physical operator was changed by this promotion.

The `twophase2d` pilot has a separate mixed advective-CH loop and case-local
periodic moment-calibrated operator. It prescribes velocity and solves no NS.
Its constant-flow base and explicit train_vortex reversed-vortex preset share
the canonical train.py entry. Velocity accepts a historical two-vector or an
explicit reversed_vortex record; serialized physical identity is preserved.
Its CPU Fourier reference imports no JAX/PINN operator; local and finite-horizon
reference errors are kept separate. It does not alter established case defaults.
Explicit train_vortex_residual/train_vortex_latent presets test bounded physical-
field weights and exact inverse-map Jacobian norms separately. The optional
phase/residual records form documented config extensions; missing records retain
the historical additive/original schema. Current two-phase base/conserved
presets explicitly set phase.map=additive. Opt in to the cubic coordinate
through train.py with --config.phase.map=inverse_cubic; no separate enable
boolean is needed. Historical cubic-only causal/initial-time/long-refinement
controls set that map explicitly. Load old models with their saved config,
since the recommended conserved preset now defaults to additive.
The leaky-tanh map is unbounded and
preserves phase IC through differences, with independently checked fixed Newton
inverse. New static network fields add no serialized parameter leaves; direct
legacy checkpoint replay is checked. Do not identify a map-only gain as evidence
for residual scaling or enforce strict tanh bounds on overshooting CH solutions.

The two-phase accuracy controls additionally support an analytic inverse-cubic
phase coordinate, frozen spatial/time-coupled CH residual metrics and explicit
FP64 PDE-only L-BFGS refinement. Research presets use conservative spectral
flux advection and optional exact full-grid mass projection through
`twophase2d.fields.predict_fields`; all physical training/evaluation/replay paths
must use this wrapper. Raw point network calls are provisional under projection.
Hard-mass presets omit `loss.mass_weight` and the loss factory computes/returns
only the phase and chemical terms; unprojected presets retain the explicit
three-term PDE/soft-mass schema. An obsolete weight in an old serialized hard
config is ignored, preserving checkpoint loading. Historical logs keep their
three components; new hard-mass logs contain two, fixed within each JIT path.
The explicit train_vortex_conserved_local control changes only diffusion to
local AD Hessian traces; spectral flux and the projected-field time JVP remain
identical. Its point-network Hessian is valid because the grid mass shift is
constant in a spatial query. Compare fields against one common reference and
keep native targets separate.

The melting local FEM solver is rewritten for the random-ellipsoid operator:
`examples/icemelting3d/solver.py` requires an explicit dotted config module,
held-out input index, mesh subdivisions, dt and output path. Its config reads
the saved training JSON and native manifest for physical/IC identity, not
future labels. Epsilon stays fixed under mesh refinement. Consistent CG2
tetrahedral mass, degree8 reaction quadrature and natural Neumann boundaries
are independent of the neural/nonlocal residual. Explicit time.scheme selects
constant-matrix IMEX BDF2 or the nonlinear implicit-midpoint control. Single
resolution files are unvalidated until independent spatial/time gates pass.
Do not reuse the removed mesh-dependent-epsilon/sphere-only FEM assumptions
or interpret a native graph target as local-PDE validation. NumPy ellipsoid
geometry imports no JAX; JAX imports are local to JAX-only geometry helpers.
The explicit train_vortex_nonlocal_flux control changes conservative first
derivatives to a moment-corrected finite-horizon nodal pair. It retains the
independent pair Laplacian, direct output, IC/mass and training budget. Its
native reference has independent sine symbols and a distinct advection identity;
old identities remain unchanged. The common local target uses spectral
derivatives. Reject the unsupported nonlocal-flux/local-diffusion combination
because no matching native reference is supplied. Centered first derivatives
have checkerboard null modes, so this is a prescribed-flow transport control,
not a pressure-stable NS operator. Time JVP and parameter AD remain.
Integer spatial feature frequencies have stopped parameter gradients; older
checkpoints with drifted frequencies remain loadable but are not hard-periodic.
Do not silently round historical modes or identify changes in spatial
discretization/constraints/optimizer budget as scalar-preconditioner gains.

The `confined_ch2d` example is a separate Neumann geometry pilot. It uses
physical triangle-centroid volumes on ellipse-minus-holes geometry, symmetric
visible regional pairs, direct phase output and weighted hard mass. Local AD
has explicit phi/mu normal BC losses; native nonlocal has two mixed PDE losses
only. Geometry preprocessing runs Gmsh in a subprocess to isolate MPI/PETSc
libraries from DOLFIN. Independent mixed-P2 FEM and independently assembled
SciPy native references must pass refinement before training. Do not infer
pointwise local Neumann enforcement from a finite-horizon pair closure, reuse
periodic FFT neighborhoods or estimate perforated area from its bounding box.
Its explicit `train_coarsening.py` preset adds `initial.kind=multimode`, with
finite integer `(kx,ky,amplitude,phase)` waves and a positive saturation scale.
This tanh defines prescribed physical initial data, not a learned phase map.
The original schema omits `initial.kind`; its baseline arithmetic is retained.
New static IC metadata adds no checkpoint parameter leaves; old runs must still
be replayed with their saved configuration. Both NumPy analytic IC Laplacians
are checked against AD. Sharper-case references use a separate data child path;
preflight files alone never authorize training.

Confined CH optional `phase` selects `lifting` and required positive
`initial_time_scale`: `exponential_blend`, `logarithmic_additive` or
`logarithmic_blend`; absence preserves legacy additive arithmetic. Optional
`formulation.chemical=eliminated` uses one phase output and derives mu from the
native operator; `predict_fields(..., D)` is then required. Learned two-head
templates remain compatible with saved legacy configs. Eliminated mode omits
`loss.chemical_weight` and rejects local AD. Optional `sample.temporal` explicitly
declares log-uniform density and horizon curriculum; no inverse-density
compensation is implied. Optional refinement `time_nodes` are physical positive
nodes; omission preserves the shared helper's legacy uniform-midpoint rule.
`loss.phase_metric` configures a frozen SPD biharmonic resolvent, not a PDE change.

The default richer-case `train_coarsening_windows.py` adds explicit
`train.windows` budgets, log boundaries, warm start, optimizer reset and native
resolvent. Each window learns a phase increment from a frozen previous PINN
endpoint; eliminated chemical law and physical-volume mass hold exactly.
State, width and Cholesky factors are fixed-shape dynamic JIT inputs. Total
SOAP/L-BFGS budgets must equal the configured sums. Only canonical train.py
dispatches this option. With an empty window run name, the canonical entry copies
the runtime config and assigns a unique timestamp/nonlocal-windows name before
dispatch; explicit names are preserved. Each stored network plus grid initial state is one slab;
load the complete direct-path `trajectory.json` through evaluate.py. Reject
partial manifests and check every initial-state link. Fixed quadrature states
have no off-grid extension, so local AD windows/classical wall diagnostics are
not available. Report derivative jumps and mu accuracy alongside phase L2.
The growing-prefix stage metric is not a full-domain convergence history.

Retained global-time research uses one checkpoint, not that window engine.
Optional `arch.model=global_basis` selects one spatial MLP whose `rank` outputs
are coefficients of globally supported Chebyshev time functions. It requires
`phase.lifting=logarithmic_additive`, positive `arch.output_scale` and exact
chemical elimination. `arch.initial_feature` explicitly adds the known IC as
a scalar spatial feature; missing field preserves legacy input dimensions.
The joint MLP also supports `phase.lifting=logarithmic_increment`, which retains
the IC plus a saturating free increment. No future trajectory is prescribed.

Optional `sample.temporal.kind=fixed_nodes` requires explicit ordered positive
`nodes` ending at T. For `phase_metric.kind=integral_resolvent`, SOAP/refinement
nodes must agree exactly; the factory rejects mismatched runtime nodes.
`time_integration.py` precomputes global log-time polynomial integration with
checked moments/positive physical weights. Optional `normalization=time_average`
means R/t; absence preserves the archived unnormalized law. Rank-r phase needs
at least3r intervals for cubic chemical interpolation. Optional nonnegative
`loss.energy_weight` adds normalized energy-dissipation balance and needs6r
intervals for a global basis. Original differential metrics include
`space_time_resolvent` (one differentiable residual recurrence, no state stepping)
and `neumann_inverse` (constant-nullspace lift, no numerical jitter).

Optional `train.refinement.optimizer=newton_cg` dispatches the case-local
matrix-free exact-Hessian helper; explicit settings are iterations, time_nodes,
time_points, step_tolerance, c1 and c2. Count actual Hessian products/function
calls. Missing optimizer field retains the shared L-BFGS helper unchanged.
All examples still enter through train.py and load saved configs/direct paths.
These global schemes are experimental; lower phase L2 alone does not justify
promotion when mu, overshoots or energy are poor.

Optional `loss.causal` declares positive `strength`, `minimum` in(0,1), and
`refinement=uniform`. Only the fixed-node eliminated Neumann-inverse control
currently supports it. SOAP uses stopped exponential prefix residual weights,
floored and normalized by their sum. This update intentionally does not
differentiate through the weights. The L-BFGS factory receives a copied config
with this record removed, so line searches use a consistent fixed scalar and
gradient. Saved refinement metadata states that weights are uniform. It is
one global network, with every time retaining positive weight; it neither
freezes intermediate fields nor dispatches the window engine. Do not conflate
this weighting variant with all details/benchmarks of the cited causal-PINN paper.

## Artifact Paths

The `precipitate2d` periodic pilot learns phase/chemical fields and eliminates
homogeneous-isotropic elastic equilibrium with a differentiable Fourier solve.
Mechanical displacement is solved, not an independent network output. Its
nonlocal phase Laplacian and L-BFGS helper explicitly reuse twophase2d modules;
source snapshots include those files. `local_spectral` is a Fourier diffusion
control, distinct from the explicit `local_ad` point-network Hessian control.
The latter differentiates both mixed fields, retains full-grid projection/time
JVP and has no exact finite-grid summation-by-parts guarantee. Both controls
retain the identical spectral mechanical subproblem and refined local target.
Odd square grids avoid Nyquist derivative null modes.
Full-grid mean projection is mandatory through `network.predict_fields` for
training/evaluation/checkpoint replay. Its independent SciPy solver assembles
acoustic matrices without importing training mechanics/operators, validates
all-field grid/time refinement and retains native/local targets separately.
This fixed-zero-macrostrain, linear-eigenstrain, equal-modulus model is a
two-way coupled simplified pilot rather than a PFHub4 reproduction.

Load models, configs and reference arrays by explicit file path. Do not introduce
file-hash computation, expected-checksum gates or hash-manifest dependencies.
Record paths and source snapshots when useful; relative checkpoint paths are
resolved from their run directory so directory retagging remains straightforward.
Preserve physical-parameter, grid, shape and time compatibility validation.
Older static checksum records are historical evidence and are not prerequisites
for loading an artifact.

The opt-in precipitate `train_multiple.py` config replaces the single-particle
IC fields with explicit `kind=multiple`, periodic `centers`, corresponding
`levels`, shared `width` and `background`. A smooth union prescribes only the
initial field; future interfaces remain learned. Legacy configs keep their
original single-particle arithmetic and serialized trainable leaves. Multi-case
checkpoint loading uses its saved config for the static IC metadata. Use the
same config with `--config.nl.method=local_ad` for the primary local control;
mechanics remains spectral in both. The simple pilot default is unchanged.

## Verification

Choose the smallest verification set that gives credible evidence for the
change. More checks are not automatically better.

- Documentation, agent instructions, and configuration-only edits normally
  need diff inspection, applicable syntax/schema validation, and
  `git diff --check`; they do not require the Python test suite or compileall.
- A localized behavior change normally needs one focused regression test or the
  narrowest existing test command that reaches the changed path. Add a test
  only when it protects meaningful behavior or a reproduced defect; do not add
  tests for wording, trivial wiring, or implementation details already covered.
- Compile only changed Python modules when syntax was not already exercised by
  the focused test. Do not compile unrelated trees by default.
- Run broader integration, numerical-equivalence, GPU, FEM, or full-suite checks
  only for changes that cross those boundaries, alter shared numerical code or
  public APIs, or when the user explicitly requests them.
- Avoid repeating expensive checks whose relevant inputs have not changed.
- Always run `git diff --check` and update the example README when its public
  API, formulation, configuration, or file structure changes.

## Markdown Math

Markdown intended for OpenCode Web uses inline `\(...\)` and display `$$`
delimiters, with each display delimiter on its own line. Do not use `$...$` or
`\[...\]`.

## Project split package layout

Reusable implementation lives in `src/nonlocal_pinn`; example directories own
case geometry, BCs, formulation and CLI. Imports use `nonlocal_pinn`, with no
`pinn` compatibility namespace and no sibling checkout dependency.
Operator implementation is preserved as one module inside `operators/` pending
separately justified subdivision.

See migration/source-map.json for old-to-new source ownership.

## SOAP implementation provenance

The shared factory currently imports pinned SOAP_JAX0.2.2 (pyproject.toml and
uv source revision). A source/parity audit against author PyTorch and independent
pytorch_optimizer is archived in tasks/archive/soap-implementation-audit.md.
It verifies the tested initialization/count/projection/refresh/moment paths and
the real factory clipping/LR recurrence; it does not establish immunity to
rank-deficient initialization sensitivity. The older upstream stale-EMA defect
is already fixed in the installed version. Reference implementations differ in
precision and nonzero weight-decay order; current factory uses zero decay.
Do not silently replace a dependency/default or compare native eigenbases
bitwise across frameworks. Preserve old checkpoint/artifact paths and record
any future algorithm/precision change separately from numerical bug repairs.

Confined CH optional `arch.model=global_mixed` requires learned chemical mode,
positive `rank/output_scale/chemical_scale`, logarithmic_additive phase lifting
and explicit `field_layout=shared|separate`. It stores both field subnetworks in
one global checkpoint; separate fields do not mean time-window decomposition.
`arch.chemical_scale` maps raw mu outputs to physical mu; `loss.chemical_scale`
normalizes chemical-unit residuals and is explicit in the new presets. Legacy
configs without either field retain their previous arithmetic/template.
Neumann-inverse evolution norms support learned as well as eliminated mu.
Optional `phase_metric.kind=mixed_block` requires fixed identical SOAP/refinement
nodes, explicit positive time_scale matching the phase log coordinate, and at
least3*rank intervals for cubic consistency. `mixed_preconditioning.py` checks
weighted spatial symmetry/PSD and the precomputed global2x2 inverse. Its losses
are separate phase-unit and chemical-unit transformed residuals. Reference
trajectory coefficients are never training inputs; time-projection probes are
explicitly validation-only.

Confined phase-only `phase.output_map={kind:tanh, amplitude:a}` selects
TanhPhaseNetwork through train.py; it requires eliminated nonlocal global_basis
and instantaneous loss metrics. `tanh_mapping.phase_state` returns projected
physical phi, its pointwise tanh Jacobian, and the implicitly differentiated
scalar mass shift. Use it through predict_fields for full-grid inference;
point calls alone omit the global shift. Boundary diagnostics include that shift
inside the physical point map. `loss.residual_transform.kind=none` retains r;
`inverse_tanh_jacobian` uses r/J before the existing spatial metric. No floor or
mean-removal is silently applied. The new static map metadata leaves legacy
constructors/parameter leaves unchanged; saved config remains necessary to know
which physical output those leaves represent. Optional amplitude1.0 is explicit,
not a universal maximum-principle assumption for quartic CH.

Confined shared mixed presets optionally select `train.loss_balance.kind` as
`fixed`, `gradient_norm` or `stagger`. They require learned shared global fields,
nonlocal instantaneous Neumann-inverse losses, and reject windows/causal weights.
Alternatively, explicit positive `loss.evolution_scale` selects raw normalized
mixed residuals and rejects a simultaneous spatial phase metric. It is separate
from `loss.chemical_scale`; omitted field preserves prior evolution arithmetic.
Case-local `loss_balancing.py` uses VJP cotangents for stopped adaptive weights
and active-loss masks, with fixed-shape state. Gradient norm means the existing
corrosion inverse-L1-gradient balancing rule, not relative-rate GradNorm.
Explicit `update_every/alpha/norm_floor/norm_ceiling/weight_floor/weight_cap`
configure adaptive state; `switch_every/stagger_steps` configure complete
alternating cycles followed by joint training. No field parameters are frozen.
`refinement=freeze` for adaptive and `joint` otherwise require one fixed joint
scalar for all L-BFGS line searches. Existing configurations lacking the record
keep the original step/refinement arithmetic. Saved physical network templates
and parameter leaves do not depend on this training-only record.

Electropolish opt-in `configs/train_operator.py` dispatches from its canonical
train.py to operator_training.py before the single-IVP shared-engine branch.
operator_data.py owns random-IC identities, analytic input fields, disjoint
partitions and strict independently time-refined same-graph reference loading.
Reusable `nn.conditional.ConditionalFieldOperator` is a raw pure Equinox module:
patch/self-attention encoding, spatial cross-attention or pooled FiLM, then
shared temporal FiLM. This is explicitly a separable PI-CViT-inspired variant,
not an exact refactor of upstream time-dependent attention. `prepare` computes
all learned spatial/condition work inside parameter AD; `decode` owns temporal
broadcasting/JVP. Known local input-function values are explicit query features.
Never cache learned encoder/prepared arrays across optimizer updates. Frequencies
are stopped coordinate metadata. Physics maps/losses stay in the example. New
operator uses unbounded normalized liquid concentration with the KKS mixture,
because random-interface references exceed the legacy sigmoid c_L bound. The
single-IVP default/serialized models stay unchanged; no future-field training
labels or test-based checkpoint selection is introduced. Preparation is CPU-only;
training is a separate ordinaryFP32 process. Strategy presets omit interface/run
seeds; per-interface IDs and dataset seed live in dataset/run provenance.

Operator optional `operator.training_indices` selects a nonempty unique subset
of the source training partition. The single-input preset chooses0/B1. Excluded
source monitors are `unused_train`; held-out IDs cannot be selected. Absence
keeps original all-source condition draws bitwise, with actual draws recorded.
Optional `operator.online` is mutually exclusive with that subset. Online mode
uses continuous amplitude/frequency/phase triples every update, constructs input
images and query-local IC through the independently checked original discrete
SDF, and conditions the distribution on declared profile height validity. A
fixed oversampling factor preserves shapes; too few accepted profiles raises
before a step, never uses a padded replacement. Nearest curve indices and pools
are precomputed. Sampler work stays outside parameter AD; the original five
physical losses and temporal JVP stay unchanged. Source dataset fields are only
an evaluation bank (`monitor`/`test`) in this mode. Record actual coefficients,
accepted counts, selected-input provenance and device allocator peaks by direct
paths. The initial online preset usesB8/5000Adam after a matched single-input
SOAP/Adam control; no established single-IVP optimizer/default is changed.

Operator `loss.use_causal=True` explicitly enables the existing shared
`training.causal.CausalWeightor` per-time rule (`causal.n_chunks=0` only).
`configs/train_operator_causal.py` is the matched continuous-online control.
Compute per-condition/per-equation free-core MAE, sorted by time, then stopped
exp(-min(epsAC,epsCH)*prefix_loss) weights and the unnormalized mean(w*MAE).
Core normalization excludes padding/halo; IC/BC terms remain unchanged.
Dynamic eps enters a case-local aux-aware step, with the same inverse-L1
gradient balancing and optimizer as the original step. Batch-mean weights
drive the shared helper's separate eps updates. Log pre-update weights/segments,
eps used/next and physical unweighted parts. Disabled causal retains
the original three-item payload and shared noncausal step. No parameter leaves,
physics, temporal sampling, hard constraints or saved model templates change.

Operator `arch.local_input=initial` feeds known query-local initial phi/c values
to the decoder; explicit `zero` disables that shortcut by zeroing only those
two feature channels. IC loss targets and global input images stay unchanged.
Both modes have identical parameter shapes/initialization and retain physical
IC supervision. Old saved configs lacking this optional field use the legacy
initial mode. Replay through `predict_fields` with saved configuration; raw
network calls alone do not apply this case-local feature option. The canonical train_operator.py preset selects online B8/5000 Adam and zero
query-local features, matching the validated image-only run. Historical initial
mode is explicit in train_operator_legacy.py and the old online controls.

Electropolish `configs/train_operator_local.py` selects opt-in `nl.method=local_ad`
through the same operator trainer. Its `operator_local.py` shares encoded images
inside parameter AD and computes phi/q coordinate Hessian traces with directional
JVPs; query rows must remain independent. Require explicit zero local-query
features, noncausal weights and reference loading. Local uses the same padded
core IDs/masks as the sparse pool and omits halo; all true IC targets remain.
Original five physical losses gain a sixth side-wall phi_n/q_n-over-c_le MSE,
with explicit positive `loss.neumann_scale` and six balancing biases. Positive
times only, top/bottom corners excluded; retain rough incompatible ICs.
The common source bank still contains graph targets, so local's periodic metrics
are labeled graph diagnostics. `operator_reference.py` accepts an independent
SciPy matrix and ordered snapshot times for local/graph reference validation.
New runs record sampling.npz and synchronized update/setup/evaluation timings;
snapshot local helper/preset sources. No shared network parameter shapes change.

Electropolish optional `sample.spatial` enables full/fixed_pool/rolling_pool
controls through the same train.py. Require explicit mode, max_nodes, max_edges,
workers and prefetch, sparse/noncausal online-condition mode. The case-local
spatial_sampling.py constructs complete one-hop scalar graphs with strict
padding from a proven degree bound; it does not call the old truncating helper.
Workers only construct NumPy data. Main-thread JAX transfers preserve fixed
shapes/dtypes and replace coordinates/edges/masks/SDF metadata together. FIFO
draw identities match the first fixed pool then advance by global update, while
condition/time keys remain independent. Record actual draws/cores and producer
wait/transfer/synchronized condition/update times. The absence of the record
retains the historical path; current canonical NO and single-IVP defaults stay
unchanged. Use shared padding for fixed/rolling ablations, and distinguish full
node coverage from interpolation/nonlocal discretization accuracy.
The recorded [composed-attention replay issue](issues/electropolish-attention-replay.md)
requires distinguishing same-GPU checkpoint replay from independent CPU
portability: never mark a failed cross-backend gate passed or silently loosen
its tolerance. Saved-field/reference metric replay is a separate validation.

Melting opt-in `configs/train_operator.py` dispatches from canonical train.py
before the single-IVP validator. `operator.action=prepare|train|memory` is explicit.
The reusable conditional model accepts2D(CYX) or3D(CZYX)volumes, with coordinate
queries in XY/XYZ. Preserve historical2Dpatch/key/parameter/position arithmetic;
3Dpatch projection and positions require matching rank and width divisible by6.
Do not tokenize every128^3physical grid node. The initial32^3volume is512tokens;
training uses fixed core/halo queries, evaluation encodes once and chunks queries.
Raw decoder local_dim is0; analytic prescribed IC enters only the case-level
hard latent wrapper. Only initial ellipsoids are parameterized; later geometry
is learned. Online parameters/SDF preparation are outside parameter AD.
Ellipsoid Euclidean distances include singular interior axes and independent
SciPy/FP32checks. Rotated/shifted shapes require a full-cube independent native
reference; never reuse sphere analytic evolution or octant symmetry. Phase and
solid/interface/volume errors remain separate. Memory probes must include PDE,
parameter backward and optimizer, and distinguish active bytes from allocator
pool/CUDAcontext. Old single-IVP defaults/checkpoints remain unchanged.

Melting optional `sample.spatial` selects full/fixed_pool/rolling_pool through
the canonical entry, with explicit mode/max_nodes/max_edges/workers/prefetch/
full_chunk_core fields. Case-local spatial_sampling.py uses uniform NumPy
Generator.choice cores, complete CSR neighbors and sorted-index remapping,
avoiding full-grid permutations/remap buffers per rolling draw. Both sampled
controls share safe1792/1536padding and deterministic first32sets. The old
configuration without this record retains its RandomState pool and trainer.
Full chunks cover the original graph once per update; accumulate core-count
weighted losses/parameter gradients before one optimizer update. Do not treat
chunks as separate stochastic updates or change IC/time keys between chunks.
Workers construct NumPy-only metadata, main thread transfers bounded arrays.
Save actual IC/time/core streams, initial/final optimizer states, synchronous
spatial/condition/GPU timers and allocator peaks. Full-grid L1training does not
make diagnostic24^3evaluation a full-grid L2 or validate continuum accuracy.


## Versioned research information

Confined CH opt-in `configs/train_operator.py` selects the accepted width256,
FP32/ordinary, SOAP50000, online rolling-pool geometry operator. It aliases the
explicit `train_operator_online_width256_cold50k.py` recipe and dispatches before
the single-IVP validator. Preserve the old mild16/8pilot in
`train_operator_mild.py`; historical presets, test fixtures and the mild reference
CLI default import that preserved baseline, not the current SOTA alias. This
avoids circular inheritance and silently changed historical controls. Canonical
operator settings match the measured saved config except runtime provenance;
load historical checkpoints with their saved config. The no-argument train.py
default remains the separate single-IVP30-window case. Each domain has its own
Gmsh triangle-centroid quadrature and visible regional graph. Padded nodes have
zero physical measure/edges and zero physical output. The ambient IC is the same
function across geometries, without a moving-hole envelope. The reference bank
contains evaluation fields; steps receive only coordinates/measures, analytic
IC, independent checked graph and resolvent, plus boundary point data.

`nn.point_conditional.PointFieldOperator` is the retained boundary-only model;
zero `latent_tokens` retains all boundary tokens. The explicit
`train_operator_heterogeneous.py` control selects `arch.model=heterogeneous` and
`HeterogeneousPointOperator`: separately lifted boundary and domain-IC point
streams, GNOT-inspired normalized linear integral attention with separate stream
measures, repeated spatial conditional reads and shared temporal FiLM. Required
`input_dims` start with the coordinates specified by `extent`; CH uses(5,3).
This is not full GNOT: no query-set self-attention or geometric MoE. It preserves
independent raw point queries/chunking. No fixed distance bias is introduced.
Do not cache learned preparation across updates. Physical output goes through
the example hard IC/mass wrapper; off-grid queries preserve the original domain
quadrature for the mass shift, not their own query average. Geometry-contrast
metrics supplement whole-field L2 because a good average field can conceal weak
conditioning. The16/8bank is finite and reused for architecture development;
do not call it an online-infinite input stream or a fresh final test.

The opt-in confined `train_operator_richer.py` changes actual mesh/ambient IC/
geometry distribution, not network priors. Its native/local reference bank is
separate. `operator_evaluate.evaluate` requires explicit `plot_style`; the CLI
defaults to `smooth`, with `flat` retained. `operator_plotting.SmoothMeshDisplay`
projects area-weighted incident cell values onto original mesh vertices and
uses original-triangle linear interpolation. It never re-triangulates across
holes; analytic circles are masked too. Reference and prediction use identical
display maps, and displayed absolute error is computed afterward. Raw physical
centroid fields/areas remain the only source of reported metrics. Do not claim
interpolated display values supply extra numerical accuracy or physical mass.

Keep source, stable Markdown, explicit configurations, compact numerical metrics
and regeneration scripts in Git. Generated media, arrays, checkpoints and HLO
under docs/tasks are ignored; no forced addition of those payloads. Existing
local files stay at their original paths. artifacts/local_files.json records
an external byte-verified backup for formerly tracked/local report outputs;
scripts/restore_local_artifacts.py restores selected/all files and refuses to
overwrite different local content. Old report image links are local artifact
links and require restoration/regeneration after a fresh clone.

Run `python scripts/check_repository_content.py` after staging. It rejects
indexed generated payloads and files larger than1MiB; small immutable migration
fixtures outside docs/tasks remain allowed. Do not rewrite published history
to reduce size without separate explicit user authorization. Ordinary new
commits removing outputs limit future growth but leave historical blobs intact.

Confined CH `train_operator_richer512.py` adds explicit `operator.precompute`
fields `workers/source_path/reserved_test_draws/monitor_train_indices`. Canonical
`train.py` dispatches prepare to spawned CPU production-operator workers, with
isolated explicit fenicsenv meshing and one BLAS/OMP thread each. Preparation
never solves a new training trajectory. Format `operator_only_v1` declares
per-record `has_reference`, source accepted-draw IDs and labeled training monitors;
train/test partitions stay disjoint and contiguous. Loaders reject evolved fields
in operator-only files and regenerate/verify analytic initial data. Optimizer
payloads omit all future fields for both labeled and unlabeled inputs. Aggregate
train L2 means the explicit monitor subset; held-out L2 uses the original eight
reserved geometries. Legacy all-labeled manifests retain their original semantics.

Confined geometry-reference preparation accepts optional explicit positive
`operator.reference_workers` and `local_reference_workers`. These change only
orchestration: spawned CPUprocesses each own one independent domain/mesh/solver,
while the parent orders records and writes manifests. Native `prepare_case`
retains independent ordered graph assembly and time-refined BDF; `local_case`
imports FEniCS in its own process and retains identical spatial/time gates.
Absent worker records preserve legacy serial paths. No JAXGPUwork overlaps FEM.
The selected opt-in `train_operator_smallscale10[_512].py` pair uses T3/10fixed
small-scale IC modes, fresh physical references and the same finite512/8operator
schema. Only geometry varies, with no IC parameterization. Earlier20/24mode
mixed-band previews are retained as untrained candidates. The generic model and
old defaults are unchanged.

An empty explicit `operator.local_validation_indices=()` defers local FEM for
a native-only operator study. Precompute then records local_reference_status=
not_requested and copies no local files; nonempty requests still require passed
matching FEM. Native reference refinement remains mandatory. `--local` rejects
an empty request, and evaluation reports local_FEM.status=not_evaluated when no
local files exist, rather than NaN or a local-accuracy claim.

The confined10mode512SOAP preset changes only the optimizer; its width96 control
changes only width. Architecture counts exclude the stopped spatial Fourier
entries. Temporal features use j*pi*t/T: horizon changes alter their explicit
physical input frequencies. Treat time-feature/width alternatives as untested
until measured, and keep slow convergence distinct from a proven capacity floor.

Confined geometry operators optionally accept `arch.time_max_frequency` as
static metadata. Omission preserves the historical integer frequency vector
`arange(1,K+1)` exactly. When present, `K>=2` Fourier frequencies are spaced from
1 to that explicit finite maximum (`>=1`), in units of `pi*t/T`; this separates
embedding width from bandwidth. It consumes no RNG and adds no parameter leaves.
Both point operator families share the same decode convention. Old configs and
checkpoints retain their arithmetic and parameter layout. The operator optional
`sample.temporal` uses the existing global log-time sampling/curriculum helper,
with a prepended exact initial node; absent record retains the old operator RNG
path. The physical horizon and hard lifting do not change during a curriculum.

Confined geometry training optionally selects `operator.metric=linear_stable_resolvent`
with required positive integer `metric_workers`. Input-only NumPy/SciPy workers
build the spectral positive-part resolvent of the CH Jacobian frozen at the
known initial mean. The physical bank/specification and old cached metric stay
unchanged; only in-memory training metric arrays are replaced before padding.
`residual_metric.json` records construction per domain. This mode is train-only;
prepare the physical bank with the original preset. The existing biharmonic
branch retains its arithmetic and performs no extra worker construction.

Confined geometry `train.continuation.parent_run` explicitly imports a complete
parent into a new run. Validate metadata/model/state files before bank setup;
by default require identical numerical settings and a larger total `train.iterations`.
Restore both states, keep global RNG/curriculum counters, and replay all saved
physical fields before new updates. `continuation.json` contains direct paths;
summary `start_step/new_updates/cumulative_wall_seconds` distinguishes stage
and lifetime cost. Fresh paths retain old arithmetic; missing optimizer state
is an error, never a reset. The shared exponential optimizer schedule depends
on its restored count and fixed decay interval, not the new total budget.

The only optional numerical exception is explicit
`train.continuation.learning_rate_factor` in(0,1). Require both lr_init and
lr_min to equal their parent values multiplied by this factor. Reject all
other optimizer/model/sampler changes and unknown continuation options. Record
before/after values in provenance; keep the imported adaptive state and count.
This is an intentionally changed ongoing schedule, distinct from strict
unchanged-schedule continuation. Without the factor, historical strict behavior
is retained. Test nonzero same-gradient update ratios and complete state equality,
in addition to the uninterrupted-versus-serialized strict path.

Geometry-operator optional `phase.lifting=logarithmic_additive` and positive
`initial_time_scale` reuse the existing normalized log coordinate for both
hard increment and decoder. `time_coordinate` is shared by grid/off-grid
wrappers; physical-time JVP retains the chain rule. Missing record retains
`t/T` arithmetic. No network leaves are added, but the wrapper semantics change:
replay checkpoints only with their saved config. Strict continuation rejects a
coordinate change. `arch.time_max_frequency` is in the chosen decoder coordinate,
which is `t/T` only for the linear-time path.

Geometry-operator optional `loss.initial_residual_fraction` sets the initial
PDE-node fraction explicitly, with the remaining weight on the positive-time
sample mean. Require a finite fraction in[0,1), at least one positive time and
the leading node0. Omission preserves the original full-node mean exactly.
Keep this fraction fixed when changing positive-time sample count if an equal
expected objective is intended. It is not an additional initial-condition loss.
First-update allocator statistics are saved for full-physics memory checks.

`operator_history.training_records` follows explicit continuation parent paths
and merges ordered global metric/loss records, removing the imported-boundary
duplicate and adding parent cumulative time to child metric times. Cycles and
unordered histories are errors. Evaluators and convergence reports use this
helper so continuation plots include the full trajectory rather than only the
latest stage. Physical prediction arrays remain stage-local.

Confined geometry optional `operator.initial_rhs_condition=True` adds a known
function channel to the heterogeneous encoder. Require domain inputdim4:
`(x,y,phi0,T*F(phi0))`, where F is the prescribed CH vector field, evaluated once
using the checked initial field and graph. `operator_conditioning.py` derives
this input without trajectory data. Loader-only `domain_fields(N,2)` contains
phi0 and the horizon-scaled RHS; `initial` remains separate for hard IC/mass.
Default absence/False retains the original8-key payload and exact encoder path.
The enabled path has9explicit label-free keys. Require an explicit boolean,
heterogeneous model and matching dimensions; currently reject augmentation
instead of reusing an initial tangent from a different transformed input.
Neither the physical PDE/loss nor the output ansatz changes. Validation/dense
query helpers must derive the same known channel from their saved config.

Incremental confined-CH operator preparation can explicitly set
`operator.precompute.reuse_mode=hardlink` (absence preserves historical copy).
Source banks may be validated full native or mixed `operator_only_v1` banks.
Reuse is keyed by the accepted continuous-prior draw identity; training prefixes,
reserved held-out draws and monitor reference roles must remain consistent.
Only missing operator inputs are built, never new evolved trajectories. Hardlink
mode fails on unsupported filesystems and makes both paths share immutable
payloads: never rewrite either source or destination files. This avoids doubling
existing dense cached metrics on disk. Array/label/partition identity is checked
in the active task's bank-growth audit before any cross-bank accuracy ranking.

Confined geometry operators explicitly accept `train.compute_dtype` as float32
or float64. Host geometry/operator/reference/residual-metric construction stays
FP64; `operator_precision.py` casts finalized floating training arrays and model
parameters, preserving integer graph indices. The initialization and temporal
RNG remain on the original FP64 path before rounding; JAX x64 being enabled
does not imply FP64 network/PDE/SOAP computation.
The local `make_compute_optimizer` adapter types FP32 SOAP beta scalars to avoid
temporary FP64 update arrays from bias correction with global x64 enabled;
shared optimizer/upstream implementation and the FP64 path stay unchanged.
Scalar configuration literals may be FP64 before conversion; reject nonscalar
FP64 tensors when auditing an FP32 training graph. Public field/query evaluation
also casts input coordinates/times to the requested compute dtype. The opt-in
width128/beta09 FP32 preset preserves all other numerical settings. Independent
saved-field checks account for rounding explicitly; original references and
FP64 checkpoint compatibility remain intact. Never continue optimizer state
across a precision change under the exact-state continuation API.

The opt-in confined-CH width192 FP32 capacity preset also sets spatial Fourier
directions96 and SOAP max_precond_dim512, preserving both feedforward axes when
their hidden width grows256->384. Four heads then have48features each. The
four condition-reading blocks are cross-attention from spatial queries into
separate boundary/function streams; time decoding shares their prepared output.
Configuration/capacity checks must record the joint backbone/Fourier change and
SOAP coverage rather than label it a width-only or head-only control. The
existing128baseline has no learned axis in257..512, so raising its cap alone
does not change coverage. A different architecture starts cold with fresh
parameters/state and is not compatible with exact-state continuation.
The explicit width256 FP32/50k preset continues this family with128spatial
directions, four64feature heads, unchanged four cross/three FiLM blocks and
axis cap512 (the FFN reaches512). Research capacity helpers select the new
case explicitly with `--case=width256_50k`; omitted case preserves the original
192/30k CLI. Memory feasibility uses production25updates including SOAPrefreshes;
disk estimates count all paired snapshots plus declared reserves. Preserve30k
checkpoint/metrics and61time evaluation when comparing to the192/30k baseline;
do not conflate larger capacity and a longer optimizer budget.

The explicit `train_operator_smallscale10_1024_width192_fp32_50k.py` preset and
researchcase `width192_bank1024_50k` change only input-bank size/provenance and
budget relative to the completed192/51230kcontrol. Reuse the immutable1024bank,
not a new set of evolved training trajectories. Source training prefix,16monitor
arrays and8reserved accepted draws must compare exactly. Fresh192weights and
temporal draws match; condition-ID draws intentionally differ with1024support
and must instead be checked against their own globalkeys. Validation indices
1024..1031 map to original512..519 by accepted geometry draw and physical arrays.
The research dense-reference view hardlinks original61time NPZs only after these
checks, preserves refinement receipts and records direct source paths; no hash
loading or new reference solve. Retain30kand50k assessments separately.

Confined-CH geometry-operator intermediate checkpoints now publish a paired
`ckpt_<step>.eqx`, `optimizer_<step>.eqx`, and `checkpoint_<step>.json` at the
existing configured checkpoint interval. Payloads are written through exclusive
temporary files and atomically renamed; the global-step completion marker is
published last. A leftover model or `.partial` file without its marker is not a
complete pair. These intermediate backups are distinct from the completed
fixed-budget `summary.json`/`model_final.eqx`/`optimizer_final.eqx`. An explicit
`train.continuation.checkpoint` imports a published intermediate pair; without
that option a completed parent is required. Full SOAP recovery through an
intermediate pair is independently tested. No checkpoint
I/O changes the numerical updates, learning-rate schedule or PRNG counters.
