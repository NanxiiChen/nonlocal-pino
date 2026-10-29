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


## Versioned research information

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
