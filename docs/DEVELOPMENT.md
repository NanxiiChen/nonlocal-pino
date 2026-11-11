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

Curated README figures live in the repository-level `assets/` directory,
currently `assets/pino/<case>.png`; example READMEs use relative links there.
Keep one selected evolution image per documented operator, below1MiB each and
2.5MiB total for the six current images. Six additional setup SVGs live in `assets/pino-setups/`, below320KiB
combined, with compact prescribed parameters in
`artifacts/pino_setup_schematics.json` and the CPU-only
`scripts/render_pino_setup_schematics.py`. These show conditions/BCs, not solved
fields; the same spinodal initial texture is shared between geometry samples.
Record evolution-figure direct inputs in `artifacts/pino_readme_figures.json` and retain
raw fields, weights and full research galleries as external artifacts. See
[`assets/README.md`](../assets/README.md) for the regeneration command.

The channel FEM reference optionally accepts a read-only `diagnostic_callback`
receiving `(time, (velocity, pressure, phi, mu), mesh, facet_tags)` at saved
snapshots. The absent-hook path retains the original equations and outputs.
Separate native weak natural-BC traces from algebraic residuals; validate this
hook with unchanged saved fields before interpreting new boundary diagnostics.

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

Channel NS--CH optionally selects `time_discretization.kind=bdf2` with explicit
dt, replacing time AD in training by BE-started BDF2 physical states from the
same current model. Actual interval identities are logged; no predecessor is
frozen. Optional `phase_transport.kind=sbp4_mapped_curl` adds six explicit
known-input arrays (two derivative matrices, two norm vectors, mapping Jacobian
and physical SBP measure) to the original ten-array geometry payload. The
encoder inputs remain unchanged. Matching quadrature and conservative mapped
phase fluxes do not imply energy/mass stability of the full mixed GMLS/NS
scheme. The explicit legacy `configs/train_operator.py` retains the original
experimental AD/GMLS recipe. The actual channel training default is
`configs/train_default.py`, selecting the validated random-wall width192/B8/
SOAP20000/integral/timeJVP fixed-normalization recipe with a fresh automatic
run name. It uses36random times in two18-time blocks and LR1e-3->1e-5;
measured active peak83.50GiB with explicit `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97`.
Historical10000/40time/block20/8/2presets retain their old numerical values;
their resource results do not qualify an unmeasured36-time block variant.
Do not redirect
the legacy factory to its own descendants or alter historical preset values.

The explicit channel integral control uses `nl.method=integral_moment`,
`derivatives={kind:integral_first,boundary:integral,velocity:unchanged_streamfunction_ad}`
and `phase_transport.kind=integral_conservative`. Its radial horizon includes all
physical neighbors, padded with zero weights to the explicit `nl.neighbors`
capacity; overflow/rank failure raises. `integral_operators.py` corrects the
original quadratic radial gradient/Laplacian coefficients to degree3 derivative
moments using physical nodal volumes and weighted QR. Payload keys and physical
conditioning arrays stay the same; stencil widths change. TimeJVP, explicit
BCs and AD streamfunction curl remain. `integral_validation` permits only the
operator/selectors/run-name differences and checks actual cold model/SOAP and
all unchanged initial/final conditioning/quadrature/time arrays. The finite
runner's `--operator-control` selects this audit independently of other controls.
No default promotion or symmetric-pair conservation claim follows.

The explicit extension `time_derivative={kind:jvp,sampling:continuous_stratified}`
instead uses `network.time_jets` for physical time derivatives and retains the
GMLS spatial PDE/BC gradients and Laplacians. It cannot coexist with BDF2.
`sample_continuous` draws one continuous time from each equal stratum afresh
on every update. A compiled sampler is shared with canonical global-key replay;
saved `sampling.npz` then has no discrete `intervals`. Forty collocation times
are not a finite-difference step. Time points do not interact in the decoder,
so the all-ones JVP tangent produces each point's own derivative without forming
a full time Jacobian. Streamfunction velocity is differentiated as a physical
field, including its mixed space/time derivative. No unused residual spatial
AD Hessians are computed in this path.

`configs/train_operator_width128_gmls_time40_adaptive_jvp_online.py` changes only
the time operator/sampling from the adaptive BDF control. The finite runner
requires that matching completed adaptive BDF run as `--comparison-control`.
Coldmodel/fullSOAP, initial/final geometry banks and every actual condition draw
must match; continuous time arrays intentionally differ and are independently
replayed. The JVP path uses `diagnose_time` for unweighted losses on the complete
trained spatial grid at240independent continuous probe times, on the nominal
and one final-pool geometry. These are canonical PDE/BC diagnostics, not an
independent discretisation/continuum certificate or a BDF full-time audit.

The explicit `train_operator_width128_gmls_time40_jvp_uniform_online.py` uses
the same timeJVP path with uniform losses, retaining the unweighted
`make_accumulated_fn` step. No causal prefix/controller is constructed. The
strict comparison admits either this uniformJVP versus matching uniformBDF,
or the existing adaptiveJVP versus matching adaptiveBDF; a uniform/causal
mismatch is rejected. The finite owner requires and audits that matching
comparison even when causal is absent. Its resource probe checks the complete
unweighted gradient and actual continuous global-key sampling, and records no
adaptive-feedback result for the uniform branch. Existing causal/BDF paths
remain separate; no model/loss-scale/physical output semantics change.

Channel optional `loss.chemical_residual_scale` must be an explicit positive
finite real scalar and is rejected with eliminated chemical rows. It overrides
only the chemical PDE-row normalizer. `loss.chemical_scale` remains the physical
mu decoder amplitude and the explicit legacy row normalizer when the new field
is absent; preserve that old branch exactly. Never describe changing legacy
`chemical_scale` as a loss-only ablation. The chemical research preset keeps
output4/boundary4 and sets row1, with unchanged nine-part/fixed-shape APIs.
`summary.json` and independent continuous-time diagnostics record both roles;
unweighted diagnostics mean no causal discount, not unnormalized physical units.
The frozen constitutive diagnostic reconstructs physical RMS and uses the saved
independent normalizer if present. Strict paired validation permits only the new
normalizer and run name, requires an original shared-scale control, and checks
complete cold model/SOAP and all actual initial/final banks and sampling arrays.

The finite channel owner separately accepts `--geometry-control` for a cold
geometry-only study. It permits only upper/lower wall amplitudes, prior bounds
and artifact paths to differ from an audited source. Initial model/fullSOAP
bytes and actual time/condition/online draw identities must match, while each
different geometry-input stream is independently reconstructed. Do not route
this through the unchanged same-geometry `--comparison-control` or claim equal
physical fields/banks. The stronger preset needs its own refined local FEM;
frozen transfer is explicitly labeled extrapolation beyond the source prior.

The channel `online.py` controller has opt-in cold-training integration, with
explicit workers/refresh_size/refresh_every/prefetch_batches and the unchanged
continuous two-amplitude prior. CPU workers rebuild the canonical geometry/GMLS
(and optional SBP) inputs without PDE/FEM trajectories, then cast finalized
arrays. FIFO counters, bounded prefetch, known-field schemas and direct draw IDs
are tested. `configs/train_operator_conservative_online.py` is an untrained
research preset; production-resolution training and unseen-geometry accuracy
remain required before claiming validated online operator learning. Canonical
train.py uses a context-managed producer, donated fixed-shape pool replacement,
actual sampled draw IDs/amplitudes and final pool inputs/state. Evaluation
geometry inputs are separately owned and stored. `audit_training_inputs`
reconstructs the FIFO stream, actual sampled geometries and final payloads.
Online continuation is explicitly rejected until full pool recovery exists.

The opt-in channel `boundary_sampling` record requires exactly
`kind=gauss_legendre`, integer `points_per_side>=2`, and
`encoder=unchanged_midpoints`, with the explicit GMLS derivative path.
The existing `boundary/boundary_weights` arrays then store the BC-loss rule;
`encoder_boundary/encoder_boundary_weights` preserve the original midpoint
conditioning stream. These extra arrays have explicit fixed shapes and are
validated/reconstructed in online input audits. Gradient targets follow the
BC-loss queries. Graph-wall quadrature includes the arc-length Jacobian;
open Gauss nodes avoid assigning a single normal to a corner vertex. Absence
preserves all old payloads, encoding and physical fields. Increasing encoder
token count is a separate architecture/conditioning change.

The GMLS/BDF2/conservative-flux joint path permits the existing explicit
`balance` settings: stopped inverse-L1-gradient-norm weights over phase,
chemical, momentum and boundary groups. This changes only the loss metric,
not any primal residual or sampling weight. It is not learned GradNorm.
Stagger remains excluded from this combination: a conservative-flux lagging
rule needs its own primal-parity validation before enabling that path.

Channel `train.microbatch={kind:time,size:8}` optionally accumulates gradients
for a complete `sample.Nt=40` random time batch. The size must divide Nt. Each
block uses the same current parameters; the averaged gradient is clipped and
sent to SOAP once. This is not sequential time integration. The initial path
requires cold joint losses, without stagger/refinement; the original uniform,
explicit termwise-balancing and causal paths remain separate.
`configs/train_operator_width128_gmls_time40_online.py` is a finite5k control
preserving the original GMLS model and boundary rule. Forty equal strata each
contain six of the240positive BDF nodes; averaging is unbiased for the original
full-time mean. It does not force the first positive node on every update.
This pilot precedes a separate matched40-time causal experiment; causal is not
enabled by increasing Nt or by accumulating gradients.

Channel `balance.grouping=terms` explicitly assigns one stopped inverse-L1-
gradient/EMA weight per soft term (nine with streamfunction). Absence retains
legacy physics-group arithmetic; `grouping=physics` names that older grouping.
The new `make_balanced_accumulated_fn` averages each term's complete parameter
gradient over all time blocks BEFORE taking its L1 norm. On refresh, reuse
these gradients to form the weighted sum; on held steps, accumulate only the
fixed-weight gradient. Clip and update SOAP once afterward. Averaging block
norms or changing weights inside a block sequence is not this algorithm.
This is not the learned-weight/training-rate objective of original GradNorm.

The explicit allterms stronger-channel preset retains outputmu4/stream.2/p1,
sets all used residual divisors1 and balances all nine terms with the existing
epsilon1e-6/cap1000/newfraction.9/refresh100 rule. Its independent
`balance_validation` audits the matched fixed-scale control, actual cold states,
all input/time arrays, refresh schedule, EMA and weighted objective. The finite
owner accepts `--balance-control` separately from prior temporal/geometry
controls. No default promotion follows from enabling adaptive weighting.

Optional channel `causal` explicitly selects `kind=coupled_prefix_mse`,
`weight_gradient=stop` and `boundary=same_time_weights`. It
requires cold original GMLS/streamfunction joint losses with BDF or explicit
time JVP and time microbatches.
`make_time_parts_fn` returns(condition,time,nineparts) without mixing geometry
prefixes; the unchanged mean-loss API retains its original reduction order.
Use shared `CausalWeightor.compute_weights` on the sum of nine scaled MSE terms.
First evaluate small per-time vectors without AD, stop global40-time weights,
then differentiate/accumulate8-time blocks with these fixed weights. Never
reset causal prefixes at block boundaries. The legacy noncausal step remains
separate. Epsilon is a dynamic scalar in the opt-in compiled step.

The historical fixed-stage preset declares positive increasing `epsilon_values`
and `steps_per_epsilon` whose stages exactly cover the training budget. Its
(.01,.1,1,10,100)/1000updates schedule is preserved for replay, not recommended
as a convergence-based controller.

`configs/train_operator_width128_gmls_time40_adaptive_causal_online.py` instead
declares `causal.adaptation` with exactly `kind=weight_feedback`, `eps_init`,
`step_size`, `max_eps`, `max_last_weight`, `min_mean_weight` and
`geometry_reduction=minimum`; fixed schedule fields cannot coexist. Reuse shared
`CausalWeightor.update_eps` independently per geometry and take the minimum
proposal for the common dynamic scalar. Every update increases only when all
last weights exceed the threshold, decreases if any mean falls below its
threshold, otherwise holds. Require `0<eps_init<=max_eps`, `step_size>1` and
`1/Nt<min_mean_weight<max_last_weight<1`; the mean has floor1/Nt because w0=1.
Current explicit settings are1,1.1,100,.99,.5. This is feedback from freshly
sampled inputs/times, not a full-time or population convergence certificate.
`causal-controller.jsonl` records EVERY used/next epsilon plus each geometry's
mean/last weight; `audit_trace` independently replays all transitions and checks
periodic full-prefix/objective records against them. Feedback is stopped and
does not add trainable leaves or change model/SOAP/physical residuals.

Physical/unweighted/weighted diagnostics remain distinct. The finite owner
additionally requires an audited
`--comparison-control` and checks coldmodel/fullSOAP bytes, all actual initial/
final geometry arrays and sampling histories, plus independent saved-array L2.
Complete unweighted240node and droplet checks remain required; sampled causal
weights do not certify full-time convergence. No default promotion is implied.

Channel optional `derivatives` explicitly selects `kind=gmls_first`,
`boundary=gmls`, `velocity=unchanged_streamfunction_ad`, together with BDF2 or the
explicit time-JVP extension and
`phase_transport={kind:gmls_conservative,quadrature:sbp4_mapped}`. Five extra known
arrays store domain x/y weights plus boundary neighbor indices/x/y weights.
Preserve integer boundary indices when casting/auditing rolling inputs. First
residual derivatives and div(phi*v) use GMLS; physical velocity remains the same
AD streamfunction curl, with no new continuity loss. SBP arrays supply the old
loss measure only in this path, not an exact conservation identity. Baseline
configs/checkpoints keep their original arithmetic. Prior Laplacian screens do
not certify new boundary gradients; every actual construction validates them.

`docs/tasks/channel_online_train_runner.py` owns one explicitly requested cold
20k online run through canonical `train.py`, followed by saved-run input/field,
phase-balance, droplet-shape and gallery checks. It requires explicit config,
run name, seed, reference root and matching input-construction screen. Its atomic
external journal reports the active child and actual latest loss/refresh; a
detached `screen` session owns the runner independently of the chat terminal.
This finite wrapper starts no FEM solve, retries or later experiments and does
not restore the cleared goal schedulers. Intermediate model checkpoints still
lack complete rolling-pool/optimizer recovery; only a completed run supplies the
final audited pool/state. Do not reinterpret those snapshots as an exact online
resume.

`docs/tasks/channel_cold20k_runner.py` separately owns the opt-in192/B8/time20
budget/LR experiment. The base `budget_validation.py` route permits only10000->20000iterations,
the declared exponential decay-rate change and run name against the completed
audited10k control. Same-GPU/same-shape historical memory evidence is explicitly
reused, without calling it a new resource run. Actual fresh model/fullSOAP and
initial/evaluation/reserved inputs must match, and final sampling must preserve
the complete10k geometry/time prefix. The802online-draw prior screen and existing
reference/rolling/replay/cohort/fresh-time audits remain required. The old
capacity-control contract is unchanged; no automatic SOTA promotion follows.

The explicit `--execution-time-block=8` extension permits only the separate
`train_operator_random_walls_width192_batch8_time8_cold20k.py` preset. The
logical40-time objective is unchanged: five complete gradient chunks precede
each optimizer update. The default20-block comparison still rejects undeclared
block changes. Probe and production auxiliary-output trees can produce different
compiled temporary-memory requirements; a matching finite probe alone cannot
certify the canonical executable's peak. Require headroom and actual production
updates. Preserve reduction-order and throughput differences in comparisons.

The separately declared `--execution-time-block=18` route requires the exact
36-time/18-block cold20k preset; geometry batch8 remains fixed. The resource
probe adds explicit `--time-samples=36 --time-block=18`. Legacy guards continue
to require40times unless that route is selected. Actual cold model/fullSOAP and
initial geometry inputs remain exact, and final geometry slot/draw prefixes
are compared. The changed-width time arrays instead require successful
independent global-key replay and36-stratum coverage in `validate_run`; never
claim the old40-time prefix is identical. Equal-block accumulation code and
the physical horizon/equations are unchanged.
The finite owner explicitly records252fresh diagnostic times for the18-block
route(14complete blocks); legacy20/8routes retain240. This is a post-training
diagnostic count, separate from the36random training times per update.

The finite owner's separate `--capacity-control`/`--capacity-probe` route
qualifies the explicit random-wall128/B2/5k/timeblock8 to192/B8/10k/timeblock2
trial. Require a production-grid seven-update resource/nominal-query check,
complete extended input-prefix screen, and actual cold model/fullSOAP bytes
equal to the probe state saved before any update. It is a combined experiment;
geometry slot draws differ, while the initial input bank and shared time prefix
are checked independently. Keep the original control routes and defaults.

The explicit hardware route adds `--hardware-trial` to the probe and finite
owner, allowing only explicit40-time divisors(2,4,5,8,10,20,40) on that same capacity problem. Require a
matching physical GPU UUID, actual full-pool gradient/SOAP/query probe and peak
memory below both90%of physical memory and95%of the JAX allocator limit. Keep
the32GB gate unchanged for the original route. Resource timings exclude cold
compilation and do not establish accuracy; pool/optimizer recovery is still
required for exact continuation after a hardware interruption.

After the user-authorized aggressive-memory study, a hardware probe may record
an explicit allocator fraction. The owner then requires the identical
`--memory-fraction` argument and checks peak<=95%physical/97%allocator memory;
legacy probes keep their previous gate. The memory probe records compiled
temporary/output/alias sizes, existing resident buffers, predicted requirement
and the actual finite-update peak separately. Insufficient-memory candidates
are skipped explicitly before executing their complete step, without changing
the declared40-time objective or retrying silently.

Online channel presets may explicitly name `operator.validation_manifest`.
`cohort.py` freezes independent uniform-prior geometry parameters and direct case
paths, with strict physical-family/draw checks; this file contains no evolved
labels. Initial and rolling inputs reject both exact and training-precision
matches with reserved amplitudes, recorded in reserved_geometries.npz and the
independent input audit. `cohort_evaluate.py` requires every reserved FEM gate,
reports individual/mean/worst and geometry-response diagnostics, and validates
physical flux/no-slip/initial constraints on each domain. Solver `--saved-config`
reads exact run JSON instead of a dotted preset; it is mutually exclusive with
`--config` and changes no FEM equations or boundary conditions.
The Python `cohort.prepare(c,path,count,query_grid)` requires an explicit pair
of positive query-grid integers; its CLI requires `--query-grid Nx Ny`. The
newer frozen manifest keeps the same16parameter draws but uses101x43midpoints
instead of96x40(which coincided with192x80training nodes). Preserve both scopes
and old files. The final pipeline includes16unseen FEM trios plus one separate
nominal-development trio, with all gates before frozen GPU evaluation.

Channel geometry-response validation now additionally freezes an explicit
`response_queries` protocol at identical physical coordinates in the maximum-
constriction common subdomain. Keep the16parameter draws/primary own-domain
queries unchanged. Optional FEM `--common-query-grid` exports additional fields
from the same trajectory; `validate_reference` requires both sets to pass the
same space/time limits when present. Legacy exports retain their original
fields/checks. Cohort evaluation uses common coordinates for the beat-blind-mean
response gate, keeps mapped-coordinate response separately and excludes
unprescribed initial pressure from flow-response metrics. The research watcher
requires an explicit absolute `--response-protocol` path and first performs a
short isolated FEM export-compatibility fixture after GPU work exits. That
fixture is not a validated reference. Physical config validation must remain
free of JAX/Equinox imports, including optional initializer path checks.

Channel `train.continuation` supports complete-parent direct-path imports with
`kind=strict`, `kind=refine`, `kind=operator` or `kind=time_refine`. All restore model AND full SOAP state, global
RNG/LR counters and parent physical replay into a new run. Strict permits only
budget/log changes; refine additionally permits integer spatial Nx/Ny increases.
Replay the parent with its original encoder quadrature before measuring the
refined-conditioning field change. Reject unsupported online/GradNorm/stagger
state recovery, missing state files or numerical-setting changes. This contract
does not reinterpret or move legacy checkpoints.
Operator refinement additionally permits higher GMLS polynomial degree, no
fewer neighbors and an explicit positive weight exponent; condition limits,
physics/model/output semantics and all other settings stay fixed. Record exact
before/after operators and independently check unchanged physical predictions
before updates. Wider condition-prior construction screening remains separate
from solution accuracy. No unknown-field fallback or conditioning relaxation.
Time refinement permits only an explicit strictly smaller integer subdivision
of the same BDF2 dt. It changes training rates and time draws, not the network's
physical output semantics, spatial operator, LR/count or moment state. Record
the before/after time discretizations; unchanged strict contracts still reject
that change. Rate consistency probes do not establish solution accuracy.

Channel `train.refinement` selects the canonical deterministic full-time L-BFGS
mode, with required parent_run/optimizer/iterations/time_batch/history_size/
max_evaluations/max_line_search/gradient_tolerance/function_tolerance. The first
mode requires one uniform, original-variable finite geometry and a complete
audited SOAP parent with unchanged numerical/physical settings. Host FP64
accumulation retains the configured neural dtype(FP32 in the current preset)
and averages gradients over every positive BDF node; it
does not freeze history states or train time windows. Keep inherited input/
sampling bytes explicitly distinct from new deterministic support. Independent
validation checks parent bytes, nodes and actual objective counts before field
replay. Retain parent SOAP state under its own name; final SciPy position/
gradient does not support exact-history recovery and is not a complete SOAP
continuation parent. Optimizer termination and accepted/evaluation counts stay
explicit, including a finite-budget exit without convergence.

Optional channel `pressure_coordinate.kind=chemical_shift` versions only the
raw pressure semantics as Pi=p+phi*mu, with physical p=Pi-phi*mu returned for
all derivatives, outlet traction and reference comparisons. It adds no parameter
leaves; missing record retains old checkpoints exactly. Its product-rule force
identity uses the current AD first derivatives, not an assumed GMLS product rule.
Frozen-basis accuracy is worse, so this option remains experimental/off.

The channel full-time finisher optionally declares `droplet_metric` with explicit
kind=frozen_soft_phase, contrast, temperature, fields, normalization=interior_mean
and source=parent_model. Its positive mean-one weights derive only from parent
phase predictions at every BDF node. They stay fixed during L-BFGS, preserving
a genuine scalar objective/gradient pair. The runtime loss batch adds a separate
droplet_weights array; saved known_inputs and inherited sampling retain the old
schema/bytes. Store phase/weights/time in frozen_droplet_metric.npz; final audit
re-evaluates the parent over all nodes and reconstructs weights. Parent whitelist
permits this metric-only extension, not combined coordinate/stress changes.
Missing option preserves original arithmetic/checkpoints. Pure physical config
validation remains independent of JAX. This is experimental residual weighting;
do not identify it as an output constraint or use reference evaluation masks in
the training metric. No online/fresh-state support is implied by this extension.

Channel optional `stress_formulation` explicitly requires kind=cauchy_mixed,
initialization=preserve_original and a positive constitutive_scale. Six raw
outputs replace pressure by three Cauchy stresses; physical five-field APIs
retain p=-trace(sigma)/2. The two independent deviatoric constitutive equations
are evaluated in the interior and at boundary queries; all original physical
BC losses stay intact. There is no extra pressure/trace/continuity loss. First
controls reject coordinate/balance/stagger combinations. Absence preserves old
four-output arithmetic. The full-time importer whitelists only this output
extension, copies all parent arrays except the declared output rows, initializes
sigma=-parent_p*I, and audits the reconstructed initial model. This adds194
parameters at width96; old SOAP state is provenance, not a compatible shape
restart. Cold online initialization preserves its original four-field function
the same way. Spatial stress gradients use AD, GMLS remains for phi/mu, and no
all-spatial nonlocal claim follows. Recipes remain experimental until measured.

Channel optional `time_embedding=linear_log` splits an even existing Fourier
width equally between normalized linear and log1p time coordinates, keeping
raw t/T and the explicit common maximum frequency. Its positive physical scale
is explicit. It changes decoder features only, not the physical BDF2 times or
hard lifting, and adds no model leaves. Missing record retains original linear
arithmetic. Feature/full-gradient tests pass; the first degree3run stopped before
updates is not an accuracy outcome. A separate cold degree4combination has
completed with worse phase/pressure accuracy and is not a promoted default.
An explicit optional `initialization=preserve_linear` retains an old linear
half-width model's function while adding zero-weight log coordinates. Transfer
targets the actual time-encoder MLP first layer, not a nonexistent Linear time
field. Copy every other array; use structural array indices for eqx.tree_at,
not a selector that branches on leaf values. For width96/8->16directions the
first input matrix grows96x17->96x33, adding1536weights. Older records without
the initialization field retain their original constructor/RNG behavior.
The full-time finisher allows only this whitelisted expansion from a matching
linear-time parent. Preserve model_parent.eqx, reconstruct/verify the padded
initial model and check physical replay; original strict continuation contracts
remain unchanged. This is extra input capacity, not a transformed PDE/loss or
an exact optimizer-state transfer. No accuracy promotion follows from parity.

Channel full-time refinement optionally declares `stress_initializer` with
exact kind/model/report/quadratic fields. `physics_auxiliary_block` imports
only a verified frozen-feature stress fit from the original four-output parent.
Require absolute paths, exact parent config/stress recipe/full BDF support,
actual FP32 objective reduction and unchanged physical replay. Reconstruct every
rounded output row from stored coefficients and compare every model array;
copy the three payloads into the run and repeat in the final audit. The original
isotropic transfer remains the missing-record behavior. No cold online default,
optimizer state, PDE metric, training labels or checkpoint semantics change.

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

The separate `channel_nsch2d` pilot solves full matched-property unsteady NS--CH
on a smooth nonperiodic graph channel; it does not replace the prescribed-flow
`twophase2d`. Canonical train.py selects configs/train_operator.py; the tested
SOAP recipe remains an accuracy-failed pilot, not SOTA. Only upper/lower inward
amplitudes vary. Boundary/domain point
streams share the common encoder; a case-local multichannel decoder retains
physical(phi,mu,ux,uy,p). The streamfunction extension instead learns raw(phi,
mu,psi,p), obtains velocity by AD, and imposes flux/divergence/no-slip/inlet and
initial velocity exactly. It omits a redundant continuity loss. Missing optional
velocity/lifting records preserve the direct5channel/linear earlier pilots;
explicit linear and normalized-direct presets retain their recipes.

This channel experiment uses a polynomial-reproducing QR-GMLS physical-node
Laplacian, with spatial AD (including psi Hessians), time JVP and explicit actual-
wall/outlet residuals. GMLS is a finite-neighborhood differentiation control,
not the established regional pair kernel or an automatically conservative/no-
flux closure. Keep that scope visible in results. Independent P2/P1/P2/P2 FEM
uses BE-started BDF2, symmetric hydrodynamic outlet traction0 and capillary force
-phi*grad(mu), with no extra pressure anchor. Training requires embedded matching
physical identities and a passing predeclared space/time gate. Never run FEM
MPI and JAX GPU together. The initial trial has one384query development FEM
geometry; a config field cannot substitute for an actual evaluation cohort.
Evolved reference labels are used only in reporting. Saved actual draws,
geometry-bank arrays, source snapshots and validate_run.py support replay and
hard-flow/phase-area diagnostics; no open-domain constant-mass projection.

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

Particle-corrosion `train_default.py` now selects `train_physical.py`: explicit
SI `physical.{D,A,L,alpha_phi,omega_phi}`, `nondim.{L_c,T_c}` and `geometry_si`
produce all four PDE groups, normalized geometry and initial interface width
through `dimensional.derive_config`. `physical.L` is actual mobility, so kappa
must equal1 in this path. Cache consistency and normalized t_max1 are validated
before reference/training; partial SI records and stale groups fail. Input and
reference loading checks units as well as existing physical fields. New SI
references include `physical_time=times*T_c`; legacy dimensionless manifests
remain readable only with matching legacy configs. `train_online.py` preserves
the old physics. Preview also selects the current preset so its IC metadata
matches preparation/reference generation. New physical accuracy must not be
inferred from historical dimensionless results.

Particle-corrosion `configs/train_online.py` adds an explicit
`operator.online` record: workers, geometry_every/size, spatial_every/size,
prefetch, halo, neighbors and worker_cache. A bounded spawn-process producer
constructs known geometry, mesh, visible-pair operator, conditioning and spatial
inputs only. Independent clocks refresh retained geometries' complete core/halo
sets and IC points as well as replacing geometries. Geometry wins same-slot
overlap at one tick; whole slot payloads install atomically before sampling.
Runtime seed namespaces determine geometry IDs and spatial revisions; worker
completion order never selects training inputs. Fixed padding fails on overflow.
Worker caches include the contour and mesh/operator, avoiding repeat geometry
rejection sampling when only cores change. Missing online record preserves the
historical fixed bank arithmetic and presets.

Inadmissible random particle candidates are normal rejection-sampling events:
check both declared contour resolutions before acceptance and retry without
changing clearances or the RNG stream. `sample_geometry(..., diagnostics=...)`
optionally records rejection counters without drawing randomness. Only the
dedicated `GeometrySamplingExhausted` budget signal is retried by the background
producer, with a warning on every budget restart. Other construction errors
propagate. Cached geometry diagnostics are reused, not counted as new draws.

Canonical `train.py --sampling-control FIXED_RUN` permits only the online record
to differ, reproducing the old cold model/optimizer and slot/pool/time draws.
Safe rolling padding can differ from the old bank's measured padding; record
that execution-shape difference when comparing accuracy or update speed.
The documented candidate-admission/retry repair is the sole shared-source
exception: all other geometry, physics, network and operator functions must
match. Preserve the original fixed input arrays and disclose the source exception
in the comparison receipt.
`--exclude-reference DIR` adds geometry-only exclusions from another reference
manifest; the primary reference geometries are always excluded. Save initial/
final pools, per-step consumed geometry/revision IDs and FIFO event metadata.
`online_validation` audits these identities and independently rebuilds every
updated final slot. Intermediate model/pool-identity snapshots do not supply
exact continuation; the trainer does not implement online resume.

Particle optional `sample.full_grid={nodes,neighbors}` uses one full mesh graph
per geometry instead of the sampled core pools. Keep every physical mesh node
available as a neighbor; only mouth Dirichlet rows and padded rows have zero
PDE measure. Other exterior and ceramic-wall rows retain physical lumped-volume
weights. The original five losses and soft mouth treatment are unchanged.
`prepare_initial_full_bank` explicitly rebuilds known mesh/operator arrays while
copying the historical IC/BC/conditioning arrays exactly. Spatial refreshes keep
the same IC draws; full graph rows do not change while the geometry is retained.
`train.py --full-grid-control ROLLING_CORE_RUN` reproduces the cold arrays and all
geometry/revision/time draws. Original pool-ID RNG draws remain recorded but
unused for full PDE selection, avoiding a shifted time stream and eight duplicate
graph copies. `full_grid_validation` independently checks complete row coverage,
physical volumes and all native CSR coefficients in initial/final pools. This
opt-in comparison does not change the rolling core/halo default.

Particle optional `operator.mouth_sampling=mesh_dirichlet` aligns the two mouth
losses with every Dirichlet node used by the discrete operator. Lumped P1 edge
lengths give positive weights to endpoints as well as interior nodes; remaining
`operator.mouth_points` capacity is zero-weight padding, never truncation.
Geometry boundary encoder tokens retain their independent midpoint quadrature.
Only the mouth-loss point/weight arrays are overridden during initial and
rolling payload construction, with no extra RNG draws. Missing selector retains
historical `midpoint` behavior; input-bank metadata must match this policy.
This experimental option does not change the physical BC, impose it hard, or
promote a new default. The speed2 study documents the endpoint gap it addresses.

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
network calls alone do not apply this case-local feature option.
`train_operator.py` selects online B8/Adam, width 96, 16 shared times,
10,000 updates and zero query-local features. The matched spatial and
time-injection presets inherit `train_operator_5k.py`, which keeps width 64,
four times and 5,000 updates. Historical initial mode is explicit in
`train_operator_legacy.py` and the old online controls.

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
retains the historical path. Canonical `configs/train_operator.py` selects the
rolling_pool path with width96,16shared times,Adam10000,256cores,32initial
entries,3328nodes/3072edges,oneworker/32prefetch. Historical full/fixed/rolling
research presets retain width64/Nt4/5000. The single-IVP uses its own preset.
Use shared padding for fixed/rolling ablations, and distinguish full node coverage from interpolation/nonlocal discretization accuracy.
The recorded [composed-attention replay issue](issues/electropolish-attention-replay.md)
requires distinguishing same-GPU checkpoint replay from independent CPU
portability: never mark a failed cross-backend gate passed or silently loosen
its tolerance. Saved-field/reference metric replay is a separate validation.

The Melting spatial control optionally accepts positive integer
log.model_every divisible by log.eval_every. It writes model-only snapshots
at that monitoring interval, alongside the existing final optimizer state.
The explicit train_operator_pool_10k.py control cold-trains10000updates and
sets model_every250. Its learning-rate schedule retains global step indexing.

Melting opt-in `configs/train_operator.py` dispatches from canonical train.py
before the single-IVP validator. `operator.action=prepare|train|memory` is explicit.
The reusable conditional model accepts2D(CYX) or3D(CZYX)volumes, with coordinate
queries in XY/XYZ. Preserve historical2Dpatch/key/parameter/position arithmetic;
3Dpatch projection and positions require matching rank and width divisible by6.
Do not tokenize every128^3physical grid node. The initial32^3volume is512tokens;
training uses fixed-shape rolling core/halo queries; evaluation encodes once and chunks queries.
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
controls share safe1792/1536padding and deterministic first32sets. The canonical
operator now selects rolling K256/32initialsets/oneworker/32prefetch with
Adam10000 and1792node/1536edgebuffers. The prior canonical5krecipe is frozen
in train_operator_5k.py; historical fixed/rolling/full and cold10k comparison
factories derive from it. Absence of the spatial record retains the original
RandomState pool and trainer.
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


Channel optional `stream_lifting={kind:normalized_inlet}` requires explicit
normalized lifting and streamfunction velocity. It reuses `lifting.inlet_length_scale`
and replaces only the streamfunction inlet envelope by the square of the
normalized inlet gate. Absence retains original `(x/L)^2` arithmetic.
The finite runner/comparison requires a completed original-envelope adaptiveJVP
control and exact cold raw state/fullSOAP/all actual times/geometry inputs.
Positive-time initial physical velocities intentionally differ; do not equate
raw-parameter parity with physical-field parity after an ansatz change.

`stream_lifting.kind=normalized_height` is a second explicit channel option,
using the existing positive channel height in the squared exponential gate.
The original `normalized_inlet` kind continues to use the phase inlet scale;
its narrow-layer NS conditioning failure is retained in the task record.
No new required default/length is introduced. Strict matched-control audits
still require the original adaptiveJVP baseline, identical rawinitial/SOAP/
actualtime/geometry streams, and explicitly different positive-time coldvelocity.

Channel `chemical_constraint={kind:interior_gmls_projection,boundary:decoder_trace,
query:mapped_bilinear}` eliminates only interior nodal mu from the chemical
relation. It requires original-variable GMLS/conservative flux/timeJVP joint
losses. Retain boundary nodal values and independent decoder boundary traces;
omit the redundant chemical soft term (eight remaining terms). Physical query
factories live in `projected_fields.py`; raw point `fields`/`potentials` reject
this flag. Precompute/save/replay `evaluation_query_rule.npz` from the actual
training-dtype coordinates. Native values use exact identity lookup; nonnative
values use mapped bilinear interpolation or retained boundary traces. Do not
interpret that query-value rule as a classical spatial-AD mu extension.

The corresponding research refinement explicitly declares
`train.initialization={kind:physics_model,source_run:/absolute/run,optimizer:fresh}`.
The source must be a completed audited pure-physics original JVP model. Strict
config comparison allows only this initializer/chemical flag/lr_init/run name.
Import the final model only, initialize complete SOAP state from its factory,
and start fresh pool/RNG/global counters. This is a global-time physics-only
model refinement, not cold training, exact optimizer continuation, or time
windowing. Record/audit parent-model bytes, complete fresh optimizer bytes and
actual sampling/input parity. Config/source-path validation must not import JAX.
Unweighted diagnostics and physical replay use projected mu consistently;
preserve all old-branch numerical gates and report the unequal cumulative
training budget when comparing the source and new stage. No default promotion
or accuracy claim follows from algebraic elimination.

Channel opt-in `geometry.kind=random_spline` uses `geometry.py` as the shared
NumPy/JAX graph-wall source. Geometry coefficient count is explicit (two sides
of `controls` coefficients); the compatibility payload key `amplitudes` carries
that vector. Use `parameter_count`, `nominal_parameters`, `initial_parameters`
and `sample_parameters` rather than assuming length2. Preserve the absent-kind
legacy RNG path. C3 walls use quartic Cox--de Boor evaluation and septic tapers;
independent host piece-polynomial extrema qualify the conditioned prior. No
SciPy work enters JIT. Fixed-size point conditioning remains generic.

Random-wall cohort manifests have their own schema and common-physical query
envelope; exclude only nominal coefficients from that envelope's family identity.
`solver --parameters` selects the complete coefficient vector (or the explicit
nominal if absent) and rejects legacy two-amplitude flags for spline walls.
Validate all queries against the actual piecewise-linear FEM domain before
factorization. `--random-wall-control` on the finite owner audits the explicit
geometry-family extension against its completed integral control: cold model/
complete SOAP and all actual slot/time/drawIDs must match, while physical inputs
are intentionally different and reconstructed independently. Reference mesh
sizes and paths differ explicitly. Do not relax existing same-geometry audits.

Particle-corrosion experimental physical presets retain SI coefficients and
use explicit optional training maps. `loss.concentration_scale` multiplies
concentration IC/BC errors before squaring; absence retains1/cLe. The optional
`phase.initial_lifting={kind:bounded_exponential,time_scale:tau}` uses
`g=1-exp(-t/tau)`, bounded convex initial-phase lifting and additive liquid
lifting. Predictions, nonlocal residuals and physical-time JVP use the same
map; no evolved trajectories enter training. Optional
`arch.time_coordinate={kind:logarithmic,time_scale:tau}` changes decoder input
only, with the physical JVP chain rule retained. These options currently reject
local_AD rather than silently applying an unmatched map. Optional
`sample.temporal={kind:logarithmic_stratified,time_scale:tau}` uses an exact
initial PDE node and Nt-1 random log-time strata covering the entire horizon;
this declares a different residual-time measure. Missing options preserve the
historical soft-IC/uniform-time recipe. They are research controls, not SOTA.

The particle physical research branch also accepts
`phase.initial_lifting.kind=latent_additive`. The explicit known initial latent
coordinate is incremented by `log1p(t/tau)*z_theta`; liquid concentration uses
an additive increment with the same gate. `arch.zero_output=True` initializes
only the final increment head to zero, preserving all other random leaves.
`boundary.dirichlet=exact_mouth` projects the prescribed reservoir values at
actual query mouth coordinates and removes both redundant soft mouth terms.
This requires the nonlocal/hard-IC branch and uses two explicit PDE loss biases.
Time JVP includes all lifting/projection operations. Existing maps are retained.
These are unpromoted optimization candidates, with no evolved reference input.

Particle original-output strategy controls accept `--strategy-control` with a
completed run and a budget no longer than its saved budget. Only explicit
PDE scales/post-balancing biases,stagger/causal settings and budget may differ;
model/physics and non-strategy settings remain fixed. Cold model/optimizer arrays and
independently replayed source sampling prefixes are required, not a reset from
a trained checkpoint. Stagger gates PDE objectives while retaining every data
term; it does not freeze one physical field. Adaptive causal reuses the shared
controller on per-time physical-volume AC/CH segments, with stopped prefix
weights and IC/BC outside causal weighting. Scalars/selectors stay dynamic in
JIT and are logged. No default promotion follows from these research controls.

An explicit Adam/SOAP strategy comparison may additionally change only
`train.optimizer`; all learning-rate and SOAP hyperparameters remain fixed.
The cold model and actual sampling still match. Each optimizer keeps its own
factory-initialized state; runtime `paired_optimizer_initialization=False`
declares that these states are intentionally different. Other control types
still require identical cold optimizer states. Do not load Adam state into SOAP.

Original-output strategy controls may independently select the existing
`arch.time_coordinate={kind:logarithmic,time_scale:tau}` option. The original
linear-time source is still required; all other architecture fields and actual
physical time draws are checked. No parameter leaf is added by this coordinate
choice, so cold model/optimizer arrays must still match. Time JVP includes the
coordinate chain rule; this is independent of any hard initial lifting.

`loss.transport_residual=ac_eliminated` is an explicit original-output
nonlocal strong-residual control. Keep Rphi and replace Rc by
Rc-dc*hprime(phi)*Rphi, evaluated as cle*s_t+Gc-dc*hprime(phi)*Gphi with
the raw liquid JVP. No inverse tanh Jacobian is evaluated. The state-dependent
mixing coefficient is differentiated; do not silently stop its gradient.
Use the `ch_ac_eliminated` label and build causal segments from these trained
components. Original physical residuals remain independent diagnostics.
Reject combinations with integral-time/hard-output variants until qualified.

Strategy sources may also be completed fresh `initial_layer` runs,with exactly
the same source/target temporal settings. Recreate the entire source schedule
using its saved budget/configuration,then take the requested prefix. Do not
transform its saved times again. Runtime time identity compares actual arrays.

`loss.gradnorm_norm=l2` selects a scaled Euclidean parameter-gradient norm;
missing selection retains the exact historical L1 sum. Both ordinary and
causal steps share the norm helper and reject nonfinite raw norms before
clipping can hide them. `train.gradnorm_update_every` remains explicit and
positive;strategy controls can change it. These are inverse-gradient balancing
controls,not original learned-weight GradNorm. L1 balancing need not balance
Euclidean update magnitudes. A metric/refresh change does not change the PDE.

Particle `train.py --seed` optionally replays runtime initialization/sampling
without adding a seed to strategy configs. Omission preserves
`secrets.randbits(32)`. Validate an integer in `[0,2**32)` and reject an override
with any existing comparison-control argument. Runtime records `seed_source`;
the default random path and saved-control paths retain their prior behavior.
Across geometry-prior changes, compare cold model/full optimizer and actual
condition indices, pool indices and times. Do not claim that physical spatial
points remain identical when the geometry itself changes. Rebuild and audit
all known geometry/operator inputs under the new prior.

Optional particle `loss.causal_steps` releases adaptive causal discounts after
that many absolute optimizer updates, including any data warmup. It requires
causal enabled and an integer strictly inside the training budget. Released
steps use effective epsilon zero (all time weights one) and freeze the positive
controller state; model and optimizer state continue unchanged. A scalar
dynamic selector avoids recompilation at release. Logs distinguish effective
epsilon used from the retained controller state and record `causal_active`.
This is a global-network optimization schedule, not time-window integration.
The absent option preserves the historical full-budget causal path.

Particle `sample.horizon_curriculum` optionally declares exactly
`kind=exponential`, `initial_fraction` and integer `ramp_steps`. After explicit
data warmup it increases the global random-time horizon from that fraction of
`t_max` to the full interval; a nonempty final full-time tail is required.
The original `initial_layer` slots are retained exactly and must fit inside
the first horizon. Only remaining global time draws are scaled. Random keys,
physical coefficients, time JVP and network time-coordinate normalization stay
unchanged. This intentionally changes the training measure during the ramp.
Initially qualify only joint original strong nonlocal losses, with no causal,
stagger, alternative output or eliminated-row combination. A strategy source
must not already use this curriculum. Logs record actual time horizons;
independent replay must distinguish transformed times from unchanged core and
condition draws. One global model is updated throughout; no field is frozen
or propagated between windows.

Particle original-output controls additionally support
`sample.temporal.kind=initial_layer` with explicit early_count/early_end:
one zero node,early_count-1random early strata,and untouched global uniform
draws. This changes the residual measure intentionally. `loss.initial_metric=
phase_liquid` uses the equivalent prescribed(phi,s)IC,s=(c-dc*h(phi))/cLe.
`loss.data_warmup_steps` masks only PDE objectives initially and freezes causal
epsilon; optimizer state is retained at the transition. Runtime records whether
source time draws are identical or explicitly transformed.

The experimental `loss.time_form=integral_simpson` keeps the original mapping
and changes only the temporal residual to y(t)-y0+int G=0. Its
`integral_initial_layer`rule supplies0,T,random early strata and random global
strata.32endpoints require63network evaluations including midpoints; spatial
preparation is shared. The prescribed G(0)is substituted from the known IC;
all later integrands remain current-network outputs. There is one global
network,no frozen predecessor,and no time JVP in this loss. Standard physical
JVP residuals remain evaluation diagnostics. Initially require nonlocal original
outputs and causal off; do not silently combine unqualified variants. Composite
Simpson consistency defects must be distinguished from trajectory accuracy.
