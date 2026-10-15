# Fracture2D: local mechanics, nonlocal damage

The main route solves displacement with local P1 plane-strain FEM and damage
with a nonlocal AT2 PINN. `nonlocal_baseline.py` uses a direct fixed-history
AT2 solve with the **same mechanics, operator, history and loading loop**.
The provisionally accepted fresh trajectory spans by load 0.0053; its complete
saved prefix ends at 0.0057 after the user-requested stop. This cleanup preserves
that numerical configuration; it does not certify the remaining field errors.
Run-specific results and the background rerun handoff are preserved in the
[closed stage](../../docs/tasks/archive/fracture2d-nonlocal-pinn.md).
A separate [architecture investigation](../../docs/tasks/nonlocal-pinn-architecture.md)
is evaluating point-cloud networks; the public solver still uses the accepted MLP.

## Files

| File | Role |
|---|---|
| `train.py` | PINN CLI and persistent damage training |
| `nonlocal_baseline.py` | Matched direct-damage CLI |
| `loading.py`, `stagger.py` | Shared loading, history and field-change stopping |
| `losses.py` | AT2 residual, L1/L2/L16, detached optional weights |
| `network.py` | History features, MLP/compact readout, output heads |
| `history_network.py`, `history_probe.py` | Optional point-patch architecture and frozen-history replay experiment |
| `history_conditioning.py` | Experimental history normalization and local-query input comparison |
| `history_training.py` | Online history solver, snapshot lifecycle and replay checkpoint store |
| `sampling.py` | Full rows and fixed-shape exact core+halo pools |
| `plot_sampling.py` | Sampling density, actual pool batch, tip zoom and profiles |
| `optimization.py`, `at2_cg.py` | Adam/SOAP/L-BFGS and checked direct AT2 CG |
| `local_mechanics.py` | Local P1 assembly, Amor driver, checked mechanics CG |
| `mesh.py`, `mesh_geometry.py` | Validated mesh/operator and grip/visibility preparation |
| `storage.py`, `monitoring.py` | Atomic checkpoints and TensorBoard figures |
| `solver.py`, `geometry.py`, `gmsh_reference_mesh.py`, `mesh.geo` | Independent local FEM comparison |
| `configs/` | `train_base.py`, `train_core_halo.py`, `nonlocal_base.py`, `solver_base.py`; optional `history_probe.py`, `train_history.py`, `train_history_conditioned.py` |
| `_legacy/` | Local ignored historical solvers, diagnostics and experiment configs |

## Model

Material and damage nodes are the original material P1 vertices, in identical
ordering; damage exists only there. Mechanics also includes the undamaged
engineering volume grips. Material degradation is integrated with degree-two
triangle quadrature; grip cells use \(1+k_{res}\). The weak equation is

$$
\int_\Omega [(1-d)^2+k_{res}]\,\sigma_0(u):\epsilon(v)\,dx=0.
$$

The history driver uses the plane-strain Amor tensile/deviatoric split,
projected from cell strains to material nodes by lumped L2 projection.
At each outer iteration the history floor is the **preceding load endpoint**:

$$
H=\max(H_{n-1},\psi^+(u)),\qquad
r_d=(G_c/l_0+2H)d+G_c l_0 D_\delta d-2H.
$$

The original reciprocal, polygon-visible cubic-kernel operator
`ExplicitVolumeSparseD` is rebuilt once and compared against prepared CSR
coefficients. \(D_\delta\) approximates the negative Laplacian; its truncated
material support supplies the natural nonlocal damage boundary treatment.
Direct damage solves the volume-weighted SPD system. PINN training uses the
same residual on its neural nodal output; it receives no direct/FEM damage
labels. This hybrid history scheme is preserved, without a new joint
variational minimization or explicit damage irreversibility constraint.
The original formulation/reference/validation record is retained in
the local ignored `_legacy/HISTORICAL_README.md`, when available.

At fixed load: solve mechanics with current damage; update/freeze/detach H;
warm-start and train damage; replace damage; check P1 field changes; repeat.
One mechanics solve occurs per outer and one final synchronization occurs per
completed load. Saved reaction and displacement therefore correspond to saved
damage. History/features/weights/probabilities receive `stop_gradient`.

## Accepted PINN defaults

- FP64; \(E=210000,\nu=0.3,G_c=2.7,l_0=0.024,k_{res}=10^{-8}\);
  damage horizon 0.008; grip extent 0.048 and spacing 0.004.
- Own-zero physical u/d/H; freshly initialized swish MLP96×3 plus compact hat
  readout; sigmoid output initialized at 0.001. Latent basis RMS scales are
  fixed, and physical H features use graph powers 0,1,2,4,8,16.
- All71307 material nodes, uniform normalized L1; Adam1e-5, 128 inner updates;
  persistent network, moments and count across outers and loads.
- du1e-4 from zero through0.0060; outer cap128 below0.0053 and1024 thereafter;
  unrelaxed omega1. Exact material P1 RMS stopping: damage1e-5,
  displacement/load1e-4, minimum3 and3 consecutive passes.
- Residual remains a diagnostic. Budget exits may advance, are explicitly
  recorded, and are not convergence certificates. Damage range tolerance1e-3
  permits the earlier small signed-output diagnostic; fields are never clipped.
- TensorBoard figures every completed load, under `/root/autodl-tmp/tf-logs/`.

## Run

Use `.venv` for JAX. Execute from the repository root with `PYTHONPATH=.:src`.
The prepared geometry and compact basis paths are explicit in the
configuration. The existing material71307/union77331-node mesh is required for
these defaults. Startup rebuilds and checks the operator and requires the saved
independent FEM parity receipt. No geometry/operator is silently substituted.

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_base.py \
  --config.output.root=/root/autodl-tmp/tf-logs/fracture2d/pinn_run

PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.nonlocal_baseline \
  --config=examples/fracture2d/configs/nonlocal_base.py \
  --config.output.root=/root/autodl-tmp/tf-logs/fracture2d/direct_run
```

Both entries use absl + `ml_collections.config_flags`; dotted CLI overrides
modify explicit ConfigDict fields. Existing output roots are refused unless
`--config.output.resume=True` is supplied. Resume checks the saved configuration and file paths, endpoint ordering,
physical field shapes/ranges and optimizer count; it restarts from
**the last completed load**, rerunning an interrupted load. Named outer
checkpoints remain available for analysis, not implicit partial restarts.
New outputs have one `state_0000..` sequence; the former prefix/tail orchestration
and separate audit processes are archived. This changes storage/monitoring
organization, not the default numerical update.

To prepare a different geometry, set a fresh `mesh.prepared_dir` and
`--config.mesh.prepare=True`; preparation exits before solving. Verify its
local FEM parity separately in `fenicsenv` using the archived
`local_fem_validate.py` and matching explicit config (`--geometry-dir` takes the prepared directory).
The matching local-gradient damage FEM control used in the accepted three-way
figures is preserved in `_legacy/local_damage_comparison.py`; `solver.py` is
an independent FEniCS comparison with its own explicit boundary configuration,
not implicitly identical to that volume-grip control. Never run FEniCS MPI and
JAX GPU concurrently. The existing independently checked geometry need not be
regenerated for ordinary training. `runtime.platform=cpu` with
`JAX_PLATFORMS=cpu` is available for small manufactured checks.

## Optional comparisons

| Configuration | Choices |
|---|---|
| `loss.norm` | `l1` (default), `l2`, `l16` |
| `sample.mode` | `full` (default), `random` |
| `train.optimizer` | `adam` (default), pinned upstream `soap`, `lbfgs` |
| `arch.output_head` | `sigmoid` (default), `exp_square` = exp(-z²), `identity` |
| `loss.weight_mode` | `uniform` (default), `damage`, capped `reciprocal` |
| `loading.omega` | 1 (default), explicitly chosen relaxation in (0,1] |
| `loading.du`, outer caps, `train.inner_steps` | Load increment/outer/inner budget controls |
| `loading.field_stop`, `advance_on_budget` | Field-change stopping and budget advancement policies |

L1 is mean absolute normalized residual; L2 is mean squared normalized residual;
L16 retains the stabilized **squared L16 norm** used in earlier trials. All use
\(2\sqrt{\operatorname{mean}(H^2)}\) as full-history forcing normalization.

Random sampling draws cores with replacement from a positive baseline plus
inverse-power density around the configured crack segment; density decreases
away from the segment. It changes sampling density, not the crack equation or
an output symmetry. Original operator rows and **all** their halo neighbors
are retained. One finite pool is built at initialization: `sample.pool_size`
defaults to 2048 batches of `sample.core_count=2048` draws, using seed1729.
Every batch is padded to the actual largest complete core+halo union in that
pool; no manual capacity, safety margin or neighbor truncation is used.
Training cycles through these pre-randomized batches, changing batch every
`sample.resample_every=1` optimizer update. The pool is never regenerated during
training, and Adam/SOAP memory persists. Resume reconstructs the same pool from
its seed and restores the slot from the optimizer count. `sampling_pool.json`
records capacity, support statistics and array storage. Only the selected
batch is forwarded; full features and operator data remain precomputed.
`1/(N p_i)` corrects nodal moments in expectation over random pool generation;
reusing a finite realized pool is an empirical approximation. The nonlinear
L16 root is not an unbiased full-loss estimate.
An optional tip bump is centered at `sample.start`: raw density is
\(1+B_s(1+r_s^2/w_s^2)^{-p_s/2}+B_t(1+r_t^2/w_t^2)^{-p_t/2}\), normalized
over all nodes; \(r_s\) is distance to the segment and \(r_t\) distance to
its start. Explicit `tip_width`, `tip_boost`, `tip_power` control the bump;
the full base config retains `tip_boost=0` among its inactive random settings.
The optional `train_core_halo.py` preset uses segment boost18, tip boost36,
both widths0.024 and powers2, with per-update resampling. Setting
`resample_every=10` holds a batch across ten Adam
updates, even across an inner/outer boundary; H remains fixed for each inner
solve. Changing density is a sampling study, with the same importance
correction and global field-stopping criterion.
Global loss/residual checks occur once after each inner solve, not each update.

In the measured fracture2d comparisons, core+halo made individual updates
cheaper but required more iterations, so total solve time did not improve.
It remains an optional comparison; full sampling and damage-change tolerance
`1e-5` remain the default. Detailed timings and accuracy checks live in the
active task. One-off count/resource/density/hold/tolerance scripts are archived
under `_legacy/sampling_experiments_20261015/`; the separate tip preset and
the earlier segment-only preset are retained in `_legacy/configs/`.

Visualize a reproducible batch without launching training or allocating the
large padded pool. The plotting entry uses the same probability function,
seeded core draws and exact operator halo as training. Colors in the density
panel show per-node probability divided by uniform probability, not raw boosts.
Repeated core draws are counted separately from unique core nodes; blue halo
nodes are determined by complete operator support. The dotted segment marks
the sampling geometry, not an imposed crack path. PNG, PDF and numerical JSON
are written together. `pool_slot` is zero-based; the default sampling config is
`train_core_halo.py` and can be replaced with `--config`.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.plot_sampling \
  --pool_slot=256 --output=/tmp/fracture2d_sampling.png
```

Adam/SOAP use `train.inner_steps`; L-BFGS uses explicit
`train.lbfgs.maxiter`/`maxfun` budgets. L-BFGS is restricted to full sampling so its objective stays fixed during each
fit. It retains parameters and starts fresh curvature for a new fixed H.
Strong damage weights previously worsened the field; they are comparisons,
not new defaults. No symmetry or prescribed straight-path constraint is added.

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_core_halo.py

PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.train \
  --config.output.root=/root/autodl-tmp/tf-logs/fracture2d/l2_random_soap \
  --config.loss.norm=l2 --config.sample.mode=random --config.train.optimizer=soap
```

## Outputs and verification

`identity.json`/`config.json` capture configuration, input/source paths and
model identity; source copies live in `code_snapshot/`. Each completed state
contains physical fields and a result report, plus PINN network/optimizer
leaves. `parameterization.npz` preserves compact scales. `result.json` has
completed-load curves and completion status; incomplete outer results are
explicitly unsynchronized. Intermediate outer checkpoints default to
`output.checkpoint_sweeps=[1, 8, 32, 64, 128, 256, 512, 1024]`; each completed
load also saves its synchronized endpoint regardless of that list. These
checkpoint positions control output only, not convergence or training budgets.
Events default to `<output.root>/events`; explicit `output.event_dir` selects
a separate TensorBoard run directory/name. PNGs in
`figures/` show damage, ux, uy, H and completed-load reaction curves. Damage and reaction optionally compare to `output.reference_root` at the exact
same physical load, with matching model settings and array shapes. Set that field to an empty string to disable comparison.
The default preserves the current direct reference. Damage
uses0..1; displacement/H ranges use each plotted field's actual extrema.
The independent TensorBoard server reads `/root/autodl-tmp/tf-logs` unchanged.

Main regressions exercise exact halo/full gradients, detached weighting,
checked direct equilibrium, zero initialization, optional optimizers/heads,
and interrupted/resumed full/random trajectories. Historical regressions are
explicitly available under `tests/_legacy/fracture2d/`; routine discovery
focuses on the active code. The main route uses direct file paths throughout, with no SHA settings or
checks. Arrays/physical ranges, actual mesh ordering and operator coefficients
are validated numerically. Historical runs can still be inspected with their
original saved code snapshots; archived legacy integrity policies are not
part of the public route.
No historical field, weight, event, figure or stopped-run receipt is modified.


## Optional cross-history prototype

The accepted train.py route remains the default. The separate experiment
uses a shared small point MLP, volume-mean/max patch pooling and coordinate
cross-attention, with sigmoid damage and the same full-node nonlocal AT2 L1
residual. It compares the same architecture with and without residual replay;
it does not establish superiority over the accepted MLP or run coupled loading.

```bash
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.fracture2d.history_probe \
  --config=examples/fracture2d/configs/history_probe.py
```

The explicit probe config uses64coarse patches(8x8),64selected fine patches
from a32x32bank,width32/2attentionheads,experimental Adam1e-3 and256updates
per snapshot. This learning rate is local to the new prototype config; the
accepted PINN remains Adam1e-5.
Fine selection uses the range of bounded H within each patch, an experimental
input heuristic with no prescribed crack path. Groups retain all their points,
padded/masked to the exact geometric maximum. No all-particle self-attention.

At each actual source load, collect only first-outer H (after its mechanics
solve) and mechanically synchronized endpoint H. Endpoint H may differ from
the last inner H_used; no old neural damage labels are used. The initial
probe replays these two fields per load at0.0050,0.0051,0.0052 from saved
pinn_baseline;0.0053start/end are held out. This stream is fixed historical
input, not the evolving trajectory generated by the new network.

A capacity128 bank has shape(128,N),with fixed current+one-old task batches
(2,N). Random old snapshots are sampled uniformly with replacement; current/
old weights are.75/.25. The initial empty bank uses the positive current H
as safe filler and zero old weight; inactive tasks are skipped with traced
array control flow. Buffer count, flags and indices remain JAX arrays. Bank
capacity never grows and overflow raises explicitly;128canstoretwo snapshots
for60loads. No H=0/RMS(H)=0 placeholder enters the physical objective.

Only geometry/raw H input data can be cached. Learned patch tokens are
recomputed as network weights update. Save network, persistent Adam and replay
bank per arm; report new-field pre-update loss/error, post-update error,
retention, held-out error and actual time including replay. The short equal-
update comparison has unequal field-evaluation work; no matched-accuracy
speedup follows from its update counts. Source and output paths are direct;
no hashes, new data-generation FEM solve or automatic baseline restart.


## Optional online history operator from zero

```bash
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_history.py
```

This fresh run has exact physical q/d/H zero, a new token network and Adam,
and an empty fixed128-slot replay bank. No saved trajectory, old network or
compact-basis coefficients supply training data. The usual reference path
is used only for same-load TensorBoard evaluation. Standard localmechanics/
nonlocalAT2 stagger,du1e-4to0.0060,128inner and128/1024outer caps remain.
The experimental token config uses Adam1e-3;accepted MLPdefaults stay1e-5.

Every outer iteration trains on its current mechanics-generated H. The first
H of a load enters replay after its first inner fit, then the mechanically
synchronized endpoint H enters at load completion. Thus bankcount is2n-1
inside loadn and2n at its endpoint,with no inner/all-outer snapshot capture.
Current and old histories always use fixed-size task/token arrays. H and
selection are frozen per fit; network/Adam warm-start across outers and loads.

Replay cadence is explicit: `history.replay_every=1` and
`history.current_only_steps=0` preserve replay at every update. For an optional
sparse policy, `replay_every=8`, `current_only_steps=32` uses replay at updates
0,8,...,88 of a 128-update fit, then finishes on the current H only. On a replay
update the old/current weights remain `replay_fraction`/`1-replay_fraction`;
otherwise the current weight is one. The cadence restarts for each frozen-H
fit without resetting Adam or changing task shapes. Required finishing length
must lie between zero and `train.inner_steps`. These are experimental controls,
not a demonstrated speed improvement. Earlier saved configs require explicit
cadence fields when used to construct a new experiment; stored runs are not
silently migrated.

Each replay snapshot is stored once in replay_store/snapshot_XXXX.npz.
Outer/endpoint checkpoints reference that store through replay_state.json
and save their own network/Adam. Endpoint resume restores the actual bank,
load phase and optimizer count (the replay RNG derives from that count),
not a reconstructed bank from external reference data. Buffer overflow or
missing/inconsistent snapshot files fail explicitly. A partial load is rerun
from the last complete endpoint by the shared loading resume workflow.

TensorBoard run pinn_history_replay_from_zero_20261015 lives under
/root/autodl-tmp/tf-logs. It logs current/replay losses and buffer count, plus
damage,ux,uy,H and reaction curve at each completed load. Field-stability and
budget-exit labels retain their existing meaning; this experimental network
has not established an accuracy or speed advantage over the accepted MLP.

## Experimental history input conditioning

`history_conditioning.py` compares encoder normalization and a local-history
decoder branch on frozen positive H fields. It is separate from the model
selected by `train_history.py`. The normalized encoder uses H/RMS(H), with
log(2 RMS(H)/(Gc/l0)) retained as a decoder amplitude input. Physical H in
the residual is untouched. A decoder may additionally receive the two local
history features at its query particle; this is a conditional local branch,
not a coordinate-only DeepONet or a prescribed crack-path constraint.

All four comparison modes share decoder width and initial weights, with
inactive features set to zero. Boolean array switches are not optimized.
The full-cloud call requires queries in H order. `predict_queries` requires
explicit int32 particle indices for reordered or subset queries and validates
their range. H, feature selection and amplitude inputs remain detached; network
parameters still receive gradients. Positive contexts are validated by
`make_context`, including when supplied as replay inputs.

The optional `configs/train_history_conditioned.py` selects
`arch.type=conditioned_history_operator` in the shared stagger loading entry:

```bash
JAX_PLATFORMS=cuda python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_history_conditioned.py
```

This bounded profile starts from exact physical zero and fresh network/Adam,
uses combined normalized/local H inputs and full-domain Adam1e-4, and targets
0.0010 with du1e-4. The first frozen-H fit performs
`history.bootstrap_l1_steps=2048` followed by `bootstrap_l2_steps=2048`.
Subsequent outer fits warm-start 128 L2 updates; switching loss never resets
Adam. Positive bootstrap lengths and Boolean input modes are explicit required
fields for this architecture. Endpoint resume retains the optimizer count and
does not repeat bootstrap. A partial first load restarts from physical zero.

Replay training is disabled in this profile, but actual first-outer/endpoint
histories still accumulate in the fixed-capacity bank for future controls.
No outer-budget advancement is allowed. Field stability alone does not establish
accuracy. The CLI uses the ordinary field stopping rule; the recorded pilot's
external driver additionally checks same-load reference error and physical
residual after each saved endpoint, stopping on a failed diagnostic guard.
Normal `train_base.py` and original `train_history.py` defaults are unchanged.
