# 2D Rounded-Notch Fracture: Reference Solvers and Nonlocal PINN

This example is the 2D stage of the fracture program. The implementation order
is local FEM reference, direct nonlocal solver, then nonlocal PINN. The
**adopted direct scheme is one self-contained GPU-only file**,
[`nonlocal_baseline.py`](nonlocal_baseline.py); the next stage (PINN) is
tracked in
[`docs/tasks/fracture2d-nonlocal-pinn.md`](../../docs/tasks/fracture2d-nonlocal-pinn.md).

The historical research stack (per-module implementation, monolithic /
arc-length / reduced solvers, durable workflow, stage-specific test suites) is
parked under the ignored `_legacy/` directory and in git history. The frozen
adopted scheme and the separate independent-horizon research branch are
described below.

**Both available reference solutions — the local FEM archive and the nonlocal
direct solution — are warning-mode, non-strict diagnostics.** Neither is a
strictly converged reference; the exact nonlocal issues are listed in
[Validation status and accepted limitations](#validation-status-and-accepted-limitations).
The separate split-reference branch accepts individual strict comparison
states; its current progress is recorded in the active task.

## Independent-horizon PINN research

Configuration B, `configs/scale_separation_b.py`, retains the SHA-bound fine
cloud (`h=0.004`), `delta_u=0.008`, `l0=0.024`, physical grip extent `0.048`
and `h_grip=0.004`, but sets `delta_d=0.024`. Validate in its fresh output root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.split_horizon_validate \
  --config examples/fracture2d/configs/scale_separation_b.py
```

After validation, run fresh-zero direct loading with both solves on GPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.split_reference \
  --config examples/fracture2d/configs/scale_separation_b.py \
  --output /root/autodl-tmp/data/fracture2d/scale_separation_20261001/reference_b_gpu_20261007 \
  --damage-backend gpu --load-step 0.0001 --last-load 0.0075 \
  --max-sweeps 1000 --omega 0.5 \
  --trial-retention sparse --trial-tail 4 --trial-stride 50
```

The legacy `--damage-backend cpu` default uses SciPy damage PCG and preserves
existing policy identities. GPU mode reuses explicitly checked volume-SPD JAX
CG; an independent host residual is an audit, never a fallback solution.
JAX does not provide a CG iteration count here, so it is reported as unavailable,
not inferred from successful return. Physical gates stop a trial immediately;
coupled convergence still requires three consecutive passing sweeps. B is a
new damage model and has its own reference/history path; current results and
validation limits live in the active task.

### Direct trial storage

Fresh direct roots default to sparse retention: preserve the first generation,
latest four published generations and every 50th generation per load. Every
accepted `committed.npz`, scalar `sweep_*.json`, result/config/progress and final
field remains available. Four trailing fields preserve the predecessor needed
to inspect three consecutive convergence confirmations. Anderson secants remain
inside each retained restart snapshot.

Use `--trial-retention all` when every-sweep full-field replay is required.
`--trial-tail` must be at least four and `--trial-stride` positive. Resume inherits
the stored policy; legacy roots without storage fields retain their original
all-generation behavior and identity. Change retention in a new fork, rather
than rewriting an existing numerical policy. Pruning follows successful atomic
progress publication and checksum validation; unpinned newer generations are
retained. Per-state `storage_pruning.jsonl` records removed intermediate fields.
Pruned generations cannot be replayed as full fields; their scalar reports remain.

### Explicit warning-only direct continuation

`warning_reference.py` inherits a verified strict prefix and explores subsequent
loads in a new diagnostic root. Original coupling/change, displacement-envelope
and healing thresholds remain in reports; they warn rather than reject.
Finite/shape/dtype, exact BC, damage range, cumulative maximum history and
checked linear solves remain hard requirements. History advances from warning
endpoints under this policy, with warning ancestry retained. No clipping or
particle removal is performed. Warning snapshots are `warning.npz`, with
`accepted_reference=false`; the strict PINN comparator rejects them.

For configuration B, initialize load51 from the saved same-load Anderson q/d
guess while retaining strict state50 H as its history floor:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.warning_reference \
  --config /root/nonlocal-KKS/examples/fracture2d/configs/scale_separation_b.py \
  --source-root /root/autodl-tmp/data/fracture2d/scale_separation_20261001/reference_b_gpu_20261007 \
  --fork-after-index 50 --last-load 0.0075 --sweeps-per-load 32 --omega 0.5 \
  --seed-state /root/autodl-tmp/data/fracture2d/scale_separation_20261001/reference_b_anderson_v2_20261008/state_0051 \
  --seed-sha256 49129bed74fe1ec9c6fde0e28ed77cbf16be123deb1164c3772a5df416e3a703 \
  --damage-backend gpu \
  --output /root/autodl-tmp/data/fracture2d/scale_separation_20261001/warning_b_from50_20261008
```

Per-load budgets finish unless three original full-gate confirmations permit
early stopping or a hard/numerical failure rejects the load. Numerical seed H
never supplies the first load's history floor. Resume restores the same policy
and progress, including an interrupted endpoint publication. Only a larger
endpoint may change on resume; other retries need a new root. Sparse storage
defaults apply. Completion of this warning path is a trend result, not strict
fracture validation or a passed PINN-reference tail.

The user-selected mainline currently returns to the original validated kernel;
this independent-horizon branch is retained as a comparison experiment.

The split branch declares mechanics horizon `delta_u`, damage horizon
`delta_d`, physical `grip_extent`, and `h_grip` independently. It retains
plane strain, A2 energy, Amor history and AT2, using the same physical collar
constraints. `split_horizon_model.py` builds its validated geometry/operators;
`split_reference.py` performs strict direct loading. An explicit `--omega`
changes only the outer Picard relaxation. `--trial-from <state-directory>`
requires an accepted-prefix fork and verifies trial model/load/parent/SHA;
only q/d initialize the solve, never the trial history lower bound. Existing
reference roots are immutable; changed numerical policies use a new root.
New progress records point to generation-named trial files, so an interrupted
field publication leaves the previous resume snapshot intact. Legacy fixed-name
trial records remain readable.

An optional `--relaxation-policy anderson_residual` uses safeguarded Type-II
Anderson on the fixed accepted-history map
\(G(d)=\operatorname{AT2}(\max(H_{committed},\operatorname{Amor}(u(d))))\).
`--anderson-depth`, `--anderson-max-backtracks` and
`--anderson-sufficient-decrease` are stored in the new root's policy; `--omega`
is the Anderson mixing beta. Defaults are depth 5, beta 0.5, four backtracks
and sufficient-decrease coefficient 0.01 (merit factor 0.995).
Pivoted QR solves the residual-difference least-squares problem. Candidate
backtracking interpolates from the ordinary Picard point toward the Anderson
proposal; it never clips fields. Raw bounds/healing and strict mechanics,
history, finite and drift checks precede acceleration selection. Invalid or
non-decreasing extrapolation restarts the secant history and takes an explicitly
recorded Picard step; physical failure of that step stops and rejects the load.
The default `--anderson-restart-policy reject_or_rank` restarts after rejected
extrapolations or deficient QR. `rank_only` retains secants from valid evaluated
Picard points after rejection, but never records rejected candidate maps as
secants; deficient QR still restarts. A changed restart/decrease policy requires
a separately identified retry root.
`--anderson-rank-strategy drop_oldest` drops oldest dependent residual/state
differences until pivoted QR has full column rank, preserving the latest valid
directions. The default `restart` retains the earlier all-history restart.
Window size remains explicit; neither option establishes stability of a
nonlinear crack branch merely by reducing the algebraic residual.
Linear-solver, structural and I/O errors propagate. All original final coupled
and physical reference gates still apply for three consecutive sweeps.

The update follows Walker/Ni (2011), DOI `10.1137/10078356X`, Eq. (1.2),
Eq. (3.1) and Section 4. The normalized-infinity merit and safeguards are
project numerical policy, not a guarantee of nonlinear convergence. The
generation-pinned progress archive also stores/checks secant history for exact
restart; each new load resets it, retaining only accepted cumulative H.

`split_pinn.py` trains separate FP64 displacement/damage networks with
full-graph physics losses. The default mechanics objective minimizes the
validated fixed-d A2 energy divided by `E * load**2 * union_volume`;
`--mechanics-objective residual` selects normalized force-residual MSE instead.
Both keep the same governing equations and raw residual reports. Each load starts from the learned predecessor;
the initial damage and history are zero. Reference q/d are held-out comparisons,
and direct history is never used in neural training. Run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.split_pinn \
  --config examples/fracture2d/configs/split_pinn_fine.py
```

The config declares load count, stage budgets, relaxation, architecture and
every reporting threshold. Research advancement requires damage RMSE <= 0.01
and reaction relative error <= 0.05. Mechanics, coupled AT2, displacement,
iteration change, range, healing, history and drift failures are retained as
warnings with raw values and original limits, following the selected research
policy. Nonfinite fields/losses, wrong input identity and violated exact BCs
stop execution. A successor of a warning state inherits that provenance even
if its own checks pass. `strict_accepted` means all configured checks and the
ancestry pass; it does not assert independent physical validation.
Online first-passage stopping is restricted to logged reports in the final
mechanics block. It uses these two comparison gates, never reference values in
the loss, and does not retroactively relabel completed rejected trials. Small
absolute damage RMSE at low load is not evidence of crack-growth accuracy;
maximum local damage error and complete fields remain available.

Outputs under `tf-logs/fracture2d/<run>/` include `reaction_curve.json/png`,
per-load `damage_field.png`, per-sweep fields/comparisons, TensorBoard events,
and identity-bound neural/optimizer snapshots. Resume with
`--resume` in the same config; `--last-index N` may extend the endpoint while
preserving prior numerical identities and snapshot bytes. Use a new root with
`--fork-from <neural-root> --fork-after-index N` for an explicit numerical-policy
retry from a verified learned prefix; it never copies direct predecessor fields.
Interrupted loads restore the last
completed training block and repeat the unfinished block; completed rejected
loads require a new explicit trial root. Reference data remain under `data`.

`split_pinn_report.py` recomputes comparison-only damage errors for a completed
research prefix, verifying cloud/comparator identity and the recorded sweep
RMSE. Besides nodal RMSE it reports volume-weighted RMSE/relative L2, peak
magnitudes/locations, and local errors in an explicit radius-`l0` disk about
the benchmark notch tip `(0.5, 0.5)`. These are diagnostics, not new losses or
acceptance thresholds. Run on a completed segment with:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python \
  -m examples.fracture2d.split_pinn_report \
  --root <neural-run-root> --last-index <completed-index> \
  --output <neural-run-root>/damage_audit.json --plots
```

`--plots` also writes a shared-color-scale damage/reference/error crop with
half-width `3*l0` about the notch tip; filenames bind the audited endpoint.

### Optional speed trial

`configs/split_pinn_fast.py` declares an independent learned-prefix fork with
`train.operator_mode="analytic_padded_damage"`, 200 mechanics / 500 damage
updates per block, and the same three sweeps and primary comparisons. This
reduced budget is validated in the small-load segment; it is not a convergence
budget for critical fracture. Keep the full-budget config for explicit
refinement trials when field/reaction quality deteriorates.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.split_pinn \
  --config examples/fracture2d/configs/split_pinn_fast.py
```

The optional operator selector supports:
- `autodiff`: original operators and automatic differentiation; legacy
  schema-v1 configs without the selector retain this path and their identity.
- `analytic`: unchanged forward operators with analytic reverse rules.
- `analytic_padded_damage`: analytic mechanics reverse rule and deterministic
  padded-row evaluation/reverse rule of the identical damage CSR coefficients.

Analytic modes require the energy mechanics objective and FP64. The q gradient
is the already verified assembled A2 residual; edge-factor gradient is strain
energy density. Reciprocal damage coefficients satisfy volume symmetry, giving
the Euclidean adjoint \(D^\top z = V D(V^{-1}z)\). Padded sums change only the
floating-point reduction order; values, parameter gradients and single Adam
updates are checked against the original implementation. Reference fields are
not optimizer targets. Changing the implementation or budgets uses a new root,
with old numerical snapshots left intact.

`split_pinn_benchmark.py` compares synchronized, warmed GPU updates at the same
learned checkpoint. `--config`, `--root`, `--state-index`, `--output`, `--warmup`
and `--repeats` are explicit. It records compilation separately and verifies
gradient/update agreement; its compiled-buffer estimates are not total GPU
allocator memory. Benchmarks include the non-adopted padded mechanics variant
for reproducibility; it gave no advantage over analytic scan assembly.

Each new completed load writes `timing.json`, separating model build and load
wall time, with per-block update counts/times. First-load timing includes JIT,
logging, checkpoints and figures; later loads reuse the compiled kernels.
Batch continuation amortizes model construction. Speed studies must report
damage/reaction and local errors as well as runtime, and retain warning labels.

## Minimal numerical refinement of the original PINN kernel

`legacy_coupling_refine.py` restores the verified original state-59 neural
prefix and a completed fixed-load-60 diagnostic as a **separate numerical
initial guess**. The latter supplies networks, optimizer states and the
damage iterate only; its H never becomes the accepted cumulative history
floor. Config and seed path/SHA bind a new output root. Original point cloud,
horizon, A2/Amor/AT2 equations, hard BCs and FP64 operator kernels are retained.

The optional `train.stagger_lr_policy="linear_tail"` scales Adam updates
linearly over a declared final block fraction to a declared final ratio.
`stagger_tail_fraction` and `stagger_tail_ratio` must be explicit in (0,1].
Legacy configs without this policy retain their constant updates and identities.
This numerical polishing does not change any residual or introduce reference
training targets. TensorBoard uses a separately declared sibling root.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.legacy_coupling_refine \
  --source-root <original-strict-59-root> \
  --seed-state <completed-fixed-load-60-state-directory> \
  --output-root <new-refinement-root> --tensorboard-root <events-root> \
  --sweeps 12 --tail-fraction 0.25 --tail-ratio 0.1
```

Each sweep stores unchanged physical reports and AT2 fixed-H/relaxation/current-H
decomposition. A separate `readiness_*.json` records research coupled readiness:
relative AT2 <=1e-4, adjacent damage change <=1e-4, q change/load <=1e-4,
H and reaction relative changes <=1e-3, existing mechanics/physical checks,
for three consecutive sweeps. These are declared neural diagnostic thresholds,
not the direct reference's 1e-6 criterion or retroactive acceptance of old
states. The run completes its explicit budget and preserves warning status;
it does not advance an unconverged trial's history to a new load.
`--coupling-omega` explicitly tests a different numerical fixed-point mixing
factor in (0,1]; omission keeps the adopted 0.5 update. This changes the
iteration, not the stationary equation or operator. `--after-stop` enables
report-time termination of mechanics-after only when the current report passes
all declared coupled/adjacent checks and the old physical/mechanics gates.
Three completed outer confirmations remain required. These controls are bound
to a new root and do not retroactively select an old passing report.

### Coupled-ready old-kernel continuation

`legacy_coupled_loading.py` solves one explicit state in 61..75 from a
SHA-bound diagnostic predecessor at the immediately previous load. It first
rechecks the strict-59 ancestry, completed checkpoint, recorded full fields,
readiness counters and original physical residuals. Three consecutive passing
outer sweeps are required to advance; physical-gate passage alone is insufficient.
The predecessor's **neural H/d** become the new load's cumulative history and
healing floor. Warning-reference fields remain comparisons only.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.legacy_coupled_loading \
  --source-root <original-strict-59-root> \
  --parent-state <coupled-ready-previous-state-directory> \
  --parent-sha256 <diagnostic-checkpoint-sha256> \
  --state-index 61 --sweeps 16 \
  --output-root <new-loading-root> --tensorboard-root <events-root>
```

Numerical taper, mixing, stage budgets and readiness limits are inherited from
the verified parent. Every load finishes its declared outer budget; the final
three confirmations decide readiness. The entry retains diagnostic/warning
status even when ready. `--resume` restores the same identity and persisted
sweep evidence after interruption; a completed root is immutable. An optional
`--seed-state <same-load-completed-trial-directory>` starts a new numerical
correction from an unready but bounded trial with the **same SHA-bound parent**.
Its networks/Adam/d are restored, but its H never replaces the prior-load floor.
Per-load `config.json`, full-sweep fields, coupling/readiness reports,
`latest.ckpt.zip`, usable `diagnostic.ckpt.zip` and `result.json` are stored in
`state_NNNN/`; the root also publishes the final result.

For an explicitly chosen **warning trend study**, `--warning-continue` permits
an unready but physically usable diagnostic predecessor. The flag is bound in
configuration/ancestry and must be supplied for later warning loads as well.
The existing readiness limits and three-confirmation counts remain unchanged;
failed checks are published in `coupling_warnings`, not relabeled as passing.
Neural diagnostic H/d advance under this warning policy. Nonfinite fields,
mechanics-gate failure, invalid damage range/healing/history or drift still stop
the solve. Such a path is diagnostic-only and is not strict field-equation,
fracture-growth or post-peak validation. Omission keeps the ready-parent guard.

### Common-history subproblem comparison

`subproblem_matrix.py` independently resets four direct/neural mechanics/damage
combinations to the same verified neural state-60 networks/Adam and d/H floor
at fixed loads 61/62. Load 62 is a controlled independent probe, not a loading
transition from a case-specific state 61. Direct shadow fields diagnose inner
errors; neural training uses the original equation residuals, without archived
or shadow-field regression targets. All numerical fields are FP64 on GPU.
The direct AT2 equation is volume-multiplied SPD and solved by Jacobi-CG with
explicit residual verification. A one-time CPU sparse solution checks the GPU
implementation only and is never a generated field or numerical fallback.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.subproblem_matrix \
  --source-root <strict-59-root> --parent-state <ready-state-60-directory> \
  --parent-sha256 <diagnostic-sha256> --load-indices 61 62 \
  --sweeps 8 --direct-sweeps 64 --mixing 0.5 --output <new-matrix-root>
```

Eight sweeps form the matched-budget comparison; the separately labeled
direct/direct extension tests convergence and cannot be presented as a matched
final-budget comparison. Every case stops on original physical gates or a
recorded numerical failure, while other cases independently continue. Results
include per-sweep full fields, direct shadow fields, driver-error maps, neural
states and diagnostic summaries. These artifacts are not production/accepted
checkpoints, and no case advances or commits its trial H to another load.

### Direct benchmark and sequential step-size control

`direct_path_probe.py` continues the saved direct-61 numerical q/d guess while
holding the original neural state-60 history floor, then independently compares
three genuine sequential load paths to 0.0062. Every subload must pass the
declared dimensionless AT2/d/q/H/reaction tolerance and original physical gates
for three consecutive sweeps before its H/d advance. Trial H never changes a
parent after a failed or unfinished subload. This tighter diagnostic contract
is not the full frozen direct acceptance contract or validation of neural
history ancestry. Original GPU A2/volume-SPD AT2 solves and mixing 0.5 are used.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.direct_path_probe \
  --source-root <strict-59-root> --parent-state <ready-neural-state-60> \
  --parent-sha256 <state-60-sha256> \
  --seed-fields <matrix-direct-61-fields_0064.npz> --seed-sha256 <seed-sha256> \
  --baseline-sweeps 192 --path-sweeps 256 --tolerance 1e-6 \
  --steps 1e-4 5e-5 2.5e-5 --output <new-direct-path-root>
```

Only q/d of the same-load numerical seed are used. Baseline and each path
attempt retain per-sweep fields, equation/adjacent checks and final reason;
failed paths stop independently. Results remain diagnostic-only rather than
accepted reference snapshots, and are not PINN training targets.

`objective_pair_probe.py` compares original free-force MSE and A2 elastic-energy
training at fixed neural load-61 damage. Both arms restore the same SHA-bound
network and initialize fresh Adam, with the same original learning rate,
mechanics-block budget and update taper. Force normalization retains the original
first-load affine RMS/load ratio; energy normalization uses the fixed-d affine
elastic energy. The actual neural state-60 history lower bound stays immutable.
Direct fixed-d q/driver/H are comparison-only; the objectives contain no labels.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv/bin/python -m examples.fracture2d.objective_pair_probe \
  --source-root <strict-59-root> --seed-state <neural-state-61> \
  --seed-sha256 <state-61-sha256> --steps 16000 --report-every 1000 \
  --output <new-objective-pair-root>
```

The diagnostic checks individual field energy derivatives and mixed directional
derivatives of mechanics versus volume-weighted frozen/current-history damage.
It reports full-domain and notch-neighborhood driver errors, cumulative-history
errors, original residuals, reaction and drift; a reported drift stops that arm.
The notch neighborhood is a radius `2*l0` around `(0.5,0.5)`. Fields, networks,
optimizer state, policy and audits are saved. No trial H or trained state is
committed; fixed-d performance is not coupled-fracture convergence. Current
history finite differences explicitly report agreement because max-history
switches can cross a nonsmooth surface. This is not an SGE model implementation.

## Reference solutions and data

The case ships with two independent reference solutions and the frozen model
inputs. Load all archives with `np.load(..., allow_pickle=False)`.

| What | Path | SHA256 |
|---|---|---|
| Adopted solver entry (GPU-only) | `examples/fracture2d/nonlocal_baseline.py` | — |
| Nonlocal material cloud (3395 nodes, `d048` graph) | `/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz` | `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267` |
| Nonlocal reference solution (compact, fresh) | `/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz` | `5f3bc12bc4bc31718cce4c4aed18d47314fd80e3a207bec0180ac6cdd93f7bab` |
| Nonlocal reference archive (research) | `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz` | `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc` |
| Local FEM reference (point anchor, warning mode) | `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz` | `e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17` |
| FEM figures + config | `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/` (`config.json`, `reaction_displacement.png`, `representative_fields_*.png`, `stress_strain.png`) | — |
| PINN task | [`docs/tasks/fracture2d-nonlocal-pinn.md`](../../docs/tasks/fracture2d-nonlocal-pinn.md) | — |
| Archived direct-stage record | [`docs/tasks/archive/fracture2d-lean-direct-solver.md`](../../docs/tasks/archive/fracture2d-lean-direct-solver.md) | — |

Archive schemas (see [Adopted solution and figures](#adopted-solution-and-figures)
for the provenance and limitations):

- **material cloud** — `coordinates (3395,2) f8`, `triangles (6284,3) i8`,
  `volumes (3395,) f8`, `bottom`/`top (3395,) bool`, `anchor_particle int`,
  `d048 (E,2) i8`.
- **nonlocal compact solution** — `loads (76,) f8` (`0 -> 0.0075`),
  `union_q (76, 7402) f8` (material prefix is the first `2*3395 = 6790`
  columns), `damage (76, 3395) f8`, `history (76, 3395) f8`,
  `top_reaction (76,) f8`, `acceptance_status (76,)`, `warning (76,) bool`,
  `coordinates`, `triangles`, `metadata_json` (contains the exact `scheme`).
- **local FEM archive** — `u_bar (127,) f8` (`0 -> 0.0075`),
  `reaction_y (127,) f8`, `snapshot_u_bar (76,) f8`, `u (76, 3395, 2) f8`,
  `d (76, 3395) f8`, cellwise `H (76, 6284)` with `H_coords (6284,2)`,
  `sigma_xx/yy/xy (76, 6284)`, `gate_warning (127,) bool`, plus the material
  and geometry scalars (`E`, `nu`, `Gc`, `l0`, `k_res`, `L0`,
  `notch_*`, `history_method`, `plane_assumption`, `gate_mode`). It is
  `plane_strain`, `gate_mode="warning"` with `13` warning loads of `127`; peak
  reaction `642.4857295` at `0.0057`, terminal `3.97718` at `0.0075`.

## Problem

The material domain is the unit square with a finite-width slot opening on the
left boundary. The slot centerline is `y=0.5`, its total width is `2*l0`, and
its right end is a semicircle of radius `l0`. The foremost point of the tip is
`(0.5, 0.5)`, so the semicircle center is `(0.5-l0, 0.5)` with `l0=0.024`.

The calculation uses plane strain. The local FEM reference applies a vertically
constrained bottom edge, one horizontal point anchor at `(0,0)`, and a vertical
displacement on the top edge:

$$
u_y=0\quad\text{on }y=0,
\qquad u_x(0,0)=0,
\qquad
u_y=\bar u\quad\text{on }y=1.
$$

In the adopted direct nonlocal scheme the unit square is not constrained
directly. Structured exterior collars ("volume grips") are added below `y=0`
and above `y=1`, and the prescribed displacement is transferred through every
added nonmaterial collar particle; material interface particles remain free.
The point anchor and loading contract are given in
[Constraints, load coordinate, and reaction](#constraints-load-coordinate-and-reaction).

## Nonlocal discretization

### Material particle cloud and visibility graph

The production material cloud is **frozen** at
`/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz`
and is loaded directly by the solver with `np.load(..., allow_pickle=False)`.
It contains the tagged Gmsh mesh data: `coordinates`, `triangles`, barycentric
volumes, `bottom`/`top` masks, the lower-left `anchor_particle`, and the frozen
`d048` visibility graph.

The historical loader (`nonlocal_geometry.py`, with `load_gmsh_particle_cloud`,
`ROUNDED_SLOT_TAGS`, the polygonized-slot visibility contract, and the
`64*eps*L_geom` machine floor) is parked under `_legacy/` and remains in git
history. The cloud's barycentric nodal volumes are

$$
V_i=\frac{1}{3}\sum_{e\ni i}A_e .
$$

The graph is reciprocal, self-edge-free, and strictly inside the horizon
(`0 < r_ij < delta`); every retained or removed pair appears in both directions.
The fracture2d direct path is frozen to float64 geometry, volumes, graph
arrays, and operator inputs with JAX x64 enabled before construction.

### Weighted nonlocal damage operator

The nonlocal damage operator is the shared
`pinn.operators.ExplicitVolumeSparseD` (the same operator the PINN stage
reuses), constructed from the material cloud with the frozen cubic kernel:

```python
from pinn.operators import ExplicitVolumeSparseD

operator = ExplicitVolumeSparseD(
    cloud["coordinates"],
    cloud["volumes"],
    cloud["d048"],
    delta,
    "cubic",
)
```

For the 2D cubic kernel,

$$
(D_\delta d)_i=\sum_{j:(i,j)\in\mathcal E}
J_\delta(r_{ij})V_j(d_i-d_j),
\qquad
J_\delta(r)=\frac{120}{\pi\delta^4}(1-r/\delta)^2
\quad(0<r<\delta).
$$

With `M=diag(volumes)` the reciprocal graph satisfies `M @ D = (M @ D).T`, so
the fixed-history damage problem is the Euclidean SPD system

$$
S=M\left[\operatorname{diag}(G_c/\ell_0+2H)+G_c\ell_0D_\delta\right],
\qquad
S d=2MH .
$$

### Bond-associated correspondence mechanics

The solver builds one target-inclusive bond-associated support per directed
bond of the union graph. For each directed bond `(i,j)`:

$$
\mathcal C_{i|j}=\{j\}\cup\{k\ne i,j:(i,k)\in\mathcal E\ \text{and}\ (j,k)\in\mathcal E\}.
$$

The support shape tensor `K`, the endpoint coefficients `beta`, the target
vectors `xi`, and the normalized support weights `eta` are frozen per directed
bond from the cubic kernel and the particle volumes. Each bond deformation
gradient is `F=A K^{-1}`; the small-strain tensor is `epsilon=sym(F-I)`. With
the plane-strain Lamé constants `lambda`, `mu`,

$$
\psi=\tfrac12\lambda(\operatorname{tr}\epsilon)^2+\mu\,\epsilon:\epsilon,
\qquad
P=\lambda(\operatorname{tr}\epsilon)I+2\mu\epsilon .
$$

Rank-deficient shape tensors fail before inversion; there is no pseudo-inverse,
determinant clipping, kernel floor, stabilization, or fallback.

### Engineering volume grips (union cloud)

The solver copies the material cloud as an **exact prefix** and adds structured
collar particles without remeshing the material triangles. Collar offsets are

```text
k * h_g  for k*h_g < delta,  followed by the exact delta
```

so `h_g=0.02`, `delta=0.048` give offsets `[0.02, 0.04, 0.048]`: three layers
below `y=0` and three above `y=1`. The union volume measure is the sum of two
phase measures,

$$
V_i^{\star}=V_i^{m}+V_i^{g},
$$

where `V_i^m` is the material barycentric volume (zero on collar particles) and
`V_i^g` is the grip-phase volume (zero on material particles). Each grip pair
has total area exactly `2*delta`, so the union area is the material area plus
`2*delta`. The union graph is rebuilt by the same visibility rule on the union
triangles, and the material-induced subgraph must equal the frozen `d048` graph.

## Governing equations (adopted engineering-grip scheme)

### Phase-measured union energy

Mechanics uses the project-owned phase-measured A2 energy on the union cloud.
For union displacement `q` and material damage `d_m`,

$$
\Pi_{A2}(q;d_m)
=\sum_{i}\left[V_i^{m}g(d_i)+V_i^{g}g(0)\right]
\sum_{j}\eta_{ij}^{\star}\,\psi_{i|j}^{\star}(q),
\qquad
g(d)=(1-d)^2+k_{\mathrm{res}} .
$$

Material particles degrade with their own damage; grip particles stay intact and
never receive damage or history fields. This is the direct-baseline hybrid
convention: the complete undamaged elastic energy is degraded in mechanics,
while the history driver uses the Amor volumetric-deviatoric positive split.

### Amor history

For every union bond strain embedded in 3D with `epsilon_zz=0`,

$$
\psi^{+}=\frac12K\langle\operatorname{tr}\boldsymbol\epsilon^{(3)}\rangle_+^2
+\mu\,\boldsymbol\epsilon_{\mathrm{dev}}^{(3)}:
\boldsymbol\epsilon_{\mathrm{dev}}^{(3)},
\qquad
K=\lambda+\frac{2\mu}{3}.
$$

The source-centered driver is eta-averaged on union edges but retained only at
material source centers:

$$
\Psi^{+}_{m,a}(q)=\sum_{j:(a,j)\in\mathcal E^{\star}}
\eta^{\star}_{aj}\,\psi^{+}_{a|j}(q),\qquad a<N_m ,
$$

and the committed history is the pure maximum update

$$
H_n=\max\left(H_{n-1},\Psi^{+}_{m,a}(q_n)\right).
$$

Material-to-grip target edges contribute to material history; grip-source
centers construct no history. No history is accumulated across sweeps.

### Nonlocal AT2 damage

At fixed history, the material damage equation is the nonlocal AT2 system on the
material cloud and its frozen `d048` graph:

$$
\frac{G_c}{\ell_0}d
+G_c\ell_0\,D_\delta d
+2H d
=2H ,
\qquad\text{i.e.}\qquad
S d=2MH .
$$

Damage is never clipped or projected; raw range and healing violations are
reported and are hard failures in strict mode. Mechanics uses the hybrid stress
`sigma=g(d) sigma_0` through the center weights above; no second
irreversibility constraint or damage bound is added.

### Constraints, load coordinate, and reaction

- every added nonmaterial collar particle has its vertical DOF prescribed:
  bottom collar `u_y=0`, top collar `u_y=\bar u`; collar horizontal DOFs are
  free;
- material particles on `y=0` and `y=1` remain completely free vertically;
- only the original material anchor particle `(0,0)` has `u_x=0`;
- all other boundaries are natural traction boundaries;
- damage uses natural zero-flux conditions.

The continuation coordinate is the prescribed collar displacement `u_bar`. The
top reaction is the residual sum over the complete top volume grip.

## Numerical algorithm

A committed state is `(u_bar, q, d, H)`. For each target load and each stagger
sweep:

1. build a fresh fixed-damage union view at `d_iter` and solve mechanics-before
   at the target load on the GPU;
2. set `H_used=max(H_committed, Psi_plus_m(q_before))`;
3. solve the exact fixed-history damage equation for `d_solved`;
4. under-relax the candidate, `d_trial = d_iter + omega * (d_solved - d_iter)`
   with `omega=0.5`;
5. solve mechanics-after at `d_trial` on the GPU;
6. set `H_next=max(H_committed, Psi_plus_m(q_after))` and evaluate every gate.

A sweep is accepted in strict mode only when every coupled gate passes. The
declared gates are displacement, damage, history, reaction, committed damage
residual, raw damage range, healing, history monotonicity, force, and moment,
plus the finite/mechanics/damage-equation/globalization hard checks. When the
stagger budget is exhausted, the separate **stabilized-warning** acceptance may
commit the last complete finite candidate and records the exact bypassed gate
list; non-finite values, failed mechanics/CG solves, structural errors, and
failed work checks remain hard failures. Cutbacks halve the increment
(`cutback_factor=0.5`, up to `8`), down to an endpoint-snapped
`du_min=6.25e-6`. A one-time centered zero-damage work check validates the
mechanics work at `load=1e-4`, `epsilon=1e-5`, `relative_limit=1e-7`.

The solver is **GPU-only**: `JaxA2CPUReplayAdapter` is required, every device
solve is replayed through a fresh CPU A2 view for energy, residual, reactions,
force/moment, and the explicit free-residual gate, and only that
CPU-authoritative result backs the acceptance gates. The CPU CG reference path
and all non-adopted relaxation policies (Anderson and backtracking variants),
matched-load schedules, and archive-budget machinery were removed.

### Parameters

| Symbol / field | Value | Role |
|---|---|---|
| `delta` | `0.048` | nonlocal horizon (material and union graph) |
| `h_g` | `0.02` | collar layer spacing; offsets `[0.02, 0.04, 0.048]` |
| material particles | `3395` | material cloud from the frozen Gmsh cloud |
| `l0` | `0.024` | phase-field length; notch half-width and tip radius |
| `Gc` | `2.7` | fracture energy |
| `youngs_modulus` | `210.0e3` | `E` |
| `poisson_ratio` | `0.3` | `nu` |
| `k_res` | `1.0e-8` | residual stiffness in `g(d)` |
| `rank_rtol` | `1.0e-12` | BA shape-tensor rank tolerance |
| `visibility_tolerance` | `1.0e-12` | graph visibility input tolerance |
| kernel | `cubic` | `J_delta` above |
| `du_initial` | `1.0e-4` | nominal load increment |
| `du_min` | `6.25e-6` | minimum increment |
| `cutback_factor` / `max_cutbacks` | `0.5` / `8` | cutback schedule |
| `max_stagger_iterations` | `144` | sweeps per attempt |
| `omega` | `0.5` | fixed damped-Picard relaxation |
| `cg_rtol` / `cg_atol` / `cg_maxiter` | `1e-10` / `1e-13` / `3000` | CG |
| coupled gate tolerance | `atol=1e-9, rtol=1e-8, floor=1.0` | displacement/damage/history/reaction/force/moment |
| damage residual tolerance | `atol=1e-10, rtol=1e-8, floor=1.0` | committed damage residual |
| `healing_atol` / `damage_range_atol` | `1e-10` / `1e-10` | raw range and healing |
| work check | `1e-4` / `1e-5` / `1e-7` | centered zero-damage work |
| `acceptance_mode` | `stabilized_warning` | honest warning labeling |
| `load_final` | `0.0075` | adopted endpoint |

All values above are the defaults of `nonlocal_baseline.py`; they reproduce the
adopted archived scheme field-by-field.

## Adopted solution and figures

The adopted complete solution covers `u_bar = 0 -> 0.0075`:

- fresh re-solve:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`,
  SHA256 `5f3bc12bc4bc31718cce4c4aed18d47314fd80e3a207bec0180ac6cdd93f7bab`;
- original research archive:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`,
  SHA256 `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`.

Statistics: `76` states (initial + `65` strict + `10` warning-accepted); peak
reaction `641.34017` at `0.0061` (local FEM warning reference: `642.48573` at
`0.0057`, archive path in [Reference solutions and data](#reference-solutions-and-data));
terminal reaction `69.15426` at `0.0075` (FEM: `3.97718`). Drift
points (`|u_i|/u_bar > 1.08`) are `0` through `0.0061`, `2` at `0.0062`,
`3/5/10` at `0.0064/0.0065/0.0066`, and `28`-`31` from `0.0067` to `0.0075`.
At the endpoint the drift-excluded ranges are `d in [6.57e-6, 0.98954]` and
`|u| in [3.26e-7, 7.726e-3]`, while the drift-inclusive maximum `|u|` is
`0.142`. Figures (coolwarm, per-panel color bars, drift points drawn as black
open circles but excluded from the color ranges):

- `.../direct_nonlocal_final/fields_drift_excluded_ranges.png`
- `.../direct_nonlocal_final/response_curve.png`
- `.../direct_nonlocal_final/fresh_v1_figures/` (fresh re-solve)

The archived solution is explicitly `field_equations_validated=false` and
`validated=false`; it is a diagnostic warning reference, not a strict or
field-validated production artifact.

## Nonlocal PINN (in progress)

### Operator and model reuse

The FP32 PINN retains the same material particles, union-collar construction,
hard constraints, A2 energy/residual assembly, Amor and AT2 laws, and physical
parameters as the direct solver. Because float32 rounding changes strict
`r < delta` membership, [`fp32.py`](fp32.py) independently rebuilds the
union/material graph, phase quadrature, BA supports and cubic damage operator.
Its graph is **not identical** to the frozen FP64 direct graph. The adopted
direct implementation remains [`nonlocal_baseline.py`](nonlocal_baseline.py):

| Need | Symbol |
|---|---|
| Union cloud, collars, constraints, `d048` prefix identity | `build_union_geometry_from_arrays`, `UnionGripGeometry`, `AffineConstraints`, `build_volume_grip_constraints`, `physical_interface_line_weights` |
| Frozen model (supports, shape tensors, static identity) | `build_engineering_grip_model`, `EngineeringGripModel`, `A2StaticMechanics` |
| Fixed-damage A2 energy / residual / tangent evaluation | `A2FixedDamageView` (`.energy`, `.residual`, `.tangent_action`, `.evaluate`) |
| Amor material-source history driver | `compute_material_source_amor` |
| Nonlocal damage operator (2D cubic kernel) | `pinn.operators.ExplicitVolumeSparseD` |
| Adopted reference generator (GPU) | `run_stabilized_engineering_grip`, `JaxA2CPUReplayAdapter` |
| Rebuilt FP32 PINN graph and residual-only mechanics backend | `fp32.build_fp32_model`, `fp32.FP32A2Backend` |

The reference model is still built in FP64 with JAX x64 enabled; the network,
new graph, BA A2/Amor residuals, and material AT2 operator are all FP32 during
training. The original `d048` graph is not cast into FP32. On this material
cloud the rebuilt union candidate pairs differ by `+209/-76` and the material
directed edges by `+76/-152`; maximum FP32 BA condition is `20.24`. The first
archived state produces FP32 reaction `12.548` versus `12.675` when evaluated
with the original FP64 operator. This is a discrete-model difference, not a
training error.

### Strong-form residuals

With union displacement `q`, material damage `d`, committed history `H_c`, the
A2 energy residual `r_{A2}`, and the Amor material-source driver
`\Psi^+_{m,a}`, the training residuals are

$$
r_q=\bigl[\,r_{A2}(q;d)\,\bigr]_{\mathrm{free}},\qquad
\rho=\frac{G_c}{\ell_0}d+G_c\ell_0D_\delta d+2H(d-1),
\qquad
r_H=H-\max\bigl(H_c,\Psi^+_{m,a}(q)\bigr).
$$

The network displacement is affinely lifted to impose bottom nonmaterial collar
`u_y=0`, top nonmaterial collar `u_y=u_bar`, and the original material anchor
`u_x=0` **exactly**, with no BC penalty. Collar `u_x` and all material top/bottom
`u_y` remain free; all other boundaries are natural. Damage has the same
material-only natural zero-flux graph treatment, without hard boundary values.
See [Constraints, load coordinate, and reaction](#constraints-load-coordinate-and-reaction).
Every symbol above is defined in
[Governing equations](#governing-equations-adopted-engineering-grip-scheme);
no new kernel, split, or degradation is introduced by the PINN.

### Reference comparison protocol

- The nonlocal compact solution and the FEM archive share the same `3395`
  material nodes and `6284` triangles, so per-state field comparison is
  node-wise after reshaping: `union_q[:, :6790] -> (76, 3395, 2)`.
- **Strict prefix** `u_bar <= 0.0059`: `59` strictly accepted reference states.
  Use these for initial quantitative field/curve comparisons.
- **Drift-free warnings** `0.0060` and `0.0061`: warning-accepted even though
  no material particle exceeds the drift envelope; compare diagnostically.
- **Drifting tail** `u_bar >= 0.0062`: `2 -> 31` drift points
  (`|u_i|/u_bar > 1.08`). Acceptance labels are mixed: warning at `0.0062`
  through `0.0068` and `0.0071`, strict at the other tail loads. Compare
  qualitatively only; never present this tail as converged physical truth or
  as an exact-solution training target. The archive contains `65` strict and
  `10` warning accepted states *in total*, not as consecutive regions.
- Curve targets: pre-peak reaction follows the FEM reference (`642.48573` at
  `0.0057`); the nonlocal reference peak is `641.34017` at `0.0061`. The
  terminal mismatch (`69.15` vs FEM `3.98` at `0.0075`) is an accepted
  direct-solver limitation, not a PINN target.

### Training convergence criteria

- The direct FP64 solver's reference gates (not yet demonstrated achievable by
  the separately rebuilt FP32 PINN) are:
  displacement/damage/history/reaction/force/moment `atol=1e-9, rtol=1e-8,
  floor=1.0`; committed damage residual `atol=1e-10, rtol=1e-8, floor=1.0`;
  raw damage range and healing `1e-10` each; linear solves `cg_rtol=1e-10,
  cg_atol=1e-13, cg_maxiter=3000`.
- A more demanding **research milestone**, distinct from the user-approved
  strict-prefix acceptance gates below, is material-field relative `L2` below
  `1e-3` for `u`, `d` and `H` at `u_bar=1e-4, 1e-3, 0.0058, 0.0059`.
  Current accepted PINN states do not imply that this milestone has been met.
  At `0.0061` the warning-accepted comparison is diagnostic only, regardless
  of field error.
- Report residual norms, field errors, and reaction together. A low training
  loss alone is not convergence evidence. Any changed gate requires an explicit
  new numerical-policy root; never retroactively accept a failed state.
- Keep training/evaluation shapes fixed in JIT paths and precompute the graph,
  supports, constraints, and kernel data outside the training step (see
  `docs/JAX_GUIDELINES.md`).

## Usage

The solver requires a visible JAX GPU; there is no CPU fallback.

```bash
source .venv/bin/activate
python -m examples.fracture2d.nonlocal_baseline run \
  --output /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz \
  --progress --progress-every 20
python -m examples.fracture2d.nonlocal_baseline verify \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz
python -m examples.fracture2d.nonlocal_baseline plot \
  --archive /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/solution.npz \
  --output-dir /root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/figures
```

The first PINN implementation is a **single-load full-graph experiment**, not
an accepted load-continuation curve. It uses the separately rebuilt FP32 model
and the previous archived history as context, never the current fields as loss targets.
Run on a strictly accepted prefix state (indices `1..59`) with the JAX virtual
environment and `PYTHONPATH=src`:

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.train \
  --config=examples/fracture2d/configs/train_base.py \
  --config.data.state_index=1 --config.train.steps=100 \
  --config.train.report_every=10
```

Both entries load `configs/train_base.py` through the shared
`ml_collections` config flag (`--config=<file>`), as in the other examples;
any field can be overridden with `--config.<path>=<value>`.
`configs/train_base.py` declares the frozen cloud/reference hashes, physical
parameters, FP32 network, optimizer, and fixed **optimization-only** loss scales.
`--config.loss.scaling=fixed` retains the original explicit scales; the default
`initial_driver_rms` freezes the initial A2 residual RMS, Amor-driver RMS and
twice the driver RMS for mechanics, history and damage respectively.
Networks are built by `_build_network` from `pinn.arch.make_pinn` with the
explicit `arch.*` fields (`type`, `hidden_dim`, `num_layers`, `activation`,
`use_spatial_emb`, `spatial_emb_dim`, `spatial_scale`, `multi_head`,
`tensor_split`); the single-network `train.py` uses four outputs
(`u_x, u_y, d, H`), while the stagger diagnostic uses separate two-output and
one-output networks. The final output layer is initialized to zero so each
field starts at the affine lift.
The script prints raw A2/AT2/Amor residual maxima, material `u/d/H` errors,
reaction, range/healing diagnostics, and the **entire** warning/drift tail with
each load's archived acceptance label. A short smoke run is not evidence of
convergence. Strict-prefix accepted-state advancement is available in
`continuation.py`.

The two governing equations are **A2 displacement equilibrium and AT2 damage**.
Amor history is an algebraic committed-history maximum, not a third PDE. An
additional single-load stagger diagnostic alternates two independent
networks (mechanics with damage fixed, then damage with history fixed), uses the
same direct-scheme damage under-relaxation, re-solves mechanics, and recomputes
history from the *unchanged* committed previous state:

$$
H_{n,a}=\max\!\left(H_{n-1,a},\Psi^{+}_{m,a}(q_n)\right).
$$

Here \(H_{n-1}\) is the **accumulated history at the last accepted load**, not
just that load's instantaneous energy. It and each fixed stagger subproblem's
other field are explicitly `stop_gradient` inputs; the current Amor driver
remains differentiable with respect to the current displacement network.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.stagger \
  --config.train.dtype=float64 --config.data.state_index=1 \
  --config.stagger.mechanics_steps=12000 \
  --config.stagger.damage_steps=12000 --config.stagger.sweeps=3 \
  --config.output.dir=/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_new_run \
  --config.output.log_every=500 --config.output.checkpoint_every=1000
```

Resume an interrupted run with the same command plus
`--config.output.resume=True`.

For strict-prefix continuation, use `continuation.py` with an explicit final
state (indices `1..59` only). The first load starts from the archived zero
state; subsequent loads reuse only the previous **accepted PINN state**,
including both networks, Adam states, displacement, damage, and cumulative
Amor history. The archive supplies the load values and diagnostic comparators,
not the current-load training targets. Each load receives a separate
`state_0001/`, `state_0002/`, ... directory under `output.dir`, an atomic
`accepted.ckpt.zip` only after the final sweep passes every gate, and
independent TensorBoard events under `<tensorboard_root>/<run-name>/state_0001/`,
etc. The first rejected load stops without
changing its accepted parent. Resume the same root with
`--config.output.resume=True` to continue an interrupted load, or increase
**only** `--config.data.end_state_index` and add the resume flag to run more
strict-prefix loads. Previously accepted loads and their numerical configs
are verified and reused without retraining; their checkpoint identities are
never rewritten. A completed rejected load requires a new run with a
different numerical policy or budget.
Historical PINN runs created with `output.dir` under `data/fracture2d/pinn_*`
are physically archived at `tf-logs/fracture2d/pinn_runs/<run-name>/`.
Their original config JSON and checkpoint bytes are unchanged: resume with
the **recorded legacy paths** and the filesystem adapter resolves artifact
access to the new archive without changing config/SHA256 identity. New runs
choose tf-logs paths explicitly. The migration file inventory is
`tf-logs/fracture2d/pinn_runs/migration_manifest.json`.
With `stagger.accept_on_first_passing_report=True`, only the **last** sweep's
mechanics-after reporting points may trigger online acceptance. All configured
gates use that point's same-pass post-update fields; the network, both Adam
states, damage and cumulative history are saved at its **actual** step.
Earlier sweeps always complete; old event curves are never searched for a
favourable state after training. Set the flag to `False` to require the
final optimizer step instead.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.continuation \
  --config=examples/fracture2d/configs/train_base.py \
  --config.train.dtype=float64 --config.data.end_state_index=2 \
  --config.stagger.sweeps=3 \
  --config.stagger.mechanics_steps=12000 \
  --config.stagger.damage_steps=12000 \
  --config.output.dir=/root/autodl-tmp/tf-logs/fracture2d/pinn_continuation_fp64
```

After loads 1–2 are accepted, run the same command with
`--config.data.end_state_index=3 --config.output.resume=True` to train only
state 3 from the accepted state-2 networks and optimizer states. All other
numerical settings (including budgets, gates and output root) must match.
If state 3 completes with a failed gate, this root remains committed only
through state 2 and further resume stops there.

To change the mechanics scaling **without retraining the accepted prefix**,
start a separate continuation root using
`--config.stagger.mechanics_scale_policy=first_load_times_load_ratio`,
`--config.output.fork_from=<previous-root>`,
`--config.output.fork_after_state_index=2`,
`--config.data.end_state_index=3` and a new `--config.output.dir`. Keep the
remaining optimizer, budget and gate settings identical to the source. The
old root and its checkpoint hashes are not rewritten; the new load starts
from the verified accepted state-2 networks, optimizer states and q/d/H.
Both root and per-state source configs and the accepted predecessor digest
are checked. A rejected trial in the old root is **not** imported.
For a later policy change, the `fork_from` root can itself be a fork. The
ancestry is followed read-only for each prefix state and its exact per-state
configuration and predecessor digest are verified. For example, to change
acceptance starting at state 7 without retraining accepted states 1–6, use
`--config.output.fork_after_state_index=6` and point `fork_from` to the
accepted state-6 continuation root, with a new output directory.

The base completed-sweep gate limits are normalized mechanics loss `1e-2`,
material displacement relative L2 `5%`, and reaction relative error `5%`,
plus finite values, zero drift, damage range `1e-10`, damage healing
`stagger.healing_atol=1e-3`, and monotone history. The healing tolerance is
separate from the damage-range tolerance: it permits finite damage decreases
of up to `1e-3` relative to the previously accepted damage field and is **not**
an exact damage-irreversibility guarantee. Raw healing is always logged.
Reference-field and reaction comparisons are acceptance diagnostics only;
the optimizer loss remains the governing-equation residual.
For damage-field accuracy, `d_rmse = sqrt(mean((d_PINN - d_ref)^2))`
over all material nodes is the primary reported comparison. It is defined at
zero reference damage and is in dimensionless damage units (`0..1`).
`d_max_abs_error = max(abs(d_PINN - d_ref))` also reveals localized mismatch
near the notch that full-field RMSE can dilute. The previous `d_relative_l2`
is retained as an auxiliary trend: at early nearly-undamaged loads it can
become large solely because the reference norm is small. These fields are
diagnostics; no reference damage is used to train a network and no numerical
damage-field acceptance threshold is set by this reporting change.
For the user-approved reaction/damage-focused continuation, set
`stagger.gate_displacement_l2=False` (still report the displacement error) and
`stagger.damage_rmse_limit=1e-2`. The RMSE gate applies on the *same final
reported state* as the reaction and other physics gates. The maximum absolute
damage error remains diagnostic-only. The base config uses `True` and `0.0`
respectively (zero disables the RMSE gate) to preserve its earlier acceptance
policy. These settings change numerical identity and require a new fork root;
the previously rejected state is never retroactively accepted.
The committed diagnostic path currently reaches the full **59-state strictly
accepted reference prefix**, with each load having an identity-bound
checkpoint. That does not validate post-peak loads: `0.0060/0.0061` are
warning-accepted in the frozen direct reference and later states include
particle drift. The full per-load reaction/damage comparison and local
near-peak damage-error limitations are in the active task
[`docs/tasks/fracture2d-nonlocal-pinn.md`](../../docs/tasks/fracture2d-nonlocal-pinn.md).
With `stagger.mechanics_scale_policy=first_load_initial`, every load
recomputes the same first-load zero-output affine free-row residual RMS
(about `0.05103` on the adopted graph). This is an **absolute path-wide**
force-residual normalization, not a load-relative accuracy criterion.
`first_load_times_load_ratio` instead divides by the same first-load RMS
times the ratio of the current prescribed displacement to the first strict
load. It changes the objective and normalized gate in the same way; raw A2
force residuals, the governing equation and the `0.01` normalized limit do
not change. The load ratio is a chosen optimization/acceptance convention,
not a proof of exact load invariance after damage evolves. An archived rejected
state is not retroactively accepted under this new rule.
`per_load_warm_start` remains an explicit alternative. A rejected completed
load cannot be resumed or silently advanced. Changing **numerical settings**
requires a new output root; increasing only the final state does not.

To inspect accepted material-node displacement against the strict direct
reference, plot the same material triangulation for reference, PINN and error:

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.plot_pinn_displacement \
  --root /root/autodl-tmp/data/fracture2d/pinn_continuation_damage_rmse_from6_20260929 \
  --states 57 58 59 \
  --output-dir /root/autodl-tmp/tf-logs/fracture2d/plot_displacement_strict_prefix
```

For a crack-tip crop, rerun with a **new** output directory and
`--zoom 0.40 0.65 0.40 0.60`. The reader verifies every preceding accepted
checkpoint SHA256 and the stored per-load configuration before plotting.
Outputs are one 3×3 PNG per requested state plus a JSON file with whole-field
material displacement relative L2, each component's relative L2, vector
RMSE and the position/magnitude of the maximum vector error. Error panels
show signed `u_x/u_y` differences and the **vector error norm**, avoiding a
pointwise relative error ill-conditioned near zero displacement. Existing
figures are not overwritten. Warning-labelled states are not plotted by
this strict-prefix entry point.

### Exploratory continuation beyond the strict reference prefix

`diagnostic_continuation.py` starts from a read-only verified strict
state-59 prefix and explores archived loads 60–75 in a separate root.
It reuses both PINN networks, both Adam states and committed q/d/H.
No archived current-load displacement or damage is used as a training target.
Reference comparison values and every archived warning label remain visible,
including tail states labelled strict but outside the pre-warning prefix.
All post-59 results are **diagnostic-only**.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.diagnostic_continuation \
  --config=examples/fracture2d/configs/train_base.py \
  --config.train.dtype=float64 --config.data.end_state_index=75 \
  --config.stagger.sweeps=3 --config.stagger.mechanics_steps=16000 \
  --config.stagger.damage_steps=12000 \
  --config.stagger.mechanics_scale_policy=first_load_times_load_ratio \
  --config.stagger.gate_displacement_l2=False --config.stagger.damage_rmse_limit=0.01 \
  --config.output.log_every=4000 --config.output.checkpoint_every=8000 \
  --config.output.fork_from=/root/autodl-tmp/data/fracture2d/pinn_continuation_damage_rmse_from6_20260929 \
  --config.output.fork_after_state_index=59 \
  --config.output.dir=/root/autodl-tmp/tf-logs/fracture2d/pinn_warning_diagnostic_new_run
```

Trial advancement requires the existing finite, normalized mechanics-loss,
damage-range, bounded-healing, monotone-history and zero-drift checks.
Warning-region reference reaction/displacement/damage comparisons do **not**
gate advancement. A passing final-sweep reporting point saves
`diagnostic.ckpt.zip` under a separate schema and parent SHA256 chain;
`accepted` is always false. No `accepted.ckpt.zip` is created beyond 59.
`diagnostic_usable` means only that these advancement checks pass; it is
not evidence of coupled damage convergence or reference agreement. The
committed same-pass AT2 residual and stagger changes are reported but are
not acceptance gates in this exploratory policy.
Resume an interrupted diagnostic run with the same flags and
`--config.output.resume=True`; only its endpoint may increase within
60–75. A completed trial that fails physical checks cannot be skipped on
resume. The strict-prefix runner retains its 1–59 limit.

### Fixed-load outer-sweep experiment

To distinguish outer-coupling insufficiency from a poorly solved fixed-field
subproblem, `fixed_load_sweeps.py` verifies the original strict state-59
parent chain, then holds prescribed displacement at `0.0060` in a separate
experiment directory. It increases only the sweep budget and disables
first-passage stopping; optimizer learning rate, individual stage budgets,
relaxation and model are copied from the source configuration.

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.fixed_load_sweeps \
  --source-root /root/autodl-tmp/data/fracture2d/pinn_continuation_damage_rmse_from6_20260929 \
  --output-root /root/autodl-tmp/tf-logs/fracture2d/pinn_fixed_load60_new_experiment \
  --tensorboard-root /root/autodl-tmp/tf-logs/fracture2d/events \
  --sweeps 12
```

The sweep budget must exceed the source's budget and is fixed before
training. Each completed sweep writes `state_0060/sweep_XXXX.json`;
the final `result.json`, `sweep_summary.json` and `sweep_summary.png`
contain the diagnostics across the whole declared budget, without picking
an intermediate passing state. Add `--resume` with identical arguments for
an interrupted experiment; a completed experiment cannot be extended or
overwritten by that flag. This entry point trains only the fixed load and
uses warning-reference comparisons diagnostically. A small mechanics loss
or a usable diagnostic checkpoint still does not imply coupled AT2/stagger
convergence.

Add `--coupling-diagnostics` to a new fixed-load experiment to save each
sweep's physical fields in `state_0060/fields_XXXX.npz` and equation-term
diagnostics in `coupling_XXXX.json`. `coupling_policy.json` declares the
reporting convention and must match on resume. The dimensionless AT2 defect
is its infinity norm divided by the sum of infinity norms of the local,
nonlocal and history terms. Adjacent-sweep diagnostics measure absolute
damage change, displacement change divided by prescribed displacement,
history change divided by the larger old/new history infinity norm, and
reaction change divided by the larger old/new reaction magnitude. The first
sweep has no same-load predecessor. Zero normalization scales fail loudly.
These are read-only diagnostics, with no new acceptance gate or early stop;
raw residuals and fixed-H/relaxed/current-H decomposition remain visible.

#### Target-free crack-growth research entries

The following entries are experimental and produce **research diagnostics**,
never strict accepted-reference states. Current runs, measured failures and
the exact resume point live in `docs/tasks/fracture2d-nonlocal-pinn.md`.
All artifacts belong under `/root/autodl-tmp/tf-logs/fracture2d/`.

- `inner_probe.py`: identity-checked frozen-subproblem probes using FP64
  residual Gauss–Newton or fixed-subproblem energy L-BFGS. Required controls:
  `--root`, `--output`, `--parent-digest`, `--steps`, `--max-cg`, `--method`.
  Energy derivatives are checked against the existing physical residuals.
- `rbf_stagger.py`: frozen inherited MLPs enriched by trained Gaussian
  readouts. Geometry-owned multi-scale features enrich the tip and collars;
  the operator/graph/BCs remain fixed. Required controls include source and
  bound initial snapshot, explicit load interval/increment/sweep budget,
  `--basis`, `--mechanics-objective`, `--damage-objective`, `--damage-link`.
  A read-only `--fork-rbf` / `--fork-load-index` selects a checked research
  parent. Failed trial history cannot advance the path. Sigmoid links bound
  damage through the ansatz; arrays are not clipped.
- `joint_rbf.py`: jointly trains displacement/damage readouts using the
  same A2/AT2 residuals and exact committed-history maximum. Required controls
  include `--base`, `--source`, `--source-index`, `--parent-digest`, `--output`,
  `--load`, `--max-evaluations`, `--lsmr-maxiter`, `--lsmr-tolerance`,
  `--parameter-scaling`, `--damage-weight`, `--extra-q-units`,
  `--mechanics-weighting`, `--q-feature-source`. `--joint-parent` advances
  only from a completed physically checked research state; `--warm-trial`
  transfers neural weights inside an unchanged committed context.
  Positive optimization weights do not change unweighted physical gates.
  At history switches, the linear solve uses an explicitly recorded
  generalized derivative (0/1 off the switch, midpoint at numerical ties),
  while forward history remains the exact maximum. Same-point agreement,
  fixed-branch finite differences and JVP/VJP adjointness are checked first.
- `joint_loading.py`: runs checked joint loads in fresh processes, with
  explicit increment and budgets. `--max-reweights` bounds separate same-load
  outlier-refinement trials; all retain the original committed parent history.
  `--resume` verifies stored experiment and parent-field digests and refuses
  to skip a rejected/incomplete state. `--first-trial` can reuse a matching
  completed rejected trial read-only.
- `research_report.py`: independently recomputes raw residuals/reactions and
  renders saved fields (`--comparison pinn_only|archive_warning`). Warning
  reference fields are comparison-only; loads
  absent from the archive get PINN-only figures rather than interpolation.
- `frozen_field_audit.py`: independently solves fixed-d mechanics and fixed-H
  AT2 for diagnosis, optionally with an explicit `--oracle-sweeps` budget.
  Oracle fields **never** become PINN training targets or advancing states.
- `operator_spectrum.py`: audits the existing tangent with explicit
  `--metric euclidean|volume`, `--operators both|damaged` and `--maxiter`;
  every reported eigenpair has an explicit Ritz defect. The volume metric
  is a displacement quadrature norm, not an added inertia term.
- `degradation_audit.py`: verifies directional positive-energy bounds and
  source contributions, then solves explicit `--widths` / `--load` synthetic
  fully damaged-strip mechanics probes. These are **prescribed-damage
  diagnostics**, not AT2 solutions or simulated crack-growth states.

The research joint checks retain finite/range/history/bounded-healing and
zero-drift requirements, with mechanics normalized MSE `<=1e-2`, mechanics
normalized infinity defect `<=0.1`, and equation-term-normalized AT2 infinity
defect `<=1e-3`. Stagger research additionally requires three consecutive
passing sweeps and explicit field/reaction change limits. Optimizer termination,
one near-unit damage value, or a usable research checkpoint alone does not
establish converged crack propagation/unloading. The all-material drift
envelope remains a hard check; changing how failed crack-core displacements
are assessed requires an explicit numerical-policy/formulation decision.

Fixed-support A2 source-energy degradation does not automatically remove a
failed point from intact neighbors' kinematic supports. Consequently,
`d=1` at sampled points is not by itself a guarantee of traction-free separation
across a thin band. Crack transmission must be checked at the declared
horizon/phase-field length and cloud resolution. Positive fixed-d stiffness
does not establish a validated post-fracture interaction model or convergence
of the coupled load path. The adopted public config still has one common
horizon; distinct mechanics/damage horizons require a separately validated
model/discretization branch.

#### Independent horizon validation branch

The user-accepted engineering comparison baseline is
`configs/scale_separation_short_damage.py`: material target size `0.004`,
`delta_u=delta_d=0.008`, `physical.l0=0.024`, and unchanged material data.
`split_reference.py` supplies its strict prefix; `warning_reference.py` supplies
the separately labeled tail. Provisional engineering acceptance preserves the
saved convergence failures and warning ancestry. The original B configuration
(`delta_d=0.024`) remains a separate comparator.

Verified artifacts for this baseline are the strict source
`/root/autodl-tmp/data/fracture2d/scale_separation_20261001/b_short_damage_fresh_20261008`
and the combined-prefix warning root
`/root/autodl-tmp/data/fracture2d/scale_separation_20261001/warning_short_damage_from52_20261008`.
The latter resolves strict states 1–52 through its recorded source and stores
warning states 53–75 locally. Its terminal field SHA256 is
`be0bed22c5a0e7ce48fa6bd8f58e900aaba94e0cb793ab520ce3129255028d68`.
The PINN comparison policy must explicitly opt into these accepted warning
fields; they retain their original diagnostic status.

The local FEM mesh controls accept `0 < mesh.lc_near <= mesh.lc_far`;
equal target sizes support uniform-resolution reference comparisons.

`split_horizon_model.py` implements the explicitly separate validation model:
mechanics radius `delta_u`, material damage radius `delta_d`, physical
`grip_extent`, and ghost spacing `h_grip` are distinct required inputs.
The damage radius is the support of the scalar operator, distinct from the
AT2 regularization length `physical.l0`. Keeping `l0` fixed while changing
`delta_d` can still change the computed damage profile and fracture energy.
Smooth second-moment consistency alone does not establish sharp-interface
energy consistency for a finite, integrable nonlocal kernel; validate spatial
refinement and coupled crack transmission as well as smooth operator patches.
It retains the engineering volume-grip BVP: every added collar particle has
prescribed vertical displacement; material interface particles remain
vertically free, with one original material horizontal anchor. Geometry,
phase quadrature, both strict visibility graphs and BA rank are independently
validated. A conservative polygon broad phase accelerates graph generation;
all potentially crossing segments use the frozen exact visibility predicate.
The new model has a separate durable identity even in common-horizon controls.

`reference_cloud.py` generates tagged reference meshes/clouds using the
installed Gmsh Python worker and explicit ASCII MSH2 parsing. Required controls
are `--geo`, `--output`, `--log`, `--h-far`, `--h-near`, `--gmsh-python` (the
interpreter with Gmsh installed). Size overrides are applied after loading
the benchmark geometry, preserving its coordinates and physical groups.
Mesh/cloud and their
SHA256 provenance belong under `data/fracture2d/`; validation/PINN logs and
reports belong under `tf-logs/fracture2d/`. It preserves the original physical
boundary names and exact anchor/tip and rejects unsupported element formats.
New cloud/model identities cannot inherit old accepted fields or certify
their new loading path from the frozen reference.

TensorBoard tags follow the repository convention (`loss/*` optimization
losses, `eval/*` reference comparisons, `diag/*` residual and state
diagnostics), matching `corrosion2d`/`fracture1d`:

- `loss/mechanics_before`, `loss/damage`, `loss/mechanics_after` — the three
  stage objectives, evaluated at the reporting step from the same post-update
  fields used by every other logged value. `loss/mechanics_before` is the
  free-row A2 residual MSE with damage held at `d_iter`; `loss/damage` is the
  AT2 residual MSE with fixed `H_used`; `loss/mechanics_after` solves the
  **same A2 equation** at under-relaxed `d_trial`.
- `eval/rel_L2_u`, `eval/rel_L2_d`, `eval/rel_L2_H`, `eval/d_rmse`,
  `eval/d_max_abs_error`, `eval/reaction` — material-node comparison against
  the FP64 archived reference, plus `eval/reaction_reference` at step zero.
- `diag/mechanics_residual_inf`, `diag/damage_residual_inf`,
  `diag/history_residual_inf` — full-graph raw residuals of the current
  (q, d, trial-history) state; `diag/history_residual_inf` is zero because the
  trial history is the exact same-pass Amor maximum.
- `diag/coupled_loss` — diagnostic sum of the normalized A2 and AT2 MSEs at
  that state; **not** the stage optimizer's loss or an acceptance gate. Its
  fixed scales are logged once as `diag/mechanics_scale` and
  `diag/coupled_damage_scale`. During the damage phase, `loss/damage` uses
  fixed `H_used`, so `diag/damage_residual_inf` (current q-driven history) can
  differ.
- `diag/drift_count`, `diag/damage_min`, `diag/damage_max`,
  `diag/damage_healing_max` — state diagnostics. `gate/*` records the
  **configured** per-sweep checks and `diag/accepted` their conjunction.
  The base policy gates material displacement L2 at 5%; the reaction-relative
  limit is 5% in both the base and reaction/damage-focused policies.

Every sweep also prints the same values as JSON, including the final coupled
diagnostics. New PINN run artifacts belong under
`/root/autodl-tmp/tf-logs/fracture2d/<run-name>/`; TensorBoard events use
the sibling `/root/autodl-tmp/tf-logs/fracture2d/events/<run-name>/`.
`/root/autodl-tmp/data/` is reserved for clouds and independent reference
solutions. View the new event tree with
`tensorboard --logdir /root/autodl-tmp/tf-logs/fracture2d/events/`. The bound
network/optimizer/trial state is atomically updated at
`<output-dir>/latest.ckpt.zip`. Restart the same command with
`--config.output.resume=True` (same settings and output directory) to continue
an interrupted run. Use a new output directory for each run; do not overwrite
an existing run. On the first
load the earlier short stagger experiments were unconverged; this does not
establish what a many-thousand-step run will achieve. A proposed fixed-damage
energy-only diagnostic was interrupted and is not an active training option.

`run` writes a compact solution NPZ (loads, material fields, top reaction,
per-state acceptance status, geometry, metadata); `verify` reports state and
warning counts, peak/terminal reaction, drift counts and drift-excluded ranges;
`plot` draws `u_x`, `u_y`, `|u|`, and `d` with drift-excluded color ranges while
keeping every drift point in the solution and drawing it.

Retained focused tests:

```text
tests/test_fracture2d_nonlocal_baseline_entry.py
tests/test_fracture2d_bond_associated_algebra.py
tests/test_fracture2d_geometry.py
tests/test_fracture2d_pinn_residual.py
```

## Validation status and accepted limitations

**Both reference solutions are warning-mode, non-strict diagnostics.** The
local FEM archive is `gate_mode="warning"` (`13` of `127` loads carry
`gate_warning=True`), and the nonlocal direct solution is
`stabilized_warning` (`10` of `75` accepted states are warning commits, with
`2 -> 31` drift points). Neither is a strictly converged reference, and the
warning tail of neither solution may be presented as physical truth.

The nonlocal direct solver's remaining issues, explicitly accepted for now and
deferred:

1. **Terminal reaction is not physical.** Nonlocal `69.15` versus local FEM
   warning `3.98` at `0.0075`; the nonlocal response collapses too slowly after
   localization. Peak and approximate onset are close (`641.34` vs `642.49`),
   which is why the solution is still used as the direct reference.
2. **Ten of seventy-five accepted states are warning-accepted.** All raw gate
   defects are retained (`field_equations_validated=false`); no failed gate is
   relabeled as converged.
3. **Particle drift after severe localization.** Two material points first
   exceed the diagnostic envelope `|u_i|/u_bar > 1.08` at `0.0062`, growing to
   `31` by `0.0075` with a drift-inclusive tip displacement of `0.142`
   (amplification about `19`). Drift points stay in the equations and the
   solution; only the plotted color ranges exclude them.
4. **Healing in the unbounded warning path.** Up to `0.02915` after `0.0067`;
   accepted only as raw diagnostic provenance, never as a physical state.
5. **The original strict path never completes.** Original strict gates first
   fail near `0.0060` (damage-map defect), and the minimum-step target
   `0.006118125` stalls after `144` sweeps. No strict `0 -> 0.0075` path exists
   under the current increments and gates.
6. **Mesh and horizon convergence is open.** On an actually refined 3711-node
   mesh with `delta=0.032`, strict continuation reaches `0.0059` but `0.0060`
   still fails. Frozen-damage diagnostics at `delta=0.040/0.032` reduce the tip
   amplification `21.84 -> 10.76 -> 6.74` but do not remove the outliers.
7. **The local FEM reference is itself warning-mode.** Its strict convergence
   and mesh refinement follow-ups are also open
   ([archived local-FEM task](../../docs/tasks/archive/fracture2d-local-fem.md)).
8. **Onset/drift trigger.** The published pre-damage/damage-jump event rule
   misses the onset in both strict and warning paths; a drift-sensitive trigger
   still requires an explicit bounded numerical choice.
9. **Model-combination provenance.** Silling's intact correspondence basis,
   bond-associated stabilization, and the nonlocal AT2/Amor split are
   source-backed separately, but the damage-aware combination is a
   project-owned discrete choice; `k_res` sensitivity (`1e-8 -> 1e-5`) was
   falsified as a drift cure rather than resolved.

## Implementation layout

`nonlocal_baseline.py` is the single maintained implementation. The merged
sections are marked with `# ===== <name> =====` comments and use module-prefixed
private names (`_g_`, `_a_`, `_j_`, `_e_`, `_s_`) from the mechanical merge:

- union geometry, constraints, BA supports, and `MechanicsEvaluation`;
- A2 fixed-damage view, CG-free CPU evaluation, Amor, and fixed-history damage;
- the JAX GPU PCG backend and `JaxA2CPUReplayAdapter`;
- the engineering-grip model, gates, state, diagnostics, and work check;
- the fixed damped-Picard continuation with `stabilized_warning` acceptance;
- the `run` / `verify` / `plot` CLI.

`src/pinn/operators.py` remains the shared operator library
(`ExplicitVolumeSparseD` and the kernel registry); this file does not redefine
it. Parked material lives in `_legacy/` (ignored) and git history: the
per-module implementation, monolithic/arc/reduced solvers, the durable
workflow, the historical Gmsh loader, and the stage-specific test suites.
Historical audit detail and rejected alternatives live in
`docs/tasks/archive/` and `docs/DECISIONS.md`.
