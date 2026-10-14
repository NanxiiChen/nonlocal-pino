# 2D Rounded-Notch Fracture: Reference Solvers and Nonlocal PINN

> 阶段评价（2026-10-13）：当前结果初步可接受，仍需大量改善。现有实验代码需要大量精简、合并和整理；本阶段保存可追溯结果，不将 provisional 状态视为充分收敛。详见[阶段报告](../../docs/reports/fracture2d-initially-accepted-20261013.md)。

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

Fresh PINN loading uses `configs/split_pinn_short_damage.py` with two FP64
networks, the same fine-cloud model and analytic energy/padded damage gradients.
Its explicit `data.reference_policy="user_accepted_warning"` and
`data.reference_policy_sha256` bind the adopted warning-root policy. Legacy
configs remain strict-only. The reader verifies field hashes, warning hard
checks and immediate parent identity; reference status/report/policy hashes
enter checkpoint identity and reporting. Warning reference ancestry prevents
strict neural labeling even if the current neural checks pass.
Reference q/d enter comparisons and first-passage reporting only; reference
history is never imported into neural losses or the learned predecessor.

For an explicitly user-authorized complete warning exploration,
`configs/split_pinn_full_warning.py` forks the usable neural prefix through53
and retrains54 onward. `reporting.comparison_gate=False` retains the original
5% reaction/0.01 damage-RMSE limits and their pass/fail diagnostics but makes
them nonblocking. Reference-first-passage stopping is disabled as well, so
unconverged direct comparisons do not decide when an inner block ends.
Reports retain `comparison_checks_pass`, `comparison_gate_enforced=False`, all
secondary defects and warning ancestry; this policy never creates strict
neural acceptance. Finiteness, exact constraints, checkpoint/model/reference
identity checks and learned maximum-history semantics remain in effect.
The old rejected54 is preserved and its H is not a new parent floor.

### Direct–PINN stagger: direct mechanics / nonlocal PINN damage

`hybrid_pinn.py` uses the same FP64 fine-cloud nonlocal mechanics and AT2
operators, but trains only one damage network in a staggered algorithm. The explicit configuration
`configs/hybrid_pinn_short_damage.py` selects `train.mechanics_solver="direct"`.
Each outer sweep performs:

1. a strict fixed-d mechanical solve with `FP64A2Backend.solve_strict`;
2. `H_used=max(own preceding-load cumulative H, current direct-q driver)`,
   followed by fixed-H nonlocal AT2 damage-network training;
3. the configured damage relaxation, a second strict mechanical solve, and
   recomputation of the final history and stagger-consistency residuals.

These are separate frozen-field subproblem solves, not a simultaneous/monolithic
u/d solve or joint loss. Damage gradients do not pass through the direct solver
or H_used. The existing report key `coupled_AT2_relative` names the damage
equation defect evaluated on the final staggered fields; it is a diagnostic,
not a coupled solution algorithm.

The default two-load control keeps three outer sweeps, 2000 damage updates per
sweep, Adam `1e-4`, and `omega=0.5`. There are no mechanics-network updates.
Damage weights/Adam and the hybrid's own q/d/H warm-start later loads; reference
q/d serve only as comparisons and reference H is never loaded. Both mechanics
phases must pass the explicit CG residual check. This subproblem check does not
certify outer convergence, so reports distinguish `fixed_H_image_AT2_relative`
from `coupled_AT2_relative` and retain change/healing/drift diagnostics. Mechanical
MSE is normalized by the first-load affine residual times the reference index;
the CG norm/tolerance is the authoritative mechanical subproblem check.

The dedicated `fracture2d-direct-mechanics-pinn-damage-v1` checkpoint schema
stores one damage network/Adam state and actual fields. Phase snapshots also
store the current direct displacement, previous damage and H_used so a resume
after damage does not repeat damage training or substitute a neural q. The
runner can start fresh or inherit an explicit SHA-verified matching hybrid
prefix through output.fork_from/fork_after_index. Full-neural checkpoints are
not compatible with this one-network schema. Completed endpoint-only resume
verifies model, reference, config and parent hashes. The default comparison gate is disabled, with the
original thresholds still reported; completed loading remains warning research.

```bash
source .venv/bin/activate
python -m examples.fracture2d.hybrid_pinn \
  --config examples/fracture2d/configs/hybrid_pinn_short_damage.py
# Re-read completed states without retraining, or extend the endpoint:
python -m examples.fracture2d.hybrid_pinn \
  --config examples/fracture2d/configs/hybrid_pinn_short_damage.py --resume
```

The initial result root is
`tf-logs/fracture2d/hybrid_short_damage_first2_20261010`; events are in the sibling
`events/hybrid_short_damage_first2_20261010`. Each saved sweep includes q/d/H,
H_used, the unrelaxed damage image and previous damage for independent checks.

The larger-network candidate `configs/hybrid_pinn_swish_fourier.py` follows
fracture1d's Swish plus trainable Fourier-feature method. The1D baseline has
200-wide4-layer MLP,32 Fourier frequencies at scale16 and33 local hat
coefficients. The2D candidate uses256-wide6-layer MLP,64 Fourier frequencies
at the same initial scale16:128 sine/cosine input features from a trainable
2x64 frequency kernel, and362369 trainable parameters. The hybrid Adam update
includes the Fourier kernel together with all trunk/output weights.

This candidate retains the same2D physical/output formulation, Adam1e-4,
2000 damage updates per block and3 stagger sweeps. Its user-selected loading
schedule is `reference_indices=[40,41,...,75]`: start fresh at0.0040, then
increments0.0001. Own d/H start fromzero; skipped reference histories are not
imported. This changed loading path is not asserted equivalent to the original
uniform path, so a curve difference cannot be attributed to architecture alone.
Its default endpoint is2 (0.0040/0.0041) for real-grid validation; full endpoint
ordinal36 reaches0.0075. It has distinct output
and event roots and must start fresh. Architecture changes cannot resume an
old128x4 checkpoint. Parameter count/forward construction alone does not
validate its damage/unloading accuracy. The1D local hats are specific to a
fixed1D grid/imperfection patch; a2D local basis for an advancing crack needs
separate design/support validation and is not included in this candidate.

```bash
python -m examples.fracture2d.hybrid_pinn \
  --config examples/fracture2d/configs/hybrid_pinn_swish_fourier.py
```

The narrower user-selected trial is
`configs/hybrid_pinn_swish_fourier_128.py`:128-wide6-layer Swish with the same
64 trainable Fourier frequencies/scale16, FP64 and99329 parameters. Its explicit
schedule `[48,49,...,75]` starts fresh at0.0048 then increments0.0001; the initial
endpoint2 tests0.0048/0.0049 and full selected endpoint28 reaches0.0075.
It uses its own output/events root and zero-floor history, not a256-wide neural
checkpoint or skipped reference histories. Both width and starting path differ
from the preceding candidate; field accuracy/runtime need actual validation.
The optimizer/physical/stagger settings are inherited unchanged.

```bash
python -m examples.fracture2d.hybrid_pinn \
  --config examples/fracture2d/configs/hybrid_pinn_swish_fourier_128.py \
  --last-index 28
```

### Convergence-controlled outer stagger

The independent `local_damage_trial.py` entry isolates local mechanics from the
existing A2 model. `configs/local_damage_trial.py` retains the71307-node material
cloud, all volume-grip vertical constraints/anchor, cubic damage radius0.008,
plane strain, material parameters and the fresh `[48,...,75]` schedule. Mechanics
uses the standard P1 weak form on the same union/grip triangulation. Material
degradation is integrated by degree2 triangle quadrature; grip cells stay
undegraded. The solver.py hybrid Amor driver is constant per P1 material cell
and projected onto material nodes using area/3 weights divided by nodal volumes.
This explicit projection and the mechanical discretization differ from A2;
neither branch is presumed to be the exact local continuum reference.

Direct and PINN damage paths start with their own zero d/H and retain the same
previous-committed maximum-history floor and relaxation0.5. They use the existing
outer gates; exhausted loads do not advance. Direct fields never enter a neural
loss or parent. Direct/PINN output roots are separate. Completed-state resume
allows endpoint extension, validates parent/field/network hashes and never skips
a rejected state. Interrupted unfinished loads require a new identified retry;
this experimental entry does not implement the older hybrid phase-recovery schema.
Root reports are snapshots of completed loads; use the curve length/ready count,
requested endpoint and worker exit to determine whether a whole run has ended.
`configs/local_damage_trial_2048.py` gives the separately identified larger
direct-solve budget, and `configs/local_damage_pinn_smoke.py` is a4-sweep own-
history neural diagnostic. It is not a physical convergence budget.
`configs/local_damage_trial_from_zero.py` checks all75 loading increments from
zero through0.0075, with its own committed history. It requests CG absolute
tolerance1e-13 instead of1e-10 for the small-load volume-weighted damage RHS;
the relative tolerance and every outer/physical gate remain unchanged.
Compare readiness, current-history residuals and crack fields together: a
declining reaction during a moving crack is not a completed outer solve.

`local_damage_comparison.py` and `configs/local_gradient_damage_from_zero.py`
provide a matched own-zero-history local damage control. Only the scalar spatial
operator is replaced by standard material P1 gradient stiffness \(S\), with
\(D_{local}=M_{lumped}^{-1}S\). Its damage weak form is

$$
[\operatorname{diag}(M_{lumped}(G_c/\ell_0+2H))+G_c\ell_0 S]d
=2M_{lumped}H.
$$

Mechanics, volume-grip constraints, nodal-lumped Amor projection, maximum
committed history, relaxation, load schedule and convergence gates match the
nonlocal direct zero-start trial. The natural scalar boundary condition is
zero flux; no damage Dirichlet constraints are introduced. Operator inputs and
code are SHA-bound, and completed-state resume supports endpoint extension.
It never imports a nonlocal physical predecessor or promotes exhausted loads.
`configs/local_gradient_damage_load_refined.py` uses the same87-load critical-
region quarter-step schedule as `configs/local_damage_load_refined.py`, allowing
loading sensitivity to be distinguished from scalar-operator differences.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.local_damage_comparison \
  --config examples/fracture2d/configs/local_gradient_damage_from_zero.py
```

`local_damage_fem_audit.py` runs in `fenicsenv` after JAX GPU work exits. It
independently assembles UFL mechanics, scalar stiffness, nodal mass and Amor
projection, checks every saved accepted state, and compares selected fixed-H
damage images with independent sparse LU solutions. Required arguments are
`--geometry-dir`, `--roots`, `--output` and `--oracle-loads`. This is independent
verification of a discrete coupled path, not a separately generated trajectory
or a spatially converged continuum reference. The older `solver.py` archive
uses boundary loading, DG0 history and consistent reaction mass; its warning
transition states cannot serve as an identical-model strict comparator.

`conditioned_damage_trial.py` is a fixed-own-history neural representation
experiment. Its separate96x3 Swish MLP receives six geometry features and
logarithmic/bounded features of the known history averaged on the existing
nonlocal graph. With \(P_{ij}=-D_{ij}/D_{ii}\) for \(i\ne j\) and
\(P_{ii}=0\), nonnegative rows sum to one; the explicit powers are
\(0,1,2,4,8,16\). These are coefficient features, not damage-equation
iterations. The known \(H\) is detached and changes as an explicit JIT input.
The neural field uses the existing positive output link/load scaling and the
same nonlocal AT2 residual. No direct target, damage correction or changed
physical/history equation is introduced.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.conditioned_damage_trial \
  --config examples/fracture2d/configs/conditioned_damage_trial.py
```

The frozen input is the earlier neural checkpoint's own `H_used`; its bytes
are validated and remain read-only. Full-field residual/range gates determine
success. Configured residual-MSE and residual-L4 stages explicitly inherit
neural weights, with bounded L-BFGS work. The latter minimizes
\(\sqrt{\operatorname{mean}((r/s)^4)+\epsilon^4}\), using the declared
\(s=2\operatorname{RMS}(H)\) and \(\epsilon=10^{-15}\), to emphasize peaks.
This changes the training objective, not the physical residual/gate. A direct
fixed-H solution is evaluated only after all training for independent error
reporting. Conditioning changes the ansatz and is not proof of unseen-history
generalization or a speed benefit. Subsequent loading requires its own history
and both inner and outer gates; fixed-H success alone does not authorize
unready physical advancement.

`compact_damage_trial.py` continues the same fixed-own-H experiment with a
frozen conditioned trunk and continuous radial hidden functions
\(\phi_j(x)=\max(1-\|x-c_j\|/r_j,0)\). Trainable readout coefficients enrich
the raw neural output before its positive link. Coarse centers cover the
material; additional centers are chosen from the preceding neural residual,
with explicit spacing/radius/budget controls. Their geometry and padded arrays
stay fixed during each optimizer. These are continuous functions, not per-node
damage correction or an imported direct solution. The code scales each readout
column by \(2\operatorname{RMS}(H)\sqrt{N}/\|A\phi_j\|_2\), where
\(A=\operatorname{diag}(G_c/l_0+2H)+G_c l_0D\). This is a checked parameter
normalization; the residual and gates remain identical. This reduced neural
ansatz should be distinguished from the plain coordinate MLP.
Different radii may share a center, including tip/corner centers; separation
applies within the adaptive scale only. `residual_source` explicitly selects
another uncorrected neural result with exactly the same own history for center
selection. Its damage values are not labels or initial readout coefficients.
`baseline_mode` is explicit: `raw_conditioned` adds readouts to the frozen raw
trunk; `positive_conditioned` adds them to its positive image before the final
positive link. The latter changes the initial field by at most the declared
link floor and gives formerly saturated negative outputs a usable derivative.
`radius_factor` sets each adaptive radius relative to its nearest material
spacing, with declared minimum/maximum bounds. `inherited_source` explicitly
preserves another neural basis and trained readouts before adding smaller
continuous units. Old columns and the initial physical field must match the
parent; coefficient scaling is transferred inversely to preserve that field.
Each optimizer still uses a fixed basis, and all source files are SHA-bound.
Available explicit `compact.methods` stages are `residual_mse`, `residual_l4`,
`residual_l16` and `residual_l32`. Zero added units explicitly retains the
inherited basis for an optimizer-only continuation.
For an even power \(p\), the peak objective is
\((\operatorname{mean}((r/(2\operatorname{RMS}(H)))^p)+\epsilon^p)^{2/p}\).
Increasing \(p\) emphasizes extremes; full-field physical gates are unchanged.
The implementation evaluates this expression with a detached maximum scale
to avoid power overflow and zero-residual floor underflow; its algebraic
objective and checked parameter derivative are unchanged.

```bash
.venv/bin/python -m examples.fracture2d.compact_damage_trial \
  --config examples/fracture2d/configs/compact_damage_trial.py
```

The source conditioned weights/history remain read-only, and direct fixed-H
damage is evaluated after training only. Fixed-H success still requires a
separate own-zero coupled loading test before any physical path claim.

`compact_damage_loading.py` supplies that separate own-zero test. It requires
a seed which has met its declared fixed-H target. Only trained parameters and
feature geometry transfer. Physical displacement, damage and committed history
start at zero; the local FEM solve generates each new driving history, using
the previous accepted load's history as its sole maximum-history floor.

`output_damage_trial.py` provides a separate fixed-own-H output-map control.
It compares the existing nonnegative map, sigmoid and \(\exp(-z^2)\), as well
as sigmoid/exp-square maps for the auxiliary neighbor field below. All groups
use the same frozen own-neural predictor, continuous compact feature geometry,
conditioned residual MLP, physical initial field and explicit optimizer budget.
The additional frozen predictor is part of the ansatz, so this experiment does
not isolate an output map against the older smaller effective representation;
its matching nonnegative control isolates the head within the new ansatz.
Direct damage/history are never training targets or physical parents.

For \(g_0=G_c/l_0\), \(g_1=G_c l_0\), \(C_i=g_1D_{ii}\) and
\(P=I-\operatorname{diag}(D_{ii})^{-1}D\), the checked nonnegative
row-stochastic graph permits the exact AT2 rearrangement

$$
(g_0+2H_i+C_i)d_i-C_i(Pd)_i=2H_i.
$$

The operator-aware output is

$$
d_{\theta,i}=\frac{2H_i+C_i\sigma_{\theta,i}}{g_0+2H_i+C_i},
\qquad r_i=C_i\bigl(\sigma_{\theta,i}-(Pd_\theta)_i\bigr).
$$

Bounding \(\sigma_\theta\) to \([0,1]\) bounds \(d_\theta\) to \([0,1)\).
This changes the neural parameterization, not the original equation, operator,
maximum history or irreversibility checks. It applies no iterative nodal
damage correction. Training and acceptance still evaluate the original AT2
residual rather than assuming that an output bound establishes convergence.

Matching initialization sets \(\sigma_0=\epsilon+(1-2\epsilon)d_{\rm own}\)
and uses its reconstructed damage as the common starting field. Each output
map is analytically inverted; the trainable residual network starts at zero.
The positive branch \(z_0=\sqrt{-\log(v_0)}\) initializes exp-square away from
\(z=0\), where its output is1 and its derivative is0. The declared
`head_trial.initialization_epsilon` makes inverse values finite; predictions
are not silently clipped. Known H/load enter all later evaluations explicitly.
Compact columns are normalized by the original residual Jacobian at that
initial field, including each head derivative and load scaling. The MLP scalar
gain is the reciprocal initial decoder-derivative RMS. Both are fixed parameter
scales, and no new weighting or physical tolerance is introduced.

```bash
.venv/bin/python -m examples.fracture2d.output_damage_trial \
  --config examples/fracture2d/configs/output_damage_trial.py
```

Input geometry/source/basis hashes and output fields/networks/scales are recorded
in a separate run identity. A rejected checkpoint supplies diagnostic own H
only. It remains rejected; subsequent coupled work must retain the last accepted
physical parent and explicitly certify the new decoder. Fixed-H head performance
alone does not establish a converged fracture path or an optimal universal map.

`output_damage_loading.py` compares certified direct sigmoid/exp-square heads
in separately identified coupled branches. It verifies the complete accepted
parent prefix, retains its physical displacement/damage/committed H, and checks
the fixed-H decoder identity, learned weights and parameter scales before
transfer. The first load's own FEM history is independently reconstructed;
later loads use each branch's own accepted history. Frozen own-neural baseline
inputs and the new residual network both receive current H/load. Seeds are
neural parameters, not diagnostic physical fields or imported direct history.
The same original inner/outer gates and updated-H raw-network check apply.
An unready inner result stops before relaxation or history commitment, and a
rejected physical state cannot become the next parent. These explicit short
controls do not provide endpoint extension/resume; further continuation needs
a declared matching-model identity and verified accepted parent.

```bash
.venv/bin/python -m examples.fracture2d.output_damage_loading \
  --config examples/fracture2d/configs/output_damage_loading.py
```

`output_damage_continue.py` supplies separately identified sigmoid continuation
from a certified bounded-head physical parent. The explicit example config
selects accepted34 and uniform loads35..75. It validates the old zero-origin
prefix and subsequent bounded-head ancestry, audited model/configuration/code,
frozen own-neural predictor, compact basis, fixed scales and residual-network
weights. Parent H/d/q and both damage images reconstruct before continuation.
The equation, committed maximum history, half-relaxation, optimizer controls and
every original inner/outer/physical gate are unchanged; no reference H is used.

`cached_damage_optimization.py` builds value/gradient and metrics functions once
for a fixed parameter-tree layout. Current parameters/H/load remain dynamic
arguments. It retains the existing SciPy L-BFGS iteration/readiness policy and
rejects changed network structure, shapes or dtypes. Startup compares actual
decoder objectives/gradients/metrics with independently traced evaluations.
Compilation reuse alone does not establish numerical or overall speed benefit.

```bash
.venv/bin/python -m examples.fracture2d.output_damage_continue \
  --config examples/fracture2d/configs/output_damage_continue.py
# Same configuration and accepted-only restart:
.venv/bin/python -m examples.fracture2d.output_damage_continue \
  --config examples/fracture2d/configs/output_damage_continue.py --resume
```

The new root records the inherited physical prefix and source/code hashes.
Completed states can be restored only when ready and unchanged; rejected states
cannot be skipped. An incomplete inner solve restarts from the last completed
outer checkpoint, with fresh L-BFGS memory; no quasi-Newton optimizer-state
continuation is claimed. Pending checkpoints preserve q/d/H, network, current
reaction, mechanics and confirmation count in atomically selected generations.
The worker retains the latest two pending generations and all final/sweep/inner
reports. A converged outer checkpoint commits without an extra physical update
on restart. Changing anything beyond the requested endpoint requires a declared
new identity. Failed initializer/range/solver checks stop loudly; no numerical
fallback, direct damage correction or history replacement is applied.

`output_damage_safe_continue.py` supplies a declared fork when the frozen
positive-only initializer exceeds its original range during later loading.
Its explicit config selects accepted52 and original loads53..75. Before the
unchanged seed reconstruction, replace the frozen prediction by
\(\min(d_{\mathrm{frozen}},1-\epsilon)\), with the existing
\(\epsilon=10^{-10}\). Thus \(\sigma_0=\epsilon+(1-2\epsilon)
\min(d_{\mathrm{frozen}},1-\epsilon)\) and
\(d_0=(2H+C\sigma_0)/(g_0+2H+C)\) remain strictly between0 and1 for
nonnegative H and positive g0/C, giving a finite logit. The final unknown is
still \(d_\theta=\operatorname{sigmoid}(\operatorname{logit}(d_0)
+\Delta_\theta)\); it is checked with the full original AT2 residual.
The cap acts on a frozen initialization coefficient, not the solved damage.
Below the cap the old and new decoder images agree exactly.

The fork validates the original independent audit, source/code snapshot,
accepted ancestry and parent hashes. Both saved parent H_used/current-H
images and exact own-FEM history must reconstruct at startup. Uncommitted
trial damage/history are never imported. The new operation/configuration and
wrapper source hash are recorded in a separate identity; the original root
and archived runner remain retained. The orchestration run API now requires
explicit parent_loader, setup_fn and extra_code_names; its original CLI passes
the original functions. Changed orchestration hashes require a new identity
instead of resuming an old root. Optimizer/physics/gates are unchanged.

```bash
.venv/bin/python -m examples.fracture2d.output_damage_safe_continue \
  --config examples/fracture2d/configs/output_damage_safe_continue.py
# Restart only this new root, with the same configuration:
.venv/bin/python -m examples.fracture2d.output_damage_safe_continue \
  --config examples/fracture2d/configs/output_damage_safe_continue.py --resume
```

The separate `plain_sigmoid_damage.py` factory uses
\(d_\theta=\operatorname{sigmoid}(f_\theta(x,H,u))\) with the same conditioned
MLP, continuous compact readouts and explicit parameter scales. Its latent
field has no frozen predictor or diagonal/H reference term. Current H is
detached throughout each inner loss; it still updates in the outer physical
solve. The same42577 parameters remain trainable; the earlier additional
frozen42577-parameter predictor is removed from the effective representation.
This is not an equal-total-capacity output-head ablation.

`plain_sigmoid_trial.py` explicitly converts old correction-head parameters:
retain trunk/readout weights, recenter the existing trainable output bias once
using the mean own-network teacher logit, then fit those own-network logits.
The teacher is used only during this one-time conversion, never in the new
runtime damage output. There are no FEM/direct training labels or per-node
damage correction. Transfer readiness and subsequent L32/L16 training both
use the original AT2/range gate; a small teacher-fit loss alone is insufficient.
Each stage retains explicit2048/check32 controls. Known H features, load
normalization, graph and physical equation stay unchanged.

The diagnostic config selects completed uncommitted53 sweep77 for parameter
conversion at its old own H, then original-residual training at failed H78.
The parent config separately converts accepted52 at its own current H.
Diagnostic physical fields are never accepted ancestry. A physical continuation
requires both fixed-H readiness and a certified matching52-parent decoder.

```bash
.venv/bin/python -m examples.fracture2d.plain_sigmoid_trial \
  --config examples/fracture2d/configs/plain_sigmoid_trial.py
.venv/bin/python -m examples.fracture2d.plain_sigmoid_trial \
  --config examples/fracture2d/configs/plain_sigmoid_parent.py
```

The explicit `_rms` trial/parent configurations instead normalize latent basis
columns by their RMS. Changing scale from s_old to s_new is accompanied by
beta_new=beta_old*s_old/s_new, preserving the initial represented field.
Inherited reference-head normalization can be unsuitable for plain logits;
each parameterization uses a separate run identity/root. Original equations
and gates remain unchanged.

`plain_sigmoid_cold.py` provides the separate fresh-start applicability test.
Its config selects only displacement0.005 (index50) as the first load. Physical
q/d/H start at zero; hidden MLP weights are freshly seeded, output weight and
compact readouts start at zero, and a trainable output bias gives uniform
initial neural damage0.01. It imports only validated mesh/operator and pinned
compact feature geometry, with no old trained weights, reference network,
teacher or physical history. Latent columns use fixed RMS scales; trunk gain
sets a unit MLP latent factor at reference displacement0.005. All42577
parameters remain trainable. Current H is detached during each damage solve.

First save the fixed-own-H network/fields/result, then proceed with half-relaxed
damage and local FEM only if the original inner residual/range gates pass.
The original coupled gates and three consecutive confirmations still determine
physical readiness. Failed points are not promoted. The fresh jump from zero
to0.005 is a distinct schedule from the earlier gradual loading prefix.
This intentionally one-load entry has no CLI restart or endpoint extension.

The `plain_sigmoid_cold_jacobian.py` configuration is a separate scaling
control with the same fresh seed, initial field and physical path. It uses
the initial readout Jacobian
\(J=[\operatorname{diag}(g_0+2H)+g_1D]
\operatorname{diag}(d_{\rm init}(1-d_{\rm init}))B\) to normalize each
column's residual RMS through the existing `residual_column_scale` function.
The resulting scales remain fixed; the output has no additional reference
field. Initial physical fields are saved for independent own-H replay.

```bash
.venv/bin/python -m examples.fracture2d.plain_sigmoid_cold \
  --config examples/fracture2d/configs/plain_sigmoid_cold.py
.venv/bin/python -m examples.fracture2d.plain_sigmoid_cold \
  --config examples/fracture2d/configs/plain_sigmoid_cold_jacobian.py
```

### Fresh gradual loading without a reference logit

`plain_sigmoid_gradual.py` initializes physical displacement, damage and
committed history exactly zero and starts training at positive load0.0001.
Its explicit uniform schedule extends through0.0075, with a milestone after
the first10 accepted loads. It stops at the first rejected inner or outer
solve. Only accepted neural parameters and physical states warm start later
loads; previous output-layer weights and histories are not imported.

The separate `make_gradual_problem` decoder is
\(d_\theta=\operatorname{sigmoid}(f_\theta(x,H)+B\beta_\theta)\).
It does not multiply the entire logit by load squared: that scaling would
also rescale the learned negative bias when increasing the load. Features
use physical, detached own history. The initial trainable bias gives uniform
neural damage1e-5; physical initial damage remains0. Readout scales normalize
the initial-own-H AT2 Jacobian once and stay fixed. These are parameterization
choices; the local FEM/nonlocal AT2 equations, committed maximum history,
omega0.5 and original full-field inner/current-history/outer gates remain.

Each fresh output directory retains zero fields, initial/final neural weights,
source/code hashes and snapshots, per-load parent hashes and pending outer
checkpoints. Existing directories are rejected; the entry point does not yet
offer automatic restart. Do not advance from an unready saved checkpoint.
The preserved single-load decoder and its prior artifacts remain separate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.plain_sigmoid_gradual \
  --config examples/fracture2d/configs/plain_sigmoid_gradual.py
```

`configs/plain_sigmoid_gradual_long.py` declares the same fresh experiment
with8192 iterations per residual objective instead of2048, in a separate
output root. It changes the training budget only; all readiness gates remain.

### Fixed-own-history saturation comparison

`plain_damage_head_trial.py` compares fresh sigmoid, \(\exp(-z^2)\) and
identity outputs at exactly the same audited initial own-zero FEM history
from the gradual experiment. It imports no trained network or direct labels.
The MLP/compact basis and trainable parameter count stay the same. Each head
initializes at uniform damage1e-5. Exp-square starts at a negative latent,
so its initial derivative is positive; identity has derivative1. Fixed
readout Jacobian scales and trunk gains match all initial parameter
sensitivities to the sigmoid control. This avoids mistaking arbitrary initial
gradient magnitude for an effect of the output map.

Sigmoid has derivative \(d(1-d)\); exp-square has derivative \(-2zd\).
Both can saturate at small damage. Reports include native/effective derivative
quantiles and original full-field AT2/range readiness. The identity head may
leave the physical range during optimization, but final readiness requires
the unchanged range gate; outputs are never clipped. These are fixed-H
diagnostics and cannot serve as accepted coupled loading states.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.plain_damage_head_trial \
  --config examples/fracture2d/configs/plain_damage_head_trial.py
```

### Linear-output range optimization and damage activity

`range_damage_trial.py` continues the rejected identity-head parameters at
the exact same audited own history. It compares2048 additional L16 updates
with a separately declared inequality augmented-Lagrangian range control.
The network keeps a linear output; the physical AT2 equation, committed
history and original residual/range gates remain unchanged. There are no
direct targets, output projection or nodal corrections.

For fixed positive history define \(S=2\operatorname{RMS}(H)/g_0\),
\(g_-=-d/S\), \(g_+=(d-1)/S\). The training objective is original residual
L16 plus

$$
\frac{1}{2\mu}\operatorname{mean}\sum_{a\in\{-,+\}}
\left[\max(\lambda_a+\mu g_a,0)^2-\lambda_a^2\right].
$$

After each inner solve update
\(\lambda_a\leftarrow\max(\lambda_a+\mu g_a,0)\).
Duals, history and penalty are fixed detached coefficients during training.
Initial duals are0; the explicit experimental penalty schedule is1,10,100,1000,
with2048 iterations per stage. This is an optimizer comparison, not a
different fracture energy or an irreversible-damage constraint.

Diagnostics separate absolute AT2 imbalance, its component denominator,
fixed forcing scale and volume-weighted damage mean/L2/activity fractions.
At \(d=0\), the current equation gives \(r=-2H\): positive history yields
relative residual1. At the exact zero-load state \(d=H=0\), the raw residual
is0 and relative0/0 is undefined, so training starts at positive load.
Growing damage and transient residual do not by themselves establish either
failure or correctness; a submitted load must still pass equilibrium and
physical checks. Fixed-H diagnostic results are never physical parents.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.range_damage_trial \
  --config examples/fracture2d/configs/range_damage_trial.py
```

`range_damage_extend.py` continues only the certified final neural parameters
and updated duals of that fixed-H diagnostic in a fresh output root. It pins
the complete parent identity/audit/fields/scales/network/dual summary and
declares additional penalties1e4,1e5,1e6 with2048 iterations each. It preserves
the same history, network output and original gates, and skips the already
completed extra-training control. No physical state is promoted by this entry.

### Own-zero loading with the linear damage head

`linear_damage_loading.py` seeds only the certified own-H neural parameters
from the range extension and initializes physical displacement/damage/history
exactly zero. It retains the identical linear decoder, physical history
features and fixed parameter scales, with no load-squared output scaling or
old-head physical history. Direct P1 mechanics alternates with fixed-H L16
range optimization, then omega0.5 physical damage relaxation. The declared
inner penalty sequence is1e3,1e4,1e5,1e6 with2048 iterations each; reset duals
to0 for each new fixed-H problem and warm start neural parameters only.

Original uniform loads1..75 and the first10 milestone advance only from
accepted states. Original inner/current-H AT2, physical range, mechanics,
history, change and three-consecutive outer gates remain. The raw network
range at the updated current H is checked explicitly, preserving the range
property that a bounded decoder previously supplied automatically. Record
both absolute/forcing-normalized imbalance and damage activity; transient
outer residual is not equated with physical failure. Commit only a final
equilibrium that passes all gates. This entry currently starts fresh roots
only and saves pending checkpoints without an automatic restart CLI.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.linear_damage_loading \
  --config examples/fracture2d/configs/linear_damage_loading.py
```

### Same-load recovery with a numerical range tolerance

`linear_damage_continue.py` preserves the strict branch and starts a new root
from its completed, uncommitted outer iteration at the same load. The last
accepted load remains the physical history/damage parent; pending fields do
not become an accepted loading state. The saved failed-inner neural parameters
are reused at the identical own history, so completed training is retained.
Source identity, final audit, accepted parent, pending fields and failed neural
checkpoint are explicitly pinned and checked before recovery.

The user-accepted range tolerance is \(2\times10^{-6}\) on both bounds, applied
to inner damage, physical relaxation, current-H raw output and mechanics input
validation. `TolerantLocalMechanics` retains the identical quadrature and
degradation law; it evaluates the actual signed values without clipping or
shifting the network output. AT2, force, history, healing, change thresholds
and three consecutive equilibrium confirmations remain the original controls.
This entry is a declared fresh continuation root, not a generic restart CLI.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.linear_damage_continue \
  --config examples/fracture2d/configs/linear_damage_continue.py
```

### Same-load recovery with additional inner training

`linear_damage_budget_continue.py` starts a separate root from the tolerant
branch's failed-inner network and last completed physical outer iteration.
It keeps the accepted loading parent and identical own history, decoder,
parameter scales and physics. Each of the four existing penalty stages has
an explicit maximum of8192 L-BFGS iterations instead of2048; training exits
early as soon as the unchanged full-field AT2 limit0.001 and range tolerance
\(2\times10^{-6}\) pass. This is an empirical budget experiment, not a
relaxation of the final equation or coupled acceptance criteria.

The recovery reuses neural parameters and explicitly restarts L-BFGS memory
and AL duals. Identity/audit, accepted-parent receipt, completed pending fields,
failed network and source code are pinned; rejected physical fields are never
promoted to a loading parent. First recovery fields are retained separately
for independent replay. A budget stop remains a rejected inner solve.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.linear_damage_budget_continue \
  --config examples/fracture2d/configs/linear_damage_budget_continue.py
```

### Full-path attempt with relaxed negative damage tolerance

`linear_damage_relaxed_continue.py` uses explicit asymmetric bounds
\(-10^{-3}\le d\le1+2\times10^{-6}\), as requested by the user for a
full remaining loading-path experiment. Only the negative bound is relaxed.
The bounds are checked consistently in fixed-H training, physical damage,
current-H neural output and FEM input validation. Signed P1 quadrature still
evaluates the actual damage; there is no clipping or output shift.

It restores the certified tolerant failed-inner124 network at the unchanged
completed physical123/own history, with accepted52 as physical loading parent.
Per-stage budget8192 and early residual stopping are retained. The old equation
limit0.001, physical equilibrium, history, healing, force, drift and three
consecutive confirmation gates remain. A successful path here uses the declared
relaxed negative range; it does not certify the stricter previous lower bound.
The requested remaining loads are53..75, ending at displacement0.0075.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.linear_damage_relaxed_continue \
  --config examples/fracture2d/configs/linear_damage_relaxed_continue.py
```

### Recovery after a process interruption

`linear_damage_pending_continue.py` restores the last serialized physical
iterate and neural weights of the interrupted relaxed-range experiment in a
fresh output root. The explicit configuration pins the interrupted identity,
prior audits, accepted parent, parameter scales and pending126 receipts. An
independent pending replay must certify the own history, neural fields and
signed FEM force before restart. A finalized or already accepted source state
is rejected by this particular recovery entry.

Accepted52 remains the physical loading parent. Pending126 is an uncommitted
iterate at the same load; restart begins at127 with its saved neural parameters,
while unsaved interrupted inner127 work is not claimed as recovered. L-BFGS
memory/AL duals restart explicitly. The original model,8192-stage budget,
negative1e-3/positive2e-6 bounds and final coupled gates are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python -m examples.fracture2d.linear_damage_pending_continue \
  --config examples/fracture2d/configs/linear_damage_pending_continue.py
```

The conditioned trunk receives the changing own H. `loading.train_trunk`
explicitly selects a frozen trunk or joint MLP/readout training; checkpoints
contain all trainable parameters and identities distinguish these controls.
`loading.history_feature_scaling` explicitly selects physical H or known-H
features normalized by \((u_\mathrm{seed}/u)^2\). The latter is identity at
the seed load and compensates the quadratic driver amplitude at small loads.
It changes neural inputs only; AT2 and committed maximum history always use
physical H. Changing damage/history along a path still requires retraining and
physical gates; this normalization is not a path homogeneity assumption.
Compact readouts
are multiplied by \((u/u_\mathrm{seed})^2\), so their physical contribution
scales with load just as the original trunk output does. The seed prediction
must reconstruct exactly at its own diagnostic H before physical loading.
The feature normalization remains fixed; all arrays have fixed JIT shapes.

```bash
.venv/bin/python -m examples.fracture2d.compact_damage_loading \
  --config examples/fracture2d/configs/compact_damage_loading.py
```

Inner MSE/L16/L32 stages train neural parameters with explicit L-BFGS budgets.
An unready inner solve saves diagnostics and stops before damage relaxation or
history commitment. Ready images enter the existing half-relaxed staggered
loop; every original field/mechanics/outer/history/healing/drift gate remains.
Additionally, the raw neural prediction must satisfy AT2 at the updated H.
Rejected load states never become parents. `--resume` checks code/config/seed,
state/network hashes and accepted ancestry. Increasing only the requested
prefix length supports continuation; all other identity fields must agree.
`configs/compact_damage_loading_original_gates.py` is a separate control which
ties the optimizer readiness target to the verified baseline's original inner
AT2 gate, \(10^{-3}\). The default loading experiment instead tests an added
\(5\times10^{-4}\) optimizer margin. All original outer/physical checks are
identical in both, including the raw network residual at updated H. A candidate
rejected in the tighter-margin root stays rejected there; the original-gate
control starts its own zero state and does not import that diagnostic H.
`configs/compact_damage_loading_full.py` requests all75 uniform load points
with the same original-gate identity. Use `--resume` only after the short
prefix is accepted; code/config/seed/state/network checks reject mismatches.

`compact_damage_fork.py` supports explicitly recorded optimizer-policy forks
from accepted own-neural prefixes. It checks parent identity, matching numerical
model/gates, field/network/report hashes, accepted gate status and exact
zero-origin/history ancestry. Only accepted states are copied; the new root
contains `fork.json` and configuration provenance. Requested optimizer stages,
budgets and output root may change; physical/model/history inputs must match.
The manifest binds parent and current numerical-runner hashes when an
independently checked optimizer-objective extension changes the runner code.
The normal loader then resumes the copied own-history states. For the declared
peak-first continuation use:

```bash
.venv/bin/python -m examples.fracture2d.compact_damage_fork \
  --config examples/fracture2d/configs/compact_damage_loading_peak_first.py
```

Use the same entry with `--resume` to recheck fork provenance before resuming
accepted checkpoints. An interrupted or rejected pending state is not an
accepted prefix. This mechanism does not import direct fields or history.
This is a continuous compact-basis neural ansatz with a conditioned MLP trunk
and trained readouts; it should be distinguished from a plain-coordinate MLP.

`damage_optimizer_trial.py` compares bounded optimizers on a read-only neural
checkpoint's own fixed `H_used`, using identical initial network weights. It
compares continued Adam, SciPy L-BFGS-B on the existing residual MSE, and L-BFGS-B
on the equivalent fixed-history damage functional. With nodal mass \(M\),
reciprocity gives \(MD=(MD)^T\), so

$$
E_H(d)=\tfrac12 d^T M[(G_c/\ell_0+2H)d+G_c\ell_0 Dd]-2d^TMH,
\qquad \nabla_d E_H=M r_{\mathrm{AT2}}.
$$

This equivalence requires the declared reciprocal volume-symmetric operator;
the runner checks symmetry and an independent CPU directional derivative. It
is only a frozen-H damage functional, not a joint hybrid mechanics/fracture
energy. A positive scalar normalization leaves stationary damage unchanged.
Full-field AT2 residual and range checks determine readiness; energy decrease
or optimizer termination alone does not. The direct fixed-H solution is
computed after training for error reporting, never as a loss or neural history.
`configs/damage_optimizer_trial.py` explicitly bounds extra Adam at2000 updates
and each L-BFGS at512 iterations/checks every16, with a5e-4 inner target. These
are experimental controls, not a general optimizer guarantee. Outputs use a
new root, preserve source hashes and never promote a diagnostic state:

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.damage_optimizer_trial \
  --config examples/fracture2d/configs/damage_optimizer_trial.py
```

`configs/local_damage_load_refined.py` is a separate own-history loading
sensitivity test:87 increments with2.5e-5 steps between.0052 and.0056, otherwise
1e-4. Its integer indices define its explicit schedule, not imported reference
fields. Material, operators, history and outer gates are unchanged.

`local_damage_corrected_trial.py` is a separately labeled neural predictor with
a nodal damage solver. Predictor weights come from the own-H optimizer trial;
physical d/H start at zero. Each load trains the predictor once with bounded
energy L-BFGS. A damped Jacobi correction then solves the same AT2 equation,
warm-started from the previous corrected image within the outer stagger:

$$
d^{k+1}=d^k-\omega_J\frac{r_{\mathrm{AT2}}(d^k,H)}
{G_c/\ell_0+2H+G_c\ell_0\operatorname{diag}(D)}.
$$

The runner checks the graph Laplacian/M-matrix assumptions. Positive H, gc0 and
bounded initial damage imply a contracting, bound-preserving iteration for
0<omega_J<=1. Controls explicitly use omega_J=.8, at most2048 corrections,
checks every16 and image residual1e-5; insufficient correction fails loudly.
The existing coupled outer gates still govern advancement. Raw neural and
corrected image residuals are reported separately. This is a hybrid nodal
solution, not evidence that the pure network satisfies AT2. Predictor training,
pretraining and correction costs must all be included in cost comparisons.
Completed-prefix resume preserves fields/networks and rejects failed loads;
unfinished-load recovery requires a new identified retry. Root reports are
completed-load snapshots, not whole-run completion flags.
`configs/local_damage_corrected_projected.py` identifies the explicit repair
for a predictor exceeding1: project only its initial nodal guess into[0,1],
then solve/verify AT2 without projecting the corrected field. Raw neural
residuals and projection magnitude remain reported. Its v2 root forks only
the six ready own-history predecessors of the retained v1 trial; it validates
model/history/solver configuration, gates and parent/field/network hashes.
The unfinished seventh v1 load is not copied. Updated v2 entries do not resume
a v1 identity in place. Use this identified fork to continue that experiment:

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_corrected_trial \
  --config examples/fracture2d/configs/local_damage_corrected_projected.py
```

```bash
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_corrected_trial \
  --config examples/fracture2d/configs/local_damage_corrected.py --last-index 2
# After readiness/exit, extend the own-history checkpoint:
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_corrected_trial \
  --config examples/fracture2d/configs/local_damage_corrected.py --resume --last-index 28
```

Prepare on CPU, perform an independent FEniCS/UFL stiffness/driver check against
the saved mesh, then use .venv GPU execution; environments must be exclusive:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_trial \
  --config examples/fracture2d/configs/local_damage_trial.py --prepare
# In fenicsenv (deactivate .venv first):
python -m examples.fracture2d.local_fem_validate \
  --geometry-dir /root/autodl-tmp/tf-logs/fracture2d/local_damage_geometry_20261011
# Switch back to .venv before GPU work:
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_trial \
  --config examples/fracture2d/configs/local_damage_trial.py --damage-solver direct
# Separate own-history PINN trial; no direct-damage checkpoint import:
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_trial \
  --config examples/fracture2d/configs/local_damage_trial.py --damage-solver pinn --last-index 2
# Complete own-history direct verification, separate output root:
PYTHONPATH=src .venv/bin/python -m examples.fracture2d.local_damage_trial \
  --config examples/fracture2d/configs/local_damage_trial_from_zero.py
```

`configs/hybrid_pinn_stagger_gated.py` selects explicit `outer_stop` controls:
minimum3 and maximum128 sweeps,3 consecutive passing confirmations, fixed-H
image AT2 limit0.001, current AT2 limit0.001 and d-change limit1e-5. Adjacent
q infinity change/load, H relative infinity change and reaction relative change
must each be <=1e-4. Mechanical explicit CG, finite/bounded damage/image,
healing/history and zero-drift checks must also pass. These are explicit
engineering readiness criteria, not a global convergence theorem.

This configuration is a conditional same-parent .0055 trial: a read-only fork
through ordinal7/.0054 of the archived128x6 path, preserving its warning ancestry.
It does not call that preceding history converged or promote exact-inner
diagnostic arrays. It keeps the128x6 network/Adam and physical model. The
preceding-load H remains the floor during all outer sweeps; maxima
from transient iterates never become a new floor.

The policy can stop early after readiness. If the maximum budget is exhausted,
result.status is `stagger_budget_exhausted`, research_advance=False, and the
next load is not attempted. Rejected results cannot be resumed as advancing
parents; changing controls requires a new identified trial. Reference curves
remain diagnostics under comparison_gate=False, but this does not bypass the
outer self-consistency gate.

Gated roots use `fracture2d-direct-mechanics-pinn-damage-gated-v1`; the legacy
fixed-budget schema/behavior is retained when outer_stop is absent/disabled.
Gated phase snapshots persist prior-sweep q/H/reaction and the confirmation
counter for recovery. Completed snapshots bind actual early-stop sweep counts,
not necessarily the maximum budget. Forks validate source schema/model/network/
physical/loading-prefix/reference/parent identity and preserve old checkpoint bytes.

```bash
python -m examples.fracture2d.hybrid_pinn \
  --config examples/fracture2d/configs/hybrid_pinn_stagger_gated.py
```

The shared checked AT2 oracle CG now packs the same FP64 CSR off-diagonals into
fixed padded rows for matvecs within CG, avoiding the scan-based nested-CG path
that failed the current GPU/JAX runtime audit. The equation remains the
volume-multiplied SPD AT2 system, with the same diagonal preconditioner and
explicit residual checks; no clipping, tolerance loosening or fallback solver.

New short-damage TensorBoard events use
`tf-logs/fracture2d/events/pinn_short_damage_first2_20261009`; checkpoints and
field plots use the sibling `pinn_short_damage_first2_20261009` run directory.

```bash
source .venv/bin/activate
python -m examples.fracture2d.split_pinn \
  --config examples/fracture2d/configs/split_pinn_short_damage.py
# After a usable two-load result, endpoint-only continuation:
python -m examples.fracture2d.split_pinn \
  --config examples/fracture2d/configs/split_pinn_short_damage.py \
  --resume --last-index 75
```

The initial run uses the existing three-sweep, `2000/2000/2000` block budget
and first-passage stopping in the final mechanics block. Primary research
advancement requires reaction error <=5% and damage RMSE <=0.01; every secondary
mechanics/coupling/field defect remains reported. A rejected completed load
requires an explicit retry root rather than endpoint-only resume.

Experimental acceleration configurations are separate from that baseline:

- `split_pinn_adaptive.py` retains uniform loading and introduces explicit
  post-update inner stopping. It checks every100 updates, keeps a200-update
  warm-start minimum (2000 on the first load), requires5 stable comparison
  intervals and retains the2000-update block cap. It reports `converged`,
  `stagnation_warning` or `budget_exhausted`; stagnation never certifies
  convergence. The former reference-first-passage block stop is disabled in
  this experiment. All numerical tolerances are explicit trial controls.
- `split_pinn_accelerated.py` combines that trial with
  `data.reference_indices=[10,20,30,40,50,51,...,75]`. The direct grid remains
  `data.load_step=0.0001`, so the actual target loads are
  `0.001,0.002,0.003,0.004,0.005,0.0051,...,0.0075` (30 loads).
  `end_state_index` and `--last-index` refer to the **neural sequence ordinal**,
  not the direct index; for this configuration the final ordinal is30.

Explicit reference indices must be positive, strictly increasing integers.
Every selected load uses its actual direct snapshot without interpolation;
new reports/checkpoints bind the reference index and physical load. Mechanical
normalization follows the physical load/reference-grid ratio, not the ordinal.
Resume preserves the entire declared schedule; a numerical fork may change
future targets only after an identical inherited schedule prefix.

Nonuniform loading preserves weights/Adam and updates learned cumulative H at
each actual selected load. It does **not** inherit histories from skipped direct
states. Larger increments can therefore change the discrete history/loading
path, especially near damage growth: at0.005 the direct peak damage is already
about0.47. The aggressive30-load example is experimental, not validated as
equivalent to the uniform path. A transition to smaller increments before0.005
may be necessary if paired accuracy/coupling checks show degradation.

Inner stopping monitors fixed-d mechanics and the **unrelaxed damage network
image at fixed H_used** separately. Stable fields/objective/residual with
inadequate accuracy can return an explicitly enabled warning plateau; current
outer coupled residuals and original load-advancement diagnostics still apply.
New per-block JSON files record actual updates, stop reason and post-update
metrics. Validation must separate stopping speed from load-path changes.

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

## Detached residual weighting diagnostic

`weighted_damage_trial.py` compares the original full-domain objective with
static crack-band and damage-dependent weights at one verified checkpoint and
identical fixed history. This diagnostic does not advance the loading solver.
The explicit configuration is `configs/weighted_damage_trial.py`; run with
`.venv`, CUDA, x64 and `PYTHONPATH=.:src`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.weighted_damage_trial \
  --config examples/fracture2d/configs/weighted_damage_trial.py
```

The original AT2 residual is retained. The stabilized degree-two \(L^{16}\)
objective receives \(\sqrt{w_i}r_i\), where \(w_i\) is positive and normalized
to nodal mean one. Uniform weights recover the original loss and gradient.
Experimental raw weights are \(1+\beta a_i\): static activity is a Gaussian
of distance to a prescribed notch-to-boundary segment; dynamic activity is
\(\operatorname{clip}(d_i,0,1)^p\) from saved physical damage. Clipping applies
only to weight construction. Parameters \(\beta,p\) and band width are explicit
optimizer controls, not material parameters or validated universal settings.

Weights and history use `stop_gradient`. Weights remain fixed throughout each
L-BFGS fit, including line search; adaptive refresh belongs between outer
iterations or separate optimizer blocks. A parameter-dependent weight can
change the loss value during line search even when its derivative is stopped.
Full-domain original residual and asymmetric range gates decide readiness.
Reports include same-budget residuals and separate shared-GPU timing status;
weighted training loss alone does not establish physical accuracy or speedup.

`damage_residual_weighting.py` also packs exact core+halo rows from the existing
CSR operator. Every original neighbor and coefficient is retained; capacity
overflow raises an error. Padded arrays/masks have fixed shapes. The diagnostic
reports support size and checks sampled-row equality, but does not yet train on
sampled subgraphs. For with-replacement sampling of node \(i\) with probability
\(p_i>0\), \(f_i/(N p_i)\) estimates the full nodal mean of the supplied moment.
The subsequent nonlinear \(L^p\) root is not claimed to be unbiased. Deliberately
uncorrected oversampling changes the training objective. Full-domain acceptance
is required for either choice, and known history features must retain their
original full-graph construction before selecting halo rows.

## Fixed-history core+halo network trial

`core_halo_damage_trial.py` compares `full_original`, `full_cached`,
`sample_uniform` and `sample_crack` from one pinned NN/physical checkpoint.
The explicit configuration is `configs/core_halo_damage_trial.py`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.core_halo_damage_trial \
  --config examples/fracture2d/configs/core_halo_damage_trial.py
```

`core_halo_damage.py` evaluates the shared trunk only on halo nodes, using
unchanged global compact-basis coefficients and column scales. Known history
features are built once on the full original graph, then their rows are
selected. Exact original AT2 rows retain every core neighbor and coefficient;
neither the operator nor its quadrature is reconstructed on the subgraph.
All pools use a common padded halo capacity, fixed core count and fixed row
widths. Unused padded nodes contribute zero through the operator coefficients.

Sampling draws with replacement. Uniform sampling uses \(p_i=1/N\).
Crack-biased sampling combines the prescribed static-band and detached saved
damage scores with a positive uniform background:

$$
p_i=\frac{1-\eta}{N}+\eta\frac{s_i}{\sum_j s_j}.
$$

Each pool, probability and known-field buffer is frozen throughout one L-BFGS
block; optimizer memory resets between pools. The sampled residual moment and
range-AL term use \(1/(N p_i)\), while forcing normalization uses the full
history RMS. This corrects the moment expectation; the subsequent nonlinear
\(L^{16}\) root is not claimed unbiased. `stop_gradient` applies to probabilities,
history, features and AL buffers. Full-domain original residual and asymmetric
range gates determine readiness, including between optimizer blocks.

The trial performs no coupled update or load advancement and never promotes
its damage output to an accepted physical state. Reports record actual
iterations, full-field metrics, optimizer-block time and warmed value/gradient
time. Use an otherwise idle GPU for the timing comparison; smaller halo size
or sampled loss alone does not establish speed to full-domain convergence.

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

### Fresh random crack sampling

`configs/core_halo_damage_resample.py` uses the same audited checkpoint but
compares fresh cores every8/32 L-BFGS updates. Run with the same entry point:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.core_halo_damage_trial \
  --config examples/fracture2d/configs/core_halo_damage_resample.py
```

The elasticity example normalizes \(1+\alpha/r^q\) around its elliptical hole.
For the crack corridor, use distance \(s_i\) to segment \((.5,.5)\)–\((1,.5)\)
with \(r_i=\sqrt{1+(s_i/.024)^2}\), \(\alpha=49\), \(q=2\), and
\(p_i=(1+49/r_i^2)/\sum_j(1+49/r_j^2)\). Density is largest on the segment,
symmetric and decreasing transversely, with strictly positive background.
These are explicit experimental sampling parameters; the PDE is unchanged.

Every block draws2048 new independent samples with replacement. Samples and
probabilities remain frozen within a line search; probabilities and importance
weights use `stop_gradient`, and L-BFGS curvature memory resets when samples
change. Samples are generated throughout training rather than cycling a fixed
pool. Save every core draw for replay; padded halo32768 fails on overflow and
retains every original operator coefficient. Full-history features are computed
once and gathered onto the halo. Fit timing includes resampling, packing and
full-domain checks. Only full-domain original residual/range gates determine
fixed-H readiness; this experiment does not advance a physical load state.

### Persistent Adam sampling control

`core_halo_adam.py` retains Adam moment/schedule state across fresh random
cores. `core_halo_adam_trial.py` shares the audited fixed-history problem
setup with the L-BFGS sampler, checks exact original/cached objectives and
sampled rows, and performs three sampled rate controls followed by a full
cached Adam control at the best checked sampled rate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.core_halo_adam_trial \
  --config examples/fracture2d/configs/core_halo_adam_trial.py
```

All required settings are explicit: initial rates1e-3/1e-4/1e-5,2048 updates,
exponential decay to10% over2048, beta1=.9/beta2=.999/eps1e-8/eps_root0,
global gradient norm cap1000,2048 core nodes, fresh draw every5 updates,
padded halo32768, full checks every128 plus final. No damage clipping,
nonfinite sanitization or reference target. History/feature/probability/
importance/AL buffers are detached; exact original full-domain AT2/range gates
remain. Adam/schedule counters must equal actual iteration throughout, so a
batch change cannot silently reset state.

Each case saves final and best checked range-admissible checkpoint, metrics,
actual core draws and matching optimizer state. Best checkpoint selection uses
original full residual; it never accepts by sampled loss. Full cached L-BFGS
polishes the best sampled/full Adam checkpoints and the audited prior cached
L-BFGS checkpoint, with explicit2048 maximum updates/early full-gate exit.
Compare measured single-route training-plus-polish time; rate-search cost is
reported separately. A rate chosen on this snapshot is not claimed generally
optimal. These are fixed-H optimization trials, not new equilibrated load states.

## User-provisional load continuation

### Certified52 restart with128 alternating updates

`restart_damage_loading.py` restarts the independently audited converged neural
state at load52/.0052 and advances53..75 in a separate root. The explicit
`configs/restart_damage_loading_128.py` selects128 alternating mechanics/damage
updates per load, with early exit on the original three consecutive coupled
convergence checks. It retains the previous experiment's512 cached full L-BFGS
updates per penalty stage, original omega.5, signed damage bounds, operators,
own committed history and original inner/outer diagnostics. Residual/coupling
failure at the finite budget may advance as user-provisional; numerical range,
nonfinite or signed FEM equilibrium failure stops. Later convergence does not
certify a path containing earlier provisional states.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.restart_damage_loading \
  --config examples/fracture2d/configs/restart_damage_loading_128.py
```

The source fields, neural weights, report and inherited parameterization are
hash-pinned. Physical damage and the raw neural image are distinct stored
objects: restore both with their corresponding own/frozen training histories,
rather than requiring the neural image to equal the relaxed physical damage.
Each load saves immutable fields/network/report at outers8/32/64/128 and at
early convergence. First-load53 outer8 versus128 uses the identical accepted52
parent/history; the former8-outer loading experiment instead began with a
partially evolved53 source. Whole-path comparison therefore includes a changed
initialization/history and must not be described as changing only outer budget.

### Earlier unready53 restart with8 alternating updates

`provisional_damage_loading.py` tests explicitly authorized advancement past a
budget-limited, unready critical-load state. It starts from the independently
audited original omega=.5 physical candidate at.0053 and the polished sampled
Adam network; original signed FEM, nonlocal AT2, parameterization and committed
previous-load maximum history are retained.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:src \
  .venv/bin/python -m examples.fracture2d.provisional_damage_loading \
  --config examples/fracture2d/configs/provisional_damage_loading.py
```

The explicit trial advances54..75 (.0054..0075), max8 alternating physical
updates per load, max512 cached full L-BFGS updates per inner fit. Known-H
features enter a fixed-shape dynamic context and are cached once per outer
fit. Original full residual/range gates still report readiness and may stop
a fit early. If damage range fails, explicit original ALpenalties1e3/1e4/1e5/
1e6 with dual updates provide another512 updates; persistent range/nonfinite/
FEM-force failures stop the run. No damage clipping or reference targets.

User authorization allows a finite range-admissible state to advance despite
failed inner/coupled convergence. Such states are `user_provisional`, with
original gate failures, healing, history, changes, drift and reaction recorded.
A later locally converged state still has provisional path ancestry; it does
not retroactively certify earlier states. The original certified branch is
unchanged. Once provisionally committed, each load's history becomes the parent
for the next load exactly as the declared committed-history formulation.

Crack extent reports connectivity of nodes with d>=.9 along original material
mesh edges, from tip radius.024 around(.5,.5) to x=1 (coordinate tolerance1e-12).
This is a stated visualization diagnostic, not a convergence or energy criterion.
Fields/network/parent hashes and separate local-readiness/path-certification
flags are saved per sweep/load. Compare this path to the direct solution at
identical load sizes; apparent later fracture can reflect insufficient coupling
iterations and must not be described as a verified material critical load.

### Fresh bounded-output morphology comparison

`bounded_damage_morphology_trial.py` with
`configs/bounded_damage_morphology_trial.py` tests no-reference sigmoid(z)
and exp(-z^2) outputs against a fresh identity-output fixed-history control.
It uses completed own histories52 and60; direct AT2 solves at those histories
are diagnostics only and never supply labels to training. Initial trainable
bias gives constant damage0.03, with matched initial residual sensitivities.
Cached full-domain L16 residual fitting uses4096 updates per fixed case.

Two separate coupled branches then use the original certified52 physical
q/d/H as parent and the newly trained bounded networks; no old weights are
loaded. This preserves the physical parent but is not a new own-zero bounded
loading trajectory or a certified converted52 network. Loads53..60 use at
most128 alternating updates and512 L-BFGS updates per inner fit, relaxation
0.5, original equations and committed previous-load maximum history. Exhausted
loads advance provisionally under the user's existing loading authorization;
finite fields, bounds and mechanics are still checked. Original residual,
healing and coupled-change gates remain diagnostics and certify only on pass.

Background mean/fraction outside abs(y-0.5)>3*l0 is a volume-weighted diagnostic
only; no zero-damage BC, horizontal-crack constraint or sparsity penalty is
imposed. Smooth positive AT2 damage away from a crack is allowed. Output bounds
do not themselves guarantee a clean or symmetric crack path.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.bounded_damage_morphology_trial \
  --config examples/fracture2d/configs/bounded_damage_morphology_trial.py
```

`configs/bounded_damage_loading.py` selects **direct loading**, skipping the
identity-head/own-H60 screening cases and direct oracles. It continues the
SHA-pinned newly trained same-head52 networks for up to8192 additional L-BFGS
updates (early exit only when the original inner gates pass). The preceding
4096 fits initialized these bounded networks fresh; no old identity-output
weights or direct fields are used as training targets. Additional adaptation
never changes the certified52 physical parent and never rejects a head solely
for residual ranking. Then both branches load53..60 with128/512 outer/inner
budgets. Results remain provisional whenever original coupled gates fail.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.bounded_damage_morphology_trial \
  --config examples/fracture2d/configs/bounded_damage_loading.py
```

### Sigmoid critical-step accuracy controls

`sigmoid_damage_accuracy_trial.py` restores the audited sigmoid load53,
outer128 physical/network checkpoint with its unchanged committed52 history.
It compares the same network at the same current own history using full-domain
L-BFGS budgets512/8192 and persistent crack-biased core+halo Adam8192 at rates
1e-4/1e-5 followed by full-domain L-BFGS4096. The original inverse-power
sampling density is positive everywhere; fresh2048-point cores are drawn
every5 updates with complete original supports and a fixed32768 halo capacity.
The sigmoid is applied to every halo value before computing diffusion.
History, features, probabilities and importance corrections are detached.
Adam moments survive resampling; full-domain metrics are checked every128
updates. Importance correction estimates the original nodal L16 moment;
its nonlinear root is not claimed to be an unbiased full-loss estimator.

The candidate with the lowest full-domain AT2 metric only changes the network
warm start. Physical q/d/H128 remain untouched before continuing at the same
load53, through at most1024 total outer updates with up to4096 L-BFGS inner
updates each and the original early stopping/readiness gates. No54 load is
advanced and committed52 history is not replaced by an unready53 candidate.
Checkpoints and source hashes preserve physical field/background/reaction
comparisons. Fixed-H direct values, when independently evaluated, never enter
training. Numerical budgets are experiment controls rather than new physics.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.sigmoid_damage_accuracy_trial \
  --config examples/fracture2d/configs/sigmoid_damage_accuracy_trial.py --preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.sigmoid_damage_accuracy_trial \
  --config examples/fracture2d/configs/sigmoid_damage_accuracy_trial.py
```

### Quarter-load sigmoid continuation

`sigmoid_damage_load_refined.py` starts from the certified physical52 fields
and the SHA-pinned adapted sigmoid52 network. It loads integer IDs209..212
times2.5e-5:0.005225,0.005250,0.005275,0.005300. Output state indices53..56
count this branch's increments; use the explicit `load` field for comparisons,
since coarse state53 instead means0.0053. The matched existing fine direct
reference uses these actual loads. No unready coarse53 field or later network
is used for initialization. The adapted52 network is not itself certified;
only the retained physical parent is certified.

The equations, local FEM mechanics, nonlocal sigmoid damage, omega.5,
previous-load maximum history with stop-gradient, and original gates stay
unchanged. Each small step allows1024 outer/4096 inner L-BFGS updates and early
exit after three readiness confirmations. This can quadruple total available
work; reports include cumulative updates and seconds. At a failed cap, the
previously authorized provisional advancement is explicit and ancestry stays
uncertified. Within a load, history uses the preceding load's saved floor,
rather than committing intermediate staggered overshoots. Network fits never
replace the physical parent or use direct training targets.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.sigmoid_damage_load_refined \
  --config examples/fracture2d/configs/sigmoid_damage_load_refined.py --preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.sigmoid_damage_load_refined \
  --config examples/fracture2d/configs/sigmoid_damage_load_refined.py
```


### Low-order residual and persistent optimizer controls

`low_order_damage.py` provides separate MAE (`l1`) and MSE (`l2`) objectives
for the same raw sigmoid nonlocal AT2 residual. History and features are frozen
inside each inner solve. The normalization remains twice the full-population
history RMS. Nonuniform with-replacement core sampling uses the exact original
CSR supports plus a padded halo; all halo outputs are mapped through sigmoid
before the core residual is assembled. The detached multiplier `1/(N*p_i)`
preserves the uniform nodal objective and its expected parameter gradient for
these linear mean reductions. Mean losses and full-domain maximum-residual
acceptance are separate quantities.

`make_optimizer` and `make_step` support persistent Adam or upstream
`pinn.optimizers.soap` (SOAP_JAX 0.2.2); neither uses L-BFGS. All schedule,
moment, preconditioner and clipping settings are explicit. Retain optimizer
state when resampling; a change of batch is not a cold start. SOAP's first call
initializes preconditioners with no parameter update. Shared SOAP now imports
the pinned [upstream package](https://github.com/haydn-jones/SOAP_JAX), including
its elementwise projected-gradient second moments. Network weights can warm
start, but legacy vendored SOAP optimizer states cannot be loaded into its new
NamedTuple state; initialize new SOAP state at this migration. Run-specific
budgets, source checkpoints, independent field/force validation and actual
outcomes belong in the active task.


`precomputed_damage_sampling.py` validates and packs immutable operator support
rows once. Fresh core sets gather cached rows and bulk-remap the exact halo;
zero padding adds no support. This avoids full-CSR copies/validation and
per-core Python loops inside the sampling loop. Core probabilities and
importance weighting are unchanged. It retains fresh random sampling rather
than cycling a finite pool; unlike a fully prebuilt pool, fresh halo union and
data transfer still have a per-batch cost. Record preparation, packing and
training costs separately when assessing sampling frequency.


`prebuilt_damage_pool.py` prepares blocks of sampled supports before optimizer
updates, and selects an entry with a dynamic scalar index inside JIT. The
compiled objective gathers current frozen history/features and fixed basis
rows on device. Use a fixed pool size for all blocks; when refreshing blocks,
retain optimizer state and the detached sampling probabilities. Fresh blocks
may preserve the exact core-draw stream while each entry is used once. Pool
preparation and transfer remain real costs and must be included in timing.

`l1_damage_loading.py` supplies a separate loading experiment from exact zero
physical displacement, damage and history, with fresh sigmoid-network weights
and precomputed spatial basis columns normalized by their latent RMS.
The fixed basis retains centers from earlier geometry/neural-residual selection;
all current readout coefficients and MLP weights are freshly initialized.
`configs/l1_damage_loading_from_zero.py` declares displacement increments
`5e-4` through `0.0050`, then `1e-4` through `0.0060`. The network's initial
constant `1e-3` damage is an optimization seed; it is not the physical state
at zero load. The first positive-load local elastic solve provides nonzero
own history, so the normalized residual loss is never evaluated at zero H.

Training uses the original nonlocal AT2 residual, normalized by twice the
full-history RMS, with an L1 mean and `1/(N*p)` importance correction. The
declared crack-segment sampling distribution is detached and positive
everywhere; each Adam update consumes a fresh core sample with full halo.
There is no extra crack/residual weighting, direct-damage label or output
reference field. Both the MLP and compact readout remain trainable. Adam
moments/count persist across pool, outer and load changes, with an explicit
constant `1e-5` rate and at most `128` outer sweeps of `512` inner updates.

The original local P1 mechanics, half-relaxed stagger scheme and committed
previous-load maximum-history floor are retained. Full inner/current AT2,
range, healing, drift, mechanics and update gates require three consecutive
passes. Budget exhaustion permits only labelled provisional advancement:
later loads retain provisional ancestry even if they pass their local gates.
Each load saves fields, network, optimizer, hashes and a reaction/readiness
report; a sampling-block journal records RNG states and core hashes without
storing the full core trace. `pending` is the latest restart record and named
outer checkpoints are immutable. Matched-load reference comparisons must use
the actual displacement, since sequential state IDs differ from older grids.

```bash
source .venv/bin/activate
PYTHONPATH=.:src JAX_ENABLE_X64=true python -m examples.fracture2d.l1_damage_loading \
  --config examples/fracture2d/configs/l1_damage_loading_from_zero.py
```

Run this on the GPU with other numerical GPU workers suspended; do not run
FEniCS MPI concurrently. Results and the exact current resume point live in
the active project task, rather than in this stable usage guide.


## Mechanics-first stagger and field-change control

`staggered_damage_trial.py` is an isolated common-parent fixed-load comparison
using the existing local P1 mechanics/nonlocal AT2 damage model. Each outer
iteration solves mechanics at the current damage, forms the committed-floor
history \(H=\max(H_{n-1},\psi^+(u))\), freezes that history during a warm-start
damage solve, then checks displacement/damage field changes. It compares
relaxation0.5 and direct replacement1 for both direct damage and L1 random Adam.
`staggered_damage.py` provides the shared sweep and exact material P1 norms.
The reference loop is [Phase_Field_Simulation.py](https://github.com/saugatsn/Phase-Field-Modeling/blob/main/Phase_Field_Simulation.py); its constitutive/history/boundary choices are not copied.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.staggered_damage_trial \
  --config examples/fracture2d/configs/staggered_damage_trial.py
```

Serialize this GPU command with other GPU solvers. The configured control
uses own-L1 physical/network/Adam state at0.0052 and its next random stream,
then holds0.0053. The source path is explicitly provisional. A budget exit
never advances load. The new configuration pins all source reports/state files,
uses direct caps2048 and neural caps128x512, and declares all field tolerances.

For a material P1 field \(v\), field changes use \(\|v\|_{L^2}/\sqrt{|\Omega_m|}\),
integrated with the consistent triangle mass matrix. Displacement change is
additionally divided by prescribed positive displacement. The configured
limits are1e-4 for normalized displacement RMS and1e-5 for damage RMS,
minimum3 sweeps and3 consecutive passes. Looser1e-3 checks are reported
separately. Original AT2 residuals remain diagnostics; a `field_stable` result
is not automatically an equation-balanced or certified physical path.

A mechanics-first outer snapshot contains q equilibrated with its pre-update
damage, then new damage. Reports explicitly label this stage pairing. There
is exactly one mechanics call per outer and one endpoint synchronization
solve per case so final fields/reaction are consistent. Current older L1
code has one initial mechanics call per load plus one per outer, not two
per outer; order changes alone need not accelerate the fixed-point map.

Independent CPU replay checks saved stage/final force/BC/history/CSR residual,
consistent-mass field norms, native NumPy neural output, Adam count and
seeded sampling streams:

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.staggered_damage_audit \
  --root /root/autodl-tmp/tf-logs/fracture2d/staggered_damage_trial_20261013 --watch
```

Original live workers and archived stage sources remain unchanged. Final
convergence comparisons must account for inner precision, parent history
and equal work, rather than interpret a smaller stopping metric as proof
of accurate fracture equilibrium. Current results/resume stay in the active task.

`staggered_damage_continue.py` extends a pinned, independently audited stage
without committing its same-load history as the preceding-load floor. The
equal-outer-count configurations hold0.0053 and use644 unrelaxed outer
updates, matching the completed common-parent direct control. The512-inner
baseline continues stage128 with unchanged Adam moments/count and next RNG;
the128-inner control starts from that baseline's original0.0052 parent.
Field stability is recorded, but does not stop this equal-count diagnostic
early. These runs compare equal outer work and different inner work.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.staggered_damage_continue \
  --config examples/fracture2d/configs/staggered_damage_644.py
```

Run the reduced-inner branch separately with
`configs/staggered_damage_644_inner128.py`; serialize both GPU commands with
other GPU solvers. `staggered_damage_continue_audit.py --root <output> --watch`
checks the pinned restart/RNG/history provenance, then reuses independent
native-network/CSR/P1/host-FEM/Adam/sampling replay. Snapshot q remains the
mechanics-stage field; final q/reaction gets one synchronization solve.
Preflight and the archived previous endpoint's reporting-only mechanics
solve are recorded separately from the644 modeled outer mechanics calls.

`full_damage_trial.py` evaluates all material nodes on every damage training
update, using the same nodal L1 residual and normalization as the random
controls. It starts from their own-L1 .0052 physical/network parent and
holds .0053 with unrelaxed mechanics-first stagger. No symmetry constraint,
straight-crack target or new residual weight is imposed.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.full_damage_trial \
  --config examples/fracture2d/configs/full_damage_trial.py
```

The explicit configuration screens full Adam128 updates and full L-BFGS128
iterations at identical frozen H/weights, then selects the smaller exact
full L1 loss. Pilot states are archived and do not advance the physical
state; coupled training begins from the original parent. Each coupled outer
allows128 inner updates/iterations through644 outers. L-BFGS uses a cached
fixed-layout value/gradient, explicit function/line-search budgets, warm
weights and fresh curvature at each new H. Its finite unsuccessful optimizer
status is recorded without being presented as equation convergence. Adam
retains parent moments/count. This empirical screen is not a universal
optimizer ranking or a pure sampling comparison if L-BFGS is selected.

`full_damage_audit.py --root <output> --watch` independently checks native
network output, the exact full L1 loss, optimizer screen, committed history,
P1 changes, CSR residuals, BCs and host FEM forces. Compare morphology at
matched load and outer/crack-growth stages: a shorter crack may look cleaner
without being more accurate. Serialize the GPU run with existing solvers.

### Frozen-history damage-weighted L1 diagnostic

`weighted_damage_fixedH.py` starts three all-node fits from the same audited
network and Adam state. The configuration explicitly selects uniform weights,
\(1/(1-d_*+10^{-3})\), and that reciprocal weight capped at50. Here
\(d_*\) is the source physical damage; weights and history remain fixed
throughout each fit. Mean normalization and `stop_gradient` are applied:

$$
\widetilde w_i=\frac{w_i}{N^{-1}\sum_j w_j},\qquad
\mathcal L=\frac{\sum_i\operatorname{stop\_gradient}(\widetilde w_i)|\rho_i|}
{\sum_i\operatorname{stop\_gradient}(\widetilde w_i)}.
$$

\(\rho_i\) retains the existing AT2 residual normalization. These weights
emphasize high damage, whereas L16 emphasizes large residuals. They have
different training objectives. Retained Adam moments are part of this warm
start diagnostic; it does not isolate optimizer transients after changing loss.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.weighted_damage_fixedH \
  --config examples/fracture2d/configs/weighted_damage_fixedH.py
```

`weighted_damage_audit.py --root <output> --watch` independently checks
native network output, weights, full CSR losses, persistent Adam counts and
field errors against a direct solve at exactly the same frozen H. That solve
is evaluation only and supplies no training targets. Compare weighted and
unweighted losses together with field accuracy. No diagnostic damage is
promoted, no loading advances, and no coupled reweighting is enabled.

### Common-parent L1/L16 optimizer comparison

`damage_norm_trial.py` uses the same .0052 sigmoid network and the archived,
audited first .0053 mechanics-generated history for all three fits: L1/Adam,
L16/Adam and L16/L-BFGS. Every update evaluates all material nodes. Adam starts
identical zero moments in both cases and retains them throughout each fit;
the original parent moments are preserved in their source archive. This
initialization isolates the loss comparison from inherited L1 moments.

`damage_norm.py` retains the previous normalized nodal L1 and the archived
stable squared L16 objective, with power16 and explicit positive floor:

$$
\rho_i=\frac{r_i}{2\sqrt{N^{-1}\sum_j H_j^2}},\qquad
\mathcal L_1=N^{-1}\sum_i|\rho_i|,\qquad
\mathcal L_{16}=\left(N^{-1}\sum_i\rho_i^{16}+\epsilon^{16}\right)^{1/8}.
$$

The implementation rescales the L16 computation to avoid overflow and detaches
the scale and all history/features; it does not change the physical residual.
The screen allows4096 Adam updates or one continuous4096-iteration L-BFGS
fit with explicit function/line-search budgets. Record actual work and status.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.damage_norm_trial \
  --config examples/fracture2d/configs/damage_norm_trial.py
```

`damage_norm_audit.py --root <output> --watch` independently replays native
outputs, full CSR objectives/residuals, source mechanics/history, Adam counts
and field errors to the unique direct solution at the same H. This oracle is
evaluation only. Loss values from different objectives cannot rank accuracy.
Eligibility for a coupled L16 followup requires same-H field L2 and normalized
max residual no worse than the final L1/Adam baseline; choose smaller field L2
if both L16 candidates qualify. Screen fits do not advance physical states.

`damage_norm_coupled.py` is a separate coupled morphology experiment. It
restarts the original .0052 physical/network parent with L16/full-node/L-BFGS
at fixed .0053, using mechanics-first omega1,128 inner iterations and644
outer updates. Networks warm-start; curvature restarts when H changes.
The completed frozen-H screen is preserved, including its selection result;
user morphology priority explicitly permits this followup despite larger L2.
Screen-fit states are discarded and no straightness/symmetry loss is added.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  .venv/bin/python -m examples.fracture2d.damage_norm_coupled \
  --config examples/fracture2d/configs/damage_norm_coupled.py
```

`damage_norm_coupled_audit.py --root <output> --watch` checks native output,
full L16/CSR residuals,committed history,BCs,host FEM,P1 changes and actual
work counts. Evaluation-only P1 transverse cuts record peak position and
contiguous half-height width. Compare overlapping active crack intervals and
adequate developed lengths; a short crack is not evidence of straighter growth.
Mesh/interpolation resolution limits these geometric measurements. Global
field error,peak residual and reaction remain reported; morphology alone
does not certify equilibrium. The fixed-load budget does not advance loading.

### Cold all-node load-step controls

`full_damage_loading.py` compares cold L1/Adam/all-node loading with
`configs/full_damage_load_coarse.py` and `configs/full_damage_load_fine.py`.
Only case identity and the increment through displacement0.005 differ:
5e-4 versus1e-4. Both then use1e-4 through0.0053 (13 versus53 positive
loads). Local P1 mechanics, nonlocal AT2 damage, sigmoid output, latent-basis
scales, physical history features and network seed remain identical. Initial
serialized network/optimizer hashes and the remaining configuration must match.
No symmetry constraint, crack-path target or reference training label is used.

Each mechanics-first unrelaxed sweep freezes the preceding-load maximum-history
floor and current mechanical driver during128 full-node Adam updates. Adam1e-5
moments/count persist across sweeps and loads. Prefix loads allow128 outer
sweeps;0.0053 allows644. Exact P1 damage RMS1e-5 and displacement RMS/load1e-4
must pass three consecutive checks after at least three sweeps to stop early.
Residuals are diagnostics, not stopping gates. Each load ends with one mechanics
synchronization. Budget-based continuation and field-stable endpoints are
provisional; this diagnostic never declares a certified physical path.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m examples.fracture2d.full_damage_loading \
  --config examples/fracture2d/configs/full_damage_load_coarse.py
```

Run the fine configuration after the coarse branch has created the shared
initialization receipt; serialize GPU work. Both refuse existing branch
identities. `full_damage_loading_audit.py --root <case-root> --watch` runs on
CPU and checks native output, host CSR loss, history floors, boundary values,
FEM forces, P1 changes, optimizer counts and matching-load reference errors.
At0.0050/0.0052/0.0053, a direct solve at the exact own history evaluates neural
accuracy without supplying training targets. Checkpoints and scalar timing
records support total-work comparisons: finer loading permits more training,
so an improvement alone does not isolate step size from its added cost.

### Best-current full-node loading from a fine prefix

`best_full_damage_loading.py` and `configs/best_full_damage_loading.py` retain
the audited own-zero fine prefix through displacement0.0052 and recompute
0.0053 through the explicit terminal displacement0.0065 with1e-4 increments.
The complete output contains the52 inherited positive-load endpoints and13
new loads. Prefix fields/network/optimizer files are copied with source hashes;
the source0.0053 endpoint is not inherited or used as a history floor.

The physical model, sigmoid network, latent scales, physical history features,
all71307 residual rows, L1/Adam1e-5 and unrelaxed mechanics-first stagger are
preserved. Restore Adam count384000 and moments with the0.0052 network.
Each inner solve uses128 updates. The archived original configuration allows
4096 outer sweeps; `configs/best_full_damage_loading_1024.py` is the
user-selected restart with1024 outer sweeps and a separate output root,
with the existing P1 field-change early stop. Frozen history, features and
uniform nodal probabilities remain detached. One final mechanical solve
synchronizes each endpoint. No symmetry constraint or straight-path loss
is introduced. Residuals and ranges remain diagnostics; advancing a budget
endpoint retains explicit provisional ancestry rather than certification.

```bash
PYTHONPATH=.:src JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m examples.fracture2d.best_full_damage_loading \
  --config examples/fracture2d/configs/best_full_damage_loading_1024.py
```

`best_full_damage_loading_audit.py --root <output>/fine` runs on CPU. It checks
fork pins and copied prefix provenance, then reuses the native/CSR/history/BC/
FEM/P1/optimizer/matching-load replay from `full_damage_loading_audit.py`.
The driver records restored-image parity, first-history loss parity and a
discarded compilation update; compiling never advances optimizer memory.
Longer coupling is a numerical trial, not a guarantee of full fracture or
passed equation gates. Existing source runs and their source snapshots remain
protected; use a fresh output root for a distinct loading policy.

### TensorBoard monitoring of the full-node loading trajectory

`field_monitor.py --config configs/best_full_damage_fields.py`
runs separately on CPU. It uses the same `tensorboardX.SummaryWriter` and
`add_figure` API as `fracture1d/train.py`. Saved scalar journals are read every
five seconds; scalars are recorded at the first outer sweep and every eight
sweeps. The global outer step derives from the persistent optimizer count.

Every completed load publishes `train/damage_field`, `train/ux_field`,
`train/uy_field`, `train/history_field` and `train/load_reaction_curve` images,
indexed by the physical load index. The0.0052 inherited endpoint supplies the
initial image. Damage panels compare PINN with the EXACT matching-load direct
reference, with shared0..1 colors. Displacement color ranges use each field's
actual material-node minimum/maximum. The reaction curve includes only
completed, mechanically synchronized endpoints; outer-stage reaction scalars
are explicitly named `diag/stage_reaction_for_previous_damage`.

Events live under `/root/autodl-tmp/tf-logs/fracture2d/events/`; the current
recovered run is `best_full_pinn_loading_1024_tbroot_20261014`. TensorBoard
on port6007 reads `/root/autodl-tmp/tf-logs` directly, without an alias. The
process entry point is named `field_monitor` so restarting TensorBoard server
processes by name does not also kill the field logger. The monitor
also saves per-load PNG files under `<solution>/tensorboard_figures/` and
publishing metadata in `tensorboard_status.json`. Plotting does not change
the history floor, sampling, JIT shapes, training loss or optimizer memory.
Native numerical validation remains the independent CPU auditor's task.

### Fresh complete loading through 0.0060

`cold_full_damage_run.py` supervises one complete, newly initialized trajectory.
It first invokes the unchanged cold-loading solver for52 positive loads at
1e-4 spacing through0.0052. Physical q/d/H are exactly zero initially;
network parameters are freshly initialized and Adam count/moments start at
zero. No earlier solved network or optimizer checkpoint is loaded. The
geometry-only compact basis and independently validated operators are reused.

After the new prefix finishes and passes its complete native CPU audit, the
existing continuation solver consumes only that same run's own0.0052 state
and computes0.0053 through0.0060. Prefix loads allow128 outer sweeps; tail
loads allow1024. Each outer uses128 full-node L1/Adam1e-5 updates, sigmoid
output, FP64 and mechanics-first omega1 stagger. Weights and Adam memory
persist across the new trajectory. The existing P1 field-change early stop
remains; capped or field-stable positive endpoints are explicitly provisional.
No symmetry constraint, prescribed straight path or reference training label
is introduced. The first prefix case directory is called `prefix/coarse`
because that name initializes the existing paired driver; its actual increment
is1e-4 throughout. The combined complete trajectory is in `fine/`.

```bash
PYTHONPATH=.:src .venv/bin/python -m examples.fracture2d.cold_full_damage_run \
  --config examples/fracture2d/configs/cold_full_damage_run.py
```

Run this supervisor detached and serialize it with all other GPU numerical
workers. It refuses an existing output identity or a busy GPU, records process
launch/exit receipts, and terminates its own children if training, auditing or
monitoring fails. The canonical output is
`/root/autodl-tmp/tf-logs/fracture2d/cold_full_pinn_loading_0060_20261014`;
change all four matching configs before reusing a new run identity. The
continuation config is frozen from the new audited prefix at the phase boundary,
not from an older run.

`cold_full_damage_monitor.py` is a read-only CPU consumer. It logs exact-zero
initial reaction/count, outer scalars from optimizer count0, and damage/ux/uy/H
plus reaction-curve images for every completed positive load. It keeps the
prefix journals during the transition so scalar steps remain monotone and
endpoint images are not duplicated. Images use synchronized q/d; reference
comparison is at the exact same physical load. Displacement colors use each
material field's actual extrema. Events are in
`/root/autodl-tmp/tf-logs/fracture2d/events/cold_full_pinn_loading_0060_20261014`;
PNG figures and global monitoring status are in the run root. The existing
TensorBoard6007 server is reused. The monitor argv contains no `tensorboard`
substring, keeping it separate from server process-name restarts.

Independent prefix and continuation auditors replay native neural output,
CSR loss, committed history, host FEM forces, optimizer counts and endpoint
ancestry. Supervisor success requires all60 positive loads and all60 published
endpoint images. Numerical accuracy limitations and live resume details remain
in the active task.
