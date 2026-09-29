# Fracture2D Nonlocal PINN

## Goal

Build the 2D nonlocal PINN stage on top of the adopted direct reference. The
PINN must reuse the **same** material cloud, union collar geometry, A2
phase-measured mechanics energy, Amor history, and nonlocal AT2 damage law
as `examples/fracture2d/nonlocal_baseline.py` and the shared
`pinn.operators.ExplicitVolumeSparseD`. Its user-approved FP32 strict-horizon
graph is separately rebuilt, not identical to the archived FP64 graph; all
discretization differences must be independently checked and reported.

## Acceptance Criteria

- Strong-form nonlocal residuals (mechanics, Amor history, AT2 damage) reuse
  the direct A2 residual method and physical laws/parameters. The approved FP32
  discretization independently rebuilds/audits its graph/operator and reports
  differences from the frozen FP64 direct graph rather than claiming identity.
- Training uses the archived material cloud (`material_cloud_d048.npz`; its
  stored `d048` FP64 edge list is a comparator, not the PINN edge list) and the
  adopted parameters (`delta=0.048`, `h_g=0.02`, `l0=0.024`, `Gc=2.7`,
  `E=210e3`, `nu=0.3`, `k_res=1e-8`, cubic kernel).
- Comparisons against the adopted direct reference distinguish the first 59
  strict accepted states (`u_bar <= 0.0059`), the drift-free but warning-accepted
  `0.0060` and `0.0061` states, and the drift/warning-mixed tail from `0.0062`.
  Drift points are never hidden or deleted from the comparison.
- No model change (kernel, split, degradation, BC, particle set) is introduced
  without an explicit decision entry in `docs/DECISIONS.md`.
- Fixed shapes/dtypes in JIT paths; CPU/GPU parity checks where applicable.

## Relevant Context

- Adopted direct solution: compact `solution_u0p0075_fresh_v1.npz` (SHA256
  `5f3bc12b...`); research archive SHA256 `9545bed7...`; peak `641.34017` at
  `0.0061`, terminal `69.15426`, drift `2 -> 31`.
- Known limits of the direct reference: warning-only acceptance,
  `field_equations_validated=false`, terminal reaction far above FEM, and
  post-localization particle drift. The PINN stage must not treat the warning
  tail as converged truth.
- The adopted direct implementation is the single GPU-only file
  `examples/fracture2d/nonlocal_baseline.py` (union geometry, BA/A2 mechanics,
  Amor, AT2, JAX PCG backend, fixed damped-Picard continuation, and the
  `run`/`verify`/`plot` CLI). The earlier `src/pinn/nonlocal2d/` package was
  merged back into this file and deleted. Reuse its functions/operators rather
  than re-deriving them.
- Reference documents: `examples/fracture2d/README.md` — the
  **Reference solutions and data** section holds every reference-solution path
  and SHA256, and **Nonlocal PINN (in progress)** holds the operator/model
  reuse, strong-form residuals, reference-comparison protocol, and training
  convergence criteria. Also `docs/NUMERICAL_METHODS.md`,
  `docs/JAX_GUIDELINES.md`.

## Status

Active (in progress). The direct stage is closed: the single GPU-only solver
`examples/fracture2d/nonlocal_baseline.py` has confirmed default behavior
against the archived solution
([archived lean task](archive/fracture2d-lean-direct-solver.md)).
Documentation handoff checked: the current reuse entry is the single-file
solver; the README's AT2 strong residual now matches its positive
`D_delta` operator convention.
The differentiable full-graph A2/Amor/AT2 residual bridge and a GPU single-load
PINN experiment are implemented in `examples/fracture2d/losses.py` and
`train.py`. The frozen 3395-node cloud and archived first state pass a CPU
same-field residual/history check. An earlier 50-step FP64 GPU experiment at
`u_bar=1e-4` ran with zero drift but was not converged: last pre-update loss
`0.0015224`, raw A2 residual infinity norm
`0.344`, material `u` relative L2 `0.394`, `d` `0.978`, `H` `0.959`, and
reaction `21.321` versus reference `12.675`. No state is accepted or advanced.
FP32 is now the training default: a separately rebuilt material/union graph,
phase quadrature and BA supports live in `examples/fracture2d/fp32.py`.
It changes candidate undirected horizon pairs by +209/-76 and material directed
edges by +76/-152 versus FP64; the material particle count and exact
collar/anchor masks are unchanged. Maximum BA condition is `20.24` and
affine/energy-gradient checks pass. On the first archived state, FP32 raw free
mechanics residual infinity norm is `0.00451` versus `1.38e-9` under FP64,
damage residual `2.79e-8`, history residual `7.96e-6`, reaction `12.548`
versus `12.675`. One FP32 GPU training step completed, without convergence.
Focused full-cloud tests pass `6/6` (`81.1 s`) for direct CPU-vs-JAX oracle,
FP32 graph counts/strict BC masks, affine-patch deformation and energy gradient,
first-state discrepancy, exact one-pass Amor history, independent q/d network
output shapes/configuration, and per-state warning labels. The post-change FP32
one-step smoke checks every trainable array and residual is FP32; predicted
reaction `8.342` versus archived FP64 reference `12.675` and material `u`
relative L2 `0.404`. Numerical precision or accuracy of later load states has
not been established. Direct-model construction and reference diagnostics still
run on the host in FP64; the compiled PINN training path is FP32.
Controlled first-load training diagnostics (`u_bar=1e-4`, seed 0, `lr=1e-4`):
with the original fixed weights, 200-step loss fell from `0.002608` to
`0.000990` but reaction drifted to `27.166` (reference `12.675`). Freezing
initial-equation RMS normalization (`mechanics=0.05105`, `damage=0.002776`,
`history=0.001388`) and running 1000 steps gives plain-MLP normalized loss
`0.2243` (`u` relative L2 `0.433`, reaction `22.762`, `9` drift points);
Fourier-MLP loss `0.1855` (`u` relative L2 `0.420`, reaction `28.200`, zero
drift points). These are lower optimization losses, not accepted states.
The fixed-damage energy-only pilot was interrupted and has no result.
Following the two-equation stagger suggestion, `stagger.py` now trains
independent FP32 displacement/damage networks with Amor history evaluated by
the exact maximum from the unchanged committed history. Short first-load
diagnostic (`1` sweep, `100/100/100` mechanics/damage/mechanics steps) gives
stage losses `0.832/0.780/0.526`, final raw mechanics/damage residual maxima
`0.334/0.0117`, reaction `22.960`; the final history defect measured by
two separate FP32 GPU reductions was `2.7e-6`. The final residual is now
evaluated with the same Amor reduction as its trial history (zero algebraic
defect by construction). Longer one-sweep diagnostic (`300/600/300` steps)
gives stage losses `0.350/0.084/0.218`, final raw mechanics/damage maxima
`0.536/0.159`, reaction `29.220` versus reference `12.675`, `u` relative L2
`0.370`. A post-reporting short `10/10/10` smoke completed with exact zero
same-pass history residual and separate raw mechanics-before/damage-solved
diagnostics. These remain *unaccepted*: at fixed history the damage network can
reduce its stage loss, yet the under-relaxed, re-equilibrated coupled trial has
a large residual. Further blind sweeps are not justified by these values.
The committed history is the *cumulative accepted* Amor maximum, not the
previous step's instantaneous energy. The current code explicitly detaches
this committed array and the fixed q/d subproblem inputs with
`jax.lax.stop_gradient`, while retaining differentiation through the current
Amor driver in the optional single-network consistency loss. A two-row JAX
gradient test checks zero derivative through committed history and nonzero
derivative through an active current driver; the forward discrete equations
and earlier FP32 reference/training artifact are unchanged. The full-cloud
focused suite passes `8/8` after the gradient change; no expensive 36k-step
suite was rerun because the forward model and archived weights are unchanged.

### Long-run evidence (2026-09-28)

The user observed that even the simpler `elasticity2d` case needs tens of
thousands of training steps; its current config specifies `40,000`
iterations (`examples/elasticity2d/configs/train_base.py:46`; the example
README's stale `20,000` entry was corrected). To test step-budget sensitivity
without changing physics or Adam `lr=1e-4`, completed one first-strict-load FP32
stagger sweep with `12,000` mechanics-before, `12,000` damage, and `12,000`
mechanics-after steps (`36,000` network updates). The run is **complete but
unaccepted**. Logged mechanics-before loss declined but its raw free residual
infinity norm reached `1.476` after 12k and material `u` relative L2 `0.577`.
The damage stage ended with normalized loss `0.862` and fixed-history raw
damage-residual infinity norm `1.308`. After the adopted `omega=0.5` damage
update and mechanics re-solve, the final *coupled* A2/AT2 residual infinity
norms are `1.764/2.019`, exact same-pass history defect is `0`, reaction
`23.145786` versus FP64 reference `12.674947`, material relative L2 errors
`u=0.714415`, `d=0.965439`, `H=17.603168`, and `22` drift points at
`u_bar=0.0001` (FP64 reference: `0`). Low stage loss is not convergence.

- Run directory: `/root/autodl-tmp/data/fracture2d/pinn_stagger_12000_v1/`.
  TensorBoard scalars were moved, at user request, to
  `/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_12000_v1/` (24 records
  per fixed stage plus one coupled record at step `36000`). Config and final
  checkpoint remain under the original data run directory. The earlier
  `pinn_stagger_resume_smoke_v1` event log was moved to the same TensorBoard
  root under its own run-name directory.
- A fresh one-step-per-phase layout smoke wrote events directly to
  `/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_tf_layout_smoke_v1/`,
  while its config/checkpoint stayed under the matching
  `/root/autodl-tmp/data/fracture2d/` run directory. The TensorBoard event
  reader confirmed `loss/before` and coupled mechanics scalar steps for all
  three runs (`36000`, `6`, and `3` respectively). The completed long-run
  checkpoint file SHA256 remains
  `16a8cb077003797b188a5131b8b66d0afbe83e93798e3a4241423e39cb53f31d`
  with a valid state payload digest; no numerical state was modified during
  event relocation. Focused stagger setup/checkpoint tests pass `2/2` and
  `git diff --check` passes.
- Atomic bound final checkpoint: `latest.ckpt.zip`, file SHA256
  `16a8cb077003797b188a5131b8b66d0afbe83e93798e3a4241423e39cb53f31d`,
  `schema=fracture2d-fp32-stagger-checkpoint-v1`, `phase=complete`,
  `sweep=1`, state payload digest verified; FP32 graph SHA256
  `0fa22faca79125d676e64931ae69378d42a564f29dee7fdf02105338905d7fe5`.
  The checkpoint retains q/d networks, Adam states and trial arrays; it is
  **not** an accepted solution or another load's committed state.
- Validation: `PYTHONPATH=src .venv/bin/python -m unittest
  tests.test_fracture2d_pinn_residual -v` passed `7/7`; a real `2/2/2`
  stagger smoke wrote TensorBoard events and successive checkpoints, and the
  event reader verified the long-run stage/coupled scalar tags, steps and
  terminal values. No FEM/JAX overlap was used.

### Network construction refactor (2026-09-28)

`train.py::_build_network` now uses `pinn.arch.make_pinn` (as in
`fracture1d`/`elasticity2d`) instead of a bespoke `eqx.nn.MLP` +
`_FourierNetwork`. `train.py` remains **one four-output network**
(`u_x, u_y, d, H`); `stagger.py` uses **two independent networks** (two-output
displacement, one-output damage) built by the same helper. New explicit
`arch.*` config fields (`type`, `hidden_dim`, `num_layers`, `activation`,
`use_spatial_emb`, `spatial_emb_dim`, `spatial_scale`, `multi_head`,
`tensor_split`, film sizes) replace `arch.width/depth/encoding/frequencies`;
the `--encoding` CLI flag was removed. The final field projection is zeroed
after construction so the initial fields still equal the affine lift. A
one-step `train.py` smoke and a `2/2/2` stagger smoke completed (different
parameter init than pre-refactor runs; physical model unchanged); focused
tests pass `8/8`.

Both entry points now load `configs/train_base.py` through a single shared
`ml_collections`/absl config flag (`train.py` defines `--config`,
`stagger.py` reuses it), exactly like the other examples; argparse was
removed, `--config.output.resume=True` replaces `--resume`, and the
stagger budget/output settings are plain config fields. Default network size
is now `hidden_dim=128, num_layers=4` (was `32 x 2`). Config-based one-step
`train.py` and `2/2/2` stagger smokes passed with the enlarged network.

Run bookkeeping: the earlier `pinn_stagger_12000_v3` directory (created
before these refactors, config shows the old `width/depth/encoding` fields)
was an aborted attempt with only a config and an empty TensorBoard event file,
so it was renamed to `pinn_stagger_12000_v3_aborted_prerefactor` in both the
data and TensorBoard roots to preserve it without overwriting.

### Wider-network v3 run (2026-09-28)

With the config-flag plumbing and `arch.hidden_dim=128, num_layers=4`, the
identical first-load budget (`12k/12k/12k`, 36,000 updates) was re-run as
`pinn_stagger_12000_v3`. The training loop took about 83 s wall time on the
RTX 4080 (the earlier estimate of tens of minutes was far too conservative for
this small per-step problem). TensorBoard again has 72 per-report points on
every `loss/*`, `eval/*` and `diag/*` tag, with `diag/accepted=0`.

Compared with the `32 x 2` v2 run, the wider network removed the displacement
drift (0 drift points throughout, versus up to 22 in v2) and drove the
fixed-history damage stage much lower (`loss/damage` `0.9379 -> 0.00916`).
The coupled diagnostic still fails to settle: `diag/coupled_loss` rises
`10.71 -> 33.02` (steps 500/1000), falls to `12.11` at 12,000, reaches its
minimum `0.358` at 24,000 (`diag/damage_residual_inf` `0.0497`), then the
mechanics-after re-solve pushes it to `3.72` at 24,500, `3.87` at 30,000 and
`6.00` at 36,000 (`diag/mechanics_residual_inf` `0.9755`,
`diag/damage_residual_inf` `0.3767`). `eval/rel_L2_u` ends `0.539`,
`eval/rel_L2_d` `1.526`, `eval/rel_L2_H` `5.406`, and `eval/reaction` `33.00`
versus the FP64 reference `12.67`. The run is complete but **unaccepted**: the
larger network improves stability and the fixed-history damage solve, not the
coupled equilibrium. Checkpoint `latest.ckpt.zip` in the run directory, file
SHA256 `ba613a4795b61b1d32339a8dc324e1f983a0fa5acece0626bcb6f0fbace89091`,
state digest verified, `phase=complete`, numerical config SHA256 prefix
`9e71f2eeb4cb43f5`.

### TensorBoard loss audit and v2 long run (2026-09-28)

Inspection of the immutable v1 36k-step run's event file confirms
`loss/before`, `loss/damage`, and `loss/after` each have 24 entries, while
every `coupled/*` tag has only the last point at global step `36000`. In that
historical code the stage losses were measured before Adam updated the
network, but raw residual and reaction tags at the same step were measured
afterward. The v1 event file is retained without reconstruction or relabeling.

The user then requested (a) a real re-run to inspect the actual loss curves
and (b) tag names consistent with the repository convention. `stagger.py` now
writes, per the `corrosion2d`/`fracture1d` convention:

- `loss/mechanics_before`, `loss/damage`, `loss/mechanics_after` — the three
  stage objectives, evaluated at the reporting step from the same post-update
  state as every other logged value. The earlier `loss_preupdate/*` group was
  removed so there is no pre/post-update ambiguity.
- `eval/rel_L2_u`, `eval/rel_L2_d`, `eval/rel_L2_H`, `eval/reaction` —
  material-node comparisons against the FP64 archived reference, plus
  `eval/reaction_reference` at step zero.
- `diag/mechanics_residual_inf`, `diag/damage_residual_inf`,
  `diag/history_residual_inf`, `diag/coupled_loss` — raw full-graph residuals
  and the normalized two-equation diagnostic from one same-pass Amor
  reduction. The old `coupled/*`, `residual/*`, `scales/*` and
  `loss_preupdate/*` groups are gone. `diag/coupled_loss` is a diagnostic, not
  the optimizer's loss or an acceptance gate; the damage-phase `loss/damage`
  uses fixed `H_used`, so it can differ from `diag/damage_residual_inf`
  (current q-driven history).
- `diag/drift_count`, `diag/damage_min`, `diag/damage_max`,
  `diag/damage_healing_max`, `diag/mechanics_scale`,
  `diag/coupled_damage_scale`, and `diag/accepted=0` at the sweep end.

A fresh `2/2/2` verification run
(`/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_tb_convention_smoke_v1/`)
confirms stage losses at steps `[1,2]`, `[3,4]`, `[5,6]`, all `diag/*` and
`eval/*` curves at every `[1..6]`, scales at `0`, and `diag/accepted=0` at
step `6`; the old groups are absent.

A full identical-budget re-run then captured the actual curve history:
`pinn_stagger_12000_v2`, `12k/12k/12k`, 36,000 updates, same seed/network/
learning rate as v1. Under the then-current (pre-rename) tags, every
per-report diagnostic had **72 entries** (24 per phase at `--log-every 500`),
not the single terminal point of v1. The curves show the fixed-`H_used`
damage-stage loss decreasing (`0.9856 -> 0.8607`) while the coupled
diagnostic rises: `coupled/loss` (now named `diag/coupled_loss`) goes from
`1.603` at step 500 to `1.850` at 1,000, `75.11` at 12,000, `64.72` at
24,000, and `156.10` at 36,000; drift goes from 0 to 22 (first drift at step
24,500), `eval/rel_L2_u` from `0.362` to `0.713`, and reaction from `30.58`
to `23.18` (reference `12.67`). The run is **complete but unaccepted**
(`accepted=false`), reproducing the v1 failure with corrected logging.
Checkpoint `latest.ckpt.zip` under
`/root/autodl-tmp/data/fracture2d/pinn_stagger_12000_v2/`, file SHA256
`f0f23f7cee0774a5aa25223f1446deedc2b912337045a77ef30809bd3a5c9fb8`,
state digest verified, `phase=complete`. This is logging and failure evidence,
**not** convergence.

### Operator audit and optimizer diagnosis (2026-09-28, after v3)

The v3 curves show `loss/mechanics_after` rising monotonically after the
damage stage (`0.2503 -> 0.3490` over 12,000 updates) and
`diag/mechanics_residual_inf` climbing with it. The implementation was audited
against `examples/fracture2d/nonlocal_baseline.py` before changing anything.

Operator equivalence, line by line:

- `fp32.py` rebuilds the BA data with exactly the baseline formulas
  (`omega = 120/(pi*delta^4)*(1-r/delta)^2`; `family_mass = sum_j omega_ij*V_j`;
  support `[j] + sorted(common neighbors)`; `beta = omega_ik*V_k`;
  `xi = x_k - x_i`; `S = sum beta*xi(x)xi`; `S^{-1}`; `eta = beta_0/m_i`),
  matching `independent_union_ba_reference` (lines 1160-1214); only the
  approved float32 coordinate/volume rounding differs.
- Mechanics deformation gradient, strain/stress, endpoint assembly, top
  reaction, center weights and grip degradation are the baseline's own
  `JaxA2UnionMechanics` methods (`_deformation_gradient` ignores damage in the
  strain; `_assemble_residual` adds `+endpoint` at support and `-endpoint` at
  source; reaction is `outer_top_selector @ residual`), fed with the FP32
  static data. Constraints/BCs reuse the same `AffineConstraints` object, and
  `lift` imposes them exactly.
- AT2 strong residual `Gc/l0*d + Gc*l0*(D@d) + 2H*(d-1)` and the
  `ExplicitVolumeSparseD` material operator match
  `_e_damage_residuals` (6126-6159) and `solve_fixed_history_damage`
  (7047-7175).
- Amor driver `0.5*(lambda+2*mu/3)*max(tr eps,0)^2 + mu*sum(dev^2)` over
  material-source edges with `eta` weights and `H = max(H_committed, driver)`
  matches `_a_amor_edge_energy` (2825-2844) and
  `compute_a2_amor_history_from_edge_strain` (2846-2886).

No operator discrepancy was found. Independent numerical checks (probe
scripts under `/tmp/opencode/`, recreated from this record if needed):

- The direct reference field evaluated under the FP32 PINN objective gives
  `loss = 2.2e-5` (mechanics residual RMS `2.4e-4`); the archived reaction
  `12.675` appears as `12.548` on the FP32 graph.
- Exact free-DOF solve of the FP32 mechanics operator with the v3 checkpoint's
  frozen damage (7,095 unknowns, unsymmetrized condition `2.46e4`, LU
  relative residual `1.8e-15`): objective `8.8e-6`, reaction `12.664`
  (`0.09%` from FP64 `12.675`), displacement rel-L2 versus the direct solution
  `0.12%`. The PINN objective is therefore solvable to float32 roundoff.
- Continuation from the v3 final checkpoint on the *same frozen* objective
  still rises: `0.3490 -> 0.3822` over 3,000 Adam steps at `lr=1e-4`, and
  `0.3490 -> 0.3570` at `lr=1e-5`.
- A single exact-gradient step with `alpha=1e-3` drops the loss to `0.2475`
  (better than 12,000 Adam steps), but plain gradient descent with that step
  diverges to NaN within 20 steps; the parameter-space problem is extremely
  stiff.
- `optax.lbfgs` is not a drop-in fix in this stack: the eager first update is
  good (`0.349 -> 0.217`) but the same update under `eqx.filter_jit` raises
  the loss to `4.45` (mixed float64/float32 scatter inside its line search).
- The FP32 GPU residual assembly (`jax.ops.segment_sum` in `_j_segment_sum`)
  is nondeterministic between evaluations: max-norm spread about `4e-5`
  (relative `1e-4`) over repeated calls. `--xla_gpu_deterministic_ops=true`
  made model build/compile impractically slow (aborted after 15 min).

Conclusion: the rising `mechanics_after` loss (and the earlier v2 coupled-loss
blow-up) is an optimizer failure on a stiff least-squares objective, not a
residual/BC/reaction/operator error. Further Adam budget or learning-rate
tuning is not justified.

### Root cause follow-up: the lift scatter compiled differently under jit (2026-09-28)

While validating the optimizer work, the jitted and eager objectives disagreed
20x on the same parameters (`0.3490` vs `7.2514`) even after the deterministic
assembly. It was not TF32 and not a captured global: `FractureResidual.lift`
imposed the collar/anchor values with
`prescribed.at[free_dofs].set(raw_q[free_dofs])`, and when that scatter was
fused into a jitted objective, XLA produced a different displacement field:
eager and jitted `q` differed by `2.07e-5`, and the jitted field exceeded the
prescribed top-collar value (`1.09e-4 > 1e-4`). The equivalent select blend
`jnp.where(free_mask, raw_q, prescribed)` agrees across paths (`q` max
difference `6.4e-9`; objective `0.34903` both; the two forms are identical
eagerly). `free_mask` is precomputed in `FractureResidual.__init__`, and
`test_jitted_lift_composition_matches_eager` guards the regression. Until this
fix, the jitted training objective and the eager stage reports were different
functions, so every pre-fix curve (v1/v2/v3 including the `mechanics_after`
rise) must be re-read as monitoring that corrupted pair, not as a valid
optimizer trace.

Two numerical-policy fixes accompany it: `train.py`/`stagger.py` set
`jax_default_matmul_precision = "highest"` (the default TF32 path made eager
and jitted network forwards differ by `~6e-5`, which the stiff residual
amplified), and `fp32.py` replaces the atomic `jax.ops.segment_sum` assembly
with deterministic per-segment scans (`_SegmentPack`/`_segmented_sum`);
repeated residuals are now bitwise identical and gradients agree to `~1e-7`
relative. Tests: `9/9`.

Post-fix optimizer status on the frozen first-load objective from the v3
checkpoint: Adam at `lr=1e-4` still rises (`0.3490 -> 0.3922` over 5,000
steps), so first-order methods still fail on the corrected objective. The new
host-side L-BFGS step (`examples/fracture2d/lbfgs.py`: two-loop recursion plus
scalar Armijo backtracking, validated on an ill-conditioned quadratic) descends
`0.3490 -> 0.0653` over about 700 iterations but plateaus there. L-BFGS is
**not yet integrated** into the stagger.

### Energy-norm diagnosis of the plateau (2026-09-28, probes)

Clean post-fix probes separate representation from optimization:

- Supervised Adam (`lr=3e-4`, 3,000 steps) fits the reference displacement to
  `0.7%` material relative L2, so the architecture can represent the reference
  field and Adam works on a well-conditioned target.
- The mechanics objective is the A-norm of the field error (A is the frozen
  tangent operator), which amplifies particle-scale error. At that `0.7%` L2
  point the free-DOF residual is `1.7e4` times the residual of the reference
  field itself. Linear A-optimal readouts of the output layer reach `4.2e3`
  (random trunk) and `1.7e3` (Adam-adapted trunk), i.e. the residual wall is
  not only in the readout. The L-BFGS plateau (`0.065` in the stagger loss
  units, ~`3e3` times the reference residual) sits at the same order as these
  readouts.
- The reference-field residual (`2.2e-5` in loss units) and the exact FP32
  solve (`8.8e-6`) are evaluated on the *assembled* FP32 residual of an
  essentially exact solution, i.e. they are dominated by float32 assembly
  rounding. Requiring the network to reach them means solving the linear
  system to machine precision, which is not a meaningful acceptance criterion
  for a neural approximation. The meaningful targets are field error,
  reaction, and a residual that is small relative to the loading scale.
- Invalidated probe results (do not cite): the earlier "supervised L-BFGS
  stalls at 13%" and "random Fourier features 81%" numbers came from probe
  bugs (one coefficient vector shared by both DOF components; a misplaced
  square). The direct-damage mechanics block is linear in the displacement,
  so a Gauss-Newton step can use the exact tangent.

Implication: the remaining mechanics bottleneck is the parameter-space
optimization of a stiff, high-frequency-sensitive residual.

### Gauss-Newton experiment (2026-09-28)

`examples/fracture2d/gauss_newton.py` implements the planned matrix-free
damped Gauss-Newton/Levenberg-Marquardt step: JVP/VJP through
`eqx.filter_jvp`/`eqx.filter_vjp`, truncated CG on the damped normal
equations in a fixed-shape `lax.while_loop`, host-side Armijo backtracking,
and geometric damping adaptation. It passes a Rosenbrock validation and
solves a well-scaled linear least-squares test.

On the frozen first-load mechanics objective it fails for a structural
reason. From the zero-output (affine base) network the first GN step is
excellent (`1.0 -> 0.155` in loss units), but at every subsequent point the
line search cannot find descent: at the first stuck point the gradient norm
is `6.7e4`, a typical GN direction has norm ~`80`, and the objective is flat
to `~1e-5` for every step size between `1` and `2^-45` of that direction
before exploding. The second accepted step from the base field needs
`step_size = 3.7e-9`; then damping climbs to `1e14` with no further
progress. The residual is the A-norm of the field error with
`kappa(A) ~ 2.5e4`, so the parameter-space objective is numerically jagged
at the `1e-5` parameter scale and no gradient/Hessian method can traverse it
in float32. The result also explains the L-BFGS plateau: that optimizer's
`0.065` field has `rel_L2_u ~ 0.5` because the residual is nearly blind to
near-rigid free-region modes, so a small residual does not imply an
accurate field.

Consequence: the next mechanics experiment should precondition the residual
before taking norms, e.g. minimize `||B R(q)||^2` with `B` an approximate
inverse of the frozen tangent operator (`B R ~ q - q*` when `B ~ A^{-1}`),
which keeps the exact solution as the unique zero but flattens the
parameter-space landscape toward the supervised L2 problem that Adam solves
(`0.7%` in 3,000 steps).

### Float64 evaluation, near-null modes, and distillation (2026-09-28)

The FP32 jaggedness is evaluation noise, not the parameterization. Rebuilding
the same experiment on the adopted float64 backend (`JaxA2UnionMechanics`,
same reference state) changes the picture completely:

- The float64 reference-field residual is `8.2e-10` in loss units (the FP32
  value is `2.2e-5`), so the FP32 floor is assembly noise.
- Raw-residual Adam in float64 decreases monotonically `1.0 -> 0.0267` over
  20,000 steps in 277 s (`~14 ms` per step). The host L-BFGS is much weaker
  here (`0.087` after 3,000 iterations in 860 s), so Adam stays the inner
  optimizer.
- The residual objective is nearly blind to near-rigid free-region modes, so
  the field does not follow: at loss `0.0267` the material rel-L2 is `0.32`
  and the reaction is `19.7` vs `12.7`. Exact preconditioning by `A^{-1}`
  would fix the field, but it is mathematically the same as training on the
  exact linear solve; approximate CG preconditioning needs about 200
  iterations per evaluation at this size and is too slow.
- Supervised distillation to the exact solution works well: float64 Adam,
  zero-output start, 8,000 steps in `~100 s` reaches `0.4-0.5%` material
  rel-L2 and reaction `12.5/11.3` vs `12.7`, with a reported residual loss
  of `0.08-0.16`. (Rigid drift is also strain-neutral, so the damage driver
  is less sensitive to it than the raw residual loss suggests.)

Decision point for the user: keep the honest residual-only mechanics stage
(float64 Adam, slow field convergence), or adopt the float64 exact solve as
the mechanics target (supervised distillation, accurate field) while the
damage stage stays a residual PINN.

Two further target-free attempts to fix the low-mode convergence were tested
and failed:

- Trainable affine field added to the ansatz (the low modes are rigid-like):
  the optimizer reduces the residual further (`0.051` vs plain `0.027` at
  20k) while the field gets worse (`48%` vs `32%`), so the affine freedom is
  spent on residual-reducing but field-wrong directions.
- Spectral preconditioner `||C R||^2` built from the 48 smallest matrix-free
  Lanczos eigenpairs of `A_ff` (`lambda_1 = 19.2` up to `lambda_max ~ 2.7e5`,
  `kappa ~ 1.4e4`, symmetric to `1.6e-15`), with weights `1/lambda` on the
  coarse subspace and `1/lambda_49` outside: the preconditioned loss drops
  `1.0 -> 6.5e-3` while the raw loss stalls at `0.30` and the field degrades
  to `60%`. The low-rank inverse re-balances the modes instead of fixing
  them; only a full `A^{-1}` makes all modes converge uniformly, which is
  equivalent to solving the linear system.

Conclusion: for the linear mechanics block, mode-balanced convergence and
field accuracy require the linear solve inside the loop (distillation or a
Newton-preconditioned gradient), while the pure residual objective in
float64 gives a monotone residual decrease with slow field/reaction
convergence. Rigid drift is strain-neutral, so damage coupling may still be
usable; the acceptance criterion decides.

### FP64 performance on the RTX 4080 (2026-09-28)

The earlier FP32 adoption was justified by a `3072^3` matmul benchmark
(`1.82 ms` FP32 vs `67.4 ms` FP64, `37x`). The actual workload is
edge-assembly dominated, so the relevant ratio is much smaller (warmed,
100 repetitions):

| workload | FP32 | FP64 | ratio |
|---|---|---|---|
| mechanics forward | 0.60 ms | 1.88 ms | 3.1x |
| mechanics value+grad | 2.10 ms | 4.33 ms | 2.1x |
| mechanics Adam step | 2.70 ms | 5.10 ms | 1.9x |
| damage Adam step | 0.59 ms | 2.77 ms | 4.7x |

A `12k/12k/12k` sweep costs about 2.6 min in FP64. FP64 is therefore
affordable for the end-to-end residual mechanics stage, and it removes both
the optimization noise and the FP32/FP64 graph mismatch (the reference
graph is the FP64 graph). Next: port the stagger to the FP64 model and run
one first-load sweep diagnostic.

### Float64 stagger port and first state-1 run (2026-09-28)

`stagger.py` now runs the adopted float64 reference graph by default
(`train.dtype = "float64"`), keeping the rebuilt float32 path behind
`train.dtype = "float32"`. The checkpoint schema is precision-specific
(`fracture2d-fp64-stagger-checkpoint-v2`), the identity binds the precision,
and `train.py` keeps the float32 single-network experiment with an explicit
error for float64. The existing residual and losses test suites pass
(`9/9` and `31/31`).

State-1 single sweep (`12k/12k/12k`, `lr=1e-4`, ~5 min):

- `before`: mechanics loss `0.077 -> 0.053` (`3k -> 12k`), drift `2 -> 16`.
- `damage`: loss `0.0038 -> 0.0028`, `damage_max ~ 5.8e-5`, no healing.
- `after`: mechanics loss `0.052 -> 0.019` and still falling at step 12k;
  drift `20 -> 0`; reaction `23.9 -> 15.0` vs reference `12.675`;
  `u_relative_l2` `0.471 -> 0.207`.
- Final coupled loss `0.170`, `accepted=false` (diagnostic by design).

For comparison, the FP32 v3 run (with the lift bug) had `mechanics_after`
rising to `156` with drift `22` and reaction `33`. The float64 end-to-end
stage is bounded and monotone, and the residual decreases through the stage;
the field/reaction gates are not met yet because the `after` loss has not
plateaued at 12k steps. Next: a larger stage budget and/or a second sweep
before judging acceptance.

### Budget comparison at state 1 (2026-09-28)

Two runs followed, both on the same adopted state-1 reference:

- Long single sweep `48k/48k/48k` (about 20 min): final `after` loss
  `0.00277`, `u_relative_l2 0.0278`, reaction `12.811` vs `12.675`, drift
  `0`, `H_relative_l2 0.039`, coupled loss `0.135`. The damage stage reached
  `damage_max ~ 6.2e-5` but `damage_min = -2.6e-8` (a slight negativity
  violation; the AT2 least-squares does not constrain the sign).
- Three sweeps `12k/12k/12k` (about 10 min): final `after` loss `0.00300`,
  `u_relative_l2 0.0427`, reaction `12.845`, drift `0`, `d_relative_l2
  0.229`, `H_relative_l2 0.044`, coupled loss `0.0316` (down from `0.170`
  after one sweep), `damage_min = 3.6e-7` positive, no healing.

Both budgets give a physically accurate mechanics field (about `3-4%`
relative L2, about `1%` reaction) with zero drift; the multi-sweep run gives
the much better coupled loss and damage consistency. The long run exposed a
damage-positivity violation (`damage_min = -2.6e-8`; the reference damage is
strictly nonnegative with a `2.3e-9` bulk minimum), because the AT2
least-squares does not constrain the sign.

Damage positivity is now imposed by a smooth one-sided floor in physical
damage units, `d = 0.5 * (raw + sqrt(raw^2 + floor^2))` with
`physical.damage_floor = 1e-8` (explicit config, validated, tested). It
equals `raw` above the floor and decays to zero from above below it, with a
finite gradient at zero. Post-fix runs:

- Three `12k` sweeps: `damage_min = 2.29e-7` positive, `u_relative_l2 =
  0.0269`, reaction `12.770` vs `12.675`, coupled loss `0.0316`, no healing,
  drift `0`.
- One `48k` sweep: `damage_min = 3.9e-11` positive, `u_relative_l2 = 0.0418`,
  reaction `12.558`, coupled loss `0.133`. The damage stage itself solves
  `damage_max = 1.2095e-4` vs the reference `1.225e-4` (1.3% low), and the
  final reported damage is the `omega = 0.5` Picard relaxation of that solve
  (`d_trial`), matching the direct scheme.
- The bulk `d_relative_l2` (`0.23-0.51`) remains dominated by the tiny
  reference bulk values (`~1e-9`) and is not yet a usable gate.

### Acceptance gates and trained-state reproducibility (2026-09-28)

`stagger.py` now evaluates explicit per-sweep gates
(`finite`, `mechanics_loss <= 1e-2`, `displacement_l2 <= 5e-2`,
`reaction_relative <= 2e-2`, `damage_range`, `healing`, `history_monotonic`,
`drift = 0`), logs each to TensorBoard, and reports `accepted`. With the
deterministic FP64 backend and a deterministic operator matvec the state-1
run accepted once: `u_relative_l2 0.0286`, reaction `12.667` (`0.06%`),
mechanics loss `0.0032`, positive damage, no healing, no drift.

Two identical follow-up runs exposed remaining training-path
nondeterminism: `(u 0.0364, reaction 13.099)` failed the `2%` reaction gate.
The forward/reporting path is bitwise reproducible; the reverse-mode XLA
graph is not. Isolated primitives are deterministic, so the cause is fused
reduction codegen; `CUBLAS_WORKSPACE_CONFIG` and autotune-off do not fix it,
and deterministic XLA ops are `>30x` too slow. The stiff residual amplifies
last-bit differences into `~1-3%` reaction spread over 36k updates. Gate
margins must account for this spread (or the stages must converge further)
before multi-load continuation.

### Strict-prefix continuation and the first state-2 attempt (2026-09-29)

The user approved a `5%` relative-reaction margin (material displacement
remains `5%`). `continuation.py` runs only archived strict-prefix loads
`1..data.end_state_index` on the FP64 graph. Each load has a separate
identity-bound stagger checkpoint and TensorBoard directory; the final sweep
writes an atomic `accepted.ckpt.zip` containing both networks and Adam states,
the final displacement, relaxed damage and cumulative Amor history **only**
if all eight gates pass. The successor uses that checkpoint, not the archived
previous-load field. A rejected load writes `result.json` but no accepted
checkpoint; resume verifies the accepted parent digest and refuses to skip the
rejected completed load. The first load starts from the archived zero state.
Implementation touches `examples/fracture2d/continuation.py`,
`stagger.py`, `stagger_checkpoint.py`, and the explicit config; the example
README documents the new CLI and snapshot layout. The accepted-snapshot
identity/parent-digest regression and strict-prefix validation are in
`tests/test_fracture2d_pinn_residual.py`. Focused verification:
`14/14` fracture2d residual tests, `31/31` fracture-loss tests, a two-load
accept/restart smoke, a strict-gate reject smoke, compilation of the changed
Python modules, and `git diff --check` all pass.

- A two-load, two-step-per-stage smoke with deliberately permissive gates
  committed states 1 and 2, verified the parent checkpoint chain and an
  idempotent resume. A separate strict-gate smoke rejected state 1 without
  creating an accepted checkpoint. The smoke initially exposed a real bug:
  the state-2 history gate compared the PINN's history to archived state-1
  history; it now compares to the PINN's own committed history.
- The first real `3 x (12k/12k/12k)` two-load attempt was rejected at state 1:
  material `u` error `2.58%`, mechanics loss `0.00398`, reaction relative
  error `7.88%` (above the approved `5%` limit). No load was committed.
- A fresh identical run committed state 1 (material `u` error `2.40%`,
  mechanics loss `0.00302`, reaction error `1.97%`, all other gates pass;
  accepted checkpoint SHA256
  `80b2c4eaaf39cd699b2e257ba46413359e16dd15d1196dd950c5269b7abf7eb7`).
  State 2 was then rejected: material `u` error `2.94%`, reaction error
  `0.23%`, but `damage_healing_max=5.97e-6` (allowed `1e-10`) and mechanics
  loss `0.826` (limit `0.01`). Its raw free-row mechanics RMS is about
  `0.0051`; the warm-started *per-load* initial scale is `0.005606` versus
  state 1's `0.05103`. Thus the unchanged `0.01` normalized threshold is
  about 83 times stricter in absolute residual at state 2. `state_0001`'s
  accepted checkpoint remains intact, `state_0002` has no accepted file, and
  a resumed run refuses to advance. Run directory:
  `/root/autodl-tmp/data/fracture2d/pinn_continuation_fp64_1to2_trial2_20260929/`.

This is a **numerical-policy blocker**, not permission to clip healing or to
weaken a gate. The direct formulation uses cumulative history and a strict
healing gate without an additional explicit damage-irreversibility constraint.
The user must choose whether to keep history-only and increase the damage/
mechanics convergence budget, or authorize a different inequality/solve
method; similarly decide whether the mechanics acceptance scale should remain
per-load or be frozen across the entire continuation before rerunning.

### User-approved damage-healing tolerance (2026-09-29)

The user explicitly approved `1e-3` as an acceptable damage-healing
**tolerance** after observing the `5.97e-6` state-2 defect. A separate required
`stagger.healing_atol=1e-3` is now used both in the completed-sweep gate and
in the accepted-successor monotonicity check. The damage-range tolerance is
still `1e-10`; cumulative Amor history must still be exactly nondecreasing.
No damage field is clipped and the AT2 equation remains history-only. The raw
healing defect is still recorded per sweep. The new tolerance can exceed the
entire early-load damage amplitude (state-2 maximum about `4.47e-4`), so a
passing healing gate means **bounded**, not strict, irreversibility.

A focused regression checks that the recorded `5.97e-6` passes the healing
gate without also loosening damage range or mechanics loss: the saved state-2
trial still fails its separate normalized mechanics gate (`0.826 > 0.01`).
The old rejected run's config hash differs from the new policy, so it must
not be relabeled or resumed under this change. New-policy physical training
must use a new output root after the mechanics-scale decision.

### Path-wide normalization and online final-sweep acceptance (2026-09-29)

The user selected `stagger.mechanics_scale_policy=first_load_initial`:
each load recomputes the first strict load's affine zero-output, zero-damage
FP64 free-residual RMS from the same model and reference identity. A two-load
smoke measured `0.051029846897393503` at both loads and verified the
accepted snapshot chain. The separate healing tolerance remains `1e-3`.

With final-step-only acceptance, two independent `3 x (12k/12k/12k)` runs
stopped at state 1 (reaction `5.56%` versus the `5%` gate) and at state 2
(mechanics loss `0.01135` versus `0.01`) respectively. An explicit budget
comparison using `3 x (16k/12k/16k)` accepted state 1 but ended state 2 at
loss `0.01045`. TensorBoard shows its last mechanics-after stage had a
complete online report at step 8,000 with mechanics loss `0.009967`, u error
`1.46%`, reaction error `1.29%`, raw healing `6.07e-6` and all other gates
passing. That historical report was not committed.

`stagger.accept_on_first_passing_report=True` now evaluates all existing
gates at each post-update report of the **final** sweep's mechanics-after
stage. The first simultaneous pass stops the optimizer and atomically saves
that exact network/Adam/field/history state with its actual step. Earlier
sweeps always complete; a false flag retains final-step-only acceptance.
The accepted-checkpoint schema is now `fracture2d-fp64-accepted-load-v2`;
no old checkpoint is retroactively accepted. Short two-load tests verified
that step 1 can be accepted with a three-step final mechanics budget while
an earlier sweep runs its full budget, and idempotent resume revalidates the
same snapshot hashes.

The first new full-budget two-load run with `3 x (16k/12k/16k)` completed
both states, under
`/root/autodl-tmp/data/fracture2d/pinn_continuation_firstpass_16kq_3x_20260929/`:

| load | accepted after step | u relative L2 | reaction error | mechanics loss | raw healing | accepted SHA256 |
|---|---:|---:|---:|---:|---:|---|
| 1 (`0.0001`) | 4,000 (sweep 3) | 2.76% | 1.05% | 0.00294 | 0 | `d26b31db57da4bb850cd433d73f39a57e69c9746fafb1fa608754557c542fe86` |
| 2 (`0.0002`) | 8,000 (sweep 3) | 1.51% | 0.88% | 0.00989 | `6.04e-6` | `eddd97052ef9081e9b57c72b692e2efdc1c16006fdcb1ea4cca6ba6a8914cf72` |

State 2 also has `d_relative_l2=0.108`, `H_relative_l2=0.0283`, zero drift,
nonnegative damage, monotone Amor history and coupled diagnostic `0.0203`.
Resume revalidated both accepted hashes without retraining. This is an
accepted **two-load** prefix under the bounded-healing policy, not proof of
an exact damage inequality or a 59-state path.

### Warm-start extension to state 3 (2026-09-29)

The user clarified that accepted networks should always warm-start the next
load (transfer learning); the numerical code already carried both networks,
separate Adam states and committed q/d/H. Only the run endpoint identity had
blocked incremental extension. `continuation.py` now permits increasing
`data.end_state_index` on resume while checking that **all other numerical
settings** are unchanged. Earlier per-state configs retain their original
endpoint hashes, and their accepted checkpoint bytes and predecessor SHA256
chain are verified without rewriting or re-training them. The root's target
config is atomically updated to the new endpoint.

A two-load short-budget root extended to state 3; both original accepted
checkpoint hashes stayed bitwise unchanged. Focused endpoint tests reject
any change in optimizer settings or a decreasing target index. The real
two-load accepted root above then extended to `data.end_state_index=3`
without retraining states 1 and 2: their SHA256 values remained
`d26b31db...`/`eddd9705...`. At `u_bar=0.0003`, the warm-started state-3
candidate completed three `16k/12k/16k` sweeps but was **rejected**. Final
metrics: material u relative L2 `0.03508`, reaction `37.2656` versus
`38.0126` (`1.965%`), damage L2 `0.08388`, history L2 `0.03853`, zero
drift, healing `4.65e-7`, damage range/finite/history gates pass; mechanics
loss `0.019838 > 0.01` is the *only* failed gate. State 3 has `result.json`
and a completed trial checkpoint but no `accepted.ckpt.zip`. Resume verified
the previous two accepted states and refused to skip the rejected state 3.
The unmodified A2 equations and fixed first-load normalization were used;
this result does not authorize changing the mechanics threshold or scale.

Verification: the focused fracture2d residual suite passes `16/16`,
two-load first-passage and final-sweep-only short runs passed, endpoint
extension preserved both parent hashes, `git diff --check` passes and the
changed Python modules compile. No subsequent strict load was trained.

### Approved load-relative normalization, read-only fork and curve (2026-09-29)

Following the user's observation that the residual and its training gradient
grow with load, the approved `first_load_times_load_ratio` policy computes
`s_n = s_1 * (u_bar_n / u_bar_1)` from the *first-load zero-damage affine*
free-row RMS `s_1=0.051029846897393503`. The exact same positive scalar
normalizes the FP64 mechanics objective, progress/coupled loss and acceptance
gate; the raw residuals and `0.01` normalized gate remain visible/unchanged.
This is a relative loading convention, not an identity for damage-evolving
fields. The earlier rejected state 3 was not retroactively accepted.

The new root
`/root/autodl-tmp/data/fracture2d/pinn_continuation_loadratio_from2_20260929/`
explicitly declares `output.fork_from` pointing to the old root and
`output.fork_after_state_index=2`. It checks the old root's policy and each
old per-state config, accepts **only** the scale-policy change and new output
identity, validates the old accepted parent SHA chain, and trains from the
actual state-2 networks, optimizer states, q/d/H without copying or rewriting
their checkpoints. Loads 3–6 passed every gate on the *new* policy and were
committed at final-sweep mechanics-after step 4000 each. For the user's
priorities, the reaction-versus-load curve and damage-field L2 are:

| load | reaction PINN / direct | reaction error | damage relative L2 | state |
|---|---:|---:|---:|---|
| 1 (`0.0001`) | 12.8081 / 12.6749 | 1.05% | 22.33% | accepted old policy |
| 2 (`0.0002`) | 25.1239 / 25.3468 | 0.88% | 10.80% | accepted old policy |
| 3 (`0.0003`) | 38.2907 / 38.0126 | 0.73% | 8.40% | accepted new policy |
| 4 (`0.0004`) | 52.4972 / 50.6692 | 3.61% | 7.05% | accepted new policy |
| 5 (`0.0005`) | 63.7694 / 63.3136 | 0.72% | 6.29% | accepted new policy |
| 6 (`0.0006`) | 76.2985 / 75.9427 | 0.47% | 5.74% | accepted new policy |
| 7 (`0.0007`) | 87.9467 / 88.5534 | 0.69% | 5.49% | rejected: u L2 5.23% > 5% |

The state-7 trial completed all three sweeps but has **no** accepted
checkpoint. Its final normalized mechanics loss is `0.001455 < 0.01`, raw
mechanics inf `0.18889`, healing `2.24e-7`, range/history/drift/finite and
reaction gates all pass. Damage relative L2 is a reported diagnostic, not an
acceptance gate; early-load damage is small, so relative L2 starts at 22.33%.
The user's primary targets are the reaction curve and damage field, and a
new numerical policy for state 7 needs a decision about the displacement
gate and a meaningful damage-field criterion. Do not reclassify its archived
trial after changing policy.

The first six SHA256 values were revalidated on an idempotent resume. New
state-3/4/5/6 accepted digests respectively: `17201a19...`,
`426e67c0...`, `1fffe9f3...`, `b86ed67b...`; old state-1/2 digests are
unchanged. Focused residual tests pass `18/18`; changed Python modules
compile and `git diff --check` passes.

### Why damage relative L2 is high at the first load

Read the actual state-1/2 accepted damage fields and the frozen reference
from their identity-bound archives (without retraining). At load 1,
`||d_ref||_2=0.000901139`, `||d_PINN-d_ref||_2=0.000201260`, hence the
reported `22.33%`. At load 2, the corresponding norms are `0.003604595`
and `0.000389174` (`10.80%`): the reference norm increased almost fourfold
with doubled prescribed displacement, while the absolute error increased
less than twofold. The large first-load percentage partly reflects a small
denominator, **but it is also a real field discrepancy**. At the first-load
reference peak `[0.5, 0.5]`, `d_ref=1.22547e-4` and `d_PINN=9.27892e-5`
(`24.3%` too low); the top 1% of reference-damage nodes contribute `35.7%`
of total squared damage error. Orthogonal projection of the PINN field onto
the reference has amplitude coefficient `0.8378` with a remaining shape
error of `15.36%` of reference norm. At load 2 these become `0.8944` and
`2.23%`: the first load has both amplitude and shape errors.

The networks optimize the A2/AT2 residual stages, not reference damage, and
the current acceptance contract has no damage relative-L2 gate. For load 1
the damage-stage fixed-history solved residual infinity is `0.000552`,
whereas the final same-pass coupled damage residual infinity is `0.006987`
after the final mechanics update and damage relaxation. These measure
different states/history; the difference establishes that the accepted
staggered final field has a non-negligible coupled damage mismatch, without
isolating it as the only source of its reference-field error. The
first-load mechanics ratio is one, so the later-load normalization policy
cannot explain the first-load discrepancy. No gate or reference was changed.

Following the user's agreement that near-zero reference damage makes relative
L2 a poor early-load headline metric, `stagger.py` now reports full-material
**absolute damage RMSE** as the primary error and **maximum absolute damage
error** as a localized-error companion; relative L2 remains auxiliary.
Retrospective calculations from the read-only accepted fields (not changes
to the old `result.json`/checkpoint files) give:

| state | damage RMSE | damage maximum absolute error | old relative L2 |
|---|---:|---:|---:|
| 1 | `3.4541e-6` | `3.2869e-5` | 22.33% |
| 2 | `6.6792e-6` | `5.9162e-5` | 10.80% |
| 3 | `1.1691e-5` | `1.1406e-4` | 8.40% |
| 4 | `1.7447e-5` | `1.8007e-4` | 7.05% |
| 5 | `2.4332e-5` | `2.6036e-4` | 6.29% |
| 6 | `3.1965e-5` | `3.3675e-4` | 5.74% |

The absolute errors rise with applied load even while relative L2 falls;
this is why both absolute and auxiliary relative trends remain visible.
Subsequent reports emit `eval/d_rmse` and `eval/d_max_abs_error` and include
both in `result.json`. No damage-field gate or threshold was inferred.
The targeted stagger setup suite passes `8/8` (including a zero-reference
damage regression), both changed Python modules compile and
`git diff --check` passes. The earlier `18/18` full fracture2d suite was
already run for the numerical-policy changes; it was not rerun for this
report-only instrumentation.

### Reaction/damage acceptance and nested fork from state 6 (2026-09-29)

The user explicitly replaced the displacement `5%` acceptance gate with
diagnostic-only displacement logging, selected `1e-2` as the **damage RMSE
gate only**, and kept maximum absolute damage error diagnostic-only. The
base config declares `gate_displacement_l2=True` and
`damage_rmse_limit=0.0` (disabled) for its prior acceptance policy; the new
root explicitly sets `False` and `0.01`. `candidate_gates` computes RMSE
at every eligible final-sweep report from the **same post-update d_trial**
as the final result; the reference damage remains an acceptance comparator,
never a training target. Reaction `5%`, mechanics loss `0.01`, bounded
healing, physical range, history and drift are still required.

The read-only nested fork
`/root/autodl-tmp/data/fracture2d/pinn_continuation_damage_rmse_from6_20260929/`
follows the previous load-ratio fork for accepted states 3–6, then the
original root for accepted states 1–2. Per-state configs, frozen graph
identity and the six predecessor SHA256 links are checked at each run;
old checkpoint bytes stay unchanged. The old rejected state-7 trial was
**not** imported. The actual new-policy continuation accepted states 7–10:

| state | reaction PINN / direct | error | damage RMSE | max abs error (diagnostic) | u L2 (diagnostic) | final-sweep after step | accepted SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| 7 (`0.0007`) | 88.0272 / 88.5534 | 0.59% | `4.1773e-5` | `4.2534e-4` | 5.20% | 4,000 | `fc5f4d65b3b292757ef5beb75fa9d05d4cf9fae5e0b13fe37b721eeb2a0c6ca6` |
| 8 (`0.0008`) | 103.0682 / 101.1428 | 1.90% | `5.2844e-5` | `5.3804e-4` | 5.34% | 4,000 | `e364670f8d5df7dcb9319009c70d64de3be99b3d3bff11af279d8aa50719b65d` |
| 9 (`0.0009`) | 112.5183 / 113.7078 | 1.05% | `6.6899e-5` | `6.4488e-4` | 5.65% | 4,000 | `c99e6f3648e87c7b8fc825e9264ad95620b5f585e7ea4b972c2f1d2f563ee994` |
| 10 (`0.0010`) | 125.9921 / 126.2453 | 0.20% | `8.2169e-5` | `7.7127e-4` | 5.71% | 4,000 | `ad4a65a70539858314b53238dead634c36f2e5f36f8208e26df98312315aa349` |

The RMSE gate is far from binding at these early loads (`8.22e-5` vs
`1e-2` at state 10); continue to show the raw RMSE, localized maximum and
reaction curve rather than equating gate passage with high damage-field
accuracy. All ten loads form an accepted prefix under their explicitly
recorded per-state policy. No warning-only state is included. The focused
fracture2d residual/continuation suite passes `21/21` (including disabled-u
and enabled-RMSE gates, and nested-fork ancestor rejection on an unauthorized
optimizer change); changed modules compile and `git diff --check` passes.
An idempotent resume at endpoint 10 revalidated all ten stored accepted
SHA256 values without retraining or rewriting any state.

### Same-policy continuation through state 30 (2026-09-29)

At the user's request, increased only the run endpoint first to 15, then
20, then 30 with `output.resume=True`; all preceding accepted snapshots and
their SHA256 parent chain were verified at each restart. Loads 11–30 all
passed the existing reaction, RMSE, mechanics and physical gates and were
accepted without retraining prior loads. Selected reaction/damage checks:

| state | reaction PINN / direct | reaction error | damage RMSE | max abs damage error | accepted SHA256 |
|---|---:|---:|---:|---:|---|
| 15 (`0.0015`) | 188.2660 / 188.4143 | 0.079% | `1.6511e-4` | `1.2738e-3` | `ff6691904ed7a7c44258ccc4545ebcfd7b40bb6d46dbbbc9550600f6048a6685` |
| 20 (`0.0020`) | 250.4138 / 249.4421 | 0.390% | `1.8813e-4` | `1.5397e-3` | `fd4c2d671711f1534b9b60d9cd35056e73b6667e4539db0373ab905785ae6e8a` |
| 25 (`0.0025`) | 306.3621 / 308.9512 | 0.838% | `1.3443e-4` | `1.3153e-3` | `8b6dfa7e5a301a170a152880bac2d89065fafcf2c311941bff43ccc19c8a04d1` |
| 30 (`0.0030`) | 365.8633 / 366.5622 | 0.191% | `1.4792e-4` | `1.9280e-3` | `4d20ef9fb0137bef7c88a5e3025374538d21ce72e29e4f3598c30823b685c48d` |

The state-30 final mechanics loss is `1.3342e-4`, damage maximum is
`0.11736` and drift count is zero. No extrapolation to loads 31–59 is
warranted. In response to the user's runtime concern, stopped after the
completed state-30 batch. Result timestamps for states 21 and 30 are
`20:47:08` and `22:51:40` local time: nine intervals take `124.5` minutes,
about `13.8` minutes per load. A further 29 equal-cost loads would require
roughly `6.7` hours *if* cost remains comparable; the cost and convergence
can change near crack growth. The current three sweeps require up to
`3*(16k+12k+16k)=132,000` updates per load; only the final sweep's
mechanics-after reports permit early acceptance. A shorter budget or fewer
sweeps changes the accepted numerical policy and needs an explicitly
authorized fork from state 30, not an in-place edit of this root.

Runtime comparison against earlier full-budget `result.json` timestamps:
state 1→2 took `12m49s` (`12:29:04`→`12:41:53`), state 3→4 took
`15m59s` (`14:01:23`→`14:17:22`), and state 7→8 took `15m36s`
(`16:30:46`→`16:46:22`). The state 21→30 nine-interval average is
`13m50s`; there is no evidence here of a new per-load slowdown. The
larger wall time comes mainly from requesting 10 loads in one sequential
call. Earlier independent FP64 step probes found mechanics Adam `5.10 ms`
and damage Adam `2.77 ms`; multiplying by 96,000 mechanics plus 36,000
damage updates gives about `9.8 min` per load of optimizer steps before
compilation, full-field reports and checkpointing. This estimate is not a
stage-by-stage runtime profile.

Before committing the current PINN milestone, the unchanged numerical code
passed `tests.test_fracture_losses` (`31/31`) and
`tests.test_scalar_operator_2d` (`18/18`) in separate processes;
`git diff --check` passed. The fracture2d continuation/residual suite had
already passed `21/21` after the gate/fork changes and was not rerun without
further numerical code changes.

## Blocker

The new damage-RMSE-focused fork has accepted strict states 1–30. The old
state-7 rejected trial remains in the previous root; the replacement state
7 was retrained and accepted under the approved policy. The `1e-2` damage
RMSE gate is loose relative to early observed errors, so monitor actual
reaction and damage field curves as continuation approaches larger loads.
States 31–59 have not been verified. Continuing the fixed budget is
computationally costly (about 14 minutes per load observed on states 21–30).
Cross-process optimizer training is
not bit-reproducible; do not label warning-only states as strict acceptance.

## Next Action

The exact resume point is the accepted `state_0030/` in the damage-RMSE fork
above. Increase **only** `--config.data.end_state_index=31` and use
`--config.output.resume=True`, keeping the root path, source root, fork
cutoff `6`, FP64, load-ratio scale, `gate_displacement_l2=False`,
`damage_rmse_limit=0.01`, 3 sweeps of `16k/12k/16k` and logging intervals
`4000/8000`. The root currently targets endpoint 30. On resume verify all
thirty accepted SHA256 values without re-training the prefix, then train
state 31 from the state-30 network/optimizer/q/d/H snapshot. Stop on the
first rejected strict state and report reaction and damage before other
diagnostics. No `0.0060/0.0061` warning state is a strict target.
