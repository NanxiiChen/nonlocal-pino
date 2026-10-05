# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- User-selected mainline is now the **old validated common-horizon kernel**:
  preserve the original 3395-node A2/Amor/AT2 model and verified state-59 PINN
  prefix; audit/fix numerical coupling with minimal changes. The split branch
  is retained as a comparator and its GPU job is stopped. The active task below
  records both branches; original-kernel warning loading is the current study.
  The old 0.006 correction now passes the declared neural coupled-readiness
  criteria for three consecutive sweeps: AT2 relative 6.66e-5, reaction error
  0.0310%, damage RMSE 0.0007352, zero drift. Reference 60 remains warning;
  this is not strict-reference/full-fracture validation. Load 61 completed an
  initial 16-sweep budget (reaction error 0.2002%, damage RMSE 0.001371, zero
  drift). State 61 now completed 56 cumulative sweeps: reaction error 0.1038%,
  damage RMSE 0.001049, AT2 relative 6.27e-5, zero drift. Only adjacent q/load
  change 1.33009e-4 >1e-4 fails, localized at the notch-tip y displacement.
  It remains not coupled-ready; independent replay correctly rejects advance
  to 62. User requested further solving: resume with a bounded three-block
  fixed-d mechanics/driver GPU audit, then select a justified numerical
  correction while retaining neural state 60 as the committed history floor.
  User explicitly selected warning continuation. Diagnostic loads 62..65 are
  complete: damage peak 0.821 ->0.908, reaction peak 651.589 at 0.0064 then
  650.241 at 0.0065, zero neural drift. Coupling residuals worsen (final AT2
  relative 0.03728); archived reaction differences reach 55.37%. No strict
  fracture/unloading validation is claimed. Final snapshot/policy ancestry
  replay passes, physical/mechanics gates pass. Further loading completed usable
  warning states 66..69: reaction falls to 602.195 at 0.0069, down 7.58% from
  sampled peak 651.589. State 70 is rejected for one drift node (ratio 1.08363
  >1.08), despite reaction 582.434; 71..75 were not attempted. Last usable 69
  is replay-verified but not coupled-ready (AT2 relative 0.10984). No job runs.
  User asks for cause/remedy analysis: prioritize identical-parent fixed-load
  inner/coupling diagnostics; reference drift and shared horizon transmission
  must be assessed separately. User approved execution: four direct/neural
  combinations at independent loads 61/62 from identical neural state-60 floor.
  All eight fixed-load cases completed. Neural fixed-H damage matches direct
  to ~6e-5; neural q has ~0.62% L2 error but local H deficits ~23/36 at loads
  61/62. Pure-direct 61 contracts slowly (64 sweeps, AT2 relative 6.39e-4),
  while pure-direct 62 hard-stops at sweep 29 with one drift node despite
  accurate inner solves. No case is coupled-ready or strictly validated.
  Next: converged direct-61 baseline, genuine small-step loading/path and
  outer-stability controls, then local neural mechanics accuracy. User requested
  execution completed: direct-61 passes the declared tight 1e-6 diagnostic
  contract after 184 cumulative sweeps, R=641.416, zero drift. All three
  Picard step sizes hard-stop on their first target (.0062/.00615/.006125).
  Safeguarded Anderson at .006125 avoids drift for 128 steps but stalls at
  AT2 relative 0.00198, not converged. Baseline/path field replay passes;
  all failed/unready trial histories remain uncommitted. No job runs.
  Sinha-paper first-stage study completed: paired fixed-d 16k training with
  identical networks/fresh Adam. Energy versus force-MSE reduces q L2 error
  .541% ->.282% and driver inf 39.92 ->36.43 (8.75%), but force MSE increases
  22%; no coupled remedy/adoption. A finite-difference-checked tip mixed-derivative
  witness confirms the original weighted residual pair is not one common energy
  gradient. All 34 saved reports and final networks replay; no job runs.
  Repository exploration (2026-10-05) cross-checked the saved latest reports;
  the active task's Blocker/Next Action now summarize the current evidence.
  User-authorized publication checkpoint: focused CPU checks ran 59 tests
  (58 passed, one expected GPU skip); changed/new Python modules compile.

- Active task: [`docs/tasks/fracture2d-nonlocal-pinn.md`](tasks/fracture2d-nonlocal-pinn.md).
  The direct stage is closed: the adopted solver is one GPU-only file,
  `examples/fracture2d/nonlocal_baseline.py` (`11,374` lines), whose default
  behavior was confirmed against the archived fresh solution (10 strict steps:
  loads identical to `4.3e-19`, reaction difference `4.2e-10`, displacement
  `3.2e-13`). The lean-refactor task is archived under
  [`docs/tasks/archive/`](tasks/archive/fracture2d-lean-direct-solver.md).
- The PINN stage reuses the A2 energy/residual method, Amor history and AT2
  law from `examples/fracture2d/nonlocal_baseline.py` and shared
  `pinn.operators.ExplicitVolumeSparseD`, on a separately rebuilt FP32 graph.
  A full-graph differentiable residual
  bridge and single-load GPU experiment now run. The approved FP32 PINN
  graph/BA/damage operator is independently rebuilt (union candidate pairs
  +209/-76 versus frozen FP64); FP32 training runs but first-load errors remain
  unconverged. An independent-network A2/AT2 stagger diagnostic now updates
  Amor history exactly without a third network. First-load 36,000-update
  diagnostics (`12k/12k/12k`) with TensorBoard and bound checkpoints are
  complete but unaccepted. The `128 x 4` v3 network removes drift entirely
  and nearly solves the fixed-history damage stage, but the coupled diagnostic
  bottoms at `0.358` (step 24,000) and ends at `6.00` with reaction `33.00`
  vs FP64 `12.67`. The 2026-09-28 audit found no operator error (exact FP32
  solve: objective `8.8e-6`, reaction `12.664` vs FP64 `12.675`) but did find a
  real implementation bug: `FractureResidual.lift` used a scatter that
  compiled to a *different* jitted displacement field than eager, so the
  training objective and the reports were different functions. The mask-blend
  lift fixes this (`q` difference `6.4e-9`), and `highest` matmul precision
  plus a deterministic segmented assembly remove the remaining
  eager/jit/TF32 mismatches. Post-fix Adam still rises on the stiff objective
  and the host-side L-BFGS descends `0.349 -> 0.065` but plateaus. Clean probes
  now show this is an energy-norm wall, not an L2-representation failure:
  supervised Adam fits the reference to `0.7%` L2, but that field's free-DOF
  residual is `1.7e4` times the reference field's, and A-optimal linear
  readouts bottom out at `1.7e3`. The `8.8e-6`/`2.2e-5` values are float32
  assembly-rounding floors of an exact solution. A validated matrix-free
  damped Gauss-Newton step (`gauss_newton.py`) was then tested and fails on
  the raw residual: the parameter-space landscape is jagged at the `1e-5`
  scale (a GN direction of norm `80` needs `step_size = 3.7e-9`), and its
  `0.065`-residual fields are `50%` off because the residual is nearly blind
  to near-rigid modes. The FP32 jaggedness is evaluation noise, and the
  actual FP64 workload is only `~2-5x` slower, so option A end-to-end
  training now runs on the adopted float64 graph: `stagger.py` defaults to
  `train.dtype = "float64"` (schema
  `fracture2d-fp64-stagger-checkpoint-v2`), with the rebuilt float32 path
  retained behind `train.dtype = "float32"`. First state-1 sweep
  (`12k/12k/12k`): `after` mechanics loss `0.052 -> 0.019` and still falling,
  drift `20 -> 0`, reaction `23.9 -> 15.0` vs `12.675`, `u_relative_l2`
  `0.471 -> 0.207`; damage loss `0.0038 -> 0.0028`, no healing. The stage is
  bounded and monotone (v3 FP32 had `mechanics_after -> 156`). Budget
  comparison: a `48k/48k/48k` single sweep reaches `u_relative_l2 0.028`,
  reaction `12.811` vs `12.675`, drift `0` (but `damage_min -2.6e-8`); three
  `12k` sweeps reach `u_relative_l2 0.043`, reaction `12.845`, coupled loss
  `0.032` (from `0.170`), `damage_min` positive, no healing. Damage
  positivity is now enforced by a smooth one-sided floor
  (`physical.damage_floor = 1e-8`), the FP64 backend and material operator
  use deterministic segmented reductions, and `stagger.py` evaluates eight
  explicit acceptance gates. State 1 accepted once
  (`u 0.0286`, reaction `12.667`, loss `0.0032`); identical reruns spread to
  `u 0.0364`, reaction `13.099` because the reverse-mode XLA graph is not
  bitwise reproducible on this GPU stack. The user approved a 5% reaction
  margin; `continuation.py` now commits only fully gated FP64 strict-prefix
  states with identity-bound, atomic snapshots and refuses to skip a rejected
  load on resume. A two-load smoke proved commit/restart/rollback. In a full
  `3 x 12k` two-load attempt, state 1 passed (u error `2.40%`, reaction error
  `1.97%`); state 2 failed healing (`5.97e-6` vs `1e-10`) and its per-load
  normalized mechanics gate (`0.826` vs `0.01`) although u/reaction errors
  were `2.94%`/`0.23%`. State 1 remains committed; state 2 is rejected.
  The user has explicitly accepted a healing tolerance of `1e-3`; this is now
  separate from damage-range tolerance `1e-10` and leaves the AT2 equation and
  strict Amor history maximum unchanged. The user selected a fixed
  first-load mechanics scale (`0.05103`) and online first-passage acceptance
  only at the final sweep's mechanics-after report points; old runs cannot
  be retroactively accepted. A new `3 x (16k/12k/16k)` run committed loads
  `0.0001` and `0.0002`: state 1 stopped at final-after step `4,000` (u
  `2.76%`, reaction error `1.05%`); state 2 stopped at step `8,000` (u
  `1.51%`, reaction error `0.88%`, mechanics loss `0.00989`, healing
  `6.04e-6`). Both accepted snapshot digests survive resume unchanged.
  Endpoint-only resume now warm-starts from the last accepted network and
  Adam states without re-training or changing old checkpoint hashes. The
  real state-3 extension (`u_bar=0.0003`) passed displacement (`3.51%`),
  reaction (`1.97%`), range, healing (`4.65e-7`), history and drift gates,
  but failed the fixed first-load-normalized mechanics gate
  (`0.01984 > 0.01`). State 3 remains unaccepted; resume refuses to skip it.
  The user approved `first_load_times_load_ratio` (first-load affine RMS
  multiplied by prescribed-displacement ratio) for training **and** gating.
  A new root read-only forks from the SHA-verified accepted state 2; loads
  3–6 now pass all gates without rewriting loads 1–2. Reaction errors on
  loads 1–6 are `1.05%, 0.88%, 0.73%, 3.61%, 0.72%, 0.47%`; damage relative
  L2 is `22.33%, 10.80%, 8.40%, 7.05%, 6.29%, 5.74%`. State 7 is
  **rejected solely by the 5% displacement gate** (`5.23%`); its reaction
  error is `0.69%`, damage L2 `5.49%`, mechanics loss `0.00146`. The user
  prioritizes the reaction curve and damage field; current damage-field L2
  is diagnostic-only. For small-load damage, new reporting leads with absolute
  RMSE and maximum absolute error (state 1: `3.454e-6`/`3.287e-5`), keeping
  relative L2 as auxiliary. The user explicitly made displacement error
  diagnostic-only and chose `1e-2` as the damage RMSE gate (max error still
  diagnostic). A new read-only nested fork retrained state 7 from accepted
  state 6 and now has an accepted **state-1–59 strict reference prefix**.
  New-policy loads 7–10
  have reaction errors `0.59%`, `1.90%`, `1.05%`, `0.20%` and damage RMSEs
  `4.18e-5`, `5.28e-5`, `6.69e-5`, `8.22e-5`. The RMSE limit is loose at
  these loads. All later strict states 11–59 passed the applicable gates;
  the full 59-point reaction curve has relative L2 error `0.934%` and a
  maximum pointwise reaction error of `3.61%`. At the final strict state
  `0.0059`, reaction is `640.304` vs `634.999` (`0.84%`), damage RMSE
  `0.00429 < 0.01`, but maximum local damage error `0.07974` is
  diagnostic-only. The 59-state SHA256 chain was reverified on resume.
  Post-peak remains **unvalidated** because `0.0060/0.0061` are warning
  reference states (and the tail later drifts); do not extrapolate strict
  acceptance beyond state 59. The precise blocker and resume point are in
  the active task. The user chose a separate diagnostic-only post-peak
  exploration. Accepted full-domain displacement/reference/error plots for
  states 57–59 and a state-59 notch crop are available under the current
  run's `displacement_figures/`; state-59 material u relative L2 is `0.740%`
  with the maximum vector error `4.08e-4` immediately above the notch tip.
  The separate diagnostic continuation now completed loading trials through
  `0.0075`: states 60–74 pass trial-advancement checks (never strict
  acceptance); terminal state 75 fails the zero-drift check with 1 node.
  At `0.0075`, PINN reaction `620.165` vs direct archive `69.154`, damage
  RMSE `0.1113` and local error `0.9381` show a very different post-peak
  response. The committed AT2 residual and stagger changes remain large;
  small mechanics loss is not coupled-fracture convergence. Last usable
  diagnostic checkpoint is state 74; state 75 cannot be skipped on resume.
  Full curve, diagnostics and exact investigation resume point are in the
  active task. Focused tests pass `25/25`.
  The fixed `0.0060` experiment from the same accepted state 59 completed
  **12 outer sweeps**, unchanged stage budgets/learning rate and disabled
  first-passage stopping. From sweep 3→12, coupled AT2 inf falls
  `9.74→4.01`, damage change `0.00972→0.00380`, and damage RMSE
  `0.00575→0.00265`; coupling improves but is not yet converged. Checkpoint
  recomputation decomposes final AT2 inf into fixed-H solved `0.074`,
  relaxed fixed-H `2.240`, and current-H `4.008`, supporting outer-coupling
  insufficiency rather than only damage-network fitting error. Next define
  a coupled stopping/budget policy before further post-peak experiments.
   Per-sweep plots and exact completed snapshot are in the active task.
   Controlled 36-sweep fixed-`0.0060` follow-up completed: AT2 inf `0.431`,
   damage RMSE `0.000899`, but adjacent reaction changes remain ~1%. Target-free
   RBF enrichment and joint residual optimization now pass explicit research
   checks through `0.0066` (damage max `0.99366`, zero drift), without reproducing
   unloading; `0.0067` is rejected. Exact-inner oracle coupling also develops
   localized crack-core drift. The user selected model/soft-mode review with
   equations and gates retained. Positive degradation-energy bounds and
   volume-norm spectra rule out a demonstrated fixed-d zero-energy instability
   in the audited states. Synthetic failed-band tests reveal strong transmission
    until band width approaches `2*delta`; source-only degradation retains
    kinematic coupling through failed points. The approved independent-horizon
    branch passes operator/work checks and suppresses thin failed-band
    transmission. Fine reference loads 1–41 are strict; load 42 remains
    unvalidated at sweep 610. Current focus: bounded outer-iteration diagnosis,
    then FP64 split PINN integration, prioritizing damage/reaction and retaining
    secondary failures as explicit warnings. Integration is now running: energy
    training plus final-block first-passage stopping gives a warning-useful
    forty-one-load warning prefix through 0.0041, curve error 1.77%, zero drift.
    The tested fast small-load fork uses analytic/padded operator gradients and
    fewer updates: paired load-21 training is 2.98x faster with explicit accuracy
    tradeoff; mechanics budget was raised to 1000 after load 25 rejected.
    State-41 reaction error is 1.13%, volume-weighted damage L2 2.49%, and
    peak damage 0.47022 vs 0.52501 (tip max error 0.07046 remains visible).
    The strict reference still ends at 41. The last safeguarded Anderson
    critical-load retry was interrupted at the user's explicit return to the
    old branch; its progress and earlier rejections remain intact.
    Undamped direct retry still rejects 0.0042 after 80 additional sweeps;
    critical fracture coupling remains unresolved. Details are in the active task.
   No post-peak strict acceptance or converged through-going opening claim.
- Amor history takes the maximum of the current positive-energy driver and
  the last accepted *cumulative* history. The committed history and each
  fixed stagger-field input are explicitly detached in JAX; current-driver
  gradients remain available. Focused history/graph checks pass `8/8`.
- Fracture2d TensorBoard events live under
  `/root/autodl-tmp/tf-logs/fracture2d/events/<run-name>/` for **new** runs;
  new PINN configs/checkpoints/figures use sibling directories under
  `/root/autodl-tmp/tf-logs/fracture2d/`. All 38 historical `pinn_*` run
  roots were moved from `data/fracture2d/` to `tf-logs/fracture2d/pinn_runs/`
  with every file hash verified; recorded config/accepted SHA256 bytes are
  unchanged and an adapter resolves old absolute PINN paths. Strict-59,
  terminal-warning and fixed-load recovery checks passed after migration.
  The fixed-load experiment's images, reports and checkpoint are also beside
  its old events for Jupyter downloads. Tags follow
  the repository convention: `loss/*` stage objectives, `eval/*` reference
  comparisons, `diag/*` residuals/state. An identical-budget v2 re-run
  (`12k/12k/12k`, same seed) logs 72 per-report points and reproduces the v1
  failure: coupled diagnostic `1.60 -> 156.10`, drift `0 -> 22`, unaccepted.
- Networks are built via `pinn.arch.make_pinn` in `_build_network` with
  explicit `arch.*` config fields, matching `fracture1d`/`elasticity2d`;
  `train.py` uses one four-output net, `stagger.py` two single-field nets.
  Parameter initialization changed, so pre-refactor losses are not
  bit-reproducible. Both entries now take the shared ml_collections config
  flag (`--config...`), and the default network is `128 x 4`.
- A coverage trace of the real strict + warning + `verify` + `plot` paths
  reaches every top-level definition, so no further free dead-code reduction
  remains; the next size reduction would be a validated core rewrite.

## Direct nonlocal reference (adopted)

- Scheme: original BA/Amor/AT2 equations on the 3395-node `delta=0.048` cloud,
  `h_g=0.02`, `l0=0.024`, `Gc=2.7`, `E=210e3`, `nu=0.3`, `k_res=1e-8`, cubic
  kernel, fixed damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, at
  most 144 sweeps, `stabilized_warning` acceptance, load path `0 -> 0.0075`.
- Artifacts: fresh compact solution
  `.../engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`
  (SHA256 `5f3bc12b...`); research archive SHA256 `9545bed7...`.
- Response: peak `641.34017` at `0.0061` (FEM `642.48573` at `0.0057`);
  terminal `69.15426` at `0.0075` (FEM `3.97718`); 76 states = 65 strict + 10
  warning; drift points `2 -> 31` from `0.0062` to `0.0075`.
- Per-state labels: strict prefix through `0.0059` (59 accepted states);
  `0.0060` and `0.0061` are drift-free warnings, while `0.0069`, `0.0070`,
  and several later drifting states have strict acceptance labels. Classify
  by archived flags and load, not by a contiguous 65/10 partition.
- Accepted limitations: warning-only acceptance,
  `field_equations_validated=false`, terminal reaction mismatch, particle
  drift, and no completed strict `0 -> 0.0075` path. All are documented in
  `examples/fracture2d/README.md`.

## Current state

- `fracture1d` is an accepted engineering baseline; default PINN dtype is
  float32 with `spatial_emb_dim=32` and `spatial_scale=16`.
- The local FEM point-anchor warning-mode archive
  (`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`)
  remains the separate continuum comparator.
- Parked and ignored: monolithic/arc/reduced research bundle, durable
  `engineering_grip_workflow.py`, and the stage-specific fracture2d test suites
  (moved to `_legacy/`). They remain in git history and on disk for reference.
- Memory: `docs/DECISIONS.md` is the append-only rationale log;
  `docs/tasks/archive/` holds the closed direct-stage records.
