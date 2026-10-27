# Matched local AD versus nonlocal vortex PINNs — completed

## Request and conclusion

The user asks for local PINN accuracy and local/nonlocal training times with
all other schemes held equal to the CURRENT recommendation. A fresh seed42
serial pair completes the requested experiment, without reference or default
changes. Local pooled phi error is slightly lower in this pair; terminal
errors are nearly equal. Nonlocal is4.38x faster by trainer wall,4.97x by
compiled SOAP updates. Historical same-config nonlocal errors vary, so this
single pair does not establish a stable accuracy ranking.

| Arm | Pooled phi L2, common local | Terminal phi L2, common local | Trainer s | Process s | Compiled SOAP ms |
|---|---:|---:|---:|---:|---:|
| nonlocal | 3.77757% | 3.26634% | 211.153 | 217.424 | 16.7317 |
| local_ad | 3.36431% | 3.22328% | 924.230 | 931.018 | 83.1561 |

Native-reference pooled errors: local3.36431%, nonlocal4.06380%. They evaluate
different finite-horizon/local equations; cross-method table uses one SAME
converged local reference. All21snapshots0..2,64x64points, t=2terminal. Metrics
are phase-field phi errors, not velocity/pressure or full coupled-flow errors.

## Controlled numerical experiment

Both use4x64GELU,3fixed integer spatial modes, inverse-cubic phase coordinate,
exact IC and exact grid-mass affine projection, original physical residual
norm/equal weights, Nt8open-stratified shared-grid time draws, FP64, ordinary
compilation. SOAP5000/lr1e-3 then L-BFGS1000 on32fixed midpoint times; both
reach the iteration limit, not convergence tolerances. No reference field
values enter the objective. Only config nl.method and log.run_name differ;
initial checkpoint bytes and36saved source files are identical.

Local changes ONLY diffusion to the point network's AD Hessian traces;
both retain conservative nodal Fourier-flux advection. In the mixed residuals
r_phi=phi_t+div_fft(v*phi)+M*(-Delta_AD)q and
r_q=q-phi^3+phi-epsilon^2*(-Delta_AD)phi, nonlocal instead uses L_delta.
The grid-mass shift is spatially constant, so its spatial Hessian is zero;
its temporal derivative is included in the physical-field JVP. This is not
an all-AD advection control or separately tuned local configuration.

Reference remains
/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz, with the previously
independently converged local/nonlocal solutions and validated identity/grid.
This investigation performs no solver/data write. Full physical context and
reference convergence are in twophase-operator-preconditioning.md and the
example README.

## Direct artifacts and reproduction

- Local config: examples/twophase2d/configs/train_vortex_conserved_local.py.
- Nonlocal config: examples/twophase2d/configs/train_vortex_conserved.py.
- Local model: /root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-local_ad/model_final.eqx.
- Nonlocal model: /root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-matched-nonlocal/model_final.eqx.
- Each directory includes config.json, model_initial.eqx, prediction.npz,
  summary.json, metrics.jsonl, loss.jsonl, refinement histories/summary and
  frozen source snapshots. Original/older directories remain unchanged.
- Archived launch/analyze/GPU-verify scripts share this task prefix. Run using
  .venv/bin/python. Launcher calls canonical train.py, seed42, CUDA/ordinary,
  persistent compilation cache disabled, one GPU process at a time. Use a new
  run/output name when rerunning; existing artifacts are never overwritten.
- reports/twophase-matched-local retains raw training logs/manifest/analysis.
- Archived analysis JSON, manifest, environment, GPU verification and convergence
  figure accompany this report. Exact Python/package/GPU versions are recorded.

## Timing interpretation

Machine reports RTX3090/49152MiB, driver580.82.09. Trainer wall starts after
network/operator setup and includes initial eval, JIT, all updates/evaluations,
saves and refinement. Process wall additionally includes imports/backend/setup.
Compiled SOAP mean excludes first100updates, time generation/eval/checkpoints,
and includes SOAP preconditioner refresh frequency5. First update includes its
JIT: local68.161s versus nonlocal30.513s. SOAP-stage elapsed through final
assessment: local498.191s versus nonlocal126.814s. L-BFGS including separate
JIT/callbacks: local425.197s versus nonlocal83.610s; SciPy evaluations1075/1079,
plus21diagnostic evaluations per arm. Same stage limits do not imply identical
line searches. Timings describe this implementation/hardware and fixed budget.

## Repeated-run variation

Earlier same-config nonlocal seed42 reached2.84283%/2.34319%, versus fresh
3.77757%/3.26634%. Direct config/initial-byte/first-loss parity passes; both
first losses equal10.29859750798121, and histories diverge by100updates.
Historical saved code differences are inactive optional endpoint/removed-anchor
branches and validation/metadata for this config, retained in the comparison
JSON. No unique cause is isolated. Ordinary GPU/adaptive training does not
promise bitwise trajectories. Do not substitute the historical favorable
checkpoint for the fresh arm or conclude stable local accuracy superiority.
No seed/budget/architecture tuning is performed in response to the outcome.

## Validation and audit corrections

- One combination-specific manufactured local-Hessian residual/mass/spectral-
  flux and parameter finite-difference test passes in16.607s:
  tests.test_twophase_preconditioning.CHPreconditioningTests.test_local_hessian_with_mass_projection_and_conservative_flux.
  Previously passed unchanged suites are not repeated.
- Both canonical runs exit0, SOAP5000 and L-BFGS1000 complete; entire configs
  equal except the two declared fields, initial bytes equal,36sources equal.
- Independent CPU checkpoint field replay passes established FP64 tolerance
  1e-10: maximum9.99e-15. Common/native L2 independently recomputed from arrays.
- GPU initial reconstruction exactly equals every saved initial leaf in both
  arms. GPU final-field replay uses the established1e-10 FP64 contract;
  maximum4.44e-16(nonlocal),0(local). Trained modes stay1,2,3; periodic edge
  gaps below1e-12. Full dtype/grid/reference/history checks pass.
- Initial audit script wrongly required bitwise CPU/GPU random initialization:
  CPU template differs by FP32 roundoff up to1.49e-8. Exact reconstruction is
  correctly performed on the training GPU backend. Another overly strict GPU
  bitwise field assertion sees4.44e-16 projected-mean reduction differences;
  use the PRE-EXISTING1e-10 prediction contract and record actual differences.
  Failed audit logs retained, no prediction/checkpoint edited, no established
  numerical tolerance loosened. These are audit assumptions, not training bugs.
- Convergence plot inspected: equal-iteration, actual-wall and final-time curves
  all use the same local reference, raw histories and terminal saved fields.
- Final schema/link/path/config/diff/worker checks are recorded in the companion
  validation JSON. No additional unchanged expensive suite was run.

## Changes, scope and task-end

Added explicit local control config and one focused combined numerical test.
Updated example README, DEVELOPMENT, artifact index, PROGRESS and DECISIONS.
Archived this task, scripts, raw comparison/validation and convergence figure.
Kept defaults, network/loss/operator production implementation, reference data,
old paths and checkpoints. No Git history operation is authorized or performed.

Status: complete. Acceptance criteria fulfilled; no blocker or next required
work. Both training and all verification processes finished; no worker remains.
A multi-seed accuracy comparison or independently tuned all-AD baseline would
be a separately scoped experiment.
