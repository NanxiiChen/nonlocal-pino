# Project Progress

> Short dashboard. Current work is self-contained in the one active task;
> completed experiments and superseded process observations are archived.

## Active task

[Nonlocal PINN history-field learning and architectures](tasks/nonlocal-pinn-architecture.md).
Opened 2026-10-15 after the user closed the fracture2d feasibility stage.

Conditioned-history retry STOPPED at 0.0053 after 1024 outers; no GPU job.
Target 0.0060 remains incomplete. User provisionally accepts the current result. True cold pilot began
at physical/network/Adam/bank zero; the current declared reduced-rate retry
inherits accepted physical states through 0.0016 and the failed 0.0017 trained
network/Adam, then redoes 0.0017. Replay training remains OFF; start/end history
snapshots are collected only. Default accepted MLP configuration is unchanged.

Learning-rate control supports reduced update oscillation: fixed-H 512-update
arms at 1e-4/1e-5 retain similar global accuracy, while only the lower rate
passes the final three field-change checks. Retry uses the existing decay
schedule, already at its 1e-5 floor at restored Adam count 35328. First three
retried endpoints independently audited (physics, state, metrics and TB).

Last field-stable accepted load is 0.0052. Saved mechanically synchronized
0.0053 candidate is budget_exhausted: damage change3.19e-4>1e-5. Endpoint
image shows a spanning, nearly straight crack. Reaction1.55179 vs same-load
reference1.57096 (1.22% difference); damage relative L2 7.130%, own-H7.098%,
own-Htip3.568%, physicalAT2.16413. All external accuracy limits pass, but
field stability fails, so driver stops normally (exit0), not a completed path.
At 0.0052: 38 outer sweeps, same-load damage error 1.874%, own-H
error 1.605%, own-H tip error 0.331%, reaction 607.029 (reference 606.591).
The 36 completed retry loads 0.0017–0.0052 need a median 6 outer sweeps
(range 5–38); this is not a matched architecture wall-time speedup benchmark.
Both old MLP and new learned multiscale encoder receive H; this run changes
its encoding and uses staged L1→L2 initialization, then L2 warm fitting.
Exact architecture, immutable controls, receipts and resume instructions:
[active task](tasks/nonlocal-pinn-architecture.md). Root/TB run basename:
`pinn_history_conditioned_decay_20261015`; owner140094/worker140110 both exited.

## Accepted fracture2d stage

- [Stage closed and archived](tasks/archive/fracture2d-nonlocal-pinn.md),
  2026-10-15. Accepted engineering result uses local P1 direct mechanics,
  nonlocal AT2 PINN damage and maximum-history stagger. Same-model direct
  damage and independent local FEM remain comparators.
- Accepted `pinn_baseline` saves 0 -> 0.0057, spans by 0.0053; the user
  interrupted 0.0058. It is not a completed 0 -> 0.0060 run. Field accuracy,
  path smoothness, convergence and efficiency remain improvement targets.
- Default: full domain, L1/Adam/sigmoid/FP64, du=1e-4, 128 inner updates,
  outer caps 128/1024, damage-change tolerance 1e-5. No symmetry constraints.
  Optional core+halo keeps a fixed 2048-pattern exact-support padded pool.
- Measured insight: cheaper core+halo iterations needed more outer work;
  matched early-load total time was 3.35 times full. Keep full as default.
- Consolidation and publication complete: final main b395e48 independently
  verified on origin. Four public configs tracked. All _legacy/ directories
  and docs/reports/ are local-only/ignored; 324 withdrawn files preserved.

## Old GPU baseline stopped by user

Saved-config repeat pinn_baseline_gpu_rerun_20261015 is CANCELLED after
through-fracture at explicit user request. Verifiedowner31071/worker31076 exited.
54completed endpoints through0.0054 preserved,partial0.0055outer600 and named
checkpoint512 retained. TensorBoard6007 continues. External stoprequest/verified
receipts and archived stage record exact state; target0.0060wasnotcompleted.
This cancellation is separate from the current short history-operator probe.

## Other retained state

fracture1d remains an accepted engineering baseline. Older fracture2d
nonlocal-mechanics, warning-accepted direct runs and detailed diagnostics are
historical controls, not the public main route; see stage archives.
[DECISIONS](DECISIONS.md) retains append-only rationale. Example READMEs
contain stable formulations and usage.
