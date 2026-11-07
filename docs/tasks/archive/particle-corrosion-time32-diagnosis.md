# Particle-corrosion early shrinkage and time sampling

Date: 2026-11-06. Status: complete; increasing random times did not restore
PINO convergence. All finite stages passed; no pending training.

## User request and scope

The user questions why uniformly doubling dynamics breaks convergence, observes
early pit shrinkage, requests confirmation that mouth BCs are correct, and wants
at least16randomly changed times per update. Diagnose the actual boundary
formulation/enforcement and sampling; use32uniform random times in an explicit
speed2/node-mouth preset. Keep PDE, architecture, optimizer, physical inputs
and reference cohorts fixed. No default promotion, commit/push or deletion.

## Context and evidence

Prior study: docs/tasks/archive/particle-corrosion-speed2.md. Original1x rolling
20k gives reused8phase1.9827%pooled/3.1209%terminal. Speed2midpoint/node trials
give42.0821%/65.8807% and41.2232%/63.2892%. Qualified local reference expands
correctly; network initially fills the pit. Node-aligned BC removes an actual
endpoint sampling gap, but early full-graph AC/CH residuals remain large.
Both PDE losses are mean absolute raw residuals with gradient balancing;
tanh bounds phase only, with no Jacobian-divided PDE preconditioner. Current
8times are already independent uniform draws for each update, pre-generated
and stored in sampling.npz; all geometries share each update's time vector.
Physical derivatives use JVP, not finite differences between those8times.

Reference/input/run roots are /root/autodl-tmp/data/particle_corrosion2d and
/root/autodl-tmp/tf-logs/particle_corrosion2d. Reuse qualified speed2-local-dev4-01,
speed2-local-reused8-01 and speed2-mouth-nodes-input-bank64-01, all with prefix
20261106-. Previous corrected run is20261106-speed2-mouth-nodes-w192-cold20000-01.
RTX5090 32GiB idle,29GiB disk free. No cleanup needed.

## Acceptance and plan

1. Independently inspect mouth group/location/targets and KKS/flux semantics
   in local reference and PINO. Quantify old and corrected trace errors and
   actual early time coverage. Distinguish evidence from causal hypotheses.
2. Add a separate Nt32preset only, validate unchanged bank compatibility and
   changed per-step sampling. Profile finite warm steps at B4/width192/32times
   before the20k cold run; no silent OOM fallback or strategy changes.
3. Run canonical train.py with rolling inputs and automatic final replay and
   reused8evaluation; inspect final convergence/fields and raw early residuals.
   Fresh seeds mean this is a practical test, not an isolated paired ablation.
4. Record timing/memory, results and limits, then archive. Do not claim more
   times guarantees convergence or that fixed-coefficient2x increases eigenvalue
   stiffness ratios; it doubles the evolution horizon represented in0..1.

## Current state

Blocker: none. OwnerPID220408 completed all five declared stages with exit0:
train,onlineaudit,reused8evaluation,report and fullmeshaudit. Final results and
limitations appear below. No pending process, automatic continuation or default
promotion. Further optimization requires a new scoped experiment; the raw-L1
objective and early dynamics remain unresolved. Previous changes are preserved.

## Boundary and early-time audit

Study directory: /root/autodl-tmp/data/particle_corrosion2d/20261106-speed2-time32-study-01.
boundary_time_audit.json confirms all eight saved FEM references have phi=c=0
on the geometric mouth y=0,xin[.445,.555], hence q=-cLe, not q=0. Native mesh
uses the same tagged opening,9Dirichlet nodes including endpoints; corrected
soft loss includes all9with total physical boundary length.11 and positive
weights. PINO/reference initial fields match exactly on the native mesh.
The remaining top/sides/bottom and particle walls use the declared closed-flux
model; no BC target was reversed or switched when doubling rates.

On the first reused geometry, independently evaluated initial smooth-corrosion
area rates are+0.0273170(local P1 on native mesh),+0.0266363(nonlocal).
Direct nonlocal BDF with identical native operator, exact mouth elimination,
rtol2e-8/atol2e-10/max_step.0005 over0...03 gives area0.0240102→0.0268213,
finite phase0..1. Runtime69.03s. This native early trajectory does not exhibit
the network's pit collapse. It is an early-dynamics check, not a complete
local/nonlocal error budget or proof of correct PINO convergence. Inputs and
raw fields are saved in native_early_reference.npz. The first audit script
mistakenly expected a saved mouth mask; retained original error log; corrected
audit derives the mask from the declared interval and saved coordinates.

Actual saved8time traces all have20000distinct adjacent rows. Fractions of
updates containing t<.01 are7.965%(1x),7.810%(speed2midpoint),7.710%(speed2node).
Theoretical hit probability1-(1-.01)^Nt is7.726%/14.854%/27.502% for8/16/32.
More uniform times reduce sampling noise and increase per-update coverage but
do not change the expected time weighting or JVP derivative accuracy.

Prepared resource_probe.py uses the production core/halo gradient path,
complete64slot device bank and32updates including warm GradNorm refresh.
run_train_audit_evaluate.py gates on its success/activepeak<=26GiB and the
boundary audit, then runs cold20k,online audit,reused8evaluation,summary/plots
and final full-mesh diagnostic. Both postprocessing scripts validate prior
stages explicitly. No prior training checkpoint is resumed.

## Nt32 resource gate and cold run

32finite production-path updates pass including two GradNorm refreshes.
First compile213.994s; held median48.7329ms, warm refresh152.7290ms;
activepeak16519942912bytes=15.386GiB. Full64slot padded bank resident; probe
does not include rolling replacements or final evaluations. Its complete logs
and resource_probe.json are in the study. No out-of-memory fallback occurred.
Config comparison proves only sample.Nt8→32 changes from the previous node
preset. Physical/boundary equations and published defaults remain unchanged.

Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-time32-w192-cold20000-01.
Automatic final evaluation: same root/20261106-speed2-time32-w192-reused8-final-01.
Automatic report/figures: same root/20261106-speed2-time32-report-01.
Runtime seed452250743; fresh model/optimizer. Actual saved sampling.npz is
(20000,32), no identical adjacent time rows;27.545%of steps hit t<.01.
No reference targets enter training; both dev4/reused8geometries excluded from
the rolling training population. All stages have logs and explicit owner status.

Same-elapsed-time comparison from prior saved fields, all reused8geometries:
old1x t=.2/.4/1 phase errors0.6806%/1.4582%/3.1209%, whereas failed speed2
node model at t=.1/.2/.5 gives19.9125%/27.3310%/43.4418%. The qualified speed2
reference on the overlap is the common target; coordinate arrays match exactly.
Thus the failure starts before the newly extended physical interval. Results
are saved in matched_elapsed_time_errors.json; cold seeds differ.

## Session changes and verification

New train_speed2_time32.py preset and README explanation only; no additional
PDE/network/optimizer changes this session. Previous uncommitted boundary fix
is retained and used. Boundary/static/sampling audits and32production gradient
updates pass. New preset parses, validates and differs only in Nt. Prior19
focused tests remain applicable to unchanged numerical code and were not rerun.
Memory updates: this archived task, PROGRESS, DECISIONS and artifact index.
No Git mutation or cleanup. The finite experiment and final review are complete.

## Final result (reviewed after the user's status request)

Final20k endpoint, same eight reused localFEM geometries. Values below are
arithmetic means of each geometry's physical-quadrature relative L2 metrics:

| Random times/update | Phase pooled | Phase terminal | Concentration pooled | Concentration terminal |
|---|---:|---:|---:|---:|
| 8, operator-node BC | 41.2232% | 63.2892% | 42.4853% | 64.5478% |
| 32, operator-node BC | 41.2407% | 63.2984% | 42.5143% | 64.5533% |

No useful improvement. Both cold runs have different seeds; the comparison
does not isolate a population effect of Nt, but this practical32time change
plainly did not repair the failure. Development4phase40.5513%/59.6958% is the
convergence-curve cohort, distinct from the reused8final table. Curves plateau
early, and the four-case final field gallery still shows pit collapse.
First reused geometry predicted smooth corrosion area is0.0235745 at t=0,
0.0067862 at t=.1,0.0033307 at t=.4,and0.0426731 at t=1; reference values
are0.0240110,0.0351528,0.0830854,0.1869636. Terminal corroded-region IoU22.1048%.

The full native graph audit confirms small mouth errors but unresolved PDEs:
at t=.01,max|phi|=.0006127,max|c/cLe|=.0034423 on actual mouth nodes, yet
ACvolumeRMS10.2385,max175.1363.99.1933%of absolute AC residual is in the
predicted interface occupying7.9311%of volume. At t=.001,CHRMS82.7825,
max1107.25. These native-operator diagnostics show failure to solve the
early dynamics despite increased random coverage; they do not prove a unique
cause or justify blaming the independently checked reference. More uniform
times leave the expected temporal loss weight unchanged; causal or residual
weighting/preconditioning changes were not tested in this experiment.

Training total1399.871936s(23.3312min, timing.json), held-update median49.0892ms,
first update including compilation196.3505s, activepeak15.6800GiB. Owner wall
including all automatic audit/evaluation/report stages1607.2662s(26.7878min).
Audit reconstructs20000updates,399events,316newgeometries,1520spatialrefreshes,
380distinctgeometries/1900geometry-spatial versions and all64final input slots;
evaluation overlap/future solution fields0. Saved per-case/final metrics are
independently recomputed by the completed report script at1e-12tolerance.

Direct report/figures:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-time32-report-01/report.json,
convergence.png and fields_first4.png. Both figures visually inspected.
Study/full_mesh_residual_audit.json contains nine audit times; owner_status.json
records every completed subprocess. All outputs remain external and defaults
unchanged. No additional test suite or training was needed for this status review.

Completion checks pass: all41snapshotted numerical Python files match current
files byte-for-byte,98local Markdown links exist,11cases/534registered artifact
directories exist, all five owner stages exited0 and the nine-time diagnostic
is complete. Both figures visually inspected; git diff --check passes. No GPU
compute processes were reported at final status inspection.
