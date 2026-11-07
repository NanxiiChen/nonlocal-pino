# Twice-speed particle-corrosion reference and cold PINO

Date: 2026-11-06 (Asia/Shanghai). Status: completed experiment; reference
qualified, both cold PINO trials failed to achieve useful accuracy.

## Goal and authorization

User approves checking the proposed uniformly doubled dynamics on reference
solutions first, then cold-training if the reference is sound. Retain t_max1,
kappa2, random eight-particle geometry, diffuse width and current rolling
core/halo network/optimizer strategy. Add explicit experiment presets; preserve
the published default, historical fields and checkpoints. No commit/push or
deletion authorized this turn.

## Acceptance and order

1. Scale n_ac1,n_ac2,n_ac3,n_ch by2 only in the training preset. Reference uses
   independently qualified local P1 FEM/lumped mass/BDF, same frozen development4
   and reused8 geometries. Keep reference labels outside gradient updates.
2. Before training, verify finite fields/phase range/mouth values, larger
   corroded area and actual interface contact; inspect a field comparison.
   Check first-geometry spatial and temporal refinement, plus the time-scaling
   identity against the existing old trajectory on its overlap. Target
   refinement phase error below0.5% pooled and1%terminal, with temporal errors
   much smaller; otherwise diagnose before training.
3. Validate fresh known-input bank and reference exclusions, then canonical
   train.py cold20000 with width192/B4/Nt8/Adam/FP32/ordinary and rolling64slots.
   Use a fresh runtime seed; do not claim a paired initialization comparison.
4. Evaluate final scheduled checkpoint on all reused8, phase/concentration
   pooled/final errors and corroded/IoU metrics, timing/memory and rolling audit.
   Present convergence and representative fields; record sources and limits.

## Resources and resume

RTX5090 32607MiB idle;29GiB free disk. Existing source/reference implementation
is unchanged. Prior same-shape rolling cold20k took959s,4.93GiB active peak.
Expected new artifacts are far below free space; no cleanup needed.

Baseline run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-02.
Original inputs: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-input-bank64-01.
Old dev/reference: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reference4-01.
Old reused8/reference: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reserved8-01.

Blocker: no pending execution. Both finite owners, final evaluations and
diagnostics completed. No training is queued. Published defaults are unchanged.
The observations below preserve the sequence of the experiment; the final
outcome and limitations are recorded at the end. Any future optimization should
start from the early-time, full-mesh residual audit rather than assuming that
the qualified reference or the mouth correction establishes PINO convergence.

## Reference gate passed / live owner

Separate train_speed2 changes only four physical coefficients by2.
reference_speed2 uses rtol2e-8,atol2e-10,max_step0.00625; temporal check uses
1e-9,1e-11,0.003125. Existing4reference focused tests pass. New64input geometry
records and all17arrays exactly reproduce the original bank from runtime seed
1307500965; initial condition/geometry do not depend on these four rates.

All12local reference trajectories complete with finite fields and physical
phase ranges. On reused8, mean final smooth-corroded-area ratio(new/old)=2.05138;
old area fractions17.21..19.34%,new35.31..38.29% of the non-ceramic domain.
The area measure integrates1-phi and includes the initial pit. First-case
spatial h.01→.0075 at256contour points changes phase0.12027%pooled/0.25819%final,
c0.15735%/0.32381%; time refinement phase2.33e-7%/3.81e-7%. All8old/new overlap
trajectories at newt<=.5 agree, maximum phase pooled relative error1.11e-11.
No out-of-domain extrapolation or interpolation through holes; conforming FEM
triangles used for spatial checks and figure. Saved gallery visually verified.

New data directories under /root/autodl-tmp/data/particle_corrosion2d/:
- 20261106-speed2-input-bank64-01
- 20261106-speed2-local-dev4-01
- 20261106-speed2-local-reused8-01
- 20261106-speed2-qualification-01 (validation.json, refinement subdirectories,
  reference_speed2_vs_original.png, generation/validation/owner scripts and logs)

Finite owner PID176421, script at qualification/run_train_audit_evaluate.py;
its owner_status.json records actual child PID/stage/exit status. After training
it runs online_validation and then explicit final-checkpoint evaluate_run.
No continuation after an unexpected failed stage; retain its error log.

Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-online-w192-cold20000-01.
Final evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-online-w192-reused8-final-01.
Environment: .venv, ordinary FP32, BLAS/OMP threads1, GPU preallocation false.
Fresh runtime training seed is in runtime.json, not a config option. No paired
initialization claim; reference cohorts were previously inspected.

## Training observation (not a final result)

At8000steps, development4phase pooled40.79%/terminal60.51%, versus its initial
56.16%/60.20%. From2000through8000, pooled remains near40%; raw PDE losses
decrease (at8000 AC0.0216,CH0.0788), while physical errors do not converge.
This is a failed early optimization trajectory, not evidence against the
qualified reference. Let the originally approved20k endpoint finish and audit
its fields before drawing a final conclusion or changing the strategy.
Warm ordinary updates are about22ms; observed peak active5275895040bytes.

## First20k complete; observed boundary gap and corrective experiment

First20k completes, owner/audit/evaluation all pass. Reused8phase42.0821%pooled/
65.8807%terminal,c43.3184%/67.0401%,corroded91.8312%/87.6075%,IoU16.802%.
Runtime1007.637s,warmheld22.4323ms,activepeak4.9222GiB. Original reference gate
is sound; this cold initialization fails optimization and is retained.
Saved first-case pit area drops0.02369→0.00314 by t=.4 despite correct initial
area0.02401; reference advances to0.08309. The volume-average chemical driving
q/cLe is approximately-1 on both, so a global lack of driving does not explain
the failure. Pointwise concentration and boundary errors still matter.

Independent full-native-mesh4081node audit finds early AC/CH residual spikes.
At t=.01 ACvolumeL1=.8005,RMS5.8605,max67.198;99.29%of absolute AC residual is
in the predicted interface occupying6.75%of volume. Most strikingly, at t=.1,
independent mouth-loss points have max|c/cLe|=.000734 while actual operator
mouth nodes reach27.0015. These discrete nodes are operator neighbors but have
no PDE rows. The old midpoint BC loss can miss the mouth endpoints.
Audit: qualification/full_mesh_residual_audit.json; diagnostic script retained.

Proceed with explicit train_speed2_mouth_nodes preset. Inputs.initial_payload
overrides only the BC-loss mouth query/weight arrays with all actual mesh
Dirichlet nodes and P1 edge-length lumped weights, fixed zero-weight padding.
Boundary encoder tokens, random IC draws, PDE/geometry and five losses stay
unchanged. Old absent selector retains midpoint semantics; load_bank rejects
an incompatible declared policy. All15focused input/online tests pass, including
the smooth trace vanishing at old midpoints but not at operator endpoints.

Corrective study: /root/autodl-tmp/data/particle_corrosion2d/20261106-speed2-mouth-nodes-study-01.
Input bank: /root/autodl-tmp/data/particle_corrosion2d/20261106-speed2-mouth-nodes-input-bank64-01.
Preparation PID194078, replay seed1307500965. Next: confirm the64geometry
records/all unchanged arrays, then launch another canonical cold20k using the
same qualified speed2 dev4/reused8 references and automatic final audits.
No promotion of defaults; fresh seed means the two trainings are not paired.

Corrective input gate passed: all64geometries and15unchanged arrays exact,
only mouth_points/mouth_weights differ. Every initial geometry has9actual
reservoir nodes, retained in16slots with positive edge weights and zero padding.
Cold corrective owner PID194241 is running from the new study directory;
its owner_status.json tracks the canonical child and all final stages.
Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-mouth-nodes-w192-cold20000-01.
Final evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-mouth-nodes-w192-reused8-final-01.
Both trainings use the same speed2 references; old default stays unchanged.
New study scripts summarize_final.py and audit_full_mesh.py are ready to run
after its finite owner reports complete. They output the counterpart report
directory20261106-speed2-mouth-nodes-report-01 and full-mesh audit JSON.

Corrective run at14000: devphase40.535%pooled/59.811%terminal, still a poor
optimization trajectory. Boundary sampling correction alone has not recovered
convergence. Keep the declared20k budget and inspect final actual-node BC and
early-interface residuals. Do not silently add further model/loss strategies or
promote these fields; the reference success and neural failure must be reported
separately. Future optimization candidates are not yet selected or launched.

## Final corrected20k result and diagnosis

Both owners report complete; all training, online replay and final evaluation
stages exited0. This is successful execution of two unsuccessful optimization
trials. On the same eight reused localFEM geometries, arithmetic means of each
geometry's relative L2 values are:

| Cold20k experiment | Phase pooled | Phase terminal | Concentration pooled | Concentration terminal |
|---|---:|---:|---:|---:|
| Speed2, original midpoint BC | 42.0821% | 65.8807% | 43.3184% | 67.0401% |
| Speed2, operator-node BC | 41.2232% | 63.2892% | 42.4853% | 64.5478% |

Corrected terminal corroded-region IoU22.0664%, corroded-field relative
L2 89.9566%pooled/84.1565%terminal. First-case predicted smooth corrosion area
still shrinks0.023724→0.003546 by t=.4 and only reaches0.042678 at t=1;
the reference grows0.024011→0.083085→0.186964. These fields are not usable
predictions. Development4finalphase40.5324%/59.6829% is distinct from the
eight-geometry result above. New cold seeds3048889704/1709019262 differ;
initial64geometries match, later rolling draws differ. Small error differences
between the two trials are not an isolated boundary-policy effect.

The corrected full4081node audit confirms the intended boundary repair:
at t=.1, max|c/cLe| on actual mouth nodes falls from27.00146 to0.0016764;
max|phi| falls0.97834→0.0013049. The corrected midpoint trace is similarly
small. Early interior/interface PDE residuals persist: at t=.01, raw AC
volume-RMS5.96864,max64.4467, with99.4213%of its absolute residual in the
predicted interface occupying6.8316%of volume. At t=.001 raw CHRMS133.1845,
max1637.28. These are native finite-horizon graph diagnostics, not continuum
certificates. They show that fixing the observed mouth-node gap is insufficient
and that the network does not solve its own early-time discrete dynamics.
They do not establish that all remaining error is from one mechanism, quantify
the complete local-versus-nonlocal discretization error, or show that speed2
is inherently unlearnable. No further weighting, hardIC, causal or optimizer
change was tested in this study.

Corrected run wall time about1019s(17.0min); synchronized held updates
median22.4913ms, process cumulative activepeak4.9214GiB. The report derives
1018.975889s from timing.json; summary.json includes final bookkeeping and
reports1019.024639s. Online audit validates20000updates,399installed events,
316newgeometries,1520spatialrefreshes,380distinctsampledgeometries,
1900distinctgeometry/spatialversions and all64exact final payloads; overlap
with evaluation and future solution fields both0.

Final report directory:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-speed2-mouth-nodes-report-01
contains report.json, fields_first4.png and convergence.png. The latter compares
the two speed2 trials on development4, and states the fresh-seed limitation.
Full-mesh diagnostic is in the corrected study/full_mesh_residual_audit.json;
owner_status.json records every finite child stage. Input_validation.json
records the64initial geometries and15unchanged-array checks. All generated
scientific figures were visually inspected; images/raw arrays/weights remain
outside Git. artifacts/index.yaml registers both reports and evaluations.

## Diff-level changes and verification

- Three explicit speed2/reference/node-BC presets; published default unchanged.
- inputs.py adds physical P1 mouth-node quadrature with validated coverage,
  positive endpoint weights and fixed padding, preserving encoder tokens/IC
  draws. problem.py validates the selector; train.py rejects mismatched bank
  policies. Historical missing selector explicitly retains midpoint semantics.
- Three new focused tests exercise physical weights/endpoint detection,
  unchanged conditioning/IC draws, and input-bank policy mismatch rejection.
  Input/online suites15tests pass; reference suite4tests pass. These19tests
  were not repeated without changes. Both full20kruns and final64slot replay
  audits exercise the production paths.
- Example README and DEVELOPMENT document the explicit option and numerical
  limitation. DECISIONS, PROGRESS, this archived task and artifact index retain
  completed results; earlier time/kinetics discussion records are preserved.
- Final checks pass:7changed/new Python files parse,94local Markdown links
  exist,11cases/530registered artifact directories exist, and git diff --check
  is clean. Both finite owner journals and all9time diagnostic snapshots per
  trial are complete. nvidia-smi reports no compute processes after completion.
  No commit/push, deletion, default promotion or pending background work is
  part of this completion.
