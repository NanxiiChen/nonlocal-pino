# Project Progress

## Private GitHub publication — 2026-10-24

Repository: [NanxiiChen/nonlocal-pinn](https://github.com/NanxiiChen/nonlocal-pinn).
Created as private; default branch main. Split code and publication records
follow dev commit -> regular merge into main -> push main only. Local dev
remains local. Original source and recovery backup are retained.

## Project split complete — 2026-10-24

Independent package, explicit configs and artifact index are delivered.
[Validation and recovery](migration/VALIDATION.md);
[completed task](tasks/archive/project-split-validation.md).
Original source/environment/data/log paths remain intact.
The split preserved physical formulations and existing artifact paths.

## Active task

None.

[Attached two-phase PINN paper reviewed](tasks/archive/two-phase-pinn-paper-review.md).
Qiu2022 uses local CHNS/four-output AD, sequential time windows and material
coefficient clipping; forward loss uses PDE+IC/BC. Reversed-vortex transport
then1000:1bubble are useful validation stages. Actual velocityL2 is3.56–15.12%,
area drift~1.12%; no exact conservation/sub-percent field claim. Rendered p7/p14
confirm printed potential/initial-profile inconsistencies; BC/chemical scaling/
coordinates require resolution before implementation. Same-group2022 MIM already
outputs chemical potential, so mixed AD is a required efficient comparator for
proposed nonlocal mixed form.35pages extracted/primary follow-ups/math/doc/diff
checks complete. No implementation/training/default/Git change or blocker.

[Next phase-field cases assessed](tasks/archive/phasefield-case-recommendations.md).
Recommend periodic2D CH spinodal decomposition for a scalar extension, or
matched-density CHNS droplet relaxation for two-phase flow. Primary PFHub/
CHNS/nonlocal-CHNS sources consulted; conservative transport/capillary energy
cancellation checked analytically. Existing Laplacian alone does not supply
compatible flow gradients/divergence; separate derivative approximation from
nonlocal interaction physics. No implementation/training/default change or
Git operation; source/doc/diff checks pass, no blocker or active task.

[Ordinary-policy defaults completed](tasks/archive/ordinary-policy-defaults.md).
Six new serial5000-update cold starts plus10KKS/1full-grid melting CPU reloads
support per-case defaults: corrosion ordinaryFP64, electro/melting ordinaryFP32.
Corrosion42/2026 phi0.002499/0.006833; electro0.004757/0.003399, while ordinary
FP64 electro42 fails0.299757. Melting new explicit ordinary full-gridphi3.9053e-5.
Original physics/residuals/data/SOAP/horizons retained; seed remains run provenance.
Corrosion late spikes/Dq persist; no uniform stability/global-optimum claim.
Shared explicit policy/conflict parsing, README/index, historical preset precision,
curves/fields/manifests/source/scripts archived.11focused tests +7affected-policy
checks,2canonical GPU smokes and default/reference/dtype/source audits pass.
All workers exit0; no Git history operation or blocker. Final JSON/link/artifact/
diff verification is recorded in the archived validation file.

[Other-case GPU policy audit completed](tasks/archive/phasefield-gpu-policy-audit.md).
Four actual-config FP32 workers finish125 updates each (50 warm-up/75 timed).
Electropolish deterministic/ordinary197.73/10.98ms (~18.0x); melting468.26/26.62ms
(~17.6x). Initial/all-pool/time/data arrays exactly match. HLO verifies3032-edge
and11929-edge loops, including backward; independent numerical errors<3e-7.
Electropolish default enforces deterministic flag; melting inherits environment
and is already ordinary when unset. Other two entries only statically inspected;
1D missing default config is documented without choosing physics. Source/raw
results/HLO/plots archived, READMEs/index updated. No long accuracy cohort,
production/default/physics/Git change; all workers exit0 and no active task.
Final21-artifact/26-source/strict-JSON/link/HLO/input/output/308-dir/diff checks pass.

[Ordinary FP64 accuracy control completed](tasks/archive/ordinary-fp64-accuracy.md).
One additional seed42 cold start finishes5000 updates/50s in196.5s through
train.py. Finalphi/c0.002499/0.002944 versus default0.008835/0.010426; observed
~6x faster with different cache histories. Latephi median0.004685 improves,
mean/max do not; Dq late spike18.665 prevents a uniform stability/transport claim.
All four initial parameter values match exactly; independent CPU reload/dtype/
config/reference/transport/topology/history/source checks pass. Curves, fields,
source/direct paths archived; README/index updated, workers exit0, no default
or Git history changes. No active task remains.
Final10-artifact/source/strict-JSON/link/output/5-case308-directory/diff checks pass.

[Compiler-policy explanation completed](tasks/archive/compiler-policy-explanation.md).
Frozen GPU/HLO probe confirms current deterministic graph3032-edge scatter
lowers to3032-iteration while; ordinary retains scatter. Ten repeats give exact
forward/input-gradient deterministic results and~6e-8 relative ordinary variation.
Both policies pass earlier SciPy checks at~1e-8. First full-run losses differ
1.5e-7, adaptive weights diverge44% by200: supports sensitive training paths,
not a guaranteed deterministic precision advantage or uniquely isolated cause.
Probe/source/HLO/metrics archived; README clarified. No retraining/default/Git
change; workers exit0 and no active task remains.

[Full local/nonlocal cold starts completed](tasks/archive/local-ad-full-training.md).
Four paired FP32/seed42 cold starts finish5000 updates/50s through train.py.
Own-reference finalphi/c: graph default0.008835/0.010426, ordinary0.020680/0.024565,
local tensor0.346112/0.373221, local joint0.411813/0.425739. Local adds matched
no-flux sixth loss; independent401 FD reference retains1.42%/1.68% spatial
refinement discrepancy, so no sub-percent continuum claim. Defaults unchanged;
ordinary faster but worsens field/late metrics while improving q/Dq. Twelve
focused checks, two canonical CPU smokes, four full runs and independent CPU
checkpoint/initial/config/transport/topology/roundoff audits pass. Curves, fields,
source and direct paths archived; all workers exit, no Git history operation.
Final source/strict-JSON/link/output/5-case307-directory and diff checks pass.

[Nonlocal advantages assessment completed](tasks/archive/nonlocal-advantages.md).
Primary-source review and actual5151-node graph checks support lower spatial
regularity requirements, reciprocal pair/positive-energy structure and finite
interaction-scale/representation flexibility. Bounded high-frequency operator
amplification is derived, not a demonstrated PINN convergence advantage.
No matched local accuracy/robustness superiority claimed; open Dirichlet
reservoirs, soft residuals, stored-diagonal roundoff and local-target bias are
explicit. Sources/check/raw JSON archived; no training/production/Git change.

[Nonlocal versus local-AD performance completed](tasks/archive/local-ad-performance.md).
All12 serial GPU workers exit0:6 primary,4 graph-policy kernels,2 ordinary
full-update controls. Same model/IC/BC/SOAP,3760–4032 active PDE pairs,
50 warm-up+75 measured updates. FP32 graph/tensor/joint229.31/11.98/16.12ms;
FP64266.11/31.32/47.77ms. Deterministic aggregation~38ms vs ordinary~0.17ms;
ordinary graph complete12.49/35.13ms shows policy overhead is a major cause.
Joint time-only grad/JVP controls agree; random pairing does not forbid JVP.
Manufactured derivatives/chain rule/data/mixed gradients, raw timing/refresh/
dtype/count/initial-loss checks pass; plots inspected. No convergence, boundary
equivalence or ordinary-policy reproducibility claim. Production code, defaults,
reference and Git history untouched. Sources/raw JSON/figures archived; GPU idle.
All27 archived files/17 JSONs, source/link/status/operator checks,5-case303-dir
inventory and git diff --check pass. Updated memory: this dashboard, DECISIONS.md
and the self-contained archived task.

[Time-JVP structure clarification completed](tasks/archive/time-jvp-structure.md).
Within-update times share a spatial subgraph; updates cycle the pool. All three
nonlocal phase-field paths already use time JVP. Independent CPU float64 toy
checks match vmapped time grad exactly and verify zero cross-time blocks.
No Nx*Nt-factor speedup claimed against properly vectorized grad; no timing,
training or production change. Evidence archived; git diff --check passes.

[Corrosion FP64 accuracy control completed](tasks/archive/corrosion-fp64-accuracy.md).
One paired seed42 cold start completes5000 updates/50s through train.py, with
identical promoted FP32 initialization/data/time draws/stored graph coefficients.
FP64 final phi/c0.002677/0.003207 vs FP32 0.008835/0.010426; free q halves,
Dq1.703->1.220. Late phi median improves~16%, but max/mean and Dq median worsen;
no stability/error-floor/cross-seed claim. Default FP32 retained; explicit
compute_dtype option, dtype records, README/index and archived scripts/results/
plots delivered. Seven focused tests, exact GPU initialization/graph parity,
independent CPU reload/AD/transport/topology and frozen-inference controls pass.
FP64 subprocess exits0 (~24.9min including compilation/evaluation); GPU idle.
All11 archived artifacts/JSON/source/link/run checks,5-case303-directory
inventory and git diff --check pass. Memory updated: this dashboard,
DECISIONS.md, DEVELOPMENT.md and the archived self-contained task.
No reference/default physics change or Git operation.

[Local FP64 performance benchmark completed](tasks/archive/fp64-performance.md).
8 serial GPU measurement workers and1 dtype-lowering audit finish. On reported
RTX3090/48GiB, same-size corrosion SOAP/GradNorm75 measured updates after50
warm-up average0.2033s FP32 vs0.2267s FP64 (~11.5% slower). JAX peak active
memory1.535->3.063GiB; compile~156s both.5000 update-only estimate16.94->18.89min,
excluding JIT/evaluation/saves. All model/PDE/SOAP tensors are true FP64; only
existing scalar Optax LR schedule stays FP32. Corrected GEMM/type controls,
finite/dtype/shape/refresh/JSON/syntax/link/diff checks pass. No full accuracy
run/default precision change or Git history operation. Sources/results archived;
GPU idle. FP64 L2 benefit remains untested.

[Corrosion robust-objective optimization completed](tasks/archive/corrosion-robust-objective.md).
Local dev snapshot e4a0764/f65e6f8 is committed; no merge/push. Four new5000-step
cold starts and a9-model CPU audit finish. Frozen bounded weighting rescues
failed7/2026 to phi~0.01 but regresses42/late metrics. Corrosion default now uses
original phase with unchanged raw-edge CH/nodal data/scales: observed42/7/2026
worst phi/c0.01074/0.01273, late phi median0.009514; historical bounded best remains
separate. Dq~1.70–1.76 remains unresolved.3 numerical/5 entry checks, default
strategy parity, paired initial/config checks,9 final phi/c reloads, topology,
strict JSON/plots/diff and artifact index5cases302dirs pass. All processes exit,
GPU idle. Sources/manifests/curves/fields archived; optimization changes uncommitted.

[Cross-seed mechanism diagnosis completed](tasks/archive/phasefield-seed-mechanism.md).
7 final/10 intermediate CPU probes show nodal IC/BC fit is not sufficient:
failed corrosion2026 BC error6.28e-5 is smaller than successful42's2.45e-3.
Same-data/initial-state seed7 original->bounded gives phi0.00422->0.422.
Failed PDE losses remain high on covered nodes;35 never-core nodes contain
only0.935% phase SSE. Stiff coupled objective/optimization-path sensitivity is
supported, unique mechanism/resampling benefit unproven. Electropolish failure
also has poor full-grid IC fit. Sources/metrics/plot archived; no training,
physical/default change or new remedy claimed. All probes/checks pass.

[Three-case independent-seed validation completed](tasks/archive/phasefield-seed-stability.md).
All3 predeclared seed2026 cold starts finish5000 updates. Corrosion bounded
phi/c0.4070/0.4310 fails with spurious late phase bands (seed7 also fails);
electropolish original/random0.002832/0.003427 succeeds but retains historical
seed42 failure; melting full-grid graph phi5.9857e-5 agrees with seed42.
Canonical configs now omit seed; run-only --seed(default42) is resolved into
saved metadata for replay. No seed shopping, math/reference changes or default
replacement. Full CPU/transport/topology audit,10 focused tests,18 legacy preset
parity checks and raw curves/field galleries pass. READMEs/index retain all
outcomes; all processes exit/GPU idle.

[Three-case SOTA defaults completed](tasks/archive/phasefield-sota-defaults.md).
All3 canonical train.py entries load train_base.py: corrosion bounded/raw-edge/
s_bc/nodal seed42(terminal phi/c0.001433/0.001719), electropolish original/
physical/random seed7(0.001138/0.001358), melting IC-only latent/SOAP seed42
(graph phi6.1097e-5). All5000 updates/lr1e-3; READMEs/index give exact configs,
paths and seed/transport caveats. Legacy configs/data/log paths preserved.
30 numerical +3 entry checks,3 real reduced CPU smoke runs, exact config/
same-device initialization/sampler audits and artifact checks pass. No full
retraining or Git history change; all processes exit.

[Electropolish3x5 best-phi panel completed](tasks/archive/electropolish-best-phi-panel.md).
Reference/prediction/absolute error at t*=0,.25,.5,.75,1 use one selected random
seed7 checkpoint005000, lowest terminal phi L2 among45 current-cohort saved
models(0.001138). Shared color scales, CPU reload, raw NPZ/JSON and PNG/SVG
exports verified. No training/reference/default changes.

[Electropolish convergence curves completed](tasks/archive/electropolish-convergence-curves.md).
Raw200-update histories for random/candidate seeds42/7 are plotted separately
for terminal5 s and pooled0–5 s phi/c L2. PNG/SVG/raw JSON preserve late
oscillations and the successful random seed7 counterexample. Histories/endpoints,
figure layouts and artifact checks pass; no retraining or default changes.

[Electropolish field optimization completed](tasks/archive/electropolish-field-optimization.md).
Nine fresh5000-update/5 s runs,11-model CPU audits and3 focused sampler checks
finish. Original residual/full-grid analytic IC/graph BC/s_bc gives seed42
phi/c0.002106/0.002518 vs current random0.290928/0.299013; seed7 candidate
0.002389/0.002869 is worse than successful random0.001138/0.001358. Candidate
q improves both seeds, but seed7 Dq/c_t regress; no universal/default promotion.
Both candidate seeds have one interface at all21 times. Temporal window helps
full latent0.5194->0.04677 but does not beat original. Sources/direct paths,
mixed outcomes and plots archived; all processes exit. Physics/reference unchanged.

[Electropolish representative physical fields completed](tasks/archive/electropolish-physical-field-gallery.md).
Saved seed42 original/transformed models at0/1/2.5/5 s: phi/c fields, signed
errors and final centerline profiles. Both positive-time fields have three
vertical interfaces in all101 columns; reference has one. Terminal phi/c
0.28604/0.29443 original versus0.34398/0.37920 transformed. CPU reload and
per-time L2 checks pass. These earlier pilots omit later corrosion s_bc/raw-edge
changes. No training, reference, numerical or default changes.

[Corrosion representative physical fields completed](tasks/archive/corrosion-physical-field-gallery.md).
Saved seed42 original/raw-edge models at0/10/25/50 s are plotted against the
unchanged graph reference: phi, c, driving q and signed errors. Positive-time
phi/c improve in this control; initial phi/c/q fits are worse. Consistent color
scales, interface contours, CPU field reload and terminal L2 checks pass.
No training, numerical or default changes.

[Transport optimization completed](tasks/archive/corrosion-transport-optimization.md).
Twelve5000-update/50 s controls,15-model CPU audit and22 focused checks finish;
full baseline and both pre-tail3000/pre-RMS4000 trajectories replay exactly.
Stable KKS/edge evaluation improves phi/c in both seeds, but seed7 q/Dq
regresses; no all-metric default promotion. Late-rate, bounded and cold/late
RMS controls do not give a robust joint field/transport improvement. Keep
original phase/KKS/scales/defaults. All processes exit; sources, commands,
metrics, plots and direct paths are archived. Remaining research: independent
concentration parameters/true field-block refinement, not yet tested.

[Normalized concentration data controls completed](tasks/archive/corrosion-normalized-data.md).
Six seed42 controls, paired seed7 validation and an exact5000-update baseline
replay finish at50 s;16 focused tests and eight independent CPU audits pass.
Original residual+s_bc lowers terminal phi/c L2 to0.01177/0.01402 (seed42)
and0.00620/0.00738 (seed7), versus~0.571/0.547 physical-data controls.
Matched graph BC/IC and explicit concentration normalization options are
delivered. Full latent/stronger IC do not improve later fields. Pit-mouth Dq
hotspots and late variability remain; no production-default or pre-scale
change. All processes exited; report, sources, metrics, plots and direct run
paths are archived. Future targeted work: near-mouth free-node transport.

[Boundary/initial drive audit completed](tasks/archive/corrosion-boundary-drive-audit.md).
Six saved-checkpoint CPU evaluations separate pit/solid BCs and initial q/Dq.
Collapsed original has pit c error~2e-9 but solid error0.035664 (whole drive
scale); initial q is nearly constant despite plausible phi. Reference IC
already supplies a q gradient. Frozen BC substitutions do not reconstruct it.
Finite tanh IC/zero-mouth corner mismatch is explicit. No training/PDE/BC/
default change. Next separate test: normalized concentration BC/IC accuracy
with explicit corner treatment and prior reproducibility limitation retained.

[Corrected stagger controls completed](tasks/archive/corrosion-stagger-experiment.md).
Production adaptive state now excludes temporary masks/bias; research stagger
is wired and verified. Four5000-update/50 s runs do not recover concentration:
phi~0.569, c~0.547, c_L near0 and Dq error~1. Fresh original joint and an
archived-step replay also fail to reproduce the earlier accurate control.
CPU first-batch gradients match; identical-input GPU gradient variation is
measured in old/new paths, without an isolated cause for final collapse.
Eleven focused tests and independent four-checkpoint/config/source/activation
and transport audits pass. Configuration defaults/pre-scales stay unchanged;
all processes exited. Next separate work: establish stable joint replay before
ranking further concentration optimizations.

[Concentration/stagger audit completed](tasks/archive/corrosion-concentration-stagger-audit.md).
Research controls all disable stagger; production single-pit uses1000/25 loss
alternation without freezing fields. Actual loss probe reproduces masked-weight
cache failure at switches without GradNorm refresh. At that audit stage the
research runner rejected unsupported nonzero stagger (now implemented above).
Seven-run decomposition shows better total
c L2 can hide worse c_L/q transport: seed42 cap11 Dq relative error2.94->8.47.
That audit made no production/default/training change; the subsequent correction
and paired experiment are recorded above.

[Residual visual explanation completed](tasks/archive/residual-visual-explainer.md).
Chinese inline compression/residual schematic and measured frozen-corrosion
loss allocation are created. Both phase ranges,12 interaction states and
schematic arithmetic pass local checks; no real browser is available. The
explanation distinguishes output mapping, PDE imbalance, weighting and physical
L2. No training/PDE/default changes.

[Bounded corrosion experiments completed](tasks/archive/corrosion-bounded-residual.md).
Eight new5000-update/50 s cold starts: eta0.1 (cap11) improves seed42 phi/c
0.03984/0.04722->0.01619/0.01928, but worsens seed7
0.02664/0.03160->0.03661/0.04343. Both no-scale arms have phi L2~0.512 and
lose the terminal interface. Retain original/KKS/pre_scale defaults; bounded
mode is research-only. Ten-checkpoint CPU audit and nine focused tests pass.
All processes exited; direct paths and figures are in the archived report.

[Corrosion latent mechanism diagnosis completed](tasks/archive/corrosion-latent-mechanism.md).
At a fixed accurate checkpoint, transformed AC allocates99.47% of absolute
residual to bulk (original8.54%); bulk/interface gradient-norm ratio0.030->2.89.
Exact graph/CH, concentration sensitivity and stiff near-solid-limit checks pass.
Evidence supports excessive bulk amplification and stiff coupling without proving
unique causality or a successful remedy. No training/PDE/default changes.
Bounded weighting was subsequently tested; see the completed experiment above.

[Corrosion KKS/scale ablation completed](tasks/archive/corrosion-kks-scale-ablation.md).
Four new5000-step/50 s controls reuse the validated graph reference. At AC1e-9,
terminal phi/c L2: KKS original0.04212/0.04992, KKS latent0.43679/0.44957;
direct-c original0.53846/0.53685, direct-c latent0.53645/0.52965. Smaller scaling
eliminates exactly-zero concentration derivatives in the KKS latent probe but
leaves extra phase bands. Direct c removes saturation yet fails to grow the pit.
No new setting beats prior KKS/original/AC1e-7, so defaults stay unchanged.
Explicit research CLI map/scale overrides and actual loss/weight logs are added;
seven focused tests and six-control checkpoint/config/initial-weight replay pass.
All GPU/FEM processes are idle; direct paths/figures are in the archived record.

[Matched corrosion/electropolish residual pilots completed](tasks/archive/coupled-latent-residual-experiments.md).
Both fresh same-graph references passed time-refinement checks; all four
matched5000-update runs finished with soft ICs. Terminal phi/c L2 original->latent:
corrosion50 s0.03484/0.04135->0.56839/0.54585; electropolish5 s
0.28604/0.29443->0.34398/0.37920. Defaults remain unchanged. Corrosion latent
concentration saturates (85.2% zero c_L derivatives); both electropolish arms
have extra interfaces. Inherited AC/CH scales were held fixed. Checkpoint probes
show phase preconditioning changes gradient balance and activates GradNorm
caps; neither scale retuning nor a causal KKS ablation was performed. All
processes exited. Detailed results, derivations, validation and direct paths
are in the completed task and replay artifacts.

[Corrosion reaction cancellation assessed](tasks/archive/corrosion-latent-residual-assessment.md).
Hard ICs excluded. Existing0-1 map is already(1+tanh(z))/2=sigmoid(2z).
Both double-well and concentration-driven phase reactions cancel its Jacobian;
the graph edge ratio is identical to melting. Derived bounded-reaction criteria
and preserved the original concentration residual. Actual1D/2D loss/operator
identity checks passed (float64~1.75e-15; FP32 convolution~9.51e-7), including
saturated-ratio verification with high precision. No corrosion training or
production configuration/code change; L2 benefits remain unmeasured.

[Residual-transformation generalization assessed](tasks/archive/icemelting-residual-generalization.md).
Scalar inverse-Jacobian residual scaling extends beyond geometry/dimension;
Allen-Cahn and thermal phase fields are natural validation targets. Glasner2001
provides direct nonlinear-preconditioning precedent. Conserved/multiphase models
need separate constraint/flux derivations. Chain-rule identities checked by AD
in1D/2D/3D (max discrepancy2.22e-16); no cross-model training was performed and
the validated melting implementation remains unchanged.

[Validated five-second default promoted](tasks/archive/icemelting-default-preset.md).
The canonical base preset is the selected5000-update SOAP configuration;
all training/model/operator/sampling parameters match the saved winner.
README/config and artifact provenance identify this default explicitly.
Configuration comparison, direct artifact paths and diff checks passed.

[Five-second SOAP versus Adam comparison completed](tasks/archive/icemelting-soap-adam-5s.md).
Current defaults: Tc5 s, lr_init0.001,5000 updates, SOAP, IC-only latent residual.
Matched5000-step full-grid terminal graph L2: SOAP6.11e-5, Adam7.40e-4;
last1000-update sampled medians differ by~5x. Adam's slightly lower analytic
error reflects cancellation against the approximate profile. Compiled step
times are close. SOAP's selected5000 checkpoint excludes later work before the
user-requested stop; Adam completed5000. No process remains running.
[Current melting strategy README completed](tasks/archive/icemelting-strategy-readme.md).
The example README foregrounds the IC-only latent default and documents loss,
sampling/optimizer/controls/L2/loading. Current defaults all use5 s; prior4 s
records remain explicitly historical. Documentation links/math/diff checks passed;
no numerical code, training or checkpoint change in this documentation task.
[Direct file-path loading without hashes completed](tasks/archive/icemelting-direct-path-loading.md).
User preference: explicit paths, no file hashing or hash-manifest dependencies.
Removed executable hash processing in the three melting runtime modules and two
replay scripts; preserved parameter/grid/shape/time checks. Two focused tests and
a config+checkpoint-only loading check passed. Old static hash records remain
historical data and are ignored by loaders. See docs/DEVELOPMENT.md for the rule.
[Melting run directory tags completed](tasks/archive/icemelting-run-tags.md):
best audited run is 20261024-ic-latent-coldstart-01-sota; former SOTA is now
legacy-raw-phase-soft-ic. All139 artifact files verified unchanged; current
paths, artifact index and replay scripts updated, original run identities retained.
[Default latent melting integration and cold starts completed](tasks/archive/icemelting-production-latent.md).
Production phase.formulation defaults to ic_latent; legacy/raw and hard-IC phase
controls remain selectable. Two fresh4 s runs finished2000 updates. Full-grid
terminal graph L2 is3.86e-4 with base batch/lr, and3.02e-5 with the bounded preset;
analytic L2 is1.40e-3 and1.03e-3 respectively. Curves show late oscillations.
Five numerical regression tests and independent checkpoint/IC/geometry checks
passed. These were historical4 s cold starts; the current5 s comparison is above.
[Original baseline metric audit completed](tasks/archive/icemelting-baseline-metric-audit.md):
historical SOTA full-grid 4 s L2=0.01412 vs analytic and 0.01466 vs the same-graph
numerical solution. Earlier phase-residual pilots also had hard IC and must not
be labelled the historical baseline. Logged pooled-time errors replayed.
[IC-only melting controls completed](tasks/archive/icemelting-weak-prior.md).
The user's strong-prior objection is accepted; the research runner now embeds
only the prescribed IC. Seven matched-budget pilots give 5 s transformed-residual
L2=8.7e-5–1.5e-4 versus 0.0038–0.0047 for phase-residual controls across three seeds.
Earlier motion-prior results remain conditional controls. The production
integration above keeps spatial discretisation and physical parameters unchanged.
