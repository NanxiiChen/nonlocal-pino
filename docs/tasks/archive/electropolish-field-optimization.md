# Electropolish field optimization — completed2026-10-25

## Status and outcome

Completed: nine fresh5000-update training controls, eleven-model independent
CPU field/topology/transport audits, three focused sampler tests and two-seed
representative physical-field figures. All processes exit0; GPU compute list
is empty. No blocker, no remaining required action. Production defaults,
physical PDEs/KKS, AC/CH scales and reference remain unchanged.

The useful optional candidate is **original phase residual + full-grid analytic
IC samples + graph-node BC samples + normalized liquid-concentration BCs**,
with full-horizon time sampling. It lowers seed42 terminal phi/c L2 from the
contemporary random-data baseline0.290928/0.299013 to0.002106/0.002518 and
eliminates extra interfaces. Seed7 candidate phi/c is0.002389/0.002869, also a
single interface, but its already accurate random baseline0.001138/0.001358
is better at the final endpoint. q improves in both seeds; seed7 Dq/c_t/CH
and pooled physical-field errors regress. There is no all-metric or every-seed
winner, so do not promote the candidate as a universally better default.

Full transformed phase residual remains worse than original. Temporal-window
sampling improves its phi L2 0.519398->0.046769 and eliminates extra bands,
but does not reach the original residual's accuracy. Original-residual window
controls do not beat their no-window endpoint/transport results. Keep these
negative outcomes and the successful seed7 original baseline visible.

## Goal, controlled model and prescribed data

Optimize the poor [earlier electropolish pilots](coupled-latent-residual-experiments.md)
after their [physical-field gallery](electropolish-physical-field-gallery.md)
showed extra phase bands. The target is physical phi/c error and interface
structure, with q/Dq/c_t diagnostics so total c does not hide driving defects.
Initial constraints remain soft; no reference interface-motion profile or
positive-time interior reference labels enter the loss.

All current controls use5 physical seconds (nondimensional t=0..1),101x51
cubic pair graph,delta0.02,Kcore256,Kpool100,Nt16,t_min0.001,SOAP,5000 updates,
lr_init1e-3,0.95-per1000 exponential rate,GradNorm every25 with alpha1/cap1000,
AC1/CH1,original KKS map and production rough surface/materials. Stagger,
causal weights and interface weights are off. At most3 GPU workers overlap,
with explicit --xla_gpu_deterministic_ops=true and no preallocation. Sources
are frozen and byte-identical for all nine new workers; within-seed initial
model bytes match exactly. Those checks do not claim an independently repeated
bitwise optimizer trajectory; optimizer states are not serialized.

The user's question about “node data” is answered precisely by these choices:

| Sampling | IC at t=0 | Spatial Dirichlet BC samples |
| --- | --- | --- |
| original random | 2000 random points, including near-interface samples | 128 random top +128 random bottom |
| graph/grid | all5151 grid points, analytic initial formula | 101 top +101 bottom graph nodes |

IC values are calculated from the prescribed rough-surface/SDF formula, with
the same declared Dirichlet overrides at t=0 as the reference. BCs remain top
phi=c=0 and bottom phi=c=1; side treatment remains in-domain graph interactions.
Only the IC/BC sampling distribution/counts change, not the initial formula,
physical BCs or spatial PDE core sampling. More IC points increase work per
update: equal update budgets are not equal-FLOP or speed comparisons. Concurrent
wall times are not optimizer benchmarks. IC-only versus BC-only sampling is
not separately isolated in this study.

The reference is the unchanged independent physical-field SciPy BDF graph solve:
/root/autodl-tmp/data/electropolish2d/results_matched_graph_20261024.npz.
Preflight validates parameters, rough geometry, graph edges/weights, ordering,
IC/BC and physical times. Earlier temporal refinement discrepancy1.52394e-8
is retained. No new FEniCS/reference solve is necessary for unchanged physics.
This measures optimization of the fixed101x51 finite-horizon graph model,
not spatial convergence to a continuum/local-FEM solution or exact pointwise
Neumann enforcement at side walls.

## Loss coordinates and time-window implementation

For current c_Se=1 and dc=1-c_Le, the unchanged KKS map is

$$
\phi=(1+\tanh z)/2,\qquad s=(1+\tanh v)/2,\qquad
c=c_{Le}s+\Delta c\,h(\phi),\qquad h(\phi)=3\phi^2-2\phi^3.
$$

Physical-data BC losses fit (phi,c). The existing s_bc option instead fits
(phi,s) to (0,0) at top and (1,1) at bottom. Zero loss gives the same physical
BC values. Away from zero this changes coordinates, coupling and concentration
scale; it is not merely a constant multiplier on total-c error. Initial data
remain physical (phi,c), so no normalized-IC or hard-IC change is mixed in.

Original AC and CH residuals are retained in the selected candidate. The
full-latent control uses r_phi/[4phi(1-phi)], evaluated with the stable existing
edge-ratio formula, and keeps CH unchanged. Every new run uses physical_difference
CH evaluation and L1 reduction; no raw-edge transport or RMS switch is tested
in this electropolish cohort. No pre_scale removal/retuning occurs: both scales
already equal1.

The only numerical runtime addition is a research-only sampling curriculum:

$$
t\sim U(t_{min},T_k),\qquad
T_k=a+(1-a)\min(k/R,1),\quad R=3000,\ a=0.05.
$$

It starts with a0.25 s upper bound, reaches5 s at index3000 and retains2000
full-horizon updates, without optimizer resets. Disabled R=0 returns exactly
the original uniform time samples. Fixed batch shapes/dtypes are preserved.
Validation rejects invalid integer ranges/early bounds and a non-unit full
endpoint. Early-window sampling intentionally changes the sampling objective;
there is no importance correction claiming a full-horizon loss before R.
The research runner uses loss.t_ramp_* for this explicit window; production
train.py's similarly named time-weight mechanism is separate and unchanged.
CLI/summary expose the research choice. README distinguishes these semantics.

## Complete endpoint comparison

All endpoints are final5000-update physical fields at exactly5 s. phi/c norms
use all5151 nodes. q,Dq,c_t norms below use4949 free PDE nodes, excluding top
and bottom Dirichlet nodes. Live q logs include boundary nodes and have a
different denominator; do not mix them with these independent free-node values.
The reference has exactly one vertical phi0.5 crossing in every x column.

| Control | terminal phi L2 | terminal c L2 | q L2 free | Dq L2 free | c_t L2 free | terminal crossings |
| --- | --- | --- | --- | --- | --- | --- |
| random / physical / original / seed42 | 0.29092777 | 0.2990127 | 0.15809172 | 0.95597171 | 0.97991845 | 3 in 101 columns |
| random / physical / original / seed7 | 0.0011378188 | 0.001357816 | 0.011507961 | 0.037761345 | 0.029407257 | 1 in 101 columns |
| grid / physical / original / seed42 | 0.0023544182 | 0.0028387627 | 0.013282672 | 0.045045975 | 0.028013929 | 1 in 101 columns |
| grid / s_bc / original / seed42 (selected) | 0.0021060144 | 0.0025178224 | 0.0031002799 | 0.044336749 | 0.032875033 | 1 in 101 columns |
| grid / s_bc / original / seed7 (validation) | 0.0023894405 | 0.0028692668 | 0.004998532 | 0.058375417 | 0.062247653 | 1 in 101 columns |
| grid / s_bc / transformed / seed42 | 0.51939806 | 0.53694831 | 0.13811628 | 1.0562508 | 1.0498642 | 3 in 101 columns |
| grid / physical / original / ramp / seed42 | 0.0065163665 | 0.0078513077 | 0.016463832 | 0.082776236 | 0.065238595 | 1 in 101 columns |
| grid / s_bc / original / ramp / seed42 | 0.0023908466 | 0.0028729241 | 0.0059136204 | 0.088987588 | 0.030624477 | 1 in 101 columns |
| grid / s_bc / transformed / ramp / seed42 | 0.046768538 | 0.055000473 | 0.017968355 | 0.45713604 | 0.45364124 | 1 in 101 columns |
| prior pilot / original / seed42 | 0.28604333 | 0.29442892 | 0.18538097 | 1.1592886 | 1.1899767 | 3 in 101 columns |
| prior pilot / transformed / seed42 | 0.34397928 | 0.37920391 | 0.18338368 | 1.2154681 | 1.2027303 | 3 in 101 columns |

The current random seed42 result reproduces the old pilot's extra-band failure
qualitatively, but not its exact values/weights (0.290928 vs0.286043 phi). The
causal data-strategy comparison uses the contemporary source/runtime-matched
random controls, not the historical source/runtime alone.

Original/grid/physical seed42 already gives phi0.002354 and lower pooled phi
and CH RMS than original/grid/s_bc. Adding s_bc improves endpoint phi/c, q and
BC accuracy for seed42, but is not best on every metric. The combined data
strategy is independently checked in seed7; a full sampling-by-normalization
factorial is not repeated in that seed, so individual causes are not uniquely
identified there. Two seeds do not establish a statistical failure rate.

In seed7 the selected candidate worsens terminal phi/c by about2.1x, Dq by
about1.55x and c_t by about2.12x relative to the successful random baseline;
q free improves0.011508->0.004999. Candidate phi remains0.24%, but this is an
explicit tradeoff, not a universal gain. The last1000-update physical-field
median is somewhat better despite the worse final endpoint:

| Current control | pooled phi L2 | pooled c L2 | late phi median | late c median |
| --- | --- | --- | --- | --- |
| random / physical / original / seed42 | 0.29700321 | 0.30722905 | 0.29196189 | 0.30012864 |
| random / physical / original / seed7 | 0.0021364124 | 0.0026017001 | 0.0033082243 | 0.0039649498 |
| grid / physical / original / seed42 | 0.0022020444 | 0.0026600443 | 0.0050413657 | 0.0060524789 |
| grid / s_bc / original / seed42 (selected) | 0.002893279 | 0.0034826948 | 0.002820676 | 0.0034069018 |
| grid / s_bc / original / seed7 (validation) | 0.0031341879 | 0.0037687299 | 0.0026965407 | 0.0032443626 |

Late medians use six saved evaluations4000/4200/.../5000, not every optimizer
step. The s_bc window has a slightly better seed42 late median than s_bc
no-window, but worse endpoint q/Dq/c and phi; do not hide this mixed result.

All selected candidate seed42/7 fields have one crossing in every column at
all21 saved physical times. The random seed7 baseline also does. Current
random seed42 and prior pilots retain extra interfaces. Full-latent window
has one crossing at all21 times while full-latent no-window has extra bands.
This is a positive topology effect of the tested window for the transformed
objective, without an original-residual accuracy win.

## Independent transport/gradient diagnosis

CPU checkpoint replay supplies raw v. Evaluate q=c_Le*(tanh(v)-1)/2 with
float64 tanh and the saved graph as a SciPy CSR matrix, independently from
training's float32 physical-difference graph evaluation. The physical c_t
comes from AD on the reloaded model. Reference c_t=-N_CH*Dq on free nodes;
CH RMS and relative c_t error are both retained. Reference terminal c_t RMS
is0.457789. This separates total-c fit, driving concentration, spatial
transport, time derivative and actual PDE imbalance.

Seed42 candidate q/Dq/c_t free errors0.003100/0.044337/0.032875 improve current
random0.158092/0.955972/0.979918. Seed7 has the explicit q-versus-transport
regression above. All eleven terminal c_L branches have zero fraction of
exactly zero float32 c_L derivatives: do not reuse corrosion's concentration-
saturation explanation for this experiment.

Frozen **old** original checkpoint probes, with the same matched-node batch,
show physical BC gradient L1 norms0.003708/0.005869 and both GradNorm proposals
at cap1000. s_bc changes them to2.9150/4.6140 and proposals101.18/63.92; physical
IC gradient0.913707 is unchanged. At the same checkpoint changing phase to
full latent raises AC loss0.197851->16.444683 and gradient L1
980.889->9202.498. These measurements support boundary-coordinate and phase-
weighting sensitivity, not a unique proof of why every run succeeds/fails.
The controlled training outcomes and adverse seed7 results remain essential.

Fixed2000-update midpoint CPU checks already found both original/grid arms
single-interface at every saved time, while s_bc/full-latent had extra bands.
They are interim diagnostics, not substituted final model selections.

## Selection, historical context and direct paths

Six initial seed42 controls were predefined: physical/original,s_bc/original,
s_bc/transformed crossed with window off/on. Before validation, the selection
rule was fixed to minimum terminal phi+c L2 among final5000 checkpoints with
one crossing in every column at all21 times. Five qualify; s_bc/original/no-window
wins. Only then launch current random baselines42/7 and that candidate seed7.
No earlier checkpoint or extra training updates are substituted for5000.

Candidate seed42 run:
/root/autodl-tmp/tf-logs/electropolish2d/20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed42

Candidate seed7 run:
/root/autodl-tmp/tf-logs/electropolish2d/20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed7

Each contains config.json,model_initial.eqx,ckpt_001000...005000.eqx,
model_final.eqx,prediction.npz,summary.json,metrics/loss histories,train.log,
and six source files with a direct files.json listing. The final model bytes
match ckpt005000. Every other new run's exact path/command/choice is recorded
in the control/validation JSON and artifact index, without file hashes.

Read-only historical SOTA context: its ckpt005000/005800 evaluated on this
reference gives phi/c0.026079/0.031064 and0.019958/0.023878, one crossing in all
columns. Those models trained301x151, full spatial batches,Nt50,lr0.005,
stagger1000 and a6000-update config, with different production checkpoint
numbering. They are cross-discretization context, not equal-budget controls
or evidence that one historical setting caused the gain. All old artifacts
and root paths remain unchanged; no checkpoint import/resume or Git history
operation occurs.

## Replay and artifacts

From repository root, use .venv only; do not run FEniCS/MPI concurrently.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  docs/tasks/archive/electropolish-optimization-audit.py
.venv/bin/python docs/tasks/archive/electropolish-optimization-gallery.py
JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m unittest discover \
  -s tests -p test_residual_time_window.py -v
```

The [README](../../../examples/electropolish2d/README.md#research-data-and-temporal-window-controls)
gives the candidate data flags and optional window flags. Existing run names
fail loudly rather than overwrite. A fresh replay uses the declared research
ConfigDict, the same flags, and a new explicit config.log.run_name before
train(config). No positive-time reference labels enter training.

- [Final audit JSON](electropolish-optimization-audit.json),
  [audit script](electropolish-optimization-audit.py),
  [training curves](electropolish-optimization-audit.png),
  [all terminal fields](electropolish-optimization-audit-fields.png).
- [Two-seed phase gallery](electropolish-optimization-gallery-phi.png),
  [concentration gallery](electropolish-optimization-gallery-c.png),
  [gallery metrics](electropolish-optimization-gallery.json),
  [gallery script](electropolish-optimization-gallery.py). Both contemporary
  random baselines are shown, including the successful seed7 counterexample.
- [Six controls](electropolish-optimization-controls.json),
  [launcher](electropolish-optimization-controls.py),
  [selection](electropolish-optimization-selection.json),
  [validation](electropolish-optimization-validation.json),
  [validation launcher](electropolish-optimization-validation.py).
- [Frozen probe](electropolish-optimization-probe.json),
  [probe script](electropolish-optimization-probe.py),
  [midpoint](electropolish-optimization-midpoint.json),
  [midpoint script](electropolish-optimization-midpoint.py),
  [historical context](electropolish-optimization-historical.json),
  [historical script](electropolish-optimization-historical.py).
- The audit-partial JSON/plots preserve the completed initial screen before
  second-seed validation; they are explicitly superseded by the final audit.

## Verification and task-end

Three focused sampler tests pass: disabled samples exact to legacy, enabled
windows/shapes/dtypes/full horizon correct, invalid settings rejected. All
nine GPU controls finish finite5000 updates; eleven CPU model reloads differ
from stored fields by at most2.802e-6 (acceptance5e-6). All terminal/pooled/
per-time physical L2 values reproduce saved summaries at1e-12 relative tolerance.
Initial bytes/source snapshots/config allowed differences, graph/time/BC/IC
counts, all5000 joint activations and final checkpoint equality pass. All
final figures are visually inspected. Strict finite JSON, local report links,
image integrity, source syntax and git diff --check are verified at closeout.

Diff-level changes: add validated research time-window functions/CLI/summary
in corrosion2d/residual_experiment.py and three focused tests; document
research semantics/data choices in electropolish README; add scripts,metrics,
plots and direct paths; update artifact index,PROGRESS,DECISIONS and this
archived task. No production/default,PDE,reference,optimizer state/history or
external artifact migration change. No debug code, commits,merges or pushes.
The task is complete. Further research may isolate IC-only/BC-only sampling,
repeat more seeds or tune phase preconditioning; none is required/pending for
this bounded study, and no universal improvement is claimed.

Final closeout counts:22 supporting artifacts plus this archived report;
10 finite JSON/index files,9 syntax/whitespace-checked Python files,6 verified
PNG files,22 report links,9 direct run records and11 CPU audits. Changed memory
files: docs/PROGRESS.md,docs/DECISIONS.md and this task record.
