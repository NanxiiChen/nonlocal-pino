# Corrosion robust-objective optimization — 2026-10-26

## Goal, status and acceptance

Complete. User requested a local commit before continued optimization.
Snapshot e4a0764 records accumulated code/results; f65e6f8 normalizes archived
SVG whitespace. Both are local on dev; no merge/push. Subsequent optimization
changes remain uncommitted. No blocker or required remaining action.

Goal: improve cross-seed corrosion behavior without selecting a seed, changing
physics/reference, adding hard ICs, or exceeding5000 updates. Test a focused
bounded-weight gradient intervention, retain failures, and choose defaults from
final checkpoints/late windows with separate concentration-transport diagnostics.
The controlled study and default revision are complete. Accurate Dq transport
is a remaining research problem, not claimed solved by this task.

## Formulation and hypothesis

Corrosion retains KKS phi=(1+tanh(z))/2, the same50s matched-graph problem,
soft full-nodal IC/BC, raw-KKS CH edge evaluation, s_bc data coordinates,
AC/CH scales1e-7/1, L1 losses, SOAP/lr1e-3/5000 and fixed spatial pool.
Grid101x51, delta0.02 cubic graph, core256/pool100/Nt16; no stagger/time
window/late-rate/CH-RMS continuation. Reference path:
`/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz`.
Reference temporal-refinement discrepancy3.64e-9 was previously verified;
this study changes no graph, physical parameters, initial or boundary model.

The current bounded residual has positive multiplier
w=(1+eta)/(4phi(1-phi)+eta), eta0.1, cap11. Full differentiation of w|r|
includes both w*d|r| and |r|*dw. The new explicit `bounded_frozen` formulation
uses stop_gradient(w): forward residual/CH values are identical to bounded,
but only the physical residual receives the AC derivative. This is a
reweighted update, not the gradient of the original state-weighted scalar
objective. Both the AC optimizer gradient and its GradNorm probe change.
For nonzero r the AC direction is E[w sign(r) grad(r)]; at zero use the
implementation's L1 subgradient.

Holding |r| fixed isolates a weight-gradient term favoring phi toward0.5,
where w is smallest. The complete PDE/network gradient also differentiates r
and couples nodes; this local observation does not predict its total direction.

## Predeclared controlled experiments

First3 cold starts: original2026, frozen7, frozen2026. Before any new final
result, add frozen42 to verify the previously successful case. Queue it until
one worker exits, at most3 GPU workers. All4 finish5000 updates/return0.
Historical original42/7 and bounded42/7/2026 complete the3x3 comparison.
Use every final5000 model and six evaluations4000,4200,...,5000; no seed
reroll, intermediate-checkpoint selection or extra updates.

Direct same-seed initial parameter bytes are equal across objectives.
Resolved configs agree except formulation/run name and explicitly verified
historical absent disabled fields. Old42 predates LR-tail options and directly
calls make_optimizer; new tail_start0 delegates to that exact optimizer.
Old42/7 predate CH continuation and always use L1; new ch_rms_after0 preserves
it. Audit checks actual old sources before normalization; no runtime fallback
or file-hash gate is introduced. Spatial data/pool and each seed's temporal
stream stay fixed. New run directories:

- `/root/autodl-tmp/tf-logs/corrosion2d/20261026-objective-phase-seed2026`
- `/root/autodl-tmp/tf-logs/corrosion2d/20261026-objective-bounded_frozen-seed7`
- `/root/autodl-tmp/tf-logs/corrosion2d/20261026-objective-bounded_frozen-seed2026`
- `/root/autodl-tmp/tf-logs/corrosion2d/20261026-objective-bounded_frozen-seed42`

Canonical commands are in the completed launch/validation JSON below. Each
run keeps resolved config, source snapshot, initial/final model,1000-update
checkpoints, metrics/loss histories, predictions and stdout; direct path loading.
Reproduce with a fresh log.run_name to avoid overwriting an existing run.

## Full-grid CPU audit

All values below are independent CPU final5000 physical-field norms. Phi/c
use all5151 nodes; q/Dq use4941 free nodes. Late phi medians use the original
six logged full-grid evaluations, not a selected checkpoint. CPU/GPU differences
are below3e-6 absolute for both phi and c. CH is reconstructed with FP64
raw-KKS q and independent SciPy graph accumulation; AC uses training's native
FP32 formula. Preserve distinctions between different residual objectives.

| Objective | Seed | Terminal phi L2 | Terminal c L2 | Terminal q L2,free | Terminal Dq L2,free | Late phi median |
|---|---:|---:|---:|---:|---:|---:|
| original | 42 | 0.008835377 | 0.0104256 | 0.01372465 | 1.702557 | 0.007190959 |
| original | 7 | 0.004224766 | 0.005005779 | 0.009900142 | 1.729812 | 0.005065322 |
| original | 2026 | 0.01074417 | 0.0127265 | 0.01222127 | 1.757147 | 0.009514079 |
| bounded | 42 | 0.001433045 | 0.001718585 | 0.01783262 | 1.500409 | 0.01813206 |
| bounded | 7 | 0.422001 | 0.4461395 | 0.1939244 | 3.942864 | 0.4221578 |
| bounded | 2026 | 0.4070161 | 0.4310253 | 0.3106974 | 4.013898 | 0.4049465 |
| frozen | 42 | 0.01456938 | 0.01727216 | 0.02673568 | 3.00751 | 0.01155886 |
| frozen | 7 | 0.01090023 | 0.01293336 | 0.00955258 | 1.357191 | 0.01463238 |
| frozen | 2026 | 0.01062355 | 0.01261652 | 0.0107694 | 1.397169 | 0.0119065 |

## Conclusions and default selection

Original phase gives the smallest worst-seed final phi/c/q/Dq and late phi
median among these3 objectives. Worst final phi/c is~0.01074/0.01273 versus
bounded~0.4220/0.4461 and frozen~0.01457/0.01727. Worst late phi median is
0.009514 versus bounded0.422158 and frozen0.014632. Therefore the canonical
train_base.py now selects `phase`; all other strategy/physical settings remain
unchanged and seed stays runtime-only. Its representative untuned seed42 run
is the historical original42 path in the audit. The lower individual bounded42
checkpoint remains a separate historical SOTA observation; no paths/tags move.
Three observed seeds do not establish a universal robustness guarantee.

Frozen weighting rescues bounded7/2026 field errors from~0.4 to~0.01 without
changing fixed IC/BC/spatial samples. This supports a role for state-weight
differentiation/its downstream optimizer trajectory; fixed IC/BC data alone
cannot explain or necessitate failure. It does not establish a unique mechanism,
a strict local minimum or that refreshing sampling could not help. The direct
AC-gradient effect cannot be separated here from its downstream GradNorm changes.
Frozen remains an explicit research option, not the default or a uniform remedy.
It improves final q/Dq/CH RMS in7/2026 versus original, but worsens42 and some
late-window metrics. In particular frozen42 Dq~3.01 versus original42~1.70.

Original and frozen models have one liquid component at all21 reference times;
bounded7/2026 retain spurious late components. All physical concentrations remain
in0..1. Original Dq errors~1.70–1.76 and CH RMS~0.98–1.01 are still substantial.
Good total-c L2 does not certify chemical-potential gradients or transport.
No pre-scale removal or concentration-accuracy cure is claimed. Electropolish
and melting were not retrained/changed in this targeted optimization.

## Diff-level changes and verification

- residuals.py adds only frozen bounded AC weighting; original/latent/bounded
  remain compatible. Pilot/CLI validation and summaries recognize the new option.
- Focused test checks exact forward equality, independent fixed-weight gradient,
  changed differentiated gradient and a real compiled SOAP update; existing
  bounded/FP64 raw-edge checks still pass (3 numerical tests).
- All5 canonical-entry checks pass; one melting subprocess initially timed out
  at60s during3 simultaneous GPU compilations, then passes isolated in1.845s.
  No timeout threshold changed. After default revision, direct/module dispatch
  checks pass again and full strategy parity with saved original42 passes.
- Native source/graph/reference/masks/ordering validated, all9 final checkpoints
  equal their model_final bytes. All9 phi/c norms reproduce GPU logs; paired
  initial bytes/configs match, including checked disabled historical fields.
- plots use raw histories, common axes/scales and the same final checkpoint
  across all displayed times; visually inspected. No curve smoothing.
- README, artifact index and project dashboard distinguish current default and
  historical single-run best; index validates5 cases/302 directories. A remaining
  snapshot EOF blank in training_data.py is removed as whitespace-only cleanup.
- Changed source/script syntax, strict JSON, links and git diff --check pass.
  All training/audit processes exit, GPU0MiB/0%; no FEniCS work was concurrent.

## Artifacts and exact resume point

- [Four-run launch source](corrosion-robust-objective-launch.py),
  [first-three manifest](corrosion-robust-objective-launch.json).
- [Queued seed42 source](corrosion-robust-objective-validation.py),
  [validation manifest](corrosion-robust-objective-validation.json),
  [parameter/config preflight](corrosion-robust-objective-preflight.json).
- [Independent CPU audit source](corrosion-robust-objective-audit.py),
  [all metrics](corrosion-robust-objective-audit.json),
  [time/IC/BC comparison](corrosion-robust-objective-audit.png).
- [Plot source](corrosion-robust-objective-plot.py),
  [raw training curves](corrosion-robust-objective-plot-convergence.png),
  [seed2026 same-scale fields](corrosion-robust-objective-plot-fields.png).

No running job or required continuation remains. Further concentration-gradient
accuracy or separate sampling interventions require a new controlled study;
this task supplies the complete baseline and negative/mixed frozen-weight results.
Memory changed: docs/PROGRESS.md,docs/DECISIONS.md and this archived task.
