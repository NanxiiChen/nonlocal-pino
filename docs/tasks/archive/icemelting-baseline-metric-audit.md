# Original melting baseline metric audit

Location update: the historical `SOTA` label below now refers to the directory
`/root/autodl-tmp/tf-logs/icemelting3d/legacy-raw-phase-soft-ic`; original run
identity and all metric values are retained. See the [rename record](icemelting-run-tags.md).

Loading update: the replay script now records checkpoint/reference paths only;
no file hashes are computed or required. Existing JSON checksum fields reflect
the original audit. See the [direct-path update](icemelting-direct-path-loading.md).

Goal: distinguish historical raw-phase/soft-IC checkpoints from the new
hard-IC original-residual controls, and verify their terminal spatial relative L2.

Acceptance criteria: replay historical logged metrics; report physical 4 s
full-grid analytic and, for nonlocal models, matching graph-reference errors;
separate physical 4 s from 5 s and avoid causal claims across unequal budgets.

Context: the earlier 0.0038–0.0047 numbers are hard-IC phase-residual pilots at
5 s, not the original production baseline. Historical production has a soft IC
loss and no hard IC. Its logged metric pools five times and 8000 spatial points.

Status: completed. No training or production code change. Blocker: none.

## Results and limits

All entries below use physical 4 s and all 128^3 grid nodes. New controls use
seed42; original SOTA is ckpt_017600. The second reference is independent direct
integration of the identical sparse graph, not a continuous-PDE exact solution.

| Model | Relative L2 vs analytic | Relative L2 vs same-graph numerical solution |
| --- | ---: | ---: |
| Historical original phase residual, soft IC | 0.01411873992 | 0.01465808296 |
| IC-only hard constraint, phase residual | 0.006160986112 | 0.005160379245 |
| IC-only hard constraint, transformed residual | 0.001077906186 | 0.00007258863847 |

Historical baseline2 ckpt_019800 gives analytic L2=0.01135536302 and graph
L2=0.01154712906. Local AD ckpt_019800 gives analytic L2=0.01431980504; the
nonlocal graph is deliberately not used as its consistent reference.

Historical logged metrics replay within 2e-9 absolute (all three checkpoints).
SOTA logs 0.009576017037 for a pooled norm across five times and 8000 sampled
nodes. Its terminal analytic error on those same nodes is 0.01379720122, versus
0.01411873992 on the full grid. Neither is the new hard-IC control's error.

The analytic profile and graph solution differ: the transformed model's
analytic error is ~0.108%, while its same-graph error is ~0.00726%. These
references must be labelled consistently. For SOTA the same-graph L2 of the
solid fraction (phi+1)/2 is ~6.971%, rather than the global phase L2 of ~1.466%;
this is a different quantity, included only to diagnose metric interpretation.

Historical models were trained on 0–4 s with different budgets/sampling; the
new controls were trained on 0–5 s. This audit aligns evaluation time, spatial
grid and reference, not training budget. No historical 5 s accuracy is claimed.
The paired hard-IC pilots remain the controlled evidence for the objective
change. Do not attribute the entire historical-to-new gap to one intervention.

## Reproduction and validation

Evidence: [JSON](icemelting-baseline-metric-audit.json) records checkpoint and
reference SHA256, physical time mapping, activation reconstruction and metrics.
[Audit script](icemelting-baseline-metric-audit.py) performs read-only inference;
run from the repository root with its installed package and examples importable:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python docs/tasks/archive/icemelting-baseline-metric-audit.py`.
Historical gelu-labelled baseline2/local_ad are explicitly reconstructed using
their replay-validated tanh implementation, as documented in the earlier audit.

The reference stores physical 0–5 s as normalized 0–1. Snapshot16 is physical4 s;
historical networks use t*=1 and new networks t*=0.8. Physical parameters,
length scale, centre, grid, and (for nonlocal models) graph settings are asserted
equal. Inference uses highest matmul precision, fixed 32768-node batches, and
float64 host norm accumulation. All historical replay assertions passed.

Diff-level changes: new archived audit record/JSON/script; historical baseline
clarification in the weak-prior report; updated progress and decisions. Production
code and checkpoints untouched. Verification: executed five checkpoint evaluations
and three logged-metric replays; git diff --check passed. No retraining needed.
Next action: any future matched-budget full baseline ablation is separate work.
