# Ordinary-policy FP64 full accuracy control — completed 2026-10-26

## Request, outcome and scope

User requested testing ordinary GPU policy + FP64. One seed42 cold start
finishes5000 updates /50 physical seconds through corrosion2d/train.py with
unchanged graph/reference/network/SOAP1e-3/data/scales. Existing three completed
controls are reused, not retrained. Fresh weights; existing compiler cache reused.
All training and audit workers exit0. No default change, new seed search,
FEniCS work, mathematical change or Git history operation.

| Policy / dtype | Final phi L2 | Final c L2 | Late phi median | Total seconds |
|---|---:|---:|---:|---:|
| Deterministic FP32 | 0.008835377 | 0.010425604 | 0.007190959 | 1179.813 |
| Ordinary FP32 | 0.020679804 | 0.024565322 | 0.012736816 | 78.026 |
| Deterministic FP64 | 0.002676716 | 0.003207012 | 0.006068563 | 1494.450 |
| Ordinary FP64 | 0.002498502 | 0.002944459 | 0.004684528 | 196.505 |

Final values come from independent CPU checkpoint reloads. Relative L2 uses
unweighted common101x51 graph-grid nodes at t*=1 (50s); late medians use six
snapshots4000,4200,...,5000. Each run has26 full L2 evaluations and identical
21 reference times. All initial parameter values, including promoted arrays,
are exactly equal across all four controls. This single ordinary FP64 run
improves final phi by71.7% and late phi median by34.9% versus default; observed
end-to-end time is6.00x lower. No cross-seed or repeated-ordinary-run guarantee.

## Limits and transport

Ordinary FP64 late phi max0.014885 and mean0.006661 exceed default0.008835/
0.006348. Late c mean/max also do not improve. Final free q/Dq0.007739/1.35699
improve on default0.013725/1.70256, but are worse than deterministic FP64
0.006861/1.22047. At4600 ordinary FP64 Dq error spikes18.6646; late Dq median
1.638 improves on default1.706, but mean4.411/max18.665 are much worse than
default1.536/1.915. Do not label improved final fields as uniformly more stable
or more accurate transport. All models retain one liquid component at21 times.

Ordinary FP64 best sampled final-time phi occurs at5000 (0.0024985); first
sampledphi<1% occurs at2200 versus4000 default,2800 ordinary FP32 and2400
deterministic FP64. The full field histories and per-time errors are archived.

Historical/new total times include actual JIT, evaluation and saves but have
different compiler-cache histories; new run training timer186.76s versus total
196.50s. They are observed costs, not a fair fresh-cache compile benchmark.
Prior independent steady-update timing35.13ms ordinary FP64 versus229.31ms
default FP32 separately supports a throughput gain. Reference generation excluded.

## Run and controls

New directory:
/root/autodl-tmp/tf-logs/corrosion2d/20261026-full-graph_ordinary-float64-seed42.
Load ckpt_005000.eqx or model_final.eqx directly; they are byte-identical.
Reference retained unchanged:
/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz.
The25 periodic checkpoints are saved every200; initial/final models, precision,
config, source, metrics/loss JSONL, TensorBoard and predictions are complete.

The [launch manifest](ordinary-fp64-accuracy-manifest.json) records exact CLI/
environment/policy. The canonical ordinary preset plus compute_dtype=float64
is sufficient; no new production configuration or implementation is required.
XLA_FLAGS is explicitly empty. Shared trainer, residuals, precision utility,
optimizer, data sampler and KKS constraint bytes exactly match saved ordinary
FP32 source. Configuration differences from current default are only dtype,
GPU policy and run name. Historical deterministic FP64 predates explicit policy/
local-dispatch fields; these absent fields and different save cadence are
reported as provenance differences, not silently injected into its config.

Network, PDE data and optimizer floating states are FP64, as independently
recorded by precision.json and CPU dtype audit. Initialization/data/graph/time
values are generated in FP32 and promoted exactly, matching the earlier control.
Preserve stored FP32 graph diagonal; no row-sum correction. Existing scalar
Optax learning-rate schedule remains FP32. Same seed42 is run provenance, not
an optimized configuration parameter. No evolved reference field trains the model.

## Independent audit and artifacts

[Launcher source](ordinary-fp64-accuracy-launch.py),
[CPU audit/plot source](ordinary-fp64-accuracy-audit.py),
[all metrics/config controls/histories](ordinary-fp64-accuracy-audit.json),
[field L2 curves](ordinary-fp64-accuracy-convergence.png),
[SVG curves](ordinary-fp64-accuracy-convergence.svg),
[transport curves](ordinary-fp64-accuracy-transport.png),
[terminal fields](ordinary-fp64-accuracy-terminal-fields.png).
Raw fields: reports/ordinary-fp64-accuracy-20261026/audited-fields.npz;
raw stdout/manifest/audit remain in the same report directory.

Independent CPU loads all four final/initial checkpoints, confirms exact initial
values, allowed config differences, reference specification, all dtype records,
finite21-time fields/time derivatives, predictions, full26-point histories and
5000 joint updates. Independently evaluates graph q/Dq/CH, pit/solid q errors,
liquid topology and all21-time L2 errors. Ordinary FP64 CPU versus saved GPU
field differences are at most4.11e-15; final L2 also reproduces. Frozen source
syntax, strict JSON, links, source/output inventories and git diff --check are
verified. No unchanged test suite rerun: production code is unchanged, and
actual full training plus independent audit exercise the requested combination.

## Changes and completion

This session adds research launch/audit scripts and results; updates example
README, artifacts/index.yaml and memory (PROGRESS, DECISIONS, this task).
Unrelated uncommitted work is preserved. All requested work completed; no
blocker or active worker remains. Defaults retained because this one run is
promising without uniform stability/transport/cross-seed validation. Any new
seed cohort, transport remedy or default promotion is separate work.

Final [verification](ordinary-fp64-accuracy-verification.json) passes:10 archived
artifacts, source/strict-JSON/links/output checks,5 cases/308 registered directories
and git diff --check. No active process remains.
