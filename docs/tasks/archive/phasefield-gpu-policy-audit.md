# Cross-case GPU compilation-policy audit — completed 2026-10-26

## Request and outcome

User asked whether electropolish, melting and other cases share the corrosion
compilation-induced speed issue. All five entries were inspected; four serial
GPU workers benchmark actual electropolish/melting FP32 default loss/step
functions. Each completes125 updates:50 warm-up and75 synchronized measurements.
All workers exit0. No long accuracy cohort, production numerical code/config
change, physical/reference modification, FEniCS work or Git history operation.

| Case | Deterministic ms/update | Ordinary ms/update | Slowdown | Actual default entry |
|---|---:|---:|---:|---|
| Electropolish2d | 197.727 | 10.977 | 18.01x | Shared KKS adapter forces deterministic |
| Icemelting3d | 468.257 | 26.616 | 17.59x | Inherits environment; no forced flag |

Inspected parent XLA_FLAGS, JAX_ENABLE_X64 and JAX_DEFAULT_DTYPE_BITS were unset.
Thus ordinary FP32 is already the normal melting behavior in this environment;
the measured large melting penalty applies when determinism is externally forced.
Do not infer which environment all older historical melting runs inherited:
those summaries lack sufficient external-flag provenance. KKS child-process
flag setup does not by itself change the parent shell environment.

## Actual workloads and protocol

Electropolish: canonical train_base original residual, physical_difference CH,
physical random IC/BC,101x51 grid,Nt16,256 core/100 pool,MLP6x64 gelu/tensor_split,
SOAP1e-3. Free core counts239–253 (3824–4048 PDE pairs). Padded support2559,
edges3032; IC2000 and128+128 BC points. Production shared make_step, losses,
optimizer and time sampler are reused directly. Seed42 is run provenance.

Melting: actual128^3 grid (2097152 nodes),12484608 directed global edges,
2000 core/200 pool,Nt64,IC-only latent residual,tensor-split MLP4x64 tanh and
production SOAP1e-3 step. Padded support13904 and11929 edges;128000 active PDE
pairs. Hard IC has no IC/GradNorm loss; returned PDE/IC weights remain1/0.
No reduced grid/pool/network substitutes the actual default workload.

Within each case both policies have exactly identical initial arrays, all pool
points/indices/weights/diagonals/masks, all125 times and all IC/BC targets.
Config changes only gpu_policy for electropolish; melting configs are identical.
All parameter/optimizer floating arrays remain FP32. Production cadence is
included:15 SOAP refreshes in75 measurements and3 GradNorm refreshes for
KKS; melting has one PDE term and no adaptive GradNorm work. Sampling is timed.
Every result is device-synchronized; logging, evaluation, saves and first compile
are excluded. Compilation/preparation costs are recorded separately. For example
training compile times ordinary/deterministic: electropolish123.61/22.28s,
melting37.86/9.90s; do not conflate these with steady throughput.

## Root cause and independent checks

| Frozen path | Electropolish ordinary/deterministic ms | Melting ordinary/deterministic ms |
|---|---:|---:|
| Graph forward | 0.145 /41.478 | 0.127 /148.771 |
| Graph scalar-loss input gradient | 0.121 /83.997 | 0.280 /451.897 |
| Initial-network PDE loss only | 1.342 /84.876 | 7.063 /150.347 |

Optimized HLO ordinary retains scatter. Deterministic forward lowers one loop
of3032 edges for electropolish and11929 for melting. Electropolish input-gradient
HLO has two3032-trip loops; melting has three11929-trip loops. This shows
reverse accumulation matters too. Kernel speed ratios are not whole-step gains;
measurements above establish whole-step effects separately.

Electropolish independent SciPy matvec/transpose gradient relative errors are
4.12e-8–8.27e-8 across both policies. Melting stable latent edge/transpose
NumPy formulas give1.70e-7–2.57e-7. For edge ratio R_ij=(tanh u_i-tanh u_j)/sech²u_i,
independently derived derivatives are dR/du_i=1+2*tanh(u_i)*R and
 dR/du_j=-exp(2*(logcosh(u_i)-logcosh(u_j))). The transpose reference uses these
edge derivatives with NumPy indexed accumulation, not JAX AD. A separate CPU
FP64 toy verifies stable forward and transpose derivatives at2.83e-17/5.34e-16.
Initial full loss vectors differ only7.54e-8/6.05e-8 relative for electropolish/
melting. Small arithmetic differences remain;125-step throughput does not prove
same final L2, deterministic superiority or ordinary convergence robustness.

## Other-case static context

Corrosion2d uses the same enforced KKS default and its prior229.31/12.49ms
measurements are reused, not repeated. Elasticity2d inherits environment and
uses ElasticD with multiple gradient/residual segment sums: potential forced
flag cost is identified, but no timing multiplier or accuracy claim is assigned.
Corrosion1d also inherits flags; both explicit activation/diffusion presets use
SparseD. Its inherited default references missing configs/train_pinn.py. We
record and document this separately, without silently selecting a physical
regime or repairing numerical configuration. No full1D timing was possible from
that default and none is claimed for either explicit regime.

## Diff-level changes and evidence

Only research benchmark/summarizer, result artifacts, README guidance, artifact
index and memory changed. All production code/defaults/checkpoints/reference
paths remain intact. Explicit ordinary KKS option is existing train.gpu_policy;
melting uses external XLA_FLAGS. All paths remain direct, without hashes.
The two measured cases have precision/performance records in artifacts/index;
all five cases have scope-qualified static policy records there.

[Benchmark source](phasefield-gpu-policy-benchmark.py),
[summarizer/source](phasefield-gpu-policy-summarize.py),
[manifest](phasefield-gpu-policy-manifest.json),
[CPU derivative preflight](phasefield-gpu-policy-preflight.json),
[summary/all timing samples/configs/HLO controls](phasefield-gpu-policy-summary.json),
[plot](phasefield-gpu-policy-comparison.png),
[SVG](phasefield-gpu-policy-comparison.svg).
Four separate worker JSONs and eight optimized forward/input-gradient HLO files
share the phasefield-gpu-policy prefix in this archive. Raw paired NPZ inputs,
initial models, logs and26 production source snapshots are retained under
/root/nonlocal-pinn/reports/phasefield-gpu-policy-20261026/.
The research source imports current production APIs; use its archived location
with --case and --policy for worker reproduction and a fresh report directory
for a new cohort. Manifest retains the original script path at execution time.

## Verification and completion

Four complete125-update workers with finite loss/gradient checks; exact paired
full input/pool/initial comparisons; config/dtype/cadence checks; eight numerical
kernel comparisons; optimized-HLO loop count checks; independent CPU derivative
preflight; three25-update block means; synchronized timing and plot inspection
pass. Production code unchanged, so no unchanged full test suite rerun.
Source/JSON/link/output/inventory/diff verification is recorded alongside this
report. No active worker, blocker or unfinished requested measurement remains.
Memory updated: PROGRESS.md, DECISIONS.md and this archived task. Future
aggregation optimization or ordinary-policy long-run accuracy controls require
separate work; no default was promoted from this speed evidence.

Final [verification](phasefield-gpu-policy-verification.json) passes:21 archived
artifacts,26 source snapshots, all strict JSON/syntax/links/HLO/numerical/input/
output checks,5 cases/308 registered directories and git diff --check. No worker
remains active.
