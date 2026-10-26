# Full nonlocal/local-AD cold starts — completed 2026-10-26

## Goal and outcome

User requested complete L2 training after throughput-only comparisons. All four
predeclared FP32 seed42 cold starts finish5000 updates /50 physical seconds
through examples/corrosion2d/train.py; all processes exit0. No checkpoint warm
start, default promotion, Git history operation or FEniCS/GPU concurrency.

| Method | Terminal phi L2 | Terminal c L2 | Late phi median | Total seconds |
|---|---:|---:|---:|---:|
| Graph deterministic | 0.008835377 | 0.010425604 | 0.007190959 | 1179.813 |
| Graph ordinary | 0.020679804 | 0.024565322 | 0.012736816 | 78.026 |
| Local tensor/JVP | 0.346112112 | 0.373221127 | 0.360578857 | 224.875 |
| Local joint/grad | 0.411813163 | 0.425738552 | 0.546657240 | 223.261 |

Final numbers are independent CPU checkpoint reloads. Relative L2 is the
unweighted nodal Euclidean norm on common101x51 samples at t*=1 (50s), scored
against each method's own reference. Late medians use six saved evaluations
4000,4200,...,5000. Full21-time/pooled/q/Dq/CH metrics and26-point histories are
in [audit](local-ad-full-training-audit.json). Defaults remain graph/deterministic/
FP32/original phase, raw-edge CH, s_bc/nodal data, SOAP1e-3, unchanged scales.

## Controls and interpretation

All initial arrays match exactly; configuration differences are limited to run
name, GPU policy, spatial operator and local sixth GradNorm bias. Network,
physical parameters, common nodal IC/Dirichlet data, SOAP/learning rate/scales,
16 BC times and3760–4032 active PDE pairs are matched. Tensor local uses common
free graph core coordinates and shared-time JVP; joint local draws iid space/time
pairs and uses scalar time grad. Graph pool only supplies common coordinates/
counts for local training; local PDE uses spatial AD Hessians, not graph actions.

Local additionally imposes phi_n=q_n=0 on90 bottom nodes outside mouth/solid
corners, traceable to original local FEM weak form solver.py:190. Its normalized
phi/s derivative MSE is a sixth GradNorm group. The earlier throughput benchmark
had five losses and omitted this local boundary closure; its timings are not
those of the completed boundary-value controls. Ordinary graph changes only
compilation policy and run name relative to baseline.

Local tensor over-expands the pit; joint under-expands it. Terminal centerline
phi=.5 heights: graph/local references0.30436/0.30740, default/ordinary
0.30251/0.30044, tensor/joint0.37694/0.20331. All fields retain one liquid component.
This supports failure to reproduce evolution rather than a topology-only
explanation. It does not identify a unique optimization cause. Inherited scales/
SOAP are not separately tuned local SOTA. One seed and different boundary
objectives cannot establish intrinsic method superiority or cross-seed stability.

Ordinary graph is faster but final and late phi/c worsen; terminal free q/Dq
improve0.01372/1.703 ->0.00934/1.238. Preserve separate field/transport criteria.
Times include observed cache-dependent JIT, evaluation and saves, reuse previous
compiler caches, and exclude independent reference generation. Fresh weights do
not mean fresh compiler cache; total times are not a clean compile benchmark.

## Independent reference and limits

Graph reference retained unchanged at
/root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz.
Local reference is independent physical-field SciPy BDF with analytic Jacobian,
centered FD negative Laplacian, mirror natural BC, same pit/solid Dirichlet and
analytic IC/Dirichlet overrides. Four solves:101,201,201 timefine,401. Raw levels:
/root/nonlocal-pinn/reports/local-ad-reference-20261026/.
Final401 solution is sampled to101x51 at21 times and written to
/root/autodl-tmp/data/corrosion2d/results_local_ad_20261026.npz plus adjacent JSON.

201-grid time refinement maxphi/c2.24e-9/2.74e-9;201->401 terminalphi/c
1.417%/1.681%,101->2012.886%/3.423%. Spatial-below1e-3 flag is explicitly false.
Do not claim sub-percent continuum accuracy or extrapolate silently. Measured
reference discrepancy is not a rigorous error bound. Graph/local final reference
phi/c differences1.277%/1.513% combine operator, boundary and discretization
choices, not isolated finite-delta bias. Against common local401 reference,
default/ordinary graph phi errors2.106%/3.276%, versus34.61%/41.18% local.
Neither reference mismatch nor refinement plausibly explains the large local
training error; finer reference is needed before fine accuracy rankings.
See [full reference metadata](local-ad-full-training-reference.json).

## Diff-level changes and evidence

Added local_ad.py, local_training.py, local_reference.py and three explicit
research presets. Canonical dispatch selects local controls; make_step supports
explicit loss count while preserving graph five-loss behavior. Added explicit
gpu_policy/spatial_operator/local_reference_path configuration; default adapter
retains deterministic policy and ordinary rejects conflicting environment flags.
Updated example README, DEVELOPMENT and artifact index; all old paths retained,
new artifacts loaded directly, no hashes or sibling dependencies. Existing
uncommitted work preserved; no Git history changes.

[Launcher](local-ad-full-training-launch.py),
[manifest/direct run paths](local-ad-full-training-manifest.json),
[CPU audit/plot source](local-ad-full-training-audit.py),
[preflight](local-ad-full-training-preflight.json),
[step/time curves](local-ad-full-training-convergence.png),
[SVG curves](local-ad-full-training-convergence.svg),
[terminal phi fields](local-ad-full-training-terminal-phi.png).
Raw NPZ fields: /root/nonlocal-pinn/reports/local-ad-full-training-20261026/audited-fields.npz.
Every run retains config, initial/final models,25 periodic checkpoints, source,
precision audits and metrics/loss JSONL in manifest-listed tf-logs directories.

## Verification and completion

Twelve focused checks pass: seven entry/precision, four local math/shared
optimizer, one ordinary-policy environment. Manufactured local Laplacian/chain
rule/no-flux and compiled six-loss optimizer are checked. Both canonical CPU
local two-update runs pass with actual reference and all output types. Default
first1000 steps replay historical six L2 evaluations exactly; final independent
CPU default fields/metrics also match historical control. All four5000-step
runs pass finite checks and complete26 evaluations; independent CPU audit verifies
initial/config controls, all predictions, final==ckpt5000 bytes, per-time fields,
transport, topology and complete histories. Local Dq uses independent401 FD q
Laplacian as target and AD network Laplacian. FD target uncertainty also applies
to these derivative comparisons; no converged transport ranking is claimed.

An initial3e-6 pointwise CPU/GPU prediction tolerance was too tight for local
tensor:7 phi nodes reached3.64e-6, c reached5.07e-6. Frozen FP64 inference of the
same weights agrees within4.21e-6 with saved FP32 GPU fields; all discrepancies
are recorded, pass explicit1e-5 absolute tolerance, and terminal L2 comparisons
retain3e-6 tolerance. This is inference roundoff, far below observed field error.
Plots inspected; final source/strict-JSON/link/inventory/diff checks pass:
11 archived artifacts,5 cases/307 registered directories. See
[verification](local-ad-full-training-verification.json). No active worker or blocker remains. Further
local scaling/sampling optimization or cross-seed controls require a separate
task; no more work is required for this requested full-budget comparison.
