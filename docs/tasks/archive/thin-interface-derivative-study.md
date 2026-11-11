# Thin-interface local-AD/nonlocal comparison

## Goal

The user requests an experiment varying diffuse-interface thickness and comparing
nonlocal versus local-AD training errors, preferably for melting. Produce paired
error-versus-thickness curves with a common local-PDE reference and record runtime.
Do not assume a nonlocal advantage before observing the measurements.

## Acceptance

- Explicit physical/nondimensional thickness convention and fixed quantities.
- Matched architectures, output constraints, optimization budget, collocation
  targets and runtime initialization within each derivative pair.
- Independent reference and operator-resolution checks before interpreting the
  thinnest-interface results. Separate discretization from learned-field error.
- Report whole-field, interface-region and interface-location errors, with
  finite-seed/budget qualifications and direct external artifact paths.
- Preserve current production defaults. Run proportionate focused tests and
  update this record plus PROGRESS/DECISIONS before finishing.

## Current status

Complete: all16cold10k runs and the final dense evaluation audit passed.
Training ran06:09:20--06:35:25UTC (14:09--14:35Asia/Shanghai),26.1minutes;
postprocessing completed06:36:22UTC. GPU is idle. Production defaults remain
unchanged. The current128^3six-axis nonlocal graph shows no thin-interface
accuracy advantage over paired localAD under this recipe.

Final positive-time full-field relative L2 percentages, mean of two seeds:

| epsilon* | Nonlocal | Local AD | Native/local baseline |
|---|---:|---:|---:|
|0.0225|0.22347|0.004970|0.20681|
|0.015|0.39252|0.008878|0.37896|
|0.010|0.65928|0.016234|0.65463|
|0.0075|0.87828|0.030574|0.89470|

The thinnest interface-band relative L2 is10.256%nonlocal versus0.3502%local;
final-time full-field is0.8657%versus0.03894%. Dense128^3evaluation gives
mean pooled0.89294%versus0.02748%, preserving the ordering. All16ray checks
have zero missing/multiple crossings. Four checkpoint replays differ by at
most3.81470e-6phase, within explicit1e-5absolute tolerance. Saved-field norms
were independently recomputed for all16runs; cold-state/input pairing and
actual saved10000/2000schedule pass.

Average of per-run warm median update+sampling is3.151msnonlocal versus
3.405mslocal (~7.5%shorter); mean complete child process88.10sversus105.88s.
These timings are for this small PINN on the recorded RTX4090, not the PINO.

Interpretation limits:
- The new slower-decay recipe already reduces thinnest5k pooled errors to
  0.754--0.867%nonlocal/0.100--0.166%local. Do not attribute all improvement
  against old5k solely to additional steps. Same-new-trajectory5k->10k mainly
  benefits the thinnest local runs; nonlocal has a plateau and can rise slightly.
- Thin local seed43 still improves0.04857->0.04076->0.03620%at8k/9k/10k.
  No universal convergence claim follows from the common10kbudget.
- Local FD384->512 discrepancies0.00542/0.00979/0.01792/0.02769%are comparable
  to tiny localADerrors, so they do not certify equally small continuum errors.
- Direct native/local errors track the nonlocal plateau, but learned/native
  errors0.09697/0.17219/0.30859/0.46431%remain nonzero. Discretization is a
  substantial contributor, not a proven sole cause or irreducible floor.
  The unchanged graph second moment0.6858547757 and finite h/epsilon require
  separate spatial-refinement/moment-normalization controls before broad claims.

## Experiment history and shared setup

The isolated single-IVP study is implemented through the normal train.py entry.
Four epsilon values (.0225/.015/.010/.0075), two runtime seeds (42/43),
K256/Nt16 and the same hard-IC latent MLP are declared. Every arm
saves actual core/time/boundary draws and cold model/optimizer states.
Local references use 384/512 cube-equivalent grids and octant reflection;
the original six-axis graph has second moment0.6858547757. PDE loss is divided
by N3 for both arms, with explicit homogeneous Neumann boundary terms.

First queue `/root/autodl-tmp/tf-logs/melting-thin-interface-20261111-01`
stopped before training: phase-coordinate epsilon.0075 reference time-halving
changed the pooled field by0.5591%. Its artifacts and failed journal are retained.
Queue `/root/autodl-tmp/tf-logs/melting-thin-interface-20261111-02` validates the
equivalent latent reference PDE and exact native difference ratio, then stops
before training because JSON lists and ConfigDict tuples were compared directly.
That schema-only comparison is corrected by normalizing both through JSON.
All four qualified references are reused without new solves by completed queue
`/root/autodl-tmp/tf-logs/melting-thin-interface-20261111-03`, screen
`melting-thin-interface-20261111-03`. Sibling .log and journal.json record stage
ownership. All16SOAP5000training arms have finished and paired initial states
and every actual input draw match. No FEM is running.

The user explicitly requests a new cold10000run of every arm. The new queue
retains the physical problem, sampling, architecture, optimizer and references;
the common exponential-decay period stretches from1000 to2000updates while
lr_init1e-3/factor.5/min1e-5 stay fixed. Expected LR at5k/10k is1.76777e-4/
3.125e-5. This is not a pure step-count ablation against the old5krecipe.
Save all1kmodels and metrics to compare5k/10kwithin the new trajectory.
No checkpoint or optimizer is loaded into new training.

The old5kthin cases were still improving: epsilon.010/seed42 local pooled
L2 decreases25.75->20.04->17.00% at3k/4k/5k; nonlocal26.20->20.71->17.42%.
At epsilon.0075 the same local sequence is45.48->43.23->41.95%; nonlocal
46.10->43.97->42.76%. These are finite-budget results, not converged limits.

Thin latent reference time-halving difference is3.9931e-15. Independent
positive-time local spatial differences across384/512 are0.00542/0.00979/
0.01792/0.02769% respectively. Thick phase/latent references agree within
5.12e-6 and2.12e-5 relative L2, with second-order spatial reduction. The failed
thin phase-coordinate solves remain unqualified.

Five focused tests pass, including exact compact/original graph agreement,
local reflected-Neumann second-order convergence, reference chain-rule checks,
paired initialization/sampling and diagnostics. Five legacy latent tests pass.
Production numerical defaults are unchanged.

Independent read-only review confirms reference algebra, boundary reflection
and paired setup. Diagnostics now preserve fields before checking ray crossings,
report missing/multiple crossings, and compute learned-versus-native metrics.
The direct native/local discrepancy is a baseline rather than a rigorous error
floor, because training also adds a Neumann secant penalty. A diagnostic-only
source revision is saved under the active queue's source/diagnostic_revision;
no training equation, objective, optimizer or sampling changes after launch.

## Historical diagnostic and run provenance

The old5kpostprocessing owner has stopped. Its strict2e-6checkpoint replay
comparison failed on the third thinnest model (seed43/nonlocal), with a maximum violating
absolute phase difference2.86475e-6; the first shown differences are at the
prescribed initial field. Its failed journal and partial dense receipt remain
intact; the two seed42 dense evaluations completed. This does not invalidate
the completed training/saved-field metrics, but the four-model128^3quadrature
audit is not complete. The new postprocessor uses explicit absolute phase
tolerance1e-5/relative0 and records initial/positive-time/all-point differences.
At the thinnest width, IC phase sensitivity to radius is at most94.3, so
3e-8radius roundoff could explain2.8e-6phase differences; this is a plausible
explanation, not a demonstrated compiler diagnosis. The old failure remains.

The completed cold10kserial queue is:
`/root/autodl-tmp/tf-logs/melting-thin-interface-cold10k-20261111-04`.
Its former same-name screen owned PID182054; first child182055 was
epsilon.0225,seed42,nonlocal. Four reference-reuse checks passed; all16children
finished10000updates and exited0.
Actual saved config confirms10000updates/2000decay; freshly constructed initial
model and optimizer bytes match the same seed in the old pilot. Runtime remains
RTX4090/49140MiB on this host; no other GPU jobs were present at launch.

Screen `melting-thin-finish-cold10k-20261111-04` completed the generalized finite
postprocessor after training. Both validated the
declared budget and decay period; legacy5kjournals alone may omit the latter.
After16training stages the owner emitted error/thickness and convergence plots
and independently checked paired inputs. The postprocessor audited128^3metric
quadrature after training released the GPU. The initial replay tolerance is
now explicit; the old failed postprocessing journal is not overwritten.

Review the completed queue's `journal.json`, `postprocess-journal.json`,
and sibling `.log`/`.finish.log`. The exact launch is:

```bash
.venv/bin/python -m docs.tasks.melting_thin_interface_study queue \
  --output /root/autodl-tmp/tf-logs/melting-thin-interface-cold10k-20261111-04 \
  --reference-root /root/autodl-tmp/tf-logs/melting-thin-interface-20261111-02/references \
  --seeds 42 43 --steps 10000 --lr-decay-every 2000
```

Do not relaunch into this existing output directory. All artifacts have
exclusive new paths; no old logs or weights were deleted. Automatic plots are
`error-vs-thickness.png`, `position-and-time.png`, and `convergence.png`.

## Session-end record

The user's request now supersedes the old5kbudget with a cold10krestart.
The old16arms complete; their saved-field metrics, cold-state/input identities
and regenerated plots pass independent analysis. Postprocessor CPU checks pass
for all16legacy configurations and a synthetic10k/2000schedule, and reject
bad budgets/schedules. New/legacy numerical regressions are unchanged from the
previous session. Reference time-step halving remains separate from training
updates and JVP temporal queries.
This turn changes the finite runner to require an explicit decay period,
snapshot the owner/config sources, record actual child commands, and plot
per-seed convergence. The generalized postprocessor checks saved budgets and
records explicit FP32 replay diagnostics. CPU schedule checks, malformed-CLI
rejection, real5kanalysis, initial-state parity, AST and git diff --check pass.
The new training and finite postprocessor are complete, with final10kaccuracy
and independently checked figures/metrics recorded above. No new numerical-kernel tests were required because
the PDE, operators and update implementation remain unchanged.
The final status turn changes documentation only, archives this completed task,
and updates `docs/PROGRESS.md` and `docs/DECISIONS.md`; git diff --check passes.
No pending stage or blocker remains. Additional graph-refinement or conditioning
experiments would be a separately selected follow-up, not silently queued work.
No commit/push or default promotion was authorized or performed in this study.
