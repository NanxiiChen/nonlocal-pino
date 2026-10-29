# Sharper, richer confined CH coarsening

Goal: address the user's criticism that the first confined CH case has weak
phase contrast, diffuse late interfaces and an overly simple initial field.
Add a richer explicit opt-in case with mature two-phase morphology and actual
merger/disappearance/wall interaction, preserving the old pilot and results.

Acceptance: deterministic multiple-mode initial phase with proven analytic
Neumann compatibility, sharper resolved interfaces and enough observation time;
independent reference preflight confirms complex evolving structure and pure
phase areas, rather than only changing plot colors. Refine the selected reference
before any training. Preserve legacy checkpoint replay and baseline defaults.
If a new training pair is run, use canonical train.py and local AD comparator,
report shared/native targets honestly. No Git authorization in this turn.

Status: complete2026-10-27. Richer opt-in physical case and independently
refined T1.5 reference delivered. Blocker: none. Next action: none in this task;
a future matched PINN study can use the validated preset through train.py.
No new PINN training/accuracy result is claimed. All FEM processes exited.

Context: base epsilon.05,T1,amplitude.3 field reaches only about+/-0.8; it has
not formed mature pure phases. The equilibrium quartic planar profile
phi=tanh(s/(sqrt(2)epsilon)) has -0.8..0.8 width2sqrt(2)atanh(.8)epsilon,
about.155 at epsilon.05. Candidate epsilon.02 gives width.062. Use the same
ellipse+two impermeable holes initially, original mixed residuals/direct phase
output, no motion prior or transformed loss. New IC is a sum of explicit
waves with tanh saturation, multiplied by the existing fourth-order wall
flattening. It is physical initial data, not a network output-coordinate trick.

First six-wave candidate at T3 produced much sharper fields (pure fraction64.6%)
but still mostly5 dominant bands (transient6), so it is an exploratory candidate,
not the selected extension. Its preflight/data/config/figures remain at the
coarsening subpath and archive/confined-ch-coarsening-sixwave-config.py.
Analytic initial-pattern screen: balanced10 and12 waves both give12 significant
regions at area>=.5% each (10-wave raw positive9/negative7); choose10 for fewer
prescribed terms, without looking at PINN accuracy. The selected10-wave
input was then covered by both focused tests, as recorded below.

Latest user steering: stop at1.5s, not3. Selected config train_coarsening.py now
uses balanced10 waves,T1.5,epsilon.02,delta.03,mesh.02,reference dt.005/.0025,
new root /root/autodl-tmp/data/confined_ch2d/coarsening_multimode10. Same equations,
BCs and optimization scheme. No T3 training was run. Prior baseline and
exploratory candidate are preserved.

Selected preflight: pure fraction |phi|>.9 grows27.06% ->46.25% by T1.5;
terminal range[-1.0262,1.0252], mass drift2.05e-16, energy.14854 ->.06997 with
all saved increments negative. Significant sign components (>=.5% domain area)
are10 initially/8 terminal on the coarse FEM; the exact analytic initial field
has12 on the same centroid grid. This topology statistic is resolution/threshold
sensitive, not a certified count of exact interfaces. Coarse P2 initial phase
interpolation L2 vs exact IC is1.131%, so refinement is needed before training.
The selected preset's two focused multimode checks pass63.038s: independent
NumPy-vs-AD Laplacian/normal compatibility, hard IC/mass/time-JVP and finite
coupled gradients for both methods. Old final checkpoints replay all21 times
and both fields with maximum differences<=2.45e-15; no old run written.
The coarse preflight plots/statistics are retained under the explicit
confined-ch-coarsening10-preflight-* names. The unsuffixed10-wave figures now
show the validated finer local FEM reference, with role explicit in title/JSON.

Initial .02/.014 P2 interpolants differ1.12569% at t0, so the phase mesh gate
necessarily fails even before the first full run completes. Refine reference
mesh sizes to .014/.01, leaving centroid training geometry.02, delta.03,
physical IC/T/epsilon and all gates unchanged. Reuse .014 refined-dt stage only
through explicit verified archive script, not a silent cache. Preserve first
failed reference. New .01 stages are independent CPU FEM processes at dt.005
and.0025; no concurrent JAX GPU/MPI work.

Explicit archive/confined-ch-coarsening10-reference.py ran --stage=fine_dt
and --stage=fine in separate fenicsenv CPU processes, then --stage=assemble
in .venv. It checked request/config/physical/query provenance, preserved
reference_mesh02_failed.npz/json, and reconstructed the same standard gates
with .014/.01 local fields and identical native dynamics. No blind cache or
hashes. The canonical solver with the final config independently regenerates
the same stage choices. Archived helper is a one-time reconstruction record;
it deliberately fails on existing stage metadata rather than overwriting it.

Final gates pass without relaxed thresholds (maximum-time relative L2):

| Check | Phase | Chemical potential |
|---|---:|---:|
| Local mesh .014/.01 | .55193% | 3.01351% |
| Local time .005/.0025 | .04796% | .17959% |
| Native time | 2.19e-6% | 1.21e-5% |

Local/native integral mass drifts5.27e-16/1.04e-15; all saved energies decline.
Local final energy.14906945 ->.06993718. Local fine mixed DOFs57544 vs coarse
29826. Finest IC interpolation vs exact analytic phase.1786%. CPU
load_reference accepts physical identity, geometry/order, fields and gate JSON;
shape(21,3682), terminal time1.5. Native/local pooled phase gap3.77886% and
mu45.4915% predate any new PINN training and are separate targets.

Final morphology: initial/terminal pure fraction27.17%/46.14%, significant
sign regions12/9 (area>=.5%), terminal range[-1.02576,1.02517]. Coarse preflight
counts10/8 show threshold/resolution sensitivity; these are diagnostics rather
than certified exact interface topology. Original case has0% pure area at
both endpoints and5 significant regions. Figures use identical[-1,1] color
limits, retain holes and label FEM; selected times0,.15,.45,.9,1.5 resolve the
early mergers and later interface maturation. This compares different physical
cases, not PINN accuracy. CH overshoots are recorded rather than clipped in data.

Full actual training-grid operator CPU audit passes: independent native matrix
maxdiff7.28e-12, V-weighted asymmetry2.22e-16, constant residual7.56e-12,
one connected component,7-23 neighbors per point. On the analytic IC, independent
finer integral quadrature at h.01/14203 centroids changes the delta.03 result
by5.664%; finite-horizon-vs-local-Laplacian discrepancy40.69%. At delta.02 the
quadrature change rises30.89%, so reducing horizon alone is unjustified.
These are IC/operator diagnostics, not native evolution spatial convergence.
No kernel normalization, training grid or physical case was changed to fit a
common target. Retain finite-horizon/pointwise-wall limitations from firstpilot.

Diff-level delivery: problem.py adds explicit multimode validation/physical IC
and independently derived NumPy Laplacian; network.py adds only static IC
metadata and the matching hard initial phase, keeping old leaves/arithmetic.
New configs/train_coarsening.py and tests/test_confined_ch_multimode.py;
README/development/index document the variant and direct paths. Reference
reconstruction/operator/morphology scripts, failed+passed checks, logs, legacy
replays and figures are archived with this record. Original defaults, old
checkpoints/data, physical equations and Git history remain unchanged.

Task-end validation: two focused selected-input tests and old final replay pass;
full-grid operator and independent reference gates pass; final config/archive
syntax, changed documentation links, artifact inventory and diff whitespace
pass. No unchanged expensive suite rerun. No new GPU trainer or Git operation.
Memory updated: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
