# All-nonlocal spatial derivative experiment

Goal: implement and test finite-horizon conservative first derivatives on the
current prescribed-vortex CH case, compared with the existing spectral flux.

Acceptance: explicit opt-in preset through train.py, direct phase output,
unchanged nonlocal pair Laplacian/IC/mass/loss/network/precision/budget. A
moment-corrected periodic nodal gradient/divergence pair is independently
checked against NumPy/symbols, mass/adjoint/nullspace behavior and parameter
gradients. New native nonlocal-flux reference passes dt refinement; common
local PDE target remains the previously refined reference. Two fresh serial
seed42 cold starts through SOAP5000+L-BFGS1000, independent checkpoint replay,
common/native pooled/terminal/interface L2 and timings with plots. No defaults,
NS implementation, tuning, SOAP investigation or Git operation requested.

Context: all first derivatives currently use spectral_flux in the recommended
preset. The assessed symmetric nodal gradient has checkerboard null modes;
this test keeps the independent coercive pair Laplacian and has no pressure
unknown, so it is a bounded transport experiment, not a validated NS operator.

Status: complete (2026-10-27). Both cold starts and CPU analyzer exit0.
Blocker: none.
Next action: none. Keep spectral flux default and the explicit research option.

Spectral final: SOAP5000+LBFGS1000, common-local pooled phi2.923619%, terminal
3.127022%; native pooled3.243504%, terminal3.461666%; trainer213.4068s.
Run directories under /root/autodl-tmp/tf-logs/twophase2d are
20261027-vortex-flux-spectral and 20261027-vortex-flux-nonlocal.

Implemented: PeriodicGradient and paired divergence; nonlocal_flux dispatcher;
explicit new preset/reference identity and independent reference sine symbols.
No old identity/default changes. Time JVP and pair Laplacian remain unchanged.
New reference path is results_periodic_vortex_nonlocal_flux.npz under the same
data root; unchanged local fields explicitly reused from validated original
reference. Native coarse/fine dt2e-4/1e-4 phase max difference1.3384e-5,
q1.4734e-4; mass drift1.11e-16. Native/common-local phi gap9.21898% pooled,
3.85773% terminal. This is a measured operator bias, not a training failure
or strict lower bound on neural error. Ten focused checks pass30.431s.

Final full-budget results (both1000 accepted L-BFGS iterations, stopping at
the budget rather than tolerance; no best-checkpoint/seed search):

| Metric | FFT flux + nonlocal diffusion | Nonlocal flux + nonlocal diffusion |
|---|---:|---:|
| Common-local pooled phi L2 | 2.923619% | 8.544797% |
| Common-local terminal phi L2 | 3.127022% | 4.514710% |
| Common-local pooled interface-band phi L2 | 12.594492% | 38.751335% |
| Common-local terminal interface-band phi L2 | 12.553031% | 19.549923% |
| Native pooled phi L2 | 3.243504% | 2.762494% |
| Native terminal phi L2 | 3.461666% | 3.018803% |
| Common-local pooled q L2 | 55.318422% | 147.583729% |
| Native pooled q L2 | 53.249822% | 43.517469% |
| Trainer wall including JIT/evaluation/refinement | 213.4068 s | 217.6332 s |
| Mean compiled SOAP update after100 | 16.8261 ms | 17.6353 ms |

Pooled is the discrete L2 across all21saved times/64x64points. The interface
band is defined from the same common local target with absolute phi below0.9.
Native targets differ. Common local fields/points/times/energy are byte-value
identical between both loaded reference files; source files and raw initial
checkpoint bytes match. Configs differ only by advection, reference path and
run name. Seed42, identical sampling code/counts; full actual GPU time-sample
trace was not stored. Training targets are PDEs/IC/mass only, no reference
field supervision. Both new loss logs have exactly two PDE parts.

Interpretation: the new first-derivative option can train against its own
semidiscrete model but worsens common-local phase/interface/q accuracy, without
speed benefit. New native/reference gap9.219% pooled is already present before
training, compared with old native gap~1.28%. Flux derivatives on the SAME
common reference differ17.5%,21.1%,11.4%,7.62% at t0.2/0.5/1.5/1.8. Given
velocity remains discretely divergence-free to1.20e-14 in these probes; the
changed flux derivative is not explained by a gross spurious compressibility.
These checks support substantial finite-horizon transport bias, not an exact
causal decomposition or a strict optimization error floor. No controlled
horizon/kernel/stabilization comparison was performed. The outcome does not
show all nonlocal first-derivative methods must be worse.

The full nonlocal native reference range[-1.54637,1.58557] and predicted range
[-1.41482,1.37797] show significant overshoot, versus FFT-flux native
[-1.06899,1.04804] and prediction[-1.05412,1.03505]. No clipping or hard phase
range bound was added; CH need not obey a maximum principle. Do not relabel
this as exclusively a network defect or automatically as time-integration
failure: actual time refinement passes at the chosen finite spatial model.
The centered gradient checkerboard null remains; diffusion retains the
independent pair Laplacian. This is not evidence of a pressure-stable NS scheme.

Verification and provenance: ten focused tests (new symbols/adjoint/mass/
nullspace, mixed physical residual and nonzero parameter AD, preserved common
target plus seven old transport checks) pass in30.431s. A later schema-only
restriction rejects nonlocal_flux with local diffusion, whose hybrid reference
is not supplied; that focused common-target/schema check passes0.008s. This
validation-only guard was added AFTER both identical snapshots were captured;
it changes no executed numerical branch. No reference/PDE changes after launch.
All19presets validate and both references pass identity/shape/time checks; old
native reference is explicitly rejected by the new preset.

Independent CPU replay uses unchanged FP64 atol1e-10: max field errors
1.0214e-14/1.2157e-14 (FFT/full-nonlocal respectively). Both history audits and
roundoff mass checks pass. Saved model/config/prediction/source paths remain in
their respective run directories; no hashes or external artifact moves.

Reproduce the native-reference generation (creates a new path and refuses
overwrite) and final audit from repository root:

    .venv/bin/python docs/tasks/archive/twophase-nonlocal-flux-reference.py
    JAX_PLATFORMS=cpu .venv/bin/python docs/tasks/archive/twophase-nonlocal-flux-analyze.py

Launches through examples/twophase2d/train.py used explicit configs
train_vortex_conserved.py / train_vortex_nonlocal_flux.py, seed42, empty
XLA_FLAGS and disabled persistent compilation cache. No warm start or concurrent
GPU training. All runs end at6000 optimizer iterations.

Artifacts alongside this record:

- [Complete metrics, parity and CPU replays](twophase-nonlocal-flux-analysis.json)
- [Common-reference convergence and per-time error](twophase-nonlocal-flux-convergence.png)
- [Common-reference physical fields](twophase-nonlocal-flux-fields.png)
- [FFT prediction versus its native reference](twophase-nonlocal-flux-spectral-native-phi.png)
- [Nonlocal prediction versus its native reference](twophase-nonlocal-flux-nonlocal-native-phi.png)
- Analysis/reference Python scripts, reference JSON/log, per-run config/summary/
  refinement/metrics/loss JSON/JSONL, focused test/schema logs with this prefix.

Diff-level changes: operators.py adds PeriodicGradient with checked fixed-grid
shape/range assumptions; advection.py dispatches the explicit flux choice;
problem.py validates and records new native identity; solver.py supplies
independent sine multipliers without importing training operators; losses.py
docstring reflects options. New research preset and numerical tests are added.
README/DEVELOPMENT explain formulation, usage, results and limitations. Artifact
index registers two new run directories and explicit reference/report paths.
Defaults, unrelated examples, existing artifacts, dependencies and Git history
are unchanged; SOAP issue remains deferred. Task-end memory: PROGRESS.md,
DECISIONS.md, DEVELOPMENT.md and this completed record. Inventory passes with
7cases/355registered directories; all88local links in five affected documents,
strict archived JSON/source/document whitespace and git diff --check pass.
All19configs validate after the guard. No GPU compute process remains and no
unchanged expensive suite was repeated.
