# More demanding coherent precipitation example

Goal: extend the mild single-particle elastic phase-field pilot into a genuinely
shape-changing multi-particle interaction/coarsening case, retaining the existing
energy-based homogeneous-elastic framework and canonical train.py entry.

Acceptance: explicit opt-in initial-particle configuration with hard phase IC
and mass, old presets/checkpoints preserved; independently solved/refined native
and local references showing a meaningful time evolution and elastic effect;
complete PINN cold start(s), phase/displacement/stress/q errors, physical fields,
energy/mass and checkpoint replay. Keep original simple default intact until
the new case is actually validated. No new constitutive law, boundary redesign,
SOAP work, Git operation or third-party dataset is requested.

Context: existing single particle uses homogeneous isotropic exact spectral
elastic elimination, anisotropic1% eigenstrain, T1 and modest axis ratio0.9586.
Multiple unequal particles and stronger elastic/interface competition provide
new interactions with the same traceable equations. Future motion is not encoded
in the network; only the supplied initial field is constrained.

Status: complete. Refined reference, both requested cold starts, CPU final
checkpoint replays and physical audits pass. No process remains running.
Blocker: none.
Next action: none for this extension. Further accuracy optimization requires a
new task; small-particle disappearance and initial q remain weak. The primary
comparison is nonlocal/local AD only, per explicit user steering. A completed
local spectral exploration is preserved/registered but excluded from the main
comparison. No Git operation was performed.

Nonlocal completed exit0: native pooled/terminal phi7.35587%/8.93540%, q
17.1837%/6.12665%, solved u8.54290%/11.3952%, stress7.48954%/9.44510%.
Common-local pooled phi8.03038%; trainer183.3399s. L-BFGS reaches500 budget.
This is a harder validated reference case, not yet a highly accurate PINN;
retain the simple default and report this limitation rather than promoting it.

Frozen physical case: three periodic tanh particles with centers(.35,.32),
(.67,.36),(.40,.68), levels0.18/0.11/0.065, width0.035/background-0.95; T4,
lambda9800/mu4900, unchanged1% E_xx. Smooth union is only an initial condition.
Reference-only native preflight: smallest particle disappears by aboutt2.6;
largest elongates and grows while middle shrinks. Zero-elasticity control
instead merges the lower pair; terminal coupled/control phase difference39.1%.
The original elastic scale shows much weaker elongation. No future motion is
encoded. Preflight records/fields are archived, not full refinement evidence.

Verification so far: seven focused new/old tests pass36.245s. Legacy single
initial arithmetic remains exact. A direct CPU replay of the original simple
checkpoint matches phi1.33e-15/q2.22e-16/u6.51e-19/stress3.91e-14 without
writing to its run. Existing model leaves remain loadable despite new static
IC metadata. New initial union/range, periodicity, hard IC/mass/time derivative
and finite coupled parameter gradients pass. Invalid particle inputs rejected.

Reference resolution: initial q127-vs255 difference0.2948% fails old0.2%
threshold. Coarse local grid increased to191 (fine255 unchanged), giving t0 q
6.57e-5. Full references use dt5e-4/2.5e-4 and all-field checks. No tolerance
relaxation or physical retuning after this preflight selection.

Full reference now passes: maximum native time errors phi7.81e-7/q2.11e-6/
u8.89e-7/stress7.36e-7. Maximum local time q3.82e-6 and local space q6.57e-5;
other fields below1.27e-6. Native/local mass drift1.11e-16/0, no saved energy
increases. Elastic-effect terminal phase difference39.1441%; native/local
pooled phase gap1.23526%. Reference stored at
/root/autodl-tmp/data/precipitate2d/results_periodic_multiple.npz with direct
JSON validation; full log/checks archived. No reference fields train the PINN.

Analysis correction implemented in the audit script: common local reference is spectrally transferred
from fine255 to63, while hard training IC/mass use direct analytic samples.
Their initial phase L2 difference0.2091%, mean difference2.88078e-6 is expected
projection/truncation error. Measure mass drift from the actual prescribed
training mean, and report the target-mean offset separately; do not loosen
roundoff conservation tolerance or modify reference data to hide it.

Primary pair precheck: initial checkpoint bytes match exactly; all21 frozen
source files match. After excluding nl.method/log.run_name, complete configs
match. Intermediate AD plateau at2750 SOAP updates was~13.2%; both full-budget runs
are now completed and separately assessed below.
Analyzer now uses local_ad only and audits mass against the training IC mean.

Artifact index now registers all three new run directories and the direct
multiple reference/config/report; completed local spectral is explicitly
exploratory and excluded from the primary comparison. Inventory passes:
7 cases/358 registered directories. README primary command uses local_ad;
DEVELOPMENT documents the new explicit IC schema/static checkpoint metadata.

## Final changes and evidence

Implementation: added explicit smooth-union multi-particle IC support in
problem.py/network.py, full-cell contour view in evaluate.py, standalone
train_multiple.py preset and focused tests. Legacy single arithmetic/trainable
serialization leaves are preserved. README and DEVELOPMENT describe the schema,
commands and limitations. Default single pilot remains unchanged.

Both requested cold starts exit0: /root/autodl-tmp/tf-logs/precipitate2d/
20261027-multiple-elastic-nonlocal-01 and20261027-multiple-elastic-local-ad-01.
Same seed42, initialized parameter bytes,21 frozen sources, network/hard IC/mass,
FP64 ordinary policy, sampling code and SOAP5000+LBFGS500. Actual full GPU
sample traces were not stored. Both LBFGS stages exhaust500 iterations.

| Common local target, final update5500 | Nonlocal | Local AD |
|---|---:|---:|
| Pooled phi relative L2 | 8.03038% | 12.89808% |
| Terminal phi relative L2 | 9.18200% | 19.24351% |
| Pooled solved-u relative L2 | 9.43082% | 13.14298% |
| Terminal solved-u relative L2 | 12.07708% | 21.03099% |
| Pooled stress relative L2 | 8.08018% | 13.84708% |
| Terminal stress relative L2 | 9.45847% | 21.57865% |
| Pooled q relative L2 | 25.37842% | 25.80487% |
| Terminal q relative L2 | 6.61354% | 6.69575% |
| Trainer time | 183.34s | 674.27s |
| Mean compiled SOAP update | 19.81ms | 91.53ms |

Nonlocal own-target pooled phi7.35587%; native/local reference gap1.23526%.
Best checkpoints areSOAP5000(nonlocal)/4750(AD), common pooled phi8.01999%/
12.83447%; final-budget fields remain the primary matched comparison. This
corrects the intermediate expectation that LBFGS would necessarily improve L2.
Single untuned seed is not a stable accuracy ranking; no SOAP investigation.
Mechanical subproblem remains solved Fourier equilibrium in both; local AD
means phase/chemical spatial Hessians, not learned-displacement derivatives.

Reference zero-contour connected-component count goes3->2 by t2.4local/
2.6native; nonlocal prediction reaches2 only at t4 and AD stays3. Late small
particle disappearance and underpredicted largest-particle elongation remain.
Initial q errors61.77%/77.04% are high; hard phi IC does not constrain q.
Both saved predicted energies decrease; this does not imply accurate trajectories
or compare different native energy functionals as a common objective.

Full final CPU replay max abs errors across all fields3.84e-13. Independent
acoustic reconstruction agrees with u/stress, mechanical divergence residual
<=1.02e-11. Own-IC mass drift<=9.99e-16. Exactly two PDE loss components remain.
All checks pass in precipitate-multiple-analysis.json. Test output and legacy
replay, full reference checks, configs/metrics/loss/refinement/replay JSON files,
reference controls and physical/convergence figures are archived beside this
record. Full raw checkpoints/source snapshots remain at direct run paths.
Completed exploratory local spectral run is20261027-multiple-elastic-local-spectral-01;
preserved without rerun and excluded from primary conclusions at user request.

Artifacts: [common fields](precipitate-multiple-common-fields.png),
[convergence](precipitate-multiple-comparison.png),
[elastic/uncoupled reference](precipitate-multiple-reference-dynamics.png),
[nonlocal native fields](precipitate-multiple-nonlocal-phi_fields.png),
[local AD fields](precipitate-multiple-local_ad-phi_fields.png),
[full audit](precipitate-multiple-analysis.json),
[reference checks](precipitate-multiple-reference.json),
[tests](precipitate-multiple-tests.txt),
[legacy replay](precipitate-multiple-legacy-replay.json).

Verification: focused7 tests passed; independent reference all-field refinement,
source/config/initial parity, final CPU replay, mass/energy/mechanics and saved
metric comparisons pass; artifact inventory7cases/358dirs; final diff/link checks
recorded at closure. Do not rerun unchanged expensive checks. No claims of
heterogeneous elasticity, full PFHub4 reproduction or cross-seed robustness.

Closure: git diff --check passes;84 local documentation links resolve and all
new archived JSON parses. Artifact inventory7cases/358directories passes.
Memory files updated: docs/PROGRESS.md, docs/DECISIONS.md and this archived
task (moved from docs/tasks/precipitate-multiparticle.md). No active task or
training process remains. Existing unrelated working-tree changes preserved.
