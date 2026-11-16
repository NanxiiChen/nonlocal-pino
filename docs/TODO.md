# Deferred research tasks

## Ceramic corrosion: training stability across geometry libraries

**Deferred by the user on 2026-11-16. Do not start further Ceramic training or
diagnostics until requested.** Preserve the successful rolling-geometry default.

The fixed64/128/256/512geometry study, with identical initialization and20kupdate
budgets, produced64test-case channel-mean space-time errors of5.1984%,32.7932%,
32.4830%,and32.7915%. The three larger banks plateaued and did not recover.
All input/final-field audits passed; a passed audit is not proof of good accuracy.
The user notes similar poor-convergence episodes in this example previously.
The reason remains unestablished; do not interpret this as a general claim that
more training geometries worsen generalization.

When resumed, compare successful/poor runs using the retained earlyAC/CH/IC/mouth
loss histories, adaptive weights and actual geometry/time/core-halo draws. Check
input/operator integrity and boundary/initial fitting before attributing the
failure to optimization. Any changes should be isolated and tested across
initializations, retaining all outcomes. Do not select seeds solely for low error.

Evidence: [completed experiment](tasks/archive/fixed-condition-library-size.md),
[all final statistics and direct paths](../artifacts/fixed_condition_library_results.json).
Run root: `/root/autodl-tmp/tf-logs/particle_corrosion2d/`, names
`fixed-library-20261115-01-ceramic-n{64,128,256,512}`.

## Spinodal: finite-horizon nonlocal versus local-PDE discrepancy

**Deferred by the user on 2026-11-15. Do not launch this study as part of the
completed five-example comparison.** Keep the current training defaults and
continue other planned experiments.

On the same eight geometries and equal-node space-time phase metric, the final
50k Nonlocal PINO gives4.5990% against its native numerical reference but31.9216%
against independently refined local FEM. FEM PINO gives5.5838% against local
FEM. The two numerical references differ by31.9633%. This supports a substantial
spatial-model/discretization discrepancy, not merely insufficient PINO fitting;
it does not isolate the kernel, quadrature and boundary contributions.

When resumed, first compare pure numerical solutions while holding the phase
thickness, physical coefficients, geometries and evaluation points fixed. Vary
horizon and spatial resolution independently, then jointly control delta/h.
Check quadrature/consistency and boundary truncation separately from interior
finite-horizon effects. Revisit PINO training only after this numerical baseline
is understood. Do not subtract relative-error magnitudes to attribute causes.

Evidence: [final dual-reference reductions](../artifacts/spinodal_final_reference_norms.json),
[qualified local references](../artifacts/spinodal_local_reference_validation.json),
and [completed comparison](../examples/confined_ch2d/README.md#spatial-method-accuracy-and-cost).
