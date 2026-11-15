# Deferred research tasks

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
