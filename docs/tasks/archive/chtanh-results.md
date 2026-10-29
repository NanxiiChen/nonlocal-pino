# Phase-only tanh/Jacobian and frequency controls

Five new complete cold starts, each5000SOAP+500L-BFGS, seed42, same T1.5 physics/reference/FP64/ordinary. The retained integral modes8 row is historical, not a fresh paired rerun. Every table entry is final, not best-checkpoint selection. Strong and integral losses are different recipes; do not attribute their cross-row gap to tanh alone.

| Model | Parameters | Phase FEM pooled % | Phase FEM terminal % | Mu native pooled % | Mu native terminal % | Trainer s | Energy max rise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong / modes8 | 16344 | 59.0121 | 65.1624 | 87.1217 | 315.9124 | 253.53 | 0.000191965 |
| Tanh / original r / modes8 | 16344 | 58.5632 | 64.9772 | 73.2626 | 265.2954 | 276.76 | 0.000130402 |
| Tanh / r/J / modes8 | 16344 | 73.0732 | 78.4242 | 162.5980 | 518.0439 | 302.69 | 0.00246175 |
| Tanh / r/J / modes5 | 15576 | 69.2708 | 75.1005 | 165.6864 | 531.3104 | 295.69 | 0.00205456 |
| Integral R/t / modes8 (historical) | 16344 | 21.5396 | 43.1779 | 258.0723 | 2716.0436 | 197.77 | 0.0670517 |
| Integral R/t / modes5 | 15576 | 17.9854 | 29.3681 | 129.2923 | 836.5967 | 190.78 | 0.00707818 |

Map amplitude1.1 is an explicit10% well-scale buffer, not a fitted trajectory; strict1.0 is a supported entry with measured range-only lower-bound phase error.3358% pooled/.4573% terminal versus FEM. Implicit scalar mass correction and cancellation-free tanh/Jacobian derivatives are independently validated. Mu remains exactly eliminated.

Tanh alone gives no convincing phase improvement; r/J before the existing spatial metric worsens the strong recipe. Lowering modes5 helps modestly in that Jacobian recipe and improves the retained integral recipe in this run. No multi-seed ranking or universal tanh/CH failure claim. Lowest pooled phase now17.9854%, terminal29.3681%, but native mu129.29%/836.60%, phase[-1.34285,1.24822] and energy rise.007078 remain unacceptable. No SOTA/default promotion.

Fourier features use pi*k*z, not2*pi*k*x. Modes8 minimum wavelengths.125/.10, about11.14/8.91 median nearest-neighbor spacings; modes5 .20/.16. Initial basic coordinate waves reach5. Spatial MLP uses no temporal Fourier embedding; time is rank24 log-Chebyshev. Measured native A² gain of mode8 relative5 is2.46--3.30 for the four axis sine/cosine probes. This is not a Nyquist theorem for irregular grids or a guarantee that nonlinear MLPs generate no higher harmonics.

All final/best CPU replays and exact chemical/mass audits pass. Trained phase time derivatives independently match NumPy Chebyshev/Brent chains (max2.3e-12). Both new mathematical tests pass. The transformed constant mode is only.052%/.020% of final Jacobian-control loss; it does not explain all degradation. Dense midpoint objectives differ by less than1%; no gross time-node aliasing appears in this check. No Jacobian underflow (min~.043) or hidden floor.

Canonical evaluate.py replay/physical boundary-normal diagnostic also passes for the mapped model; native wall normals remain local-limit diagnostics. Source/initial parity is in chtanh-provenance.json; map controls have identical51-source snapshots. The first unmapped control differs only in unused map/diagnostic helper updates; its active network/training/operator/optimizer code agrees. All3 modes8 initial parameter files are identical. Modes5 changes feature dimension/initialization, so this is not parameter matched.

Exact follow-up is in the active task. No Git operation, reference/geometry change, supervised initialization or frozen time windows.

Field figure shares a phase colour scale clipped at±1; actual extrema are
reported above. The new low-band working preset exactly matches the completed
CLI strategy. All26 config schemas, artifact paths, strict/scaled entries,
independent final objectives and git diff --check pass. No GPU/FEM worker remains.
