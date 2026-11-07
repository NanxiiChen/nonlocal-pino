# Particle-corrosion dimensional parameter audit

Date: 2026-11-06. Status: complete. Uniform2x scaling passes; the original
dimensionless pilot is not the electropolish material tuple in different units.

## Request and scope

The user questions whether the failed speed2 training comes from a wrong
parameter scaling and requests a dimensional-first check against corrosion and
electropolish, including the actual time represented if the same physical
coefficients are used. Derive and numerically verify the mapping; distinguish
uniform time rescaling from whether the original groups represent those same
materials. Do not change physics/defaults, retrain, publish Git or delete data.

## Context

Previous particle speed2 and Nt32 studies completed with poor PINO convergence.
Qualified local reference and a direct native early solve expand correctly.
Original particle defines n_ac1=4*2*A/omega, n_ac2=4,
n_ac3=4/(2*30**2), n_ch=80, kappa=2. All three AC terms multiply by kappa;
CH does not. Speed2 doubles all four n coefficients, not kappa. Initially these
were dimensionless pilot choices, not a calibrated material/time tuple.
Relevant archived records: particle-corrosion-time-kinetics.md,
particle-corrosion-faster-normalized-time.md, particle-corrosion-speed2.md,
particle-corrosion-time32-diagnosis.md.

## Acceptance and next action

1. Inspect actual default configs and code paths for particle, heterogeneous,
   corrosion2d and electropolish2d, including physical-time JVP and concentration
   normalization, and saved speed2 run configs.
2. Derive the four groups from the physical PDE and compare all coefficients,
   including the kappa multiplier. Solve for characteristic length/time and
   show any overdetermination/incompatible dimensionless ratio explicitly.
3. Provide a physically consistent inverse interpretation and state what must
   change to use exactly electropolish's material constants. Do not report a
   unique physical time unless the chosen dimensional anchors are explicit.
4. Save compact reproducible calculations/source pointers, update relevant
   documentation if incorrect, then archive the audit. No new PDE solve needed.

Blocker: none. Derivations, independent residual checks and saved-configuration
audit complete. No numerical-default change, new solve, training or Git action.
Future calibrated studies must first choose a dimensional material/length/time
tuple and derive coefficients and initial width consistently; none launched here.

## Dimensional formulation and units

Concentration is already normalized to the solid concentration, so c,phi,q are
dimensionless fields. Coordinates and time in the following equations have SI
units. The existing corrosion/electropolish convention is

$$
q=c-(c_{Se}-c_{Le})h(\phi)-c_{Le},\quad
f=Aq^2+\omega_\phi g(\phi)+\frac{\alpha_\phi}{2}|\nabla\phi|^2,
$$
$$
\phi_t=L_{\mathrm{eff}}\left[\alpha_\phi\Delta\phi-
\omega_\phi g'(\phi)+2A(c_{Se}-c_{Le})q h'(\phi)\right],\qquad
c_t=D\Delta q.
$$

Here A,omega have units J/m³,alpha has J/m,and L_eff has m³/(J·s).
D is the diffusivity appearing in this CH equation, not a bare mobility
requiring another factor2A. These units follow independently from f and phi_t.
The old README labels J/(m³·s) for A/omega and m/(J·s) forL are typos; corrected
in both corrosion/electropolish README tables with an explicit explanation.
No parameter values were changed to compensate for those labels.

With x=Lc*xstar,t=Tc*tstar, the effective coefficient vector is

$$
B_{\rm CH}=\frac{DT_c}{L_c^2},\quad
B_1=2A L_{\rm eff}T_c,\quad B_2=L_{\rm eff}\omega_\phi T_c,\quad
B_3=\frac{L_{\rm eff}\alpha_\phi T_c}{L_c^2}.
$$

For particle, B_i=kappa*n_aci and B_CH=n_ch. If L0 is the base mobility,
L_eff=kappa*L0; do not count kappa a second time in the effective coefficients.
Both current JVP input times and PDE coefficients are dimensionless. The old
network docstring's "physical-time" meant differentiation in the input time;
it did not assign seconds. Docstring/README now state the time unit explicitly.

## Actual current configurations

Electropolish PINN and PINO inherit A5.35e7,D8.5e-10,L1e-6,
alpha2.575e-5,omega1.76e7,Lc100um,Tc5s. Corrosion's current1pit preset uses
the same A,D,omega but L2,alpha1.03e-4,Lc100um,Tc50s.
The heterogeneous pilot explicitly chooses the independent dimensionless
groups and an activation-dominated regime; particle inherited those choices.

| Effective group | Corrosion50s | Electropolish5s | Particle original | Particle speed2 |
|---|---:|---:|---:|---:|
| B_CH | 4.25 | 0.425 | 80 | 160 |
| B_1 | 1.07e10 | 535 | 48.6363636 | 97.2727273 |
| B_2 | 1.76e9 | 88 | 8 | 16 |
| B_3 | 1.03e6 | 0.012875 | 0.00444444 | 0.00888889 |

Production legacy nondim() agrees with an independent formula evaluation.
All four completed particle run config.json physical sections match the
corresponding current original/speed2 factory; t_max remains1. n_ac1,n_ac2,
n_ac3,n_ch all double exactly, while kappa,cLe,cSe and width remain unchanged.
There is no missed multiplier, only-AC acceleration, or accidental extra kappa
in this implementation. Uniform2x is Tc_new=2Tc on the same autonomous dynamics.

## Scale-invariant inconsistency with an identical material tuple

The identities

$$
\frac{B_1}{2B_2}=\frac A{\omega_\phi},\qquad
\frac{B_3}{B_{\rm CH}}=\frac{L_{\rm eff}\alpha_\phi}{D}
$$

are independent of Lc andTc. The chemical energy ratio agrees. The second
ratio is5.55555556e-5 for particle, versus0.03029411765 for electropolish:
a factor545.29411765 discrepancy. It also disagrees with corrosion's tuple.
Therefore there is no exact Lc,Tc pair which maps all current particle groups
to all electropolish dimensional coefficients. Speed2 preserves the same ratio.
This is a different dimensionless physical regime, not merely changed units.

## Conditional inverse maps and physical time

Fixing A,omega,alpha,D to a chosen material leaves Lc,Tc,L_eff determined:

$$
L_c^2=\frac{\alpha_\phi}{\omega_\phi}\frac{B_2}{B_3},\qquad
T_c=\frac{B_{\rm CH}L_c^2}{D},\qquad
L_{\rm eff}=\frac{B_2}{\omega_\phi T_c}.
$$

| Fixed anchors | Lc | Original Tc | Speed2 Tc | Fitted actual matrix mobility |
|---|---:|---:|---:|---:|
| Electropolish A,omega,alpha,D | 51.3178597um | 247.860963s | 495.721925s | 1.83387271e-9 |
| Corrosion A,omega,alpha,D | 102.635719um | 991.443850s | 1982.887701s | 4.58468177e-10 |
| Electropolish A,omega,D and Lc100um | 100um | 941.176471s | 1882.352941s | 4.82954545e-10 |

Mobility units are m³/(J·s). Third row additionally fits alpha=9.77777778e-5J/m;
it does not preserve the electropolish alpha or mobility. First row preserves
its energy/diffusion constants but requires mobility545.294times smaller than
electropolish's1e-6. Its L0 before kappa2 is9.16936354e-10. Each mapped speed2
case has the same physical mobility and length as its original and twiceTc.
Every inverse map is substituted back into all four groups and checked.

If instead one insists on electropolishL1e-6 andomega1.76e7, the AC barrier
alone gives Tc8/(L*omega)=0.454545s (speed2:0.909091s). This is NOT a valid
complete interpretation: combined with the matching alpha/interface width,
CH would require Tc247.860963s. Equivalently,D would have to be4.635e-7m²/s,
not8.5e-10. Reporting only the AC-derived time would hide the inconsistency.

To use the entire electropolish material tuple at Lc100um, choose a physical
horizon first.5s implies the electropolish column above;10s implies
(B_CH,B1,B2,B3)=(.85,1070,176,.02575). The consistent initial coefficient is
k=sqrt(omega/(2alpha))*Lc=58.4591800, not30. With kappa2 retained as a base
mobility parameterization, stored n_aci=B_i/2. These are calculated candidates,
not new presets, solved trajectories or selected training defaults.

## What this establishes about failed training

The user's concern identifies missing physical calibration in the pilot;
the earlier dimensionless groups were selected before assigning an SI tuple.
They are mathematically consistent with other positive material constants,
but cannot be called the same electropolish material by choosing a time scale.
The uniform2x change itself is correct. Both original and speed2 occupy the
same scale-invariant regime, and their independent references evolve normally.
Consequently this audit does not prove that calibration mismatch caused the
abrupt PINO failure, nor that adopting the electropolish tuple will restore
convergence. The latter is a new physical problem requiring fresh references.
Keep the original low-error pilot result scoped to its dimensionless model.

## Reproducibility, changed files and verification

[CPU audit script](particle_corrosion_dimensional_audit.py) and
[numeric receipt](particle-corrosion-dimensional-audit.json) contain the actual
configs, conditional maps and assertions. From the repo root:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m docs.tasks.archive.particle_corrosion_dimensional_audit \
  --output /tmp/particle-corrosion-dimensional-audit.json
```

Besides scalar groups, an independent dimensional SI residual on a synthetic
four-node positive graph matches production particle residuals after multiplying
byTc: max absolute AC1.55e-15,CH1.07e-14. Production residuals with doubled
time derivatives and coefficients equal2xoriginal; independent solver RHS and
Jacobian also scale2x. This is an algebra/unit test, not an empirical
convergence claim. It runs CPU-only and does not train or integrate a reference.

Changes: three example README clarifications/unit corrections, one time-unit
docstring, this task/receipt/reproduction script, PROGRESS and DECISIONS.
No loss, PDE, configuration, geometry, reference fields, checkpoints or Git
history changed in this audit. Prior uncommitted experimental changes remain.

Syntax/JSON and registered external-directory checks pass. Local document and
artifact links pass after excluding a math expression mistakenly matched by
the initial generic link regex; no reference path was changed to hide a failed
check. git diff --check passes. The subsequent user request explicitly
authorizes a new dimensional-first physical-case implementation; see the new
active physical-corrosion task rather than continuing this completed audit.
