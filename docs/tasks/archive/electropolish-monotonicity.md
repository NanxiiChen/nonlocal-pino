# Electropolish phase monotonicity diagnosis

## Goal and acceptance criteria

User observes apparent liquid-to-solid conversion in the main figure. Determine
whether physical phi (metal=1) is actually non-increasing at fixed points; quantify
PINO and reference reversals and total solid-area changes. Check existing refined
references and decompose the implemented AC RHS before attributing causes.
Distinguish numerical accuracy against the model from corrosion irreversibility.
Do not silently change PDE, add an inequality constraint or retrain.

## Context

Main figure uses current selected rolling 10k checkpoint and samples 0/21/42/63,
local FD101 references, 25 saved times including early snapshots, five shown.
Source paths: artifacts/electropolish_main_figure.json. Existing 64-input evaluation
and four spatial refinements can supply evidence without a new solve. No Git
operation authorized. Other local work and paused process remain untouched.

## Status

Completed diagnosis, 2026-11-12. No training, inference, new reference solve,
PDE/configuration change, clipping, or Git operation. User's follow-up asks
whether reference agreement proves physical correctness: explicitly answered
no; it establishes agreement with a model that lacks unilateral dissolution.

## Findings

Physical phi=1 denotes metal. A tanh output bounds phi but does not enforce
its time derivative's sign. At fixed nodes, define upward excursion as the
maximum of phi(t_j)-phi(t_i) for saved t_i<=t_j. Include all25saved times,
not only the five main-figure columns; no display interpolation or smoothing.

| Method,64inputs | Excursion >0.01 | Ever below then above phi=0.5 | Increasing solid area |
|---|---:|---:|---:|
| Selected PINO |62|57|0|
| Independent local FD101 |62|58|0|

Solid area here is the tensor-trapezoidal integral of phi; it is not total
solute mass or an experimental dissolution-rate measurement. Counts concern
saved snapshots, not a proof of behavior between snapshots.

For figure row1(sample0), the interior point(x*,y*)=(0.46,0.23) has PINO
phi0.07990→0.89491 from t*=0→0.5 and FD101 phi0.10688→0.82444 from t*=0→0.6.
Thus it really crosses the diffuse-interface midpoint, rather than merely
showing tiny noise in a bulk phase. T_c=5s and L_c=100micrometres.

Maximum upward excursion over each whole space-time array:

| Input | PINO | FD101 | FD201 | FD401 | Native graph101 |
|---|---:|---:|---:|---:|---:|
|0|0.8150|0.7176|0.7436|0.7468|0.7087|
|21|0.7802|0.8030|0.8043|0.8064|0.7993|
|42|0.7392|0.7079|0.7175|0.7244|0.7018|
|63|0.2916|0.3096|0.3077|0.3070|0.3005|

Different methods can attain maxima at different nodes/times. This supports
a model-level effect, not a claim that all discretization error is zero.
The local reference is centered FD with mirrored side Neumann treatment,
not FEM. Saved reference metadata documents tolerance tightening and maximum
time-step halving; sample0FD401 reports3.18e-9for its existing temporal metric.

## Driving-term diagnosis

The exact declared dimensionless local equation is

$$
\phi_{t^*}=N_3\Delta\phi-N_2g'(\phi)+N_1\Delta c\,q\,h'(\phi),
\quad q=c-\Delta c\,h(\phi)-c_{Le},
$$

where \(h=3\phi^2-2\phi^3\), \(g=\phi^2(1-\phi)^2\),
\(N_1=535\), \(N_2=88\), \(N_3=0.012875\),
\(\Delta c=1-0.03566433566433566\), and \(N_{CH}=0.425\).
These are computed from the saved dimensional coefficients, not newly chosen.

At sample0FD401, the above interior witness and t*=0.1:

| AC RHS contribution, per unit t* | Value |
|---|---:|
| N3 Laplacian(phi) |18.72413|
| -N2 g'(phi) |-11.57562|
| N1 dc q h'(phi) |-4.78985|
| Sum |+2.35866|

Here q=-0.00674343: the chemical term favors dissolution, but the combined
gradient/double-well interfacial-energy contribution is larger and positive.
The positive local rate is allowed by the model. It is not evidence of a
physically justified cathodic deposition mechanism. Nor does gradient-flow
structure guarantee pointwise irreversibility; open concentration Dirichlet
boundaries also preclude assuming an unqualified closed-system energy law.

An independent reflected-stencil reconstruction was compared with the actual
`operator_reference.make_system(...).rhs` using the production sparse negative
Laplacian at six saved times and4949free FD101 nodes: maximum absolute phase
RHS difference1.281e-13. Equation signs and coefficient scaling agree.

## Physical interpretation and limits

The observed valley filling is consistent with the implemented AC/KKS model.
For an intended anodic dissolution-only experiment, its physical suitability
still needs assessment: interfacial-energy/chemical-force balance and whether
unilateral kinetics is appropriate. Reference refinement and low PINO L2 do
not settle that question. Do not label the effect automatically correct physics
or blame PINO alone; do not silently add a sign constraint or clip the curves.
Any changed kinetics needs a checked formulation and new reference solutions.

## Diff-level changes and reproduction

- `scripts/audit_electropolish_monotonicity.py`: raw-array CPU audit of64inputs,
  four selected refinement cohorts, signed FD401AC terms, and diagnostic figure.
- `artifacts/electropolish_monotonicity.json`: direct input/output paths,
  summary, refinement and validation evidence; full per-input data is external.
- Example README: explicitly document the model's irreversibility limitation.
- DECISIONS and PROGRESS: preserve the modeling qualification and completion.

```bash
.venv/bin/python scripts/audit_electropolish_monotonicity.py \
  --manifest artifacts/electropolish_main_figure.json \
  --output-dir /root/autodl-tmp/tf-logs/electropolish2d/monotonicity-20261112
```

External directory contains `electropolish2d_monotonicity.json`, matching
PDF/PNG, and `rhs_validation.json`. Main figure manifest records all input
paths without hashes. New figure shows sample0at the above fixed point (left)
and integrated solid area normalized by each method's own initial area(right).
FD401and native traces are independent numerical solutions. Curves connect
saved values; there is no fitted smoothing. PDF170x61mm, NPG colors,
6–7pt embedded Helvetica TrueType, editable vector text/curves; no images
embedded in PDF. Font/dimension extraction and PNG visual inspection pass.

Full audit executed successfully, exact input/time/grid checks pass,
independent production-RHS comparison passes, and `git diff --check` passes.
Unrelated Melting changes and paused process group618216 were untouched.

## Blocker

None.

## Next action

Await user direction on whether to retain this gradient-flow benchmark with
an explicit limitation or investigate dissolution-only model changes. The
present numerical diagnosis is complete; no changed physics has been selected.
