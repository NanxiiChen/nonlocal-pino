# Intuitive residual-transformation explanation — completed 2026-10-25

## Goal and outcome

Created two inline Chinese visuals and an accompanying explanation for the
user's request to understand residual transformation without a long formula
sequence. No training, PDE, default or user-checkpoint changes. No hard IC is
introduced. Current corrosion original/KKS/pre_scale defaults remain unchanged.

## Visual paths and interpretation

- Interactive compression/residual schematic:
  /root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/phase-residual-intuition.html
- Measured corrosion AC allocation:
  /root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/corrosion-loss-focus.html

These are literal fragments in the authorized thread-scoped directory, intended
for inline content references, not repository runtime modules or Library files.
No CDN, remote data or browser API call is needed. Colors follow host tokens.
Native controls persist model-relevant choices through the optional host state
API. They have useful default content without any user interaction.

The first diagram separates two quantities. A curve/markers show a fixed
internal-output shift0.5 and its physical-phase response. The bars separately
fix the normalized latent PDE residual to1 and show original a, transformed1,
and bounded a*(1+eta)/(a+eta), with a=sech^2(u), eta0.1. They are schematic,
not simulated training or physical-field errors. Both maps are selectable:
phi=tanh(u) for melting and phi=(1+tanh(u))/2 for corrosion. The latter uses
the already-checked interface-centre residual normalization r_phi/[4phi(1-phi)].
Thus the same a describes both bar comparisons despite different output ranges.
The bounded comparison is a distinct objective, not the exact latent residual.

The second diagram uses the frozen accurate corrosion checkpoint probe from
[mechanism report](corrosion-latent-mechanism.md) and its JSON. On all free
nodes at10/25/50 s, original absolute AC residual is91.46% interface/8.54% bulk;
transformed is0.53% interface/99.47% bulk. Each objective total is normalized
separately to100%; this is not an absolute-loss comparison, a field-error
fraction, or an assertion about optimizer direction. Network outputs are held
fixed. Bulk phase classification uses reference phi<0.01 or>0.99.

## Explanation boundaries

Use the PDE-accounting analogy: the residual measures imbalance of time,
diffusion, reaction and forcing terms. Output tanh compresses near pure phases;
changing the residual additionally changes how imbalances are scored. Finite
latent-variable zero equivalence does not guarantee equal finite-budget
minimizers, gradients or physical relative L2. The tanh output alone is not the
same objective as an inverse-Jacobian residual. Do not claim all gradient
saturation vanishes, invent future interface motion or imply reference labels
are supplied during training.

Melting's matched controls showed improvement under their shared representation.
Corrosion additionally couples a learned concentration and stiff reactions;
amplifying pure-phase imbalance may overemphasize already accurate physical
tails and concentration errors. The bounded pilot improved seed42 and worsened
seed7; removing scales failed in both formulations. Link that experiment when
numerical outcomes are needed rather than reinterpreting this diagram as proof.

## Verification and limits

Both fragments pass the visualize renderer's official _read_fragment check,
HTML parsing, unique-ID/literal-markup checks and1 MB limits (10.5/2.0 KB).
A local Node VM with synthetic DOM executes the real interaction script:
12 states across both output maps and u=-3.5,-3,0,2,3,3.5; exact schematic
residual identities to displayed precision; state save/restore handlers; and
four widths320/360/736/1024. Marker/frame bounds and approximate text bounds
pass. No initial state save occurs. Actual font metrics and browser runtime
were not verified: no local browser/Playwright/Puppeteer was available. Do not
claim real screenshots or browser-layout verification. The curve's responsive
renderer also measures actual text bounds at client runtime and drops optional
annotations on collision. No network dependency is required.

Measured fractions were independently read from the archived JSON; interface
plus bulk fractions sum to1 within1e-7. Node verification outputs are archived
in [interaction-check results](residual-visual-explainer-check.json). Verification
script was /tmp/check-residual-visual.cjs; it is a session check, not a production
test dependency. No Python numerical suite is needed for explanation-only work.

## Session diff, task-end and exact continuation

Added the two thread-local fragments, this archived task and check JSON;
updated PROGRESS/DECISIONS. Corrected a stale PROGRESS sentence saying bounded
weighting was still untested by pointing to the subsequent completed experiment.
No production API or example README change. Final diff/whitespace checks pass
at session end. No blocker or remaining work for this explanation task. For
future edits, reload the full visualize skill before authoring after compaction.
