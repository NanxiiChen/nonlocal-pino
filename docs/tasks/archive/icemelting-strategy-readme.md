# Document the current melting strategy

Goal: update examples/icemelting3d/README.md with the current production strategy
and clear separation between defaults, controls, research pilots and evaluation.

Acceptance: README matches the current IC-only latent field and transformed
residual, default loss/sampling/optimizer/configuration, analytic versus graph
L2 definitions, physical time conventions and direct-path loading. Preserve
stable problem/solver guidance and link run-specific evidence instead of
duplicating experimental results. Documentation only; no training/code changes.

Context: existing README describes much of the method but puts historical pilots
before production and mixes a 5 s FEM parameter table with 4 s training defaults.
All defaults and formulas have been checked against phase.py, losses.py, train.py,
evaluation.py and both configuration files.

Status: completed. Blocker: none. Next: keep the README synchronized with future
configuration or formulation changes; run-specific results stay in task archives.

## Task end

Updated examples/icemelting3d/README.md with a current-strategy overview and
organized the IC-only formulation/default production settings before historical
research controls and the independent FEM workflow. Documented exact IC, zero
correction initialization, learnable initial derivative, transformed versus
original residuals, stable log-domain edge ratios, and PDE-only core MAE.

Added selectable-control and base/cold-start configuration tables, network and
Fourier settings, core/halo pool sampling, SOAP/learning-rate/precision settings,
disabled interface/causal weighting, boundary handling, and evaluation-only use
of reference fields. Detailed terminal versus pooled phase L2, sampling cadence,
full-grid evaluation, approximate analytic versus direct graph/FEM references,
and direct-path loading without hash requirements.

Corrected the prior mixed parameter table to production4 s/128^3 nodes/h=1/127/
eps*=0.0225/lambda*=0.20. The FEM64-element/5 s defaults and mesh-derived thickness
are described separately. Normalized existing math to repository Markdown rules.
Archived validation and artifact-index links carry transient numerical evidence
and sota locations instead of copying run results into this stable README.

Verification: compared statements against current phase/loss/train/evaluation,
optimizer and train/solver configs; all12 local links resolve; code fences,
display/inline math delimiters and required strategy sections validated;
git diff --check passed. Documentation-only change: no tests, compilation,
training, checkpoint changes or Git history operations were needed.

Memory files updated: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
