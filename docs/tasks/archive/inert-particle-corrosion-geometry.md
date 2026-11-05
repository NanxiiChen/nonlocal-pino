# Inert ceramic particles as parameterized corrosion geometry

Date: 2026-11-06. Status: complete conceptual assessment.

## Goal and acceptance

Preserve the completed heterogeneous-material comparison. Explain whether the
paper's noncorroding SiC particles in dissolving aluminum can be modeled by
excluding their interiors, and which boundary conditions that requires.
Distinguish source statements from proposed assumptions and no-reaction from
no-transport. No implementation, training, deletion or Git operation requested.

## Context

Previous results: [matched optimizer/AD study](heterogeneous-corrosion-local-ad.md).
Paper audit: [heterogeneous pitting candidate](heterogeneous-pitting-candidate.md).
Attachment: `/root/.codex/attachments/fd49538e-3318-4d8f-9499-bad2980205a2/weijie.pdf`.
Mai et al. (2016), DOI10.1016/j.corsci.2016.04.001, section4.3/Figure8.

## Source and independent check

Read section4.3 of the supplied PDF (extracted text lines604--622). It explicitly
assumes noncorrodible ceramic particles and states exterior-domain zero-flux
conditions; it does not specify the inclusion-internal equations, particle
surface conditions or whether numerical interiors were removed. Publisher
metadata/abstract verified at https://doi.org/10.1016/j.corsci.2016.04.001;
publisher full-page fetch was denied, so exact section claims use the attachment.
The earlier comparison, checkpoints and defaults remain intact.

## Proposed interpretation, not an implementation decision

For fixed, impermeable particles without surface electrochemical reactions,
solve the existing metal/electrolyte KKS fields only on the exterior box minus
particle interiors. These excluded regions still represent physical ceramic,
not electrolyte-filled cavities. The computational domain is fixed in time for
each sample; the metal/electrolyte phase interface evolves inside it. Random
particle locations, sizes and shapes therefore define a geometry-conditioned
operator with a fixed initial pit and homogeneous matrix kinetics.

Transport boundary: zero normal species flux. For the current constant-mobility
model this is normal derivative of q zero, with
q=c-Delta_c*h(phi)-cLe, proportional to chemical potential. It is not generally
the same as setting the total concentration derivative zero in isolation.
With the separate neutral phase condition normal derivative of phi zero,
the two concentration forms are equivalent for this KKS model.

Phase boundary: variation of bulk gradient energy alpha*|grad(phi)|^2/2 plus
wall energy gamma_wall(phi) gives alpha*normal_derivative(phi)+gamma'_wall(phi)=0.
A constant wall energy yields homogeneous Neumann. This imposes neutral wall
preference and an orthogonal diffuse interface where its gradient is nonzero;
it is an additional assumption, not a consequence of noncorrodibility.
General phase-field wall-energy support: Yue, Zhou & Feng (2010),
https://personal.math.ubc.ca/~jfeng/Publications/PDFs/10_JFM1.pdf.
Independent agent audit agrees with the KKS and variational reasoning.

Only setting kinetic mobility L or kappa to zero freezes reaction but leaves
CH transport, so it does not implement an impermeable particle. A nonlocal
implementation needs geometry-dependent operators with particle-interior nodes
excluded and interaction segments crossing solid particles removed; endpoint
masks alone can transmit through a thin obstacle. Conservative closure and
local-limit boundary consistency require validation, not a claim of exact
finite-horizon equivalence to local Neumann. Existing geometry point-token
encoder and online operator-pool workflow can be reused conceptually.

Fixed particles omit detachment/motion, non-neutral wall energetics and galvanic
surface reactions. These are explicit limitations of the suggested first model.

## Completion and resume point

Conceptual question answered and previous results retained. No code/config,
training, solver, artifacts or Git history changed. Only task memory and the
decision log were updated; link/existence and diff checks apply. No blocker or
required next action; implementation awaits a subsequent user request.
