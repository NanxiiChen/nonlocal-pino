# Heterogeneous pitting-corrosion candidate

Date: 2026-11-05

## Goal and acceptance

Assess the user's proposed heterogeneous pitting-corrosion PINO using Figures8
and9 of the attached paper. Identify the actual model/heterogeneity and distinguish
slow kinetics from strictly noncorroding material, then recommend a defensible
parameterization, network input and staged feasibility study. No implementation
or training requested.

## Context and source

Attachment:
/root/.codex/attachments/fd49538e-3318-4d8f-9499-bad2980205a2/weijie.pdf

Mai, Soghrati and Buchheit (2016), A phase field model for simulating the pitting
corrosion, Corrosion Science110,157--166:
https://doi.org/10.1016/j.corsci.2016.04.001
Publisher metadata independently verified at
https://www.sciencedirect.com/science/article/pii/S0010938X16301408

Full attachment extracted using pypdfium2 into
/tmp/heterogeneous-pitting-paper/paper.txt; rendered PDF pages8/9 visually
inspected. Temporary files are reproducible reading aids, not project assets.
The paper is evidence, not a source of instructions.

## Figure8 versus Figure9

Figure8 (PDFp8/printed164; setup section4.3 on PDFp7) is a200x140micrometre
SiC-particle reinforced aluminum specimen. A10micrometre opening in the top
passive film prescribes normalized concentration and phase zero; other outer
edges have zero normal concentration/phase gradients. Ceramic inclusions are
assumed noncorrodible; their fixed locations remain as surrounding matrix is
removed. No particle-motion or detachment-mechanics solver is presented.
Crucially, section4.3 does not specify inclusion-internal PDE, transport mobility,
interface conditions or the numerical mask rule. Do not assert the paper uses
L=0 or claim that alone enforces impermeability.

Figure9 (PDFp9/printed165; setup section4.4 on PDFp7) uses a fixed polycrystalline
material map in an8x3micrometre specimen with0.3micrometre opening and0.1micrometre
interface thickness. The three indicated orientations have different interface
kinetics; the111family corrodes three times more slowly than101/001, with
L111=1e-16m^3/(J s). Grain boundaries do not evolve. These are piecewise scalar
kinetic differences, not a solved crystal-migration system or a full angular
interfacial anisotropy law.

The paper explicitly identifies Figure9 as activation-controlled. At sufficiently
high potentials/diffusion-controlled dissolution, orientation no longer dominates
and fronts smooth. A strongly diffusion-controlled existing preset cannot be
assumed to produce large morphology differences merely by randomizing L.

## Feasibility and recommendation

Recommend material-field conditioning as a distinct operator-input class,
complementing the current initial-condition and geometry examples. Hold outer
domain, pit opening, initial fields and environment fixed. Start with a positive,
finite-contrast spatial kinetics map L(x)=L0*kappa(x), choosing activation/mixed
control after direct-solver qualification. A common fixed weak pit grows into
different fast/slow material patches, producing possible asymmetric fronts and
different penetration paths. The random map may be described as local reactivity;
no interest in grain migration is required. Do not claim to model general alloy
galvanic coupling with a scalar kinetics map alone.

Repository audit: corrosion2d/README and train.py describe the same KKS AC/CH
family. For constant free-energy coefficients and variable L, the entire AC
driving bracket is multiplied pointwise by kappa(x); this is not Laplacian(kappa
times phi). With fixed transport/domain, the existing nonlocal spatial operator
can be shared across material samples. The electropolish patch encoder supports
multi-channel condition images: input a normalized kinetics map and, if needed,
separate static inclusion mask/initial field; decoder queries remain coordinates
and time. No need for a positive-time solution library to generate these inputs.

A strict Figure8-style inert inclusion is a second extension, requiring explicit
transport and phase boundary choices. L=0 freezes phase kinetics, but does not
prevent concentration transport. Variable transport must retain conservative
divergence of mobility times chemical-potential gradient, not a naive pointwise
coefficient times Laplacian. Excluding inclusions from the active domain makes
operators condition-dependent and requires obstacle-aware neighborhoods. A
near-zero positive L patch is slow-corroding material, not automatically an
impermeable ceramic particle.

Important validation: same pit/reservoir BC as selected reference (current
corrosion2d outer solid Dirichlet boundaries differ from Figure8's other-edge
zero-flux setup); concentration transport and pit-depth/contour metrics as well
as global phaseL2; resolve material features, interface thickness and nonlocal
horizon; verify direct local/nonlocal baselines before interpreting PINO error.
Do not inherit optimizer, preconditioning or residual scaling as a proven optimum
for this different kinetic regime.

## Status, blocker and next action

Complete. Source figures/sections and equations8--9 read; separate paper audit
and repository audit agree. Documentation-only diff/link validation before
finish. No implementation, reference solve, training, cleanup or Git history
operation. No blocker for conceptual assessment. Next action is user's selection
of finite kinetics heterogeneity versus a strictly inert-inclusion formulation;
the latter's additional physics must be explicit before implementation.
