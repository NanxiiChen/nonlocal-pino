# Design a physically meaningful richer random-wall channel family

## Goal and acceptance

User asks whether the current two-amplitude geometry has insufficient diversity
and what random upper/lower wall parameterization would increase difficulty
while retaining physical meaning. Check existing geometry/scales and primary
literature. Recommend a controlled random smooth-channel family, explain expected
effects and model limitations, and provide a geometry-only preview if useful.
No PDE, optimizer/default, reference, training or Git changes in this design task.

## Context

- Prior integral control: [completed comparison](channel-integral-nonlocal-control.md).
- Current fixed bump center2, halfwidth.9; independent upper/lower amplitudes
  [.06,.20]. Nominal .18/.12; R0=.22,H=1,epsilon=.05,T=6,fixed inlet flux.
- Hence all samples share one broad aligned constriction; throat heights lie
  [.60,.88] and initial core diameter .44 is smaller than every throat.
- One nominal solved geometry does not measure response diversity of the entire
  prior or establish that the neural encoder ignores geometry.
- Preserve all accumulated uncommitted implementation/results and saved paths.

## Status, blocker and next action

Complete design recommendation and geometry-only preview. No blocker or queued
training/solver action. Numerical defaults and all previous results are unchanged.

## Recommendation

Frame the next problem as droplet transport in a geometrically disordered smooth
microchannel. Independently sample upper/lower stationary inward wall profiles;
control roughness amplitude, axial correlation length and minimum aperture.
Use low-dimensional smooth B-spline/random-function coefficients (approximately
6–8 per wall initially), not only two amplitudes multiplying one shared bump.
Peaks, widths and upper/lower alignment vary naturally; several moderate
contractions/expansions can produce different deformation histories. No separate
eccentricity category or new physics-specific attention is needed.

- Random position alone may mainly shift encounter times; aperture, contraction
  slope/width and spacing between successive throats change deformation forcing.
- Keep initial droplet, inlet flux, materials and wall BCs fixed. At fixed flux,
  horizontal cross-section mean velocity is Q/h; narrower gaps strengthen local
  flow. Independent wall profiles also bend the passage and permit lateral motion.
- Choose correlation lengths comparable to droplet diameter; avoid tiny roughness
  below interface/operator resolution. Bound gap, slopes and curvature; preserve
  straight inlet/outlet and a flat upstream interval around the fixed diffuse IC.
- Stationary walls vary between samples, not over time. Existing NS–CH remains
  the model. Initial regime should emphasize connected-droplet transport and
  varying stretch/tilt/recovery, before pursuing pinchoff or trapping.
- Surface-tension-driven recovery can make late shapes similar even when prior
  histories differ. Compare centroid paths, deformation/angle versus time and
  pressure drop, plus interface/shape errors; do not assess diversity only from
  terminal global phi L2 or raw sample count. Align shapes by centroid when
  separating shape diversity from translational/timing differences.

The strongest direct physical analogue is a corrugated/constricted microfluidic
passage, with geometric disorder representing nonuniform fabrication or a family
of pore-throat shapes. Rough-fracture transport is a broader qualitative analogue;
actual wetting/trapping claims require wall wettability/thin-film/3D checks beyond
the present neutral-wetting matched-fluid 2D model. Do not label this a validated
fracture, biological-cell or elastic-membrane simulation.

## Primary literature checked

- [Trégouët et al. 2018, transient deformation near a microfluidic constriction](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.3.053603):
  experiments/numerical flow stress and transient droplet deformation; supports
  the microfluidic motivation, not the specific proposed stochastic prior.
- [Salari et al. 2020, expansion-mediated breakup](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.5.013602):
  geometry/flow-regime-induced breakup exists, but its high-inertia discussion
  does not establish breakup for our Re1/parameters; a later difficulty level.
- [2023, wetting transition and trapping in a microfluidic fracture](https://pmc.ncbi.nlm.nih.gov/articles/PMC10235991/):
  experimental roughness/wetting morphology, with distinct physical ingredients
  from the current single-droplet neutral-wetting channel.

## Geometry-only preview and verification

Added docs/tasks/channel_random_wall_preview.py; executed once on CPU and
visually inspected its PNG. No fields were solved or predicted. Figure shows
the actual nominal single bump and5legal illustrative random-wall samples,
all with the same initial circle. Eight positive cubic B-spline coefficients
per wall, smooth C2 tapers, active x interval[1.15,2.75]. The actual diffuse IC
ends before x1.15. Proposal-only sampled-gap bounds[.55,.70], slope bound1.25;
5samples accepted from414cheap curve draws. These are illustrative choices,
not a qualified training prior, operator validity gate or reference benchmark.

Artifacts: /root/autodl-tmp/data/channel_nsch2d/random-wall-design-20261102-01/
(geometry-preview.png/.pdf, geometry-preview.json, preview-source.py).
Plots/data remain external; no generated binary was added to the repository.
The script verifies finite positive gaps, straight upstream portions and IC
separation. AST, JSON/provenance/figure and git diff --check pass; no code tests,
JAX GPU training, FEM, Git/history or cleanup operation.

## Future implementation boundary

The existing point-token encoder can stay generic: wall coordinates/normals/type
and domain coordinates/fixed phi0. Geometry coefficients generate these data;
no need to put a fixed list of throat identities into the encoder. The mapped
channel coordinate remains available as long as upper>lower and walls are graphs.

Implementation will need a single JAX-compatible wall value/slope definition
shared by sampling, lifting, normals, operator construction and FEM geometry;
the existing two-amplitude shape/identity/sampling contracts must be extended.
Preview cubic splines are C2, so production regularity and curvature need explicit
validation (or a smoother spline family). Recheck spatial resolution, integral
horizon/capacity and rank/reproduction throughout the wider prior; legal curve
plots alone do not qualify the current128-neighbor operator or old FEM files.

First inspect 3–5 physically contrasting numerical trajectories from this family
(narrow aligned, staggered, repeated throat) and confirm separation in deformation
history/pressure response at common material/flux parameters. Then qualify the
random-wall neural-operator experiment with new independent reference gates.
This design task does not authorize an automatic expensive solve/training queue.
