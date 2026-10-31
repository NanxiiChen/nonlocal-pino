# Encode random initial CH fields on the confined irregular domain

Date:2026-10-29. Status: COMPLETE (design only); no blocker.
User defers coupled two-phase work and asks how to condition an encoder on
random initial fields in the prior irregular coarsening domain. Acceptance:
inspect geometry/input arrays, compare masked patches vs unstructured tokens,
recommend a compatible encoder and identify fixed/variable-geometry scope.
No implementation/training/default change or Git history authorization.

Current domain: fixed ellipse minus two circular holes. Physical sampling uses
triangle centroids and corresponding cell areas, not regular pixels or vertex
mass. Reference /root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz
has3682points/volumes,1932meshvertices/3682triangles; volume min8.65356e-5,
max2.26324e-4,total.6013278963. Different initial fields can share point/area/
operator/BC metadata. Existing single-IVP default uses30time windows, phase-only
prediction and exact chemical elimination/physical-volume mass projection.

Next: propose pointwise input lifting and small latent cross-attention encoder,
preserving shared spatial preparation/time JVP; compare minimally invasive
mask-aware image patches and geometry-aware local grouping. Check primary
Perceiver/GINO/GNOT sources; avoid implying encoding solves long-time training.

Recommendation:
- Tokens need not be regular image patches. Direct input samples are
  (x_i,y_i,phi0_i) with fixed physical areas m_i. Shared pointwise lifting and
  coordinate positional encoding are permutation-equivariant. A small latent
  query array (illustratively128tokens) cross-attends to all input points, then
  self-attention runs only over those latents. Query-coordinate spatial decoder
  attends to the resulting condition tokens; time modulation/JVP remains split.
  Keep learned input lifting/preparation inside parameter AD on every update;
  no cache across optimizer updates. Raw decoder queries remain coordinate/time.
- For3682inputs, N*128=471296attention scores vs N*N=13557124, about28.8xless
  at that stage for fixed heads/width; include latent M*M and decoder Nq*M
  work, activations/backward/optimizer before any actual GPU memory claim.
  Existing conditional model is image-specific; new point encoder is proposed,
  not already supported by its current reshape/patch_projection/encode API.
- Use area-aware aggregation on irregular sampling. One proposed normalized
  integral-attention quadrature is z_k=sum_i m_i exp(s_ki) V(h_i) /
  sum_i m_i exp(s_ki), equivalently softmax(s_ki+log m_i). This follows by
  quadrature of normalized domain integrals, is not a claimed verbatim formula
  from cited models and does not prove mesh-independent accuracy. Keep exact
  volume-weighted mean as explicit metadata/physical mass target; normalized
  latent attention need not preserve mass by itself. Do not confuse passing
  weights as an input feature with weighting the aggregation measure.
- Fixed geometry: coordinates/areas/anchor metadata/neighbor lists/operators
  precomputed and shared across samples; only initial field values vary.
  Boundary/SDF tokens become useful for future geometry families but geometry
  transfer is not established by fixed-domain training. Holes are absent from
  the physical input support, not filled with an actual phase value.
- Alternative minimal-change control: sample initial field on a bounding-box
  image, input mask*phi0 and domain mask (optional actual boundary distance),
  retain regular patches. Partial patches are valid with explicit occupancy;
  all-void patches contribute no condition data. This is an encoder embedding,
  not changing physical domain/quadrature or extending CH through holes. Need
  verify image resolution/interpolation preserves sharp interfaces/wall detail.
- Alternative mesh patches: group nearby points, area-pool local lifted features
  and patch position/area into tokens. Visible/geodesic neighborhoods can avoid
  joining points across impermeable holes when imposing a local geometric bias.
  Global attention is not itself mass transport and need not prohibit every
  across-hole information link; physical connectivity stays in the PDE operator.
- Neither encoder nor image mask enforces Neumann conditions; keep current
  native closed pair flux and distinguish local wall normals as before.
  Encoder choice does not solve prior global-time CH optimization failures.
  Retain the time-window route; encoding each window's start field suggests a
  short-time evolution operator, with rollout accumulation/distribution shift
  separately validated. No autoregressive training scheme claimed implemented.

Primary sources checked:
- Perceiver https://arxiv.org/abs/2103.03206 : asymmetric cross-attention into a
  small latent array, including point-cloud inputs.
- GNOT https://proceedings.mlr.press/v202/hao23c.html : operator transformer
  handling irregular meshes/multiple functions.
- GINO https://arxiv.org/abs/2309.00583 : graph mapping irregular grids to/from
  regular latent grids, demonstrates a further route; not implemented here.
These support architectural feasibility, not pure-physics CH convergence or
matching all original architecture details in the proposed adaptation.

Task-end: inspected actual geometry/quadrature, saved array shapes/areas,
window formulation and shared image-conditional model. Documentation-only;
no code tests, training or numerical/default changes. Diff whitespace checked;
no binary artifact generated. Archive this design, update PROGRESS/DECISIONS.
Next action only if requested: add a separate raw point-condition encoder with
explicit area/fixed-shape contracts, verify permutation/gradient/time-split
behavior, then pilot random initial fields with current physical operator.
NSCH work remains deferred; no Git operation.
