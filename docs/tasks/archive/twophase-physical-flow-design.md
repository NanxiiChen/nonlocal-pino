# Physically interpretable random flow families

Date:2026-10-29. Status: COMPLETE (proposal only); no blocker.
Goal: replace arbitrary Fourier-coefficient randomization in the proposed
prescribed-flow CH operator with an interpretable physical flow scenario.
Acceptance: compare candidate scenarios, respect current periodic boundary and
one-way coupling, state meaningful random variables and necessary constraints.
No implementation, training/default changes or Git history operations requested.

Context: prior recommendation fixes initial droplet and conditions on velocity
images; current twophase2d is periodic prescribed-flow CH, not coupled NSCH.
Fourier remains a numerical representation and need not define the input prior.
Next: check primary droplet/straining/vortex literature, recommend an initial
flow family and distinguish physical motivation from experimental fidelity.

Recommendation:
- A fixed droplet exposed to a controlled counter-rotating vortex pair, inspired
  by stirring and vortex/drop encounter studies. Randomize interpretable
  circulation magnitude Gamma, Gaussian core radius a, pair separation d and
  lateral offset b relative to the initial droplet. Keep the known temporal
  actuation envelope fixed initially; changing it later requires input that
  identifies the entire forcing law, not only a snapshot.
- Periodic construction: omega=Gamma*(G_a(x-c_plus)-G_a(x-c_minus)), G_a a
  periodized unit-integral Gaussian. Equal opposite circulations imply zero
  cell-mean vorticity, needed for periodic -Delta psi=omega; choose psi mean0
  and v=(psi_y,-psi_x), with fixed known mean velocity if present. Mixed
  derivatives cancel divergence. Do not randomize independent net circulations
  without explicitly addressing periodic compatibility.
- Fourier is only a numerical Poisson/periodization tool here, not the random
  prior. Images of physical velocity remain encoder conditions; coefficients
  generate a continuous finite-dimensional physical family, not arbitrary flow
  coverage. Parameter support needs pilot resolution/strain checks, not numbers
  borrowed from the illustration. Broad enough d/b variations avoid a mere
  amplitude sweep or rotated copies.
- The recommended first variant is externally prescribed controlled stirring:
  vortex motion/decay and phase feedback are not solved. A static maintained
  Gaussian pair is not an exact freely evolving multi-vortex NS solution. No
  physical breakup threshold, viscosity-ratio, Weber/capillary-number validation
  follows from the current one-way CH model. Coupled NSCH is a separate upgrade.
- Four-roll-mill droplet stretching is a second strong physical motivation, with
  random extension/rotation rates or actuation settings. A literal device has
  walls/rollers and needs a different flow/boundary setup; label a periodic
  analogue as inspired, never an exact reproduction. Prefer the vortex family
  for current periodic implementation; do not apply nonperiodic linear strain
  directly to the present periodic domain.

Primary sources checked:
- On the dynamics of vortex-droplet co-axial interaction, JFM:
  https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-dynamics-of-vortexdroplet-coaxial-interaction-insights-into-droplet-and-vortex-dynamics/39D430437DFD87BC4D43A8A9761517BB
  Experiments involve3Dvortex rings/two-way coupling; motivates encounter and
  circulation parameters, not validation of the proposed2Dprescribed-flow model.
- Bentley & Leal1986, computer-controlled four-roll mill:
  https://doi.org/10.1017/S002211208600280X
  Controlled strain/vorticity and drop experiments motivate alternative device
  family; no claim current periodic flow solves that device.
- Topological bifurcations of vortex pair interactions, JFM917,A11:
  https://glass.ruc.dk/pdf/articles/2021_JFM_917_A11.pdf
  Gaussian vortices/analytic single-vortex profile versus synthetic multi-Gaussian
  models are distinguished explicitly in the source.

Visualization/verification:
- twophase-physical-flow-schematic.py produces three illustrative periodic
  velocity/vorticity fields with identical circle overlay: weaker/stronger
  centered pair and stronger offset pair. No CH/reference/network prediction.
  Dimensionless numbers are display examples only, not chosen training bounds;
  circle overlay is schematic, not a changed initial-condition implementation.
- N160 independent NumPy spectral construction: max divergence<=5.89e-17,
  mean vorticity<=5.72e-18. Circulation doubling doubles RMS speed from.04024
  to.08048; translation retains it. Measured receipts are in adjacent JSON.
- Inspected generated figure. PNG remains ignored/local; source/provenance are
  lightweight text. No code suite required for documentation/design work;
  git diff --check passes. No solver/default/training/history change.

Task-end: archived design, plot source/provenance and local figure; PROGRESS
cleared and DECISIONS updated. Next action only on implementation request:
choose physical distribution/resolution/forcing and independent reference gates,
then implement opt-in train.py operator. No active simulation is queued.
