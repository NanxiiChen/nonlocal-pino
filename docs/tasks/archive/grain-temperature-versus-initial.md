# Temperature versus initial grain parameterization

## Goal
Compare physical meaning and useful operator diversity of parameterizing a
uniform mean temperature versus initial grain geometry/orientations.

## Acceptance criteria
- Distinguish heat-treatment design from initial-microstructure response.
- Independently check the limited scalar-mobility time-rescaling equivalence.
- State why mean temperature alone does not specify a nonuniform thermal field.
- Recommend scope matched to desired morphology and mechanics, without selecting
  material parameters or launching numerical work.

## Context
The user is considering a grain-growth PINO after discussing quasistatic
elasticity and migration kinetics. Neither a physical model nor a new run has
been selected. The latest comparison explicitly says temperature.

## Status
Complete (2026-11-05). Primary model references checked; the restricted
temperature/time equivalence independently derived and agent-audited.

## Blocker
None for conceptual advice.

## Next action
Await user selection of thermal-processing versus initial-microstructure scope.
No physical parameters, architecture, code or experiment selected.

## Conclusions

Both choices have direct physical meaning. Uniform temperature/history concerns
annealing/heat-treatment process conditions; initial grain sizes, neighbors,
topology and crystal orientations concern material state and variability. For
the user's aim of diverse grain evolution in a field-to-field PINO, recommend
initial microstructure as the main input first, holding temperature and a
mechanical protocol fixed. This is a recommendation, not an implementation.

Uniform temperature can enter grain mobility through an Arrhenius factor.
Let all kinetic coefficients share a positive factor a(T), and eliminate
quasistatic mechanics under temperature-independent properties and fixed
mechanical boundary inputs. If the reduced equation is

$$
\partial_t\boldsymbol\eta=-a(T(t))\,\boldsymbol G[\boldsymbol\eta],
\qquad \tau(t)=\int_0^t a(T(s))\,ds,
$$

then the chain rule gives d eta/d tau = -G[eta]. Same initial state follows
the same trajectory, merely at a different rate. For constant temperatures,

$$
\boldsymbol\eta_T(t)=\boldsymbol\eta_{T_r}
\!\left(\frac{a(T)}{a(T_r)}t\right).
$$

This independently checked limiting case does not apply in general: a fixed
physical-time mechanical ramp remains an explicit time-dependent input after
the transformation; distinct activation energies can change relative boundary
mobilities; thermal expansion, temperature-dependent elastic constants or
energies can change driving forces. Do not invent these effects merely to
increase operator complexity. Thermal timing remains physically useful even
when trajectories coincide under time rescaling.

Initial-state conditioning should use coherent grain structures with controlled
size/shape/orientation statistics rather than arbitrary pixel noise. Grain
identity and crystal orientation are different; labels alone have no physical
meaning. Random geometry does not guarantee substantially different average
mechanical behavior, especially when bulk statistics are similar. Validate
morphology and averaged response separately. An initial-state operator supplies
material-state conditioning; temperature is a thermal condition; neither should
be mislabeled as arbitrary mechanical boundary-function conditioning.

"Mean temperature" is interpreted here as a spatially uniform prescribed cell
temperature. If the temperature field varies spatially, its mean is insufficient
and one needs a field/history or a thermal model and boundary conditions.

## Sources

- https://mooseframework.inl.gov/source/materials/GBEvolution.html
  (temperature-dependent Arrhenius mobility).
- https://mooseframework.inl.gov/modules/phase_field/Grain_Growth_Model.html
  (grain kinetics, anisotropy and coherent polycrystal initializations).
- https://mooseframework.inl.gov/modules/phase_field/Elastic_Driving_Force_Grain_Growth.html
  (orientation-dependent stiffness and elastic driving).

## Completion verification

Documentation-only changes to PROGRESS, DECISIONS and this task. Verify archive
paths, inspect records and git diff whitespace. No tests or numerical work.
