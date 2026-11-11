Five phase-field case studies show how to learn an evolution operator from governing equations, prescribed data and geometry. The report covers the network, the spatial discretization and the evidence behind each selected result.

## Operator learning from physics

A condition \(a\) specifies an initial field on a fixed domain or specifies the domain and its known inputs. The network is trained over a family of conditions and returns a trajectory sampled at query coordinates.

$$
\mathcal G_\theta(a)(\mathbf x,t)
  \longmapsto \bigl(\phi_\theta(\mathbf x,t),\,\ldots\bigr),
  \qquad \mathbf x\in\Omega_a .
$$

The auxiliary fields depend on the PDE. Training evaluates physical residuals at sampled times and points, alongside any soft initial or boundary constraints. Positive-time solver trajectories enter evaluation only. A single trained operator handles multiple inputs from its condition family.

<figure>
  <a href="assets/architecture.svg"><img src="assets/architecture.svg" width="1024" height="748" alt="A fixed-domain query reads patches. A parameterized-domain query averages separate boundary and domain reads. Both temporal decoders modulate three hidden layers with FiLM."></a>
  <figcaption>Coordinate queries read one patch stream for a fixed domain and two separately weighted point streams for a parameterized domain. Both paths reuse spatial features across times and apply FiLM in each temporal hidden layer.</figcaption>
</figure>

## Method chapters

| Chapter | Contents |
| --- | --- |
| [Conditional architecture](architecture.html) | 2D/3D patch encoding, weighted point streams, cross-attention and shared spatial features |
| [Nonlocal spatial calculus](spatial-calculus.html) | Interior Taylor moments, weighted graph assembly, finite-horizon wall closure and moment-corrected channel stencils |
| [Training and validation](training.html) | Sampling and rolling pools, hard lifts, inverse-gradient balancing, residual preconditioning and evaluation protocols |

The spatial operator evaluates derivatives from neighboring predictions, avoiding repeated coordinate Hessians for Laplacians in the loss. Time differentiation still uses a JVP. Parameter gradients remain part of training, and the channel still uses an AD curl to recover velocity from its streamfunction.

## Case studies

| Case | Varying condition | Model | Phase relative \(L^2\), pooled / terminal | Reference and cohort |
| --- | --- | --- | ---: | --- |
| [01 · Ice melting](ice-melting.html) | Initial ellipsoid | Driven Allen–Cahn | 0.1002% / 0.1711% | Direct nonlocal, 8 held-out shapes |
| [02 · Electropolish](electropolish.html) | Initial rough interface | KKS AC/CH | 1.547% / 1.326% | Local finite differences, 64 unseen interfaces |
| [03 · Spinodal decomposition](spinodal.html) | Two hole positions and radii | Confined Cahn–Hilliard | 4.8329% / 5.1935% | Direct nonlocal, 8 reused development geometries |
| [04 · Two-phase flow](two-phase-flow.html) | Upper and lower channel walls | Incompressible NS–CH | 1.6435% / 2.1961% | Local mixed FEM, 3 reused reserved geometries |
| [05 · Ceramic-particle corrosion](particle-corrosion.html) | Six inert particle shapes and poses | KKS AC/CH | 2.933% / 5.757% | Local P1 FEM, 4 development geometries |

Each row averages per-sample errors from the selected final checkpoint. Time weights, diagnostic grids and reference solvers vary by case. The corresponding chapters give the comparison protocol, field images and limitations.

## Reading the evidence

- **Direct nonlocal integration** tests how accurately the network reproduces the specified spatial discretization.
- **An independent local solver** also exposes the discrepancy between spatial models and their discretizations. Some cases report both targets on matched inputs.
- **A training control** isolates only the changes documented in that experiment. The electropolish study reports a matched local-AD control, and other cases report optimizer, architecture or sampling comparisons.
