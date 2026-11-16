Five phase-field case studies show how to learn an evolution operator from governing equations, prescribed data and geometry. The report covers the network, the spatial discretization and the evidence behind each selected result.

## Operator learning from physics

The completed [Electropolish 15k study](electropolish.html#fixed-initial-condition-library-size)
compares 32/64/128/256/512 fixed initial-profile libraries against continuous online
sampling on the same 64 local-FD tests. Final channel means are 16.45%/12.27%/5.46%/2.12%/2.02%; online gives
1.53%. The standalone figure shows means and one sample SD. Its paired 15k time-feature
study compares spatial attention/time FiLM with joint(x,t)queries on a complete
cold-wall-time axis. Defaults remain unchanged; one initialization does not establish
a general accuracy ranking. [Ceramic library stability](particle-corrosion.html#fixed-geometry-library-and-training-stability)
remains deferred, with its historical numerical outcomes retained and no library-size plot.

A condition \(a\) specifies an initial field on a fixed domain or specifies the domain and its known inputs. The network is trained over a family of conditions and returns a trajectory sampled at query coordinates.

$$
\mathcal G_\theta(a)(\mathbf x,t)
  \longmapsto \bigl(\phi_\theta(\mathbf x,t),\,\ldots\bigr),
  \qquad \mathbf x\in\Omega_a .
$$

The auxiliary fields depend on the PDE. Training evaluates physical residuals at sampled times and points, alongside any soft initial or boundary constraints. Positive-time solver trajectories enter evaluation only. A single trained operator handles multiple inputs from its condition family.

<figure>
  <a href="assets/architecture.svg"><img src="assets/architecture.png" width="1024" height="904" alt="Master figure: background geometry and operator preparation, rolling input pools, core-halo sampling, patch or two-stream point encoders, spatial cross-attention, three time-FiLM decoder layers, and nonlocal physics training."></a>
  <figcaption>Architecture and training workflow. Patch encoding and two-stream point encoding are alternatives; both prepare spatial features before per-layer time FiLM. The lower panels summarize nonlocal residual evaluation and case-dependent training strategies. Field textures are schematic. <a href="assets/architecture.pdf">Editable PDF</a> · <a href="assets/architecture.svg">SVG</a>.</figcaption>
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
| [05 · Ceramic-particle corrosion](particle-corrosion.html) | Six inert particle shapes and poses | KKS AC/CH | 2.905% / 5.436% | Local P1 FEM, 4 development geometries; current default |

Each row averages per-sample errors from the selected final checkpoint. Time weights, diagnostic grids and reference solvers vary by case. The corresponding chapters give the comparison protocol, field images and limitations.

The five field galleries show **Best / Middle / Worst** samples ranked by equal-point, phase-only relative L² over the complete space-time evaluation grid. Their cohorts contain 64 melting shapes, 64 electropolish interfaces, 8 spinodal geometries, 3 reserved channels and 64 ceramic geometries. Middle is the lower central actual sample. Figure labels show each snapshot’s spatial error, not the ranking score or a channel mean. Original field-model checkpoints are retained; Melting and Spinodal contours still use native nonlocal references. [Sample IDs and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_ranked_main_preview.json).

The [current ceramic SOTA default](particle-corrosion.html#current-sota-default) also completes a 64-case local-FEM evaluation: phase 3.6965%, concentration 4.4839%, and channel mean **4.0902%**. These use equally weighted full space-time node norms, distinct from the four-case quadrature-weighted row above. The actual default entrypoint was validated from a cold start for 20,000 steps; one paired initialization is reported.

## Reading the evidence

- **Direct nonlocal integration** tests how accurately the network reproduces the specified spatial discretization.
- **An independent local solver** also exposes the discrepancy between spatial models and their discretizations. Some cases report both targets on matched inputs.
- **A training control** isolates only the changes documented in that experiment. The electropolish study reports a matched local-AD control, and other cases report optimizer, architecture or sampling comparisons.


## Additional spatial-method comparison

The [two-phase channel comparison](two-phase-flow.html#three-method-comparison)
shows development convergence and final errors on three reserved geometries.
At 20k, nonlocal/local AD/FEM PINO phase errors average 1.64%/2.15%/4.65% and
velocity errors1.52%/2.09%/4.39% against independent numerical FEM.
Current-card warm probe updates cost1.60/1.74/1.42s; AD samples2,048interior
queries with independent boundary losses, while the other arms use15,633nodes.
This finite comparison favors the nonlocal recipe's accuracy;
its limitations and separate timing protocol are documented in the chapter.

The [Electropolish three-method control](electropolish.html#three-method-comparison)
uses 10k steps and the same 64 local-FD test inputs at all 41 evaluations.
Nonlocal / Local AD / FEM PINO phase errors are 1.9713% / 4.3904% / 1.8486%; concentration
errors are 2.3563% / 5.1374% / 2.2072%. FEM and Nonlocal are close in this one run.
AD has faster warm updates and lower active memory, but longer compilation;
complete cold times are 15.63 / 18.27 / 15.44 minutes. The chapter separates these
costs and retains the selected SOTA model independently of the comparison.

The [Melting three-method comparison](ice-melting.html#three-method-comparison)
uses final 10k weights and 64 native test ellipsoids at all 41 evaluations. Phase means
are 0.2860% / 0.0850% / 0.1093%(Nonlocal / Local AD / FEM PINO), on 24³ queries and five times.
The separate local CG2 case005 gives 0.1928% / 0.1208% / 0.0671%; it is not a 64-case local
study. Cold times are 6.89 / 10.37 / 7.35 minutes. Both reference scopes, component
figures and timing limitations are reported without changing the SOTA defaults.

The [Spinodal spatial-method comparison](spinodal.html#spatial-method-comparison)
contains matched final 50k Nonlocal/FEM PINO controls on eight reused geometries,
with all 101 evaluations against independently refined local FEM. Phase errors
are 31.9216% / 5.5838%; against the native nonlocal reference, the same equal-node
metric gives 4.5990% / 29.3864%. The numerical references themselves differ by
31.9633% in phase. Native fitting accuracy therefore does not establish local-PDE
accuracy in this finite-horizon case. Complete cold times are 2.2464 / 1.9035 h
on the same GPU. Local AD was omitted after the user reviewed its resource cost;
no final AD accuracy is claimed. The original SOTA recipe remains unchanged.

See [complete cold training costs](training.html#cold-training-wall-clock-cost)
for process-wide timing, including initialization, compilation, in-training
evaluation and final writes. Warm-update costs are reported separately;
historical GPU and timing-resolution differences remain explicit.

## Temporal feature controls

The [Electropolish temporal-control study](electropolish.html#time-feature-fusion-comparison)
keeps nonlocal derivatives and compares spatial attention with decoder FiLM to
joint space-time queries. Final 10k / 64 local-FD phase errors are 1.9713% versus 2.0577%,
with channel means 2.1638% versus 2.2549%. Cold times are 15.63 versus 23.65 minutes,
warm inclusive medians 64.61 versus 105.76 ms, and active peaks 12.03 versus 32.45 GiB.
These models have different temporal capacities; one initialization supports
close accuracy here, with lower measured training cost for spatial attention/FiLM.

The [Melting temporal control](ice-melting.html#time-feature-fusion-comparison)
finds a different accuracy tradeoff: FiLM/joint-query phase means are
0.2860% / 0.0935% on the 64 native references, and 0.1928% / 0.1302% on the
separate local CG2 case005. Cold costs are 6.89 / 14.99 minutes, warm inclusive
medians 15.81 / 60.44 ms, and active peaks 1.45 / 9.36 GiB. Both cases favor FiLM
on measured training cost; final accuracy depends on the case. All controls
use nonlocal derivatives, one initialization, final 10k weights and complete
64-case histories, with temporal-capacity and reference qualifications explicit. The selected temporal convergence plots use
elapsed wall-clock seconds from cold launch, including initialization, compilation,
in-training evaluation and final process completion; raw errors are unchanged.

The [Ceramic temporal control](particle-corrosion.html#time-film-versus-joint-query)
extends this test to two-stream geometry conditioning and rolling geometry/core-halo
inputs. Final 20k FiLM/joint-query channel means are **4.0902% / 4.4832%** on
64 common local-FEM cases. Complete cold times are **2629.49 / 12475.49 s** on
the same physical GPU: FiLM is **4.74 times as fast** in this paired run.
All 41 records use equal-point full-space-time norms and direct parent clocks.
Temporal capacities differ; defaults remain FiLM. The case chapter includes
the selected convergence and final-error distributions.
