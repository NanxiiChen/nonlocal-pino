A geometry-conditioned operator predicts the growth of a corrosion pit around six smooth inert ceramic inclusions. Particle shapes and poses vary while the physical coefficients, reservoir and initial pit stay fixed.

## Setup and boundary data

The reactive region is \([0,1]\times[0,0.6]\) minus the ceramic interiors, or \(50\times30\,\mu\mathrm m\) in physical dimensions. Depth increases downward. The initial semicircular pit has radius \(6\,\mu\mathrm m\), and the reservoir mouth is at \(y=0\), \(|x-0.5|\leq0.055\) in normalized coordinates. It prescribes \(\phi=c=0\). The remaining walls are neutral and impermeable.

Each particle uses a smooth radial perturbation with modes 3–5, area normalization, aspect stretch and rotation. Equivalent radii range from 2.25 to \(3.75\,\mu\mathrm m\). Clearance, curvature and near-pit requirements screen layouts using geometry alone. The reservoir can exchange concentration, so global concentration is free to change.

<figure>
  <a href="assets/particle_corrosion2d-setup.png"><img src="assets/particle_corrosion2d-setup.png" width="1181" height="885" alt="Gray metal, blue electrolyte pit and six smooth ivory ceramic particles with dark outlines"></a>
  <figcaption>Ivory ceramic regions are excluded from the reactive domain. Particle positions, shapes and orientations vary; the initial pit and narrow reservoir mouth are fixed. <a href="assets/particle_corrosion2d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

The KKS mixture uses \(h(\phi)=3\phi^2-2\phi^3\), \(g(\phi)=\phi^2(1-\phi)^2\), \(\Delta c=c_{Se}-c_{Le}\), and \(q=c-\Delta c\,h(\phi)-c_{Le}\). The dimensional phase and concentration laws are

$$
\begin{aligned}
\partial_t\phi&=L\,[\alpha_\phi\Delta_x\phi-\omega_\phi g'(\phi)
                    +2A\Delta c\,q h'(\phi)],\\
\partial_t c&=D\Delta_x q .
\end{aligned}
$$

The displayed case sets \(L_c=50\,\mu\mathrm m\), \(T_c=350\,\mathrm s\), and \(L=2\times10^{-9}\,\mathrm{m^3/(J\,s)}\). After substituting \(\mathbf x^*=\mathbf x/L_c\) and \(t^*=t/T_c\), the coefficient groups are

$$
\begin{aligned}
N_1&=2ALT_c=74.9, & N_2&=L\omega_\phi T_c=12.32,\\
N_3&=L\alpha_\phi T_c/L_c^2=0.00721, &
N_{\mathrm{CH}}&=DT_c/L_c^2=119 .
\end{aligned}
$$

The initial phase uses a tanh of signed radial distance from the pit and \(c_0=c_{Se}h(\phi_0)\) away from the mouth. At the mouth the prescribed values take precedence, including at the initial time.

## Nonlocal residual and loss

The raw network channels \(z,s\) supply the physical fields through the algebraic KKS map.

$$
\phi=\tfrac12(1+\tanh z),\qquad
c=c_{Le}s+\Delta c\,h(\phi),\qquad
q=c_{Le}(s-1).
$$

The concentration channel \(s\) is unrestricted. Positive \(D_\delta\) uses visible, symmetric edges with physical volumes. The regional graph excludes segments crossing ceramic interiors. Evaluating \(D_\delta q=c_{Le}D_\delta s\) exploits constant cancellation and reduces FP32 loss of significance.

$$
\begin{aligned}
r_{\mathrm{AC}} &= \phi_{t^*}+N_3D_\delta\phi+N_2g'(\phi)
                     -N_1\Delta c\,q h'(\phi),\\
r_{\mathrm{CH}} &= c_{t^*}+N_{\mathrm{CH}}c_{Le}D_\delta s .
\end{aligned}
$$

The objective uses absolute AC and CH residuals with unit prescales. Mouth phase, mouth concentration and initial-field errors give three soft terms. Scale the concentration parts of prescribed-data errors by \(c_{Le}\) before squaring. Inverse-L1 gradient-norm weights refresh every 25 updates. AC and CH train jointly, with the original physical residual despite the bounded phase output.

## Field evolution

<figure>
  <a href="assets/particle_corrosion2d.png"><img src="assets/particle_corrosion2d.png" width="2007" height="1086" alt="Predicted pit growth around warm-ivory ceramic particles with dark outlines for four development geometries across five times, with local-FEM interfaces"></a>
  <figcaption>The selected width-256, 20,000-step model on all four development geometries at t* = 0, 0.2, 0.5, 0.8 and 1 (0, 70, 175, 280 and 350 s). Colors show predicted phase; warm-ivory interiors with dark outlines are inert ceramic, and dashed curves are independent local-FEM φ = 0.5 interfaces. Labels give instantaneous phase relative L² on the original spatial nodes, excluding ceramic interiors. <a href="assets/particle_corrosion2d.pdf">Editable PDF</a>.</figcaption>
</figure>

## References and comparisons

Both targets use the same four development geometries, excluded from the rolling training stream. Pooled errors use spatial quadrature and trapezoidal time weights over 11 saved snapshots. The independent local P1 FEM solver uses weak-form stiffness, lumped mass and sparse BDF integration. A first-geometry mesh refinement from spacing 0.01 to 0.0075 changes the local reference phase by 0.1007% pooled.

| Reference | Phase pooled / terminal | Concentration pooled / terminal |
| --- | ---: | ---: |
| Independent local P1 FEM | 2.933% / 5.757% | 3.566% / 6.871% |
| Direct production nonlocal solver | 1.450% / 2.781% | 1.763% / 3.278% |

The two references differ by 1.72–1.80% pooled phase over these four geometries. Terminal corroded-region IoU is 94.99% against FEM and 98.10% against direct nonlocal integration.

| Same-stream cold 20k comparison | Width 192 | Width 256 |
| --- | ---: | ---: |
| Local phase pooled / terminal | 3.091% / 5.985% | 2.933% / 5.757% |
| Warm held-update median | 49.06 ms | 62.94 ms |
| Complete training process | 23.35 min | 28.47 min |
| Peak active allocation | 15.66 GiB | 20.69 GiB |

Both width and Fourier feature capacity change, and the parameter shapes initialize separately. The table uses the final 20k model on the already-inspected development cohort.

## Spatial derivative comparison

This three-method comparison uses final 20,000-step neural weights and 64 common held-out geometries. **FEM PINO** replaces spatial differentiation with the assembled mass-normalized P1 weak operator \(M_L^{-1}K\); natural homogeneous Neumann data enter through the weak form. **FEM numerical reference** is the independent refined P1/BDF solver used for evaluation. It is not a trained model, and its trajectories do not enter training.

For each sample and physical channel, relative \(L^2\) is computed over all space-time evaluation nodes, then averaged over the phase and concentration channels. The values below average those sample errors. These equally weighted node norms differ from the four-case volume/time-weighted results above.

| Method | Phase mean | Concentration mean | Channel mean ± sample SD | Training time |
| --- | ---: | ---: | ---: | ---: |
| Nonlocal PINO | 3.8175% | 4.6205% | 4.2190% ± 0.5719 pp | 26.77 min |
| Local AD PINO | 32.2070% | 32.6060% | 32.4065% ± 0.6047 pp | 28.42 min |
| FEM PINO | 5.5535% | 6.5937% | 6.0736% ± 0.8166 pp | 20.97 min |

<figure>
  <a href="assets/particle_corrosion2d_fem_distribution.png"><img src="assets/particle_corrosion2d_fem_distribution.png" width="1181" height="755" alt="Final 64-sample channel-averaged error distributions for Nonlocal PINO, Local AD PINO and FEM PINO, evaluated against the same FEM numerical reference"></a>
  <figcaption>All 64 errors from the final 20k weights; logarithmic error axis with unscaled ratios. Violin widths show a KDE in log-error coordinates, with dashed quartiles. <a href="assets/particle_corrosion2d_fem_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

The shared recipe uses width 256, Adam, rolling geometry/core-halo pools, and random times. These are seed-42 measurements on the same RTX PRO 6000 Blackwell Server Edition. Training time includes compilation and in-training evaluation, but excludes the separate final 64-sample evaluation. The displayed nonlocal model reproduces the historical source and launch settings; an unsuccessful execution and the unresolved reproducibility investigation are retained in the [experiment record](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/tasks/fem-weak-pino-comparison-plan.md). This is a finite single-initialization comparison, not a general method ranking. [Exact model, configuration, metric and figure provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/ceramic_fem_comparison_publication.json).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Network | Two point streams, width 256, four heads, two spatial cross-attention blocks, three time-FiLM layers |
| Spatial work | Target mesh spacing 0.015, horizon 0.026, 256 sampled cores, complete halo |
| Samples per update | 4 geometry draws and 32 freshly sampled times |
| Online pool | 64 geometry slots, four CPU workers, geometry and spatial payloads refreshed on separate schedules |
| Optimizer | Adam, 20,000 cold updates, initial \(10^{-3}\), halve every 3,000 updates, floor \(10^{-5}\), ordinary FP32 |

Prepare the initial known-input bank and independent local reference from the explicit six-particle recipe. The [repository run guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/particle_corrosion2d/README.md#run-the-documented-case) gives the preview, input preparation and local/native qualification commands. Then train from the repository root.

~~~bash
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
python -m examples.particle_corrosion2d.train \
  --config examples/particle_corrosion2d/configs/train_physical_spaced_w256.py \
  --inputs /path/to/input-bank --reference /path/to/local-reference \
  --output /path/to/new-run
~~~

The explicit width-256 preset selects the 350 s result on this page. The no-argument example entry selects a different higher-mobility 5 s experiment.
