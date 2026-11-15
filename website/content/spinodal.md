The operator learns spinodal decomposition in an elliptical vessel with two impermeable holes. Each domain receives the same ambient ten-mode initial texture restricted to its own material region.

## Setup and boundary data

The outer ellipse is centered at \((0.5,0.5)\) with semiaxes \((0.5,0.4)\). Hole centers are sampled from \([0.22,0.78]\times[0.27,0.73]\) and radii from \([0.045,0.12]\). Containment and a 0.025 clearance condition screen proposals. Walls are neutral and impermeable. The dimensionless horizon is \(T=3\).

<figure>
  <a href="assets/confined_ch2d-setup.png"><img src="assets/confined_ch2d-setup.png" width="1594" height="1435" alt="Fixed blue-orange initial fluctuations in an ellipse with two random holes and gray hatched walls"></a>
  <figcaption>Each geometry restricts the same initial function. Only the hole positions and radii vary. Colors show initial fluctuations; gray wall-band thickness is illustrative. <a href="assets/confined_ch2d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

The Cahn–Hilliard chemical potential is the variational derivative of the double-well and gradient energy. With mobility \(M\), concentration-like phase \(\phi\) evolves by the divergence of its chemical-potential flux.

$$
\begin{aligned}
E[\phi]&=\int_\Omega
  \left[\tfrac{\beta}{4}(\phi^2-1)^2
        +\tfrac{\epsilon^2}{2}|\nabla\phi|^2\right]\,d\mathbf x,\\
\mu&=\beta(\phi^3-\phi)-\epsilon^2\Delta\phi,\qquad
\phi_t=M\Delta\mu,\qquad \partial_n\phi=\partial_n\mu=0 .
\end{aligned}
$$

The modeled parameters are dimensionless, with \(\beta=1\), \(M=0.02\), and \(\epsilon=0.06\). Natural neutral wetting gives the phase condition, and no chemical flux closes the mass balance.

## Nonlocal residual and loss

Physical triangle-cell areas define the quadrature. The symmetric graph excludes segments that cross the holes. With \(D_\delta\approx-\Delta\), eliminate the chemical potential before computing the phase residual.

$$
\mu_\delta=\beta(\phi^3-\phi)+\epsilon^2D_\delta\phi,\qquad
r=\phi_t+M D_\delta\mu_\delta .
$$

The output hard-imposes the initial phase and projects onto its per-geometry weighted mean \(m_0\). The exact initial field and mass need no additional loss. The closed graph also satisfies the weighted row identity described in [spatial calculus](spatial-calculus.html#regional-quadrature-and-walls).

The phase loss uses a frozen linear stable-part resolvent. Let \(V\) contain cell areas on its diagonal and \(S=V^{1/2}D_\delta V^{-1/2}\). Linearizing the phase equation about the known mean gives the modal rates \(q(\lambda)=M\lambda[\epsilon^2\lambda+\beta(3m_0^2-1)]\).

$$
P=V^{-1/2}\bigl[I+Tq(S)_+\bigr]^{-1}V^{1/2},
\qquad q(S)_+=U\,\mathrm{diag}\!\bigl(\max(q(\lambda_j),0)\bigr)U^\mathsf T.
$$

The loss evaluates \(Pr\). Growing modes retain unit gain, while stiff decaying modes are downweighted. The positive spectral weights preserve the zero set of the residual. The metric is built from the graph and known initial phase.

## Field evolution

<figure>
  <a href="assets/confined_ch2d.png"><img src="assets/confined_ch2d.png" width="2007" height="1334" alt="Predicted spinodal phase for four reserved geometries with two holes across five times, with native reference interfaces"></a>
  <figcaption>The selected 50,000-step model on the first four reserved development geometries, 1024–1027, at t = 0, 0.75, 1.5, 2.25 and 3. Colors show predicted phase; dashed curves are native nonlocal φ = 0 interfaces. Labels use instantaneous, equally weighted original-cell phase relative L²; display smoothing does not enter these errors. <a href="assets/confined_ch2d.pdf">Editable PDF</a>.</figcaption>
</figure>

## References and comparisons

Direct integration of the graph dynamics is time-refined. Physical cell areas weight field errors over five saved snapshots. The eight development geometries were reserved from training and reused in configuration studies.

| Mean relative \(L^2\) against direct nonlocal | Pooled | Terminal |
| --- | ---: | ---: |
| Phase, 8 development geometries | 4.8329% | 5.1935% |
| Chemical potential, same 8 geometries | 10.7329% | 14.2757% |
| Phase, 16 frozen historical training-domain monitors | 5.0529% | 5.4186% |

The worst development phase errors are 9.5686% pooled and 10.5862% terminal. Independent local FEM is now qualified for the [completed spatial-method comparison](#spatial-method-comparison) below; this historical table retains its native target. The historical fixed-bank and rolling-pool experiments also differ in network size or starting stage. The two cohorts in the table assess coverage and fit without isolating the causal effect of rolling the pool.

## Spatial-method comparison

Matched Nonlocal and FEM PINO controls each complete 50,000 cold SOAP updates
with one paired initialization, width 256, B16/Nt16 and the rolling geometry
pool. Both retain hard phase IC/mass and use positive-time PDE collocation;
the original SOTA also allocates 20% of its PDE measure to the initial time.
Defaults are unchanged. Local AD was omitted by the user after full-size
resource probes; no trained AD error is reported.

The independent numerical reference uses mixed-P2/BDF2 local FEM on the same
eight reused geometries 1024–1031 and five times \(0,0.75,1.5,2.25,3\).
All references pass spatial/time and physical checks: maximum phase mesh gap
0.67923%, phase time gap 0.0010383%, positive-time chemical mesh gap 0.74673%,
mass drift below 3.50e-16, and no observed energy increase. The trained FEM PINO
instead uses P1 stiffness/lumped mass and natural weak Neumann conditions; it
is distinct from this numerical reference solver.

For each channel and sample, take the equal-node full space-time relative norm,
then average samples. Both methods evaluate all eight geometries at all 101
actual records. The final audit checks paired cold states, every actual draw,
16,000 online inputs and all 144 source files per run.

| Final50k local-reference error | Nonlocal PINO | FEM PINO |
|---|---:|---:|
| Phase |31.9216%|5.5838%|
| Chemical potential, all five times |46.5728%|45.8887%|
| Mean over the two channels |39.2472%|25.7362%|
| Chemical potential, positive times only (diagnostic) |31.3091%|14.9354%|

The initial local chemical trace uses analytic continuum derivatives; the
models eliminate chemical potential with their own spatial operators. Generic
wall normals are incompatible with the prescribed initial phase. Report the
positive-time diagnostic separately without dropping the initial trace from
the primary error.

| Equal-node phase error against each target | Nonlocal PINO | FEM PINO |
|---|---:|---:|
| Native nonlocal numerical reference |4.5990%|29.3864%|
| Independent local FEM numerical reference |31.9216%|5.5838%|

The numerical targets differ by 31.9633% in phase under this same metric. This
finite-horizon case exposes a substantial spatial-model/discretization gap;
native fitting accuracy does not establish local-PDE accuracy. Do not subtract
relative errors to attribute causes. The historical SOTA likewise gives
4.9174% native /32.2170% local under the same equal-node metric.
Further horizon, resolution and boundary investigation is deferred by the user
and recorded in the [research TODO](https://github.com/NanxiiChen/nonlocal-pino/blob/main/docs/TODO.md); no additional Spinodal run is started.

<figure>
  <a href="assets/confined_ch2d_nonlocal_fem_convergence.png"><img src="assets/confined_ch2d_nonlocal_fem_convergence.png" width="2007" height="708" alt="Spinodal phase and chemical-potential convergence for Nonlocal and FEM PINO against local numerical FEM"></a>
  <figcaption>All 101 measured means remain as thin solid lines; thick lines show log-LOWESS with span0.12. Shading uses2,000 paired resamples of complete geometry trajectories, conditional on eight reused cases and one training run. It is not cross-seed or population-wide95%coverage. Chemical error includes the declared initial trace. <a href="assets/confined_ch2d_nonlocal_fem_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/confined_ch2d_nonlocal_fem_distribution.png"><img src="assets/confined_ch2d_nonlocal_fem_distribution.png" width="2007" height="708" alt="Final50k phase and chemical-potential relative-error violins on eight common local-FEM geometries"></a>
  <figcaption>Final50k weights; eight geometries per method and channel. Log10 Gaussian KDE, Scott bandwidth, three-bandwidth tails and quartiles; raw error-ratio axes with automatic limits. Eight-case densities are descriptive. Both figures are170×60mm. <a href="assets/confined_ch2d_nonlocal_fem_distribution.pdf">Editable PDF</a>.</figcaption>
</figure>

| Same RTX PRO6000 Blackwell Server GPU | Nonlocal PINO | FEM PINO |
|---|---:|---:|
| Complete cold training process |8,086.91s (2.2464h)|6,852.67s (1.9035h)|
| Full-run peak active allocation |20.44GiB|21.89GiB|

Times cover initialization, preparation, compilation, all updates/rolling waits,
in-training evaluation, final writes and teardown. Separate offline banks and
post-training audits/plots are excluded. Local AD's approximately114/55hour
update projections for nested/Taylor differentiation are resource probes, not
complete training costs. One paired run does not establish a universal ranking.

[Saved configurations, checkpoints, selected figures and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_three_method_figures.json),
[dual-reference norms](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/spinodal_final_reference_norms.json),
and [complete timing provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_training_wall_costs.json).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Network | Two point streams, width 256, four heads, four spatial cross-attention blocks, three time-FiLM layers |
| Geometry | Gmsh target spacing 0.035, horizon 0.11, padding 1,536 nodes and 128 neighbors |
| Known-input pool | 1,024 initial slots, six CPU workers, 32 replacements every 100 updates |
| Samples per update | 16 geometries and 16 positive times plus the initial time |
| Optimizer | SOAP, 50,000 cold updates, initial \(10^{-3}\), \(\beta_1=\beta_2=0.9\), ordinary FP32 |

The selected run took 102.26 min and peaked at 20.43 GiB active allocation on RTX 6000D. It sampled 15,994 new online geometries. The operator predicts a full time interval; the repository also has a separate single-problem time-window example.

~~~bash
source .venv/bin/activate
python -m examples.confined_ch2d.train \
  --config=examples/confined_ch2d/configs/train_operator.py
~~~

Use the prepared 1,024-slot input bank and separate evaluation bank. See the [example configuration and native reference preparation notes](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/confined_ch2d/README.md#geometry-operator-sota).
