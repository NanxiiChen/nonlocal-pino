The operator learns spinodal decomposition in an elliptical vessel with two impermeable holes. Each domain receives the same ambient ten-mode initial texture restricted to its own material region.

## Setup and boundary data

The outer ellipse is centered at \((0.5,0.5)\) with semiaxes \((0.5,0.4)\). Hole centers are sampled from \([0.22,0.78]\times[0.27,0.73]\) and radii from \([0.045,0.12]\). Containment and a 0.025 clearance condition screen proposals. Walls are neutral and impermeable. The dimensionless horizon is \(T=3\).

<figure>
  <a href="assets/confined_ch2d-setup.svg"><img src="assets/confined_ch2d-setup.svg" alt="The same ten-mode initial phase texture in three elliptical vessels with different hole positions and sizes"></a>
  <figcaption>Each geometry restricts the same initial function. Only the two holes vary.</figcaption>
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
  <a href="assets/confined_ch2d.png"><img src="assets/confined_ch2d.png" width="1950" height="1200" alt="Spinodal patterns evolving for three geometries with different hole positions and radii"></a>
  <figcaption>Development geometries 1027, 1028 and 1026 at t = 0, 0.75, 1.5, 2.25 and 3, selected as low, upper-median and worst pooled-error examples. Dashed contours are direct-nonlocal interfaces. Display interpolation is separate from the error quadrature.</figcaption>
</figure>

## References and comparisons

Direct integration of the graph dynamics is time-refined. Physical cell areas weight field errors over five saved snapshots. The eight development geometries were reserved from training and reused in configuration studies.

| Mean relative \(L^2\) against direct nonlocal | Pooled | Terminal |
| --- | ---: | ---: |
| Phase, 8 development geometries | 4.8329% | 5.1935% |
| Chemical potential, same 8 geometries | 10.7329% | 14.2757% |
| Phase, 16 frozen historical training-domain monitors | 5.0529% | 5.4186% |

The worst development phase errors are 9.5686% pooled and 10.5862% terminal. Local FEM validation of this harder ten-mode case is deferred. The historical fixed-bank and rolling-pool experiments also differ in network size or starting stage. The two cohorts in the table assess coverage and fit without isolating the causal effect of rolling the pool.

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
