The operator predicts three-dimensional interface retreat for a family of initial ellipsoids. The phase field is near \(+1\) in ice and \(-1\) in water.

## Setup and boundary data

The cube \([-50,50]^3\) maps to \([0,1]^3\), and the physical horizon is 5 s. Normalized semiaxes are sampled independently from \([0.24,0.36]\), center coordinates from \([0.45,0.55]\), and the orientation is Haar-random. With the signed Euclidean distance \(d_E\) positive inside the ellipsoid, the initial profile is \(\tanh(d_E/(\sqrt{2}\epsilon^*))\). The native graph has no exterior exchange. The local FEM comparison uses homogeneous natural Neumann data.

<figure>
  <a href="assets/icemelting3d-setup.svg"><img src="assets/icemelting3d-setup.svg" alt="Three ellipsoids with different semiaxes, centers and orientations in the same cube"></a>
  <figcaption>Three prescribed initial ellipsoids. The cube and physical coefficients are fixed.</figcaption>
</figure>

## Governing equations and scaling

Let \(F(\phi)=(1-\phi^2)^2/4\). Its derivative \(\phi^3-\phi\) gives the double-well term. The driving term uses \(\sqrt{2F}=(1-\phi^2)/\sqrt{2}\) on \([-1,1]\).

$$
\partial_t\phi=M\left[\Delta\phi-\frac{\phi^3-\phi}{\epsilon^2}\right]
-\lambda\,\frac{1-\phi^2}{\sqrt{2}\epsilon}.
$$

Set \(\mathbf x^*=(\mathbf x+50)/L_c\), \(t^*=t/T_c\), \(\epsilon^*=\epsilon/L_c\), and \(\lambda^*=\lambda T_c/L_c\). The coefficients are

$$
N_1=\frac{MT_c}{L_c^2},\qquad
N_2=\frac{N_1}{(\epsilon^*)^2},\qquad
N_3=\frac{\lambda^*}{\sqrt{2}\epsilon^*}.
$$

The selected setup uses \(L_c=100\), \(T_c=5\,\mathrm{s}\) and \(\epsilon^*=0.0225\).

## Nonlocal residual and loss

Use \(D_\delta\approx-\Delta^*\). The physical residual is \(r_\phi=\phi_{t^*}+N_1D_\delta\phi+N_2(\phi^3-\phi)+N_3(1-\phi^2)\). A hard latent lift supplies the initial condition for every sampled shape.

$$
u_\theta(\mathbf x,t^*)=\frac{d_E(\mathbf x)}{\sqrt{2}\epsilon^*}
    +t^*N_\theta(\mathbf x,t^*),\qquad
\phi_\theta=\tanh u_\theta .
$$

Since \(\partial_{t^*}\phi=(1-\phi^2)\partial_{t^*}u\), the transformed residual at a node has the exact factorization \(r_\phi=(1-\phi_i^2)r_u\) for finite \(u_i\).

$$
(r_u)_i=\partial_{t^*}u_i+
N_1\sum_j w_{ij}
\frac{\sinh(u_i-u_j)\cosh u_i}{\cosh u_j}
-N_2\tanh u_i+N_3 .
$$

The edge formula follows from the difference of two hyperbolic tangents divided by the output Jacobian at node \(i\). Log-domain arithmetic evaluates the cosh ratio without dividing by a saturated rounded phase value. The loss minimizes the transformed absolute residual on the core nodes. The hard lift supplies the exact initial phase without an IC penalty.

## Field evolution

<figure>
  <a href="assets/icemelting3d.png"><img src="assets/icemelting3d.png" width="1800" height="1170" alt="Three predicted ice ellipsoids shrinking across five times"></a>
  <figcaption>Historical visualization from the earlier fixed-pool 5,000-step model. Held-out inputs 5, 9 and 2 at 0, 1.25, 2.5, 3.75 and 5 s. Surfaces are predicted φ = 0 isosurfaces extracted from 96³ visualization queries.</figcaption>
</figure>

## References and comparisons

The selected cold 10,000-step rolling model gives **0.1002% pooled / 0.1711% terminal phase L2** and **0.2908% / 8.4058% solid-indicator L2** on eight reference inputs. The native reference integrates the full cubic graph; neural error is measured on a 24³ diagnostic grid at five times.

| Pool, cold 10,000 updates | Pooled phase L2 | Terminal phase L2 | Warm effective median |
| --- | ---: | ---: | ---: |
| Fixed | 0.1203% | 0.1132% | 16.26 ms |
| Rolling, selected | 0.1002% | 0.1711% | 16.68 ms |

Both arms share runtime seed 43, initial model/Adam state, input shapes and time draws. Timings were measured serially on the same RTX 4090; active peaks are about 1.494 GiB. Late checkpoints fluctuate and their pooled medians favor fixed (0.1203% versus 0.1439%). The selected rolling recipe reflects the user's choice after this finite-budget comparison.

The historical 5,000-step optimizer comparison below uses the earlier fixed spatial pool.

The eight held-out shapes use time-refined direct nonlocal integration on the \(128^3\) graph. The field errors below pool five times on a \(24^3\) diagnostic grid. The optimizer control uses the same input draws and final budget.

| Mean relative error | Adam, historical 5k | SOAP, historical 5k |
| --- | ---: | ---: |
| Pooled phase \(L^2\) | 0.2266% | 0.0785% |
| Terminal phase \(L^2\) | 0.0922% | 0.1291% |
| Pooled solid-indicator \(L^2\) | 0.6425% | 0.2235% |
| Terminal solid-indicator \(L^2\) | 4.5017% | 6.5036% |
| Warm synchronized update | 22.43 ms | 33.66 ms |

In the separate one-shape local FEM check, the two references differ by 0.1474% pooled phase on the common nodes. Adam and SOAP give 0.1479% and 0.1861% against that FEM reference. The eight-shape table and the one-shape FEM check answer different questions. The approximate shrinking-sphere profile omits curvature effects and is unsuitable as the reference for arbitrary initial ellipsoids.

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Condition | \(32^3\) volume, \(4^3\) patches, 512 tokens |
| Model | Width 48, four heads, two encoder blocks, three time-FiLM layers |
| Spatial work | \(128^3\) graph, horizon 0.01, 256 cores plus full halo, 32 initial sets and a background rolling FIFO |
| Background pool | One CPU worker, 32 prefetched sets, 1,792 node slots / 1,536 edges |
| Samples per update | 16 fresh ellipsoids and 8 shared times |
| Optimizer | Adam, \(10^{-3}\) initial rate, 10,000 updates, ordinary FP32 |

~~~bash
source .venv/bin/activate
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python -m examples.icemelting3d.train \
  --config=examples/icemelting3d/configs/train_operator.py \
  --config.operator.action=prepare
python -m examples.icemelting3d.train \
  --config=examples/icemelting3d/configs/train_operator.py
~~~

Preparation creates a new reference bank at the configured path. See the [example source and full run notes](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/icemelting3d/README.md#pino-overview).
