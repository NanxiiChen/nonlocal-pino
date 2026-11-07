The operator maps a random rough metal–electrolyte interface to coupled phase and concentration trajectories. The geometry, material coefficients and boundary drive stay fixed.

## Setup and boundary data

The dimensionless rectangle is \([0,1]\times[0,0.5]\). The upper boundary prescribes electrolyte \((\phi,c)=(0,0)\), the lower boundary prescribes metal \((1,1)\), and the side walls close transport. A six-mode random height function generates a polyline interface. The initial diffuse phase uses its signed segment distance, positive below the profile, and \(c_0=c_{Se}h(\phi_0)\). Initial endpoint slopes are preserved.

<figure>
  <a href="assets/electropolish2d-setup.svg"><img src="assets/electropolish2d-setup.svg" alt="Three different initial metal profiles with the same rectangular geometry and boundary values"></a>
  <figcaption>The fixed rectangle and prescribed top and bottom values are shared by all profiles.</figcaption>
</figure>

## Governing equations and scaling

With \(\Delta c=c_{Se}-c_{Le}\), set

$$
h(\phi)=3\phi^2-2\phi^3,\qquad
g(\phi)=\phi^2(1-\phi)^2,\qquad
q=c-\Delta c\,h(\phi)-c_{Le}.
$$

The chemical free energy \(Aq^2\) gives the phase driving force \(2A\Delta c\,q h'(\phi)\) at fixed \(c\). The gradient and double-well terms give \(\alpha_\phi\Delta\phi-\omega_\phi g'(\phi)\). With \(L_c=100\,\mu\mathrm m\), \(T_c=5\,\mathrm s\), \(\mathbf x^*=\mathbf x/L_c\) and \(t^*=t/T_c\), the local equations are

$$
\begin{aligned}
\phi_{t^*}&=N_3\Delta^*\phi-N_2g'(\phi)+N_1\Delta c\,q h'(\phi),\\
c_{t^*}&=N_{\mathrm{CH}}\Delta^*q,
\end{aligned}
$$

where \(N_1=2ALT_c\), \(N_2=L\omega_\phi T_c\), \(N_3=L\alpha_\phi T_c/L_c^2\), and \(N_{\mathrm{CH}}=DT_c/L_c^2\).

## Nonlocal residual and loss

The bounded phase and unrestricted liquid concentration are mapped to the total concentration using the algebraic KKS mixture. Apply the graph to the physical phase and to \(q\).

$$
\begin{aligned}
r_{\mathrm{AC}}&=\phi_{t^*}+N_3D_\delta\phi+N_2g'(\phi)
                       -N_1\Delta c\,q h'(\phi),\\
r_{\mathrm{CH}}&=c_{t^*}+N_{\mathrm{CH}}D_\delta q .
\end{aligned}
$$

Mean absolute PDE residuals and three physical MSE terms (initial, top and bottom) give five losses. Inverse parameter-gradient norms balance these terms. The side walls use regional no-exchange closure, including for input profiles with incompatible initial normal slopes. The local-AD control enforces its side-wall derivatives through an additional positive-time penalty.

## Field evolution

<figure>
  <a href="assets/electropolish2d.png"><img src="assets/electropolish2d.png" width="2025" height="1095" alt="Phase evolution for four rough initial interfaces over five times"></a>
  <figcaption>Unseen inputs 0, 21, 42 and 63 at 0, 1.25, 2.5, 3.75 and 5 s. Dashed curves are local-FD101 phase interfaces.</figcaption>
</figure>

## References and comparisons

Both training arms are scored against the same independent centered local FD101 reference with mirrored side-wall closure, 64 unseen interfaces, \(101\times51\) nodes and 21 saved times. Finer FD201 and FD401 grids are checked on four declared inputs. The arms share the initial model and actual condition, time, core, IC and prescribed-boundary draws. The AD arm computes coordinate Hessian traces and adds the side-wall loss.

| Quantity | Nonlocal PINO | Local-AD PINO |
| --- | ---: | ---: |
| Phase \(L^2\), pooled / terminal | 4.427% / 3.176% | 8.595% / 5.498% |
| Concentration \(L^2\), pooled / terminal | 5.222% / 3.779% | 9.835% / 6.431% |
| Cold training process | 368.35 s | 693.03 s |
| First update, compilation plus call | 223.35 s | 533.87 s |
| Warm synchronized update mean | 16.71 ms | 19.35 ms |

The reported cold time includes setup and periodic evaluation. It excludes reference generation and the final 64-input test. On the direct nonlocal target, the selected graph model gives 4.392% pooled and 3.041% terminal phase error. The paired experiment measures this geometry family and one initialization.

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Condition | \(32\times64\) image, \(4\times8\) patches, 64 tokens |
| Model | Width 64, four heads, two encoder blocks, three time-FiLM layers |
| Spatial work | \(101\times51\) graph, horizon 0.02, 256 cores plus full halo, 32 fixed sets |
| Samples per update | 8 fresh initial profiles and 4 shared times |
| Optimizer | Adam, \(10^{-3}\) initial rate, 5,000 updates, ordinary FP32 |

~~~bash
source .venv/bin/activate
python -m examples.electropolish2d.train \
  --config=examples/electropolish2d/configs/train_operator.py
~~~

The preset expects its prepared monitoring bank. The independent native/local evaluation banks and local-AD control are documented in the [example and reference preparation guide](https://github.com/NanxiiChen/nonlocal-pinn/blob/main/examples/electropolish2d/README.md).
