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
  <a href="assets/electropolish2d.png"><img src="assets/electropolish2d.png" width="2007" height="1003" alt="Predicted phase for four rough initial interfaces across five times, with dashed local-FD reference contours and per-panel errors"></a>
  <figcaption>The selected rolling-pool 10,000-step model on unseen inputs 0, 21, 42 and 63 at t* = 0, 0.25, 0.5, 0.75 and 1 (0–5 s). Colors show predicted phase; dashed curves are local-FD101 φ = 0.5 interfaces. Labels give instantaneous spatial phase relative L² on the original reference nodes. <a href="assets/electropolish2d.pdf">Editable PDF</a>.</figcaption>
</figure>

## References and comparisons

The selected 10,000-step operator scores **1.547% pooled and 1.326% terminal phase error**, and **1.860% pooled and 1.580% terminal concentration error**. Evaluation uses an independent centered local FD101 reference with mirrored side-wall closure, 64 unseen interfaces, \(101\times51\) nodes and 21 saved times. The same checkpoint gives 1.637% pooled phase error on the four finer FD401 cases. The representative result uses the better observed final run of two paired seeds.

The current spatial study compares fixed and rolling pools with identical cold model and Adam state, initial profiles, times, IC/BC queries and safe padding. Both use width 96, 16 shared times and 10,000 updates.

| Runtime seed | Pool | Phase pooled / terminal | Concentration pooled / terminal | Warm step median |
| --- | --- | ---: | ---: | ---: |
| 42 | Fixed | 1.747% / 1.526% | 2.089% / 1.827% | 40.06 ms |
| 42 | Rolling | **1.547% / 1.326%** | **1.860% / 1.580%** | 41.61 ms |
| 43 | Fixed | 2.633% / 2.050% | 3.142% / 2.440% | 39.78 ms |
| 43 | Rolling | **2.465% / 1.672%** | **2.930% / 2.001%** | 41.93 ms |

All four arms ran serially on the same RTX 4090. Warm steps include synchronized condition preparation, spatial transfer and updates after the first 100 steps. Active peak memory is about 13.0 GiB. Rolling improves errors in both pairs with about 4–5% more step time. Seed-to-seed spread is larger than the pooled improvement within each pair.

The historical 5,000-step local-AD comparison used the same FD101 bank. Its two arms share the initial model and actual condition, time, core, IC and prescribed-boundary draws. The AD arm computes coordinate Hessian traces and adds the side-wall loss.

| Quantity | Nonlocal PINO, 5,000 steps | Local-AD PINO, 5,000 steps |
| --- | ---: | ---: |
| Phase \(L^2\), pooled / terminal | 4.427% / 3.176% | 8.595% / 5.498% |
| Concentration \(L^2\), pooled / terminal | 5.222% / 3.779% | 9.835% / 6.431% |
| Cold training process | 368.35 s | 693.03 s |
| First update, compilation plus call | 223.35 s | 533.87 s |
| Warm synchronized update mean | 16.71 ms | 19.35 ms |

The reported cold time includes setup and periodic evaluation. It excludes reference generation and the final 64-input test. On the direct nonlocal target, the historical 5,000-step graph model gives 4.392% pooled and 3.041% terminal phase error. The paired experiment measures this geometry family and one initialization.

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Condition | \(32\times64\) image, \(4\times8\) patches, 64 tokens |
| Model | Width 96, four heads, two encoder blocks, spatial cross-attention and three time-FiLM layers |
| Spatial work | \(101\times51\) graph, horizon 0.02, 256 cores plus full halo, 32 initial sets and a background rolling FIFO |
| Samples per update | 8 fresh initial profiles and 16 shared times |
| Background pool | One CPU worker, 32 prefetched sets, fixed 3,328-node / 3,072-edge capacities |
| Optimizer | Adam, \(10^{-3}\) initial rate, 10,000 updates, ordinary FP32 |

~~~bash
source .venv/bin/activate
python -m examples.electropolish2d.train \
  --config=examples/electropolish2d/configs/train_operator.py
~~~

The preset expects its prepared monitoring bank. The independent native/local evaluation banks and local-AD control are documented in the [example and reference preparation guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/electropolish2d/README.md).
