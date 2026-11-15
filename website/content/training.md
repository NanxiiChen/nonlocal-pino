The report uses a physics-only objective over a family of conditions. Initial and boundary data enter through exact output constraints or through explicit loss terms, depending on the case.

## Objective and constraints

Each residual uses the norm, scales and quadrature of its own recipe. A schematic objective is

$$
\mathcal L(\theta)
=\mathbb E_{a,t,\mathbf x}\!\left[
  \sum_{\ell}\lambda_\ell
  \,\ell_\ell\!\left(P_\ell r_\ell(\theta;a,\mathbf x,t)\right)
\right]+\mathcal L_{\mathrm{soft\ data}}.
$$

Here \(P_\ell\) denotes a case-specific residual scaling or transform. Soft-data terms constrain prescribed initial or boundary values. Hard lifts embed such values in the output and omit the matching loss term. Spinodal additionally projects onto a fixed physical-volume mean. Melting hard-imposes only the initial profile in its latent output and learns the subsequent interface motion.

## Sampling and reuse

- **Shared time queries.** Prepare spatial query features once per condition, then evaluate a fresh batch of times. Time JVP differentiates in physical or nondimensional time as specified by the model.
- **Core+halo.** Evaluate residuals on sampled core nodes and retain the full neighbors needed by those rows. Halo values support the stencil. Fixed padding preserves JIT shapes.
- **Rolling spatial pools.** Electropolish starts from 32 complete core+halo sets and replaces them using one background CPU worker. Paired runs at two seeds improve phase and concentration errors at about 4–5% more warm step time on the measured RTX 4090. The selected Melting recipe also rolls 256-core sets and trains for 10,000 updates. Its endpoint pooled error is lower than the paired fixed control, while terminal error and late-window medians favor fixed in that run.
- **Online inputs.** Fixed-domain cases draw fresh initial fields. Geometry cases roll a bounded pool of meshes, quadrature, known inputs and precomputed operators. Workers prepare these inputs independently of evolved reference fields.
- **Time microbatches.** The selected channel averages the two 18-time gradient blocks at fixed parameters before its single optimizer update. It still trains one global-time operator.

## Balancing and preconditioning

Electropolish and particle corrosion use **stopped inverse gradient-norm balancing**. A typical refresh forms weights proportional to the mean gradient norm divided by the term's clipped parameter-gradient norm, then smooths and caps them as configured. This keeps the weights outside the gradient of the weighted objective. Channel NS–CH uses fixed normalizers, with separate scales for a field output, its PDE row and its boundary residual.

Melting optimizes a residual scaled by the Jacobian of \(\phi=\tanh u\). The stable edge formula avoids division by the numerically saturated value of \(1-\phi^2\). Spinodal instead applies a frozen stable-part resolvent to its eliminated phase equation. Their objectives emphasize different residual components and magnitudes, so compare physical residuals or field errors when comparing recipes.

**Causal weighting** is available in research controls. For time-ordered loss segments \(\ell_k\), stopped weights reduce the influence of later times until earlier residuals improve.

$$
w_0=1,\qquad w_k=\exp\!\left(-\epsilon_c\sum_{i<k}\ell_i\right).
$$

All five selected recipes use their documented objectives with causal weighting disabled. See the [shared controller](https://github.com/NanxiiChen/nonlocal-pino/blob/main/src/nonlocal_pinn/training/causal.py) for its adaptation rule.

## Validation protocol

The case tables report means of per-condition relative errors. A representative pooled metric for one sample is

$$
e_{\mathrm{pool}}
=\left[
\frac{\sum_{k,i}\tau_k m_i\,\lvert\widehat u_{ki}-u^{\mathrm{ref}}_{ki}\rvert^2}
     {\sum_{k,i}\tau_k m_i\,\lvert u^{\mathrm{ref}}_{ki}\rvert^2}
\right]^{1/2}.
$$

Here \(\tau_k\) and \(m_i\) denote the evaluation time and spatial weights. The terminal metric restricts the sums to the final snapshot. The precise weights, masks and saved times belong to each case. Read interface or minority-phase metrics alongside whole-field \(L^2\) when the bulk dominates.

Direct nonlocal integration holds the spatial model fixed and checks temporal refinement. Independent local FD or FEM tests a different discretization of the local PDE. Their mismatch can be measured on common samples but cannot be removed by subtracting reported relative error magnitudes. Report compilation and first-update cost separately from warmed updates, and distinguish active memory from allocator reservation.

## Cold training wall-clock cost

Primary training cost covers the complete cold process: imports/device and model
initialization, in-process input/operator preparation, compilation, all updates
and rolling-pool waits, in-training evaluation, final output/checkpoint writing,
and teardown. Use the parent's timer from before process creation through exit;
warm-update measurements are auxiliary and must not replace this total.
Separately prepared input/reference banks, qualification probes, and separately
launched post-training evaluations/audits/plots are outside this interval.

| Example | Updates | Nonlocal | Local AD | FEM PINO |
|---|---:|---:|---:|---:|
| Electropolish | 10,000 | 15.63 min | 18.27 min | 15.44 min |
| Melting 3D | 10,000 | 6.89 min | 10.37 min | 7.35 min |
| Ceramic corrosion | 20,000 | 26.81 min | 28.47 min | 21.01 min |
| Two-phase flow | 20,000 | approximately 9.18 h* | 10.16 h | 8.03 h |
| Spinodal | 50,000 | 2.2464 h | Omitted after resource probes | 1.9035 h |

*The historical Nonlocal channel interval is reconstructed from external UTC
records (33,044.11 s), with approximately ten-second exit polling plus monitor
work and a timestamp immediately after process creation. It is approximate,
not a strict sub-second pre-spawn timer, and uses a different physical GPU from
the current AD/FEM controls. Ceramic's three controls share an older physical
RTX PRO 6000; Electropolish/Melting and current channel AD/FEM use the current
RTX PRO 6000. Same model name alone does not establish same-card timing.
The other table values use recorded pre-launch/post-exit parent timers.
Spinodal uses this same current physical GPU for both completed controls. Different examples retain their
declared monitoring cohorts; these totals include that evaluation workload.

For the two 10k temporal controls (both nonlocal), complete cold times are:

| Example | Spatial attention + decoder FiLM | Joint space-time query |
|---|---:|---:|
| Electropolish | 15.63 min | 23.65 min |
| Melting 3D | 6.89 min | 14.99 min |

[Exact timing provenance](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_training_wall_costs.json).
