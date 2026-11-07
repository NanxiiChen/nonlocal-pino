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
- **Online inputs.** Fixed-domain cases draw fresh initial fields. Geometry cases roll a bounded pool of meshes, quadrature, known inputs and precomputed operators. Workers prepare these inputs independently of evolved reference fields.
- **Time microbatches.** The selected channel averages the two 18-time gradient blocks at fixed parameters before its single optimizer update. It still trains one global-time operator.

## Balancing and preconditioning

Electropolish and particle corrosion use **stopped inverse gradient-norm balancing**. A typical refresh forms weights proportional to the mean gradient norm divided by the term's clipped parameter-gradient norm, then smooths and caps them as configured. This keeps the weights outside the gradient of the weighted objective. Channel NS–CH uses fixed normalizers, with separate scales for a field output, its PDE row and its boundary residual.

Melting optimizes a residual scaled by the Jacobian of \(\phi=\tanh u\). The stable edge formula avoids division by the numerically saturated value of \(1-\phi^2\). Spinodal instead applies a frozen stable-part resolvent to its eliminated phase equation. Their objectives emphasize different residual components and magnitudes, so compare physical residuals or field errors when comparing recipes.

**Causal weighting** is available in research controls. For time-ordered loss segments \(\ell_k\), stopped weights reduce the influence of later times until earlier residuals improve.

$$
w_0=1,\qquad w_k=\exp\!\left(-\epsilon_c\sum_{i<k}\ell_i\right).
$$

All five selected recipes use their documented objectives with causal weighting disabled. See the [shared controller](https://github.com/NanxiiChen/nonlocal-pinn/blob/main/src/nonlocal_pinn/training/causal.py) for its adaptation rule.

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
